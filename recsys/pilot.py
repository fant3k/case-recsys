"""Bounded end-to-end pilot; selection validation only, never final test."""

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import statistics
import sys
import time
import traceback
import zipfile

import numpy as np
import torch
from torch.nn import functional as F
import transformers

from .data import OFFSET, load_sequences, pack_contexts, personal_frequency
from .metrics import ranking_metrics
from .models import Policy, RewardModel, exact_kl, policy_loss, state_hash


ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    temporary.replace(path)


def source_manifest():
    """SHA256 of the shipped sources: the SSH transfer manifest when present, otherwise computed locally."""
    transferred = ROOT/"transfer-manifest.json"
    if transferred.exists():
        return json.loads(transferred.read_text())
    manifest = {}
    for folder in ["recsys", "tests", "scripts", "research/protocol"]:
        for source_file in sorted((ROOT/folder).rglob("*")):
            if source_file.is_file() and "__pycache__" not in source_file.parts:
                manifest[source_file.relative_to(ROOT).as_posix()] = hashlib.sha256(source_file.read_bytes()).hexdigest()
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--sft-steps", type=int, default=80)
    parser.add_argument("--rm-steps", type=int, default=200)
    parser.add_argument("--rl-steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--reward-users", type=int, default=64)
    parser.add_argument("--max-seconds", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = ROOT/args.out
    out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(out/"run.log", encoding="utf-8")])
    started = time.perf_counter()
    report = {"status": "running", "kind": "technical_pilot_not_final_experiment", "started_utc": datetime.now(timezone.utc).isoformat(), "arguments": vars(args), "final_test_scored": False, "full_protocol_complete": False}
    atomic_json(out/"status.json", report)

    def budget():
        if time.perf_counter()-started > args.max_seconds:
            raise TimeoutError("Pilot wall-time budget reached")

    def note(message):
        logging.info(message)
        report["last_event"] = message
        report["elapsed_seconds"] = time.perf_counter()-started
        atomic_json(out/"status.json", report)

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; refusing to silently train on CPU")
        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats()
        report["environment"] = {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__, "cuda_build": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "vram_bytes": torch.cuda.get_device_properties(0).total_memory}
        protocol_file = ROOT/"research/protocol/case2-v1.json"
        protocol = json.loads(protocol_file.read_text())
        report["protocol_sha256"] = hashlib.sha256(protocol_file.read_bytes()).hexdigest()
        report["protocol_id"] = protocol["protocol_id"]
        atomic_json(out/"configuration.json", {"protocol": protocol, "pilot": vars(args), "environment": report["environment"]})
        atomic_json(out/"source-manifest.json", source_manifest())
        with zipfile.ZipFile(out/"code-snapshot.zip", "w", compression=zipfile.ZIP_DEFLATED) as snapshot:
            for folder in ["recsys", "tests", "scripts", "research/protocol"]:
                for source_file in (ROOT/folder).glob("*"):
                    if source_file.is_file():
                        snapshot.write(source_file, source_file.relative_to(ROOT))
            snapshot.write(ROOT/"requirements-windows.txt", "requirements-windows.txt")
        note("Preparing past-only sequences and train-only catalog")
        load_start = time.perf_counter()
        data, features, item_means, global_mean, support = load_sequences(
            ROOT/protocol["data"]["sample_path"], protocol["time"]["train_end_exclusive"], protocol["catalog"]["size"]
        )
        support["preparation_seconds"] = time.perf_counter()-load_start
        report["data"] = support
        atomic_json(out/"data-support.json", support)
        cutoff = protocol["time"]["query_cutoffs"]["selection_validation"]
        label_start = cutoff+protocol["time"]["gap_seconds"]
        label_end = label_start+protocol["time"]["horizon_seconds"]
        uids, query_x, query_mask, raw_contexts = data.queries(cutoff)
        truth = data.truth(label_start, label_end)
        report["selection_validation"] = {"query_users": len(uids), "ranking_users": len(set(uids)&truth.keys()), "cutoff": cutoff, "label_start": label_start, "label_end_exclusive": label_end}
        rng = np.random.default_rng(args.seed)

        def tensors(batch):
            return tuple(torch.as_tensor(a, device=device) for a in batch[:4])

        def amp():
            return torch.autocast(device_type="cuda", dtype=torch.float16)

        def update(loss, optimizer, scaler, parameters):
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            return scaler.get_scale() < previous_scale

        def train_stage(name, steps, iteration):
            times, losses, skips = [], [], 0
            torch.cuda.reset_peak_memory_stats()
            note(f"Starting {name}: {steps} steps")
            for step in range(steps):
                budget()
                torch.cuda.synchronize()
                t = time.perf_counter()
                loss, skipped = iteration(step)
                torch.cuda.synchronize()
                times.append(time.perf_counter()-t)
                losses.append(float(loss))
                skips += int(skipped)
                if (step+1) % max(10, steps//20) == 0 or step+1 == steps:
                    note(f"{name} {step+1}/{steps}, loss={losses[-1]:.5f}, step={times[-1]:.3f}s")
            result = {"steps_attempted": steps, "optimizer_steps_skipped_by_scaler": skips, "optimizer_updates": steps-skips, "seconds": sum(times), "median_step_seconds_excluding_first_five": statistics.median(times[5:] or times), "first_ten_mean_loss": float(np.mean(losses[:10])), "last_ten_mean_loss": float(np.mean(losses[-10:])), "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
            atomic_json(out/(name+"-trace.json"), {"summary": result, "losses": losses, "step_seconds": times})
            report.setdefault("training", {})[name] = result
            return result

        policy = Policy(len(data.catalog)).to(device)
        report["policy_parameters"] = sum(p.numel() for p in policy.parameters())
        optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=0.01)
        scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)

        def sft_iteration(_):
            x, mask, actions, _ = tensors(data.sample(rng, args.batch_size))
            with amp():
                loss = F.cross_entropy(policy(x, mask).float(), actions)
            return loss.detach().item(), update(loss, optimizer, scaler, policy.parameters())

        train_stage("sft", args.sft_steps, sft_iteration)
        policy.eval()
        sft_hash = state_hash(policy)
        checkpoint = {"model": policy.state_dict(), "settings": policy.settings, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(), "sampler_rng": rng.bit_generator.state, "catalog": torch.from_numpy(data.catalog), "protocol_sha256": report["protocol_sha256"], "step": args.sft_steps}
        torch.save(checkpoint, out/"sft-checkpoint.tmp")
        (out/"sft-checkpoint.tmp").replace(out/"sft-checkpoint.pt")
        probe_x = torch.as_tensor(query_x[:2], device=device)
        probe_mask = torch.as_tensor(query_mask[:2], device=device)
        with torch.no_grad(), amp():
            before_reload = policy(probe_x, probe_mask).float()
        restored = torch.load(out/"sft-checkpoint.pt", map_location=device, weights_only=True)
        policy.load_state_dict(restored["model"])
        with torch.no_grad(), amp():
            torch.testing.assert_close(before_reload, policy(probe_x, probe_mask).float())
        restored_optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4)
        restored_optimizer.load_state_dict(restored["optimizer"])
        if len(restored_optimizer.state) != len(optimizer.state):
            raise AssertionError("Optimizer state did not restore")
        del restored_optimizer, restored, checkpoint, optimizer, scaler
        report["checkpoint_reload"] = "PASS: predictions and optimizer state restored"
        note("GPT-2 checkpoint saved and reloaded successfully")

        @torch.no_grad()
        def predict(model, x=query_x, mask=query_mask):
            result = []
            model.eval()
            for start in range(0, len(x), args.batch_size):
                budget()
                with amp():
                    logits = model(torch.as_tensor(x[start:start+args.batch_size], device=device), torch.as_tensor(mask[start:start+args.batch_size], device=device)).float()
                result.extend(data.catalog[logits.topk(10, dim=-1).indices.cpu().numpy()].tolist())
            return result

        predictions = {"MostPop": [data.catalog[:10].tolist() for _ in uids], "PersonalFrequency128": personal_frequency(raw_contexts, data.catalog, 10), "GPT2_SFT_pilot": predict(policy)}
        next_uids, next_x, next_mask, next_truth = data.next_group_queries(label_start, label_end)
        next_predictions = {"GPT2_SFT_pilot": predict(policy, next_x, next_mask)}
        note("Selection-window ranking evaluated; confirmation and final test untouched")

        rm = RewardModel(len(data.catalog), features).to(device)
        rm_optimizer = torch.optim.AdamW(rm.parameters(), lr=1e-3, weight_decay=0.01)
        rm_scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
        rm_rng = np.random.default_rng(args.seed+1)

        def rm_iteration(_):
            x, mask, actions, response = tensors(data.sample(rm_rng, args.batch_size*4))
            with amp():
                loss = F.mse_loss(rm(x, mask, actions).float(), response)
            return loss.detach().item(), update(loss, rm_optimizer, rm_scaler, rm.parameters())

        train_stage("reward_model", args.rm_steps, rm_iteration)
        rm.eval()
        val = data.sample(np.random.default_rng(args.seed+2), 2048, bounds=(label_start, label_end))
        vx, vm, va, vr = tensors(val)
        with torch.no_grad(), amp():
            rm_pred = rm(vx, vm, va).float().cpu().numpy()
            permutation = torch.randperm(len(vx), device=device)
            shuffled_pred = rm(vx[permutation], vm[permutation], va).float().cpu().numpy()
        rm_result = {"scope": "Selection observed in-catalog sampled events; user/group-balanced with replacement; not all heldout events", "sampled_records": len(val[-1]), "unique_user_event_records": len(set((uid, event) for uid, event, _ in val[-1]))}
        def regression(p):
            err = np.asarray(p)-val[3]
            users = {}
            for (uid, _, _), error in zip(val[-1], err):
                users.setdefault(uid, []).append(abs(float(error)))
            return {"MAE_sample_weighted": float(np.mean(np.abs(err))), "RMSE_sample_weighted": float(np.sqrt(np.mean(err**2))), "mean_prediction_bias": float(np.mean(err)), "MAE_macro_sampled_users": float(np.mean([np.mean(v) for v in users.values()]))}
        rm_result.update({"constant_train_mean": regression(np.full(len(val[3]), global_mean)), "item_train_mean": regression(item_means[val[2]]), "history_RM": regression(rm_pred), "history_RM_with_shuffled_histories": regression(shuffled_pred)})
        report["reward_validation"] = rm_result
        atomic_json(out/"reward-validation.json", rm_result)
        torch.save({"model": rm.state_dict(), "width": 64, "catalog": torch.from_numpy(data.catalog)}, out/"reward-model.pt")
        del rm_optimizer, rm_scaler
        for parameter in rm.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        rm_hash = state_hash(rm)
        reference = copy.deepcopy(policy).eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        reference_hash = state_hash(reference)

        matched_rng = np.random.default_rng(args.seed+3)
        matched_batches = [data.sample(matched_rng, args.batch_size) for _ in range(args.rl_steps)]
        continued = copy.deepcopy(policy)
        optimizer_more = torch.optim.AdamW(continued.parameters(), lr=1e-4, weight_decay=0.01)
        scaler_more = torch.amp.GradScaler("cuda", init_scale=1024.0)
        def continue_iteration(step):
            x, mask, actions, _ = tensors(matched_batches[step])
            with amp():
                loss = F.cross_entropy(continued(x, mask).float(), actions)
            return loss.detach().item(), update(loss, optimizer_more, scaler_more, continued.parameters())
        train_stage("continued_sft", args.rl_steps, continue_iteration)
        predictions["GPT2_continued_SFT_pilot"] = predict(continued)
        next_predictions["GPT2_continued_SFT_pilot"] = predict(continued, next_x, next_mask)
        continued_state = {k: v.detach().cpu() for k, v in continued.state_dict().items()}
        torch.save({"model": continued_state, "settings": continued.settings}, out/"continued-sft.pt")
        del continued, optimizer_more, scaler_more

        optimizer_rl = torch.optim.AdamW(policy.parameters(), lr=1e-4, weight_decay=0.01)
        scaler_rl = torch.amp.GradScaler("cuda", init_scale=1024.0)
        baseline = global_mean
        rl_trace = []
        def rl_iteration(step):
            nonlocal baseline
            x, mask, _, _ = tensors(matched_batches[step])
            with amp():
                logits = policy(x, mask).float()
                with torch.no_grad():
                    reference_logits = reference(x, mask).float()
                    actions = torch.distributions.Categorical(logits=logits).sample()
                    rewards = rm(x, mask, actions).float()
                loss, kl = policy_loss(logits, reference_logits, actions, rewards, baseline, 0.03)
            skipped = update(loss, optimizer_rl, scaler_rl, policy.parameters())
            observed = rewards.mean().item()
            baseline = 0.95*baseline+0.05*observed
            rl_trace.append({"step": step+1, "predicted_reward": observed, "exact_kl": kl.item(), "baseline_for_next_step": baseline})
            return loss.detach().item(), skipped
        train_stage("rl_completion_beta003", args.rl_steps, rl_iteration)
        if state_hash(reference) != reference_hash or state_hash(rm) != rm_hash:
            raise AssertionError("Frozen reference or reward model changed")
        if any(p.grad is not None for p in list(reference.parameters())+list(rm.parameters())):
            raise AssertionError("Gradient reached frozen components")
        if state_hash(policy) == sft_hash:
            raise AssertionError("RL did not update policy weights")
        report["frozen_components_check"] = "PASS: hashes identical and gradients absent"
        torch.save({"model": policy.state_dict(), "settings": policy.settings, "optimizer": optimizer_rl.state_dict(), "scaler": scaler_rl.state_dict(), "steps": args.rl_steps, "baseline": baseline, "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(), "protocol_sha256": report["protocol_sha256"]}, out/"rl-checkpoint.pt")
        atomic_json(out/"rl-reward-trace.json", rl_trace)
        predictions["GPT2_RL_pilot"] = predict(policy)
        next_predictions["GPT2_RL_pilot"] = predict(policy, next_x, next_mask)

        rows = {}
        report["ranking"] = {}
        report["next_group_ranking"] = {}
        report["recommendation_behavior"] = {}
        catalog_action = {int(item): j for j, item in enumerate(data.catalog)}
        for name, recs in predictions.items():
            score, user_rows = ranking_metrics(uids, recs, truth)
            report["ranking"][name] = score
            rows[name] = user_rows
            counts = Counter(item for ranking in recs for item in ranking)
            total = sum(counts.values())
            report["recommendation_behavior"][name] = {
                "query_users": len(uids),
                "unique_recommended_items": len(counts),
                "catalog_coverage": len(counts)/len(data.catalog),
                "most_frequent_item_share": counts.most_common(1)[0][1]/total,
                "top10_items_share": sum(n for _, n in counts.most_common(10))/total,
                "seen_in_last128_fraction": sum(item in set(map(int, h)) for h, ranking in zip(raw_contexts, recs) for item in ranking)/total,
                "mean_capped_length_seconds": float(np.mean([features[catalog_action[item], 0]*600 for ranking in recs for item in ranking])),
            }
        for name, recs in next_predictions.items():
            report["next_group_ranking"][name] = ranking_metrics(next_uids, recs, next_truth)[0]
        atomic_json(out/"ranking-per-user.json", rows)
        atomic_json(out/"rankings.json", {"query_users": uids, "rankings": predictions})

        # Predetermined user subset, independent of future activity; reuse RM scores.
        reward_indices = np.sort(np.random.default_rng(args.seed+4).choice(len(uids), min(args.reward_users, len(uids)), replace=False))
        reward_x = torch.as_tensor(query_x[reward_indices], device=device)
        reward_mask = torch.as_tensor(query_mask[reward_indices], device=device)
        note("Computing exact full-catalog reward on a predefined pilot user subset")
        with torch.no_grad(), amp():
            all_rewards = rm.all_scores(reward_x, reward_mask).float()
            ref_logits = reference(reward_x, reward_mask).float()
            rl_logits = policy(reward_x, reward_mask).float()
            continued_eval = Policy(**policy.settings).to(device)
            continued_eval.load_state_dict(continued_state)
            continued_eval.eval()
            more_logits = continued_eval(reward_x, reward_mask).float()
        log_ref = F.log_softmax(ref_logits, dim=-1)
        tilted_logits = log_ref+all_rewards/0.03
        reward_summary = {"scope": "Pilot subset, not the full-protocol all-warm-user reward evaluation", "users": [uids[j] for j in reward_indices], "beta": 0.03, "reward": "completion", "models": {}}
        for name, logits in [("SFT", ref_logits), ("continued_SFT", more_logits), ("RL", rl_logits), ("analytic_reweighting", tilted_logits)]:
            lp = F.log_softmax(logits, dim=-1)
            expected = (lp.exp()*all_rewards).sum(-1)
            kl = exact_kl(logits, ref_logits)
            top = logits.topk(10, dim=-1).indices
            reward_summary["models"][name] = {"expected_predicted_reward": expected.mean().item(), "top10_predicted_reward": all_rewards.gather(1, top).mean().item(), "exact_kl_to_SFT": kl.mean().item(), "regularized_objective": (expected-0.03*kl).mean().item()}
        oracle = reward_summary["models"]["analytic_reweighting"]["regularized_objective"]
        if any(v["regularized_objective"] > oracle+1e-4 for v in reward_summary["models"].values()):
            raise AssertionError("A policy exceeded the exact H1 proxy optimum")
        report["reward_policy_comparison"] = reward_summary
        atomic_json(out/"reward-policy-comparison.json", reward_summary)

        rollouts = {}
        for name, model in [("SFT", reference), ("RL", policy)]:
            torch.manual_seed(args.seed+100)
            contexts = [query_x[j][query_mask[j].astype(bool)].tolist() for j in reward_indices[:16]]
            generated, step_rewards = [[] for _ in contexts], []
            for step in range(5):
                budget()
                xx, mm = pack_contexts(contexts, 128)
                tx, tm = torch.as_tensor(xx, device=device), torch.as_tensor(mm, device=device)
                with torch.no_grad(), amp():
                    actions = torch.distributions.Categorical(logits=model(tx, tm).float()).sample()
                    step_rewards.append(rm(tx, tm, actions).float().mean().item())
                for j, action in enumerate(actions.cpu().tolist()):
                    generated[j].append(int(data.catalog[action]))
                    contexts[j].append(action+OFFSET)
                    contexts[j] = contexts[j][-128:]
            rollouts[name] = {"hypothetical_recommendations_only": True, "generated_item_ids": generated, "mean_predicted_reward_by_step": step_rewards, "within_rollout_repeat_fraction": sum(len(g)-len(set(g)) for g in generated)/(len(generated)*5)}
        report["h5_diagnostic"] = rollouts
        atomic_json(out/"h5-diagnostic.json", rollouts)
        report["cautions"] = ["Short training only: these are pilot diagnostics, not final model-quality conclusions.", "Only completion reward and beta=0.03 trained; the full planned matrix and item-only neural RM ablation remain.", "Reward expectation uses a predefined subset, unlike the main protocol's all-warm-user estimate.", "No confirmation or final-test model scores computed; no confidence intervals or replicated seeds yet.", "H5 recommendations are not observed user actions."]
        report["status"] = "completed"
        report["elapsed_seconds"] = time.perf_counter()-started
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(out/"report.json", report)
        atomic_json(out/"status.json", report)
        note("PILOT COMPLETED")
    except BaseException:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        report["elapsed_seconds"] = time.perf_counter()-started
        atomic_json(out/"status.json", report)
        logging.exception("Pilot failed; existing checkpoints and logs preserved")
        raise


if __name__ == "__main__":
    main()
