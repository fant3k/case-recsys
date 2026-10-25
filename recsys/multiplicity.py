"""All confirmatory hypotheses together, from the saved per-user results (no models, no GPU).

The final test (H1, H2) and the new-cohort checks (N1, N2) were each corrected for two hypotheses (97.5% intervals);
N3a and N3b were added later. This recomputes every confirmatory interval with a Bonferroni correction over all six
hypotheses (H1, H2, N1, N2, N3a, N3b; H2 is one hypothesis made of two conditions, an intersection-union test), level
1 - 0.05/6, with the original bootstrap seeds and 10 000 draws. It first reproduces the published 97.5% intervals
from the same rows as a self-check. Writes research/results/multiplicity-2026-10-25.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "research/results/runs"
DRAWS, M = 10000, 6
FINAL_SEED, NEW_SEED = 20261025, 20261026


def interval(values, level, seed, draws=DRAWS):
    """Paired user bootstrap of a mean (same algorithm and seeds as final_test.interval / new_cohort.interval)."""
    v = np.asarray(values, dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, len(v), size=(draws, len(v)))
    tail = (1 - level) / 2
    lo, hi = np.quantile(v[idx].mean(1), [tail, 1 - tail])
    return {"mean": float(v.mean()), "level": level, "interval": [float(lo), float(hi)], "users": len(v)}


def paired_ndcg(rows, ref_rows, scale=1.0):
    if [r["uid"] for r in rows] != [r["uid"] for r in ref_rows]:
        raise ValueError("Paired users do not match")
    return [a["ndcg"] - scale * b["ndcg"] for a, b in zip(rows, ref_rows)]


def build_tests():
    final_rows = json.loads((RUNS / "final-test-20261023-01/ranking-per-user.json").read_text())
    final_reward = json.loads((RUNS / "final-test-20261023-01/reward-per-user.json").read_text())
    new_rows = json.loads((RUNS / "new-cohort-20261024-01/ranking-per-user.json").read_text())
    day2_rows = json.loads((RUNS / "new-cohort-confirmation-20261025-01/ranking-per-user.json").read_text())
    seconds = lambda name: np.asarray(final_reward[name]["judge_equivalent_seconds"], dtype=np.float64)
    # name: (values, seed, saved report, key path, lower bound must be > 0 (or >= 0 for the margin))
    return {
        "H1_A_minus_reference_ndcg": (paired_ndcg(final_rows["A_rm_rerank_top50"], final_rows["reference"]), FINAL_SEED,
                                      "final-test-20261023-01", ("confirmatory", "H1_ndcg_delta_A_minus_reference")),
        "H2_judge_equivalent_seconds_B_minus_reference": (seconds("B_rl_seconds_beta0.3") - seconds("reference"), FINAL_SEED,
                                                          "final-test-20261023-01", ("confirmatory", "H2_judge_equivalent_seconds_delta")),
        "H2_margin_B_minus_0.95_reference_ndcg": (paired_ndcg(final_rows["B_rl_seconds_beta0.3"], final_rows["reference"], 0.95), FINAL_SEED,
                                                  "final-test-20261023-01", ("confirmatory", "H2_margin_B_minus_0.95_reference")),
        "N1_A_minus_reference_new_users": (paired_ndcg(new_rows["A_rm_rerank_top50"], new_rows["reference"]), NEW_SEED,
                                           "new-cohort-20261024-01", ("confirmatory", "N1_A_minus_reference")),
        "N2_E1_head_minus_reference_new_users": (paired_ndcg(new_rows["E1_head_all_features"], new_rows["reference"]), NEW_SEED,
                                                 "new-cohort-20261024-01", ("confirmatory", "N2_E1_head_minus_reference")),
        "N3a_A_minus_reference_other_day": (paired_ndcg(day2_rows["A_rm_rerank_top50"], day2_rows["reference"]), NEW_SEED,
                                            "new-cohort-confirmation-20261025-01", ("confirmatory", "N3a_A_minus_reference")),
        "N3b_E1_head_minus_reference_other_day": (paired_ndcg(day2_rows["E1_head_all_features"], day2_rows["reference"]), NEW_SEED,
                                                  "new-cohort-confirmation-20261025-01", ("confirmatory", "N3b_E1_head_minus_reference")),
    }


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    level_all = 1 - 0.05 / M
    out = {"hypotheses": ["H1", "H2 (two conditions)", "N1", "N2", "N3a", "N3b"], "m": M, "level_all_together": level_all,
           "draws": DRAWS, "seeds": {"final": FINAL_SEED, "new_cohort": NEW_SEED}, "tests": {}}
    for name, (values, seed, run, path) in build_tests().items():
        saved = json.loads((RUNS / run / "report.json").read_text())
        for key in path:
            saved = saved[key]
        published = interval(values, saved["level"], seed)
        # The self-check: the published 97.5% interval is reproduced exactly from the same rows.
        if abs(published["mean"] - saved["mean"]) > 1e-9 or any(abs(a - b) > 1e-9 for a, b in zip(published["interval"], saved["interval"])):
            raise ValueError("Published interval is not reproduced for " + name)
        together = interval(values, level_all, seed)
        margin = name.startswith("H2_margin")
        out["tests"][name] = {"published_975": published, "all_six_together": together,
                              "lower_bound_passes": bool(together["interval"][0] >= 0 if margin else together["interval"][0] > 0)}
    out["all_pass"] = all(t["lower_bound_passes"] for t in out["tests"].values())
    path = ROOT / "research/results/multiplicity-2026-10-25.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    for name, t in out["tests"].items():
        lo, hi = t["all_six_together"]["interval"]
        print(f"{name:48s} {t['all_six_together']['mean']:+.4f} [{lo:+.4f}; {hi:+.4f}] passes={t['lower_bound_passes']}")
    print("all pass:", out["all_pass"])


if __name__ == "__main__":
    main()
