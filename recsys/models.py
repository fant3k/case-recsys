import hashlib

import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2Config, GPT2Model

from .data import OFFSET


class Policy(nn.Module):
    def __init__(self, vocab_size, context_size=128, width=256, layers=4, heads=4, dropout=0.0):
        super().__init__()
        self.settings = dict(vocab_size=vocab_size, context_size=context_size, width=width, layers=layers, heads=heads)
        if dropout:
            self.settings["dropout"] = dropout
        config = GPT2Config(
            vocab_size=vocab_size + OFFSET, n_positions=context_size, n_ctx=context_size,
            n_embd=width, n_layer=layers, n_head=heads, resid_pdrop=dropout,
            embd_pdrop=dropout, attn_pdrop=dropout, use_cache=False,
            bos_token_id=2, eos_token_id=None, pad_token_id=0,
        )
        config._attn_implementation = "sdpa"
        self.transformer = GPT2Model(config)

    def forward(self, x, mask):
        hidden = self.transformer(input_ids=x, attention_mask=mask, use_cache=False).last_hidden_state
        last = hidden[torch.arange(len(x), device=x.device), mask.sum(1) - 1]
        return F.linear(last, self.transformer.wte.weight[OFFSET:])


RATIO_BUCKETS, GAP_BUCKETS = 5, 22


def pps_bias(rows, positions, size, epsilon):
    """ln(1 + c/epsilon) for items counted in rows[:, :position+1]; zero for unseen items.

    Equal, up to a per-row constant that cancels in softmax, to the PPS score
    ln((c+epsilon)/max(C+epsilon)) of Abbattista et al. (RecSys 2024).
    """
    length = rows.shape[1]
    valid = (torch.arange(length, device=rows.device)[None] <= positions[:, None]) & (rows >= OFFSET)
    counts = torch.zeros(len(rows), size, device=rows.device, dtype=torch.float32)
    counts.scatter_add_(1, (rows - OFFSET).clamp(min=0), valid.float())
    return torch.log1p(counts / epsilon)


class FeedbackPolicy(Policy):
    """GPT-2 policy with optional PPS logit prior and past-feedback input embeddings."""

    def __init__(self, vocab_size, context_size=128, width=256, layers=4, heads=4, dropout=0.0,
                 pps=False, feedback=False, pps_epsilon=0.01):
        super().__init__(vocab_size, context_size, width, layers, heads, dropout)
        self.settings.update(pps=pps, feedback=feedback, pps_epsilon=pps_epsilon)
        if feedback:
            self.ratio_embedding = nn.Embedding(RATIO_BUCKETS + 1, width, padding_idx=0)
            self.gap_embedding = nn.Embedding(GAP_BUCKETS + 1, width, padding_idx=0)
            for e in (self.ratio_embedding, self.gap_embedding):
                nn.init.normal_(e.weight, std=0.02)
                with torch.no_grad():
                    e.weight[0].zero_()

    def states(self, x, mask, ratio=None, gap=None):
        if self.settings["feedback"]:
            embeds = self.transformer.wte(x) + self.ratio_embedding(ratio) + self.gap_embedding(gap)
            return self.transformer(inputs_embeds=embeds, attention_mask=mask, use_cache=False).last_hidden_state
        return self.transformer(input_ids=x, attention_mask=mask, use_cache=False).last_hidden_state

    def forward(self, x, mask, ratio=None, gap=None):
        hidden = self.states(x, mask, ratio, gap)
        position = mask.sum(1) - 1
        logits = F.linear(hidden[torch.arange(len(x), device=x.device), position], self.transformer.wte.weight[OFFSET:])
        if self.settings["pps"]:
            logits = logits.float() + pps_bias(x, position, logits.shape[1], self.settings["pps_epsilon"])
        return logits


def make_policy(settings):
    return FeedbackPolicy(**settings) if "pps" in settings else Policy(**settings)


class RewardModel(nn.Module):
    def __init__(self, vocab_size, features, width=64, use_history=True):
        super().__init__()
        self.use_history = use_history
        self.embedding = nn.Embedding(vocab_size + OFFSET, width, padding_idx=0)
        self.context = nn.Sequential(nn.Linear(width * 2, width), nn.Tanh())
        self.scorer = nn.Sequential(nn.Linear(width * 3 + 2, 128), nn.ReLU(), nn.Linear(128, 1))
        self.register_buffer("features", torch.as_tensor(features, dtype=torch.float32))

    def encode(self, x, mask):
        e = self.embedding(x)
        mean = (e * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        last = e[torch.arange(len(x), device=x.device), mask.sum(1)-1]
        h = self.context(torch.cat([mean, last], dim=-1))
        return h if self.use_history else torch.zeros_like(h)

    def score(self, h, actions):
        candidate = self.embedding(actions + OFFSET)
        f = self.features[actions]
        return torch.sigmoid(self.scorer(torch.cat([h, candidate, h * candidate, f], dim=-1))).squeeze(-1)

    def forward(self, x, mask, actions):
        return self.score(self.encode(x, mask), actions)

    def all_scores(self, x, mask, chunk_size=1024):
        h = self.encode(x, mask)
        outputs = []
        for start in range(0, len(self.features), chunk_size):
            actions = torch.arange(start, min(start+chunk_size, len(self.features)), device=x.device)
            candidates = self.embedding(actions + OFFSET).unsqueeze(0).expand(len(x), -1, -1)
            history = h.unsqueeze(1).expand_as(candidates)
            f = self.features[actions].unsqueeze(0).expand(len(x), -1, -1)
            inputs = torch.cat([history, candidates, history*candidates, f], dim=-1)
            outputs.append(torch.sigmoid(self.scorer(inputs)).squeeze(-1))
        return torch.cat(outputs, dim=1)


def exact_kl(logits, reference_logits):
    logp = F.log_softmax(logits.float(), dim=-1)
    logq = F.log_softmax(reference_logits.float(), dim=-1)
    return (logp.exp() * (logp - logq)).sum(-1)


def policy_loss(logits, reference_logits, actions, rewards, baseline, beta):
    logp = F.log_softmax(logits.float(), dim=-1)
    chosen = logp.gather(1, actions[:, None]).squeeze(1)
    reinforce = -((rewards.detach() - baseline) * chosen).mean()
    kl = exact_kl(logits, reference_logits.detach()).mean()
    return reinforce + beta * kl, kl


def state_hash(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
