"""Speculative decoding (Leviathan et al. 2023 / Chen et al. 2023).

A cheap draft model proposes `k` tokens; the expensive target model verifies all `k` in a
*single* forward pass. Each proposal x_i is accepted with probability min(1, p_i/q_i) where p
is the target dist and q the draft dist; the first rejection resamples from the normalized
residual max(0, p−q). This provably samples from exactly the target distribution while doing
~1 target forward per several tokens — the latency win for memory-bound decode.

Cache discipline: after each round the last committed token is intentionally left
"unprocessed" by both models and re-fed at the top of the next round, so a rejected
speculation simply gets overwritten in the KV cache (writes are addressed by start_pos).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _dist(logits: torch.Tensor, temperature: float, top_k: int | None) -> torch.Tensor:
    """Return a probability distribution (one-hot argmax when temperature == 0)."""
    logits = logits.squeeze(0).squeeze(0) if logits.dim() == 3 else logits.squeeze(0)
    if temperature <= 0.0:
        out = torch.zeros_like(logits)
        out[logits.argmax(-1)] = 1.0
        return out
    logits = logits / temperature
    if top_k is not None:
        k = min(top_k, logits.shape[-1])
        thresh = torch.topk(logits, k).values[-1]
        logits = logits.masked_fill(logits < thresh, float("-inf"))
    return F.softmax(logits, dim=-1)


def _sample(dist: torch.Tensor) -> int:
    return int(torch.multinomial(dist, 1).item())


@dataclass
class SpecResult:
    tokens: list[int]
    num_accepted: int
    num_proposed: int

    @property
    def acceptance_rate(self) -> float:
        return self.num_accepted / max(1, self.num_proposed)


@torch.no_grad()
def speculative_generate(
    target,
    draft,
    prompt_tokens: list[int],
    max_new_tokens: int,
    n_draft: int = 4,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> SpecResult:
    device = target.device
    out = list(prompt_tokens)
    t_cache = target.new_cache(1)
    d_cache = draft.new_cache(1)
    t_proc = 0  # positions already in the target KV cache
    d_proc = 0
    n_acc = n_prop = 0

    def feed(model, cache, tokens, start):
        x = torch.tensor([tokens], device=device, dtype=torch.long)
        return model(x, cache=cache, start_pos=start)

    while len(out) - len(prompt_tokens) < max_new_tokens:
        # 1. Target processes everything committed but not yet seen -> p1 (next-token dist).
        t_logits = feed(target, t_cache, out[t_proc:], t_proc)
        p1 = _dist(t_logits[:, -1], temperature, top_k)
        t_proc = len(out)

        # 2. Draft proposes n_draft tokens, recording q_i.
        d_logits = feed(draft, d_cache, out[d_proc:], d_proc)
        d_proc = len(out)
        q_dists, proposals = [], []
        cur = _dist(d_logits[:, -1], temperature, top_k)
        for i in range(n_draft):
            q_dists.append(cur)
            x = _sample(cur)
            proposals.append(x)
            if i < n_draft - 1:
                nxt = feed(draft, d_cache, [x], d_proc)
                d_proc += 1
                cur = _dist(nxt[:, -1], temperature, top_k)

        # 3. Target verifies all proposals in one pass -> p2..p_{k+1}.
        t_seq = feed(target, t_cache, proposals, t_proc)  # (1, k, V)
        p_dists = [p1] + [_dist(t_seq[:, j], temperature, top_k) for j in range(n_draft - 1)]
        p_bonus = _dist(t_seq[:, n_draft - 1], temperature, top_k)

        # 4. Accept / reject.
        accepted = 0
        rejected_token: int | None = None
        for i in range(n_draft):
            n_prop += 1
            xi = proposals[i]
            p_i, q_i = p_dists[i][xi], q_dists[i][xi]
            ratio = (p_i / q_i).clamp(max=1.0) if q_i > 0 else torch.zeros(())
            if torch.rand((), device=device) <= ratio:
                accepted += 1
                n_acc += 1
            else:
                residual = torch.clamp(p_dists[i] - q_dists[i], min=0.0)
                residual = residual / residual.sum().clamp(min=1e-8)
                rejected_token = _sample(residual)
                break

        new_tokens = proposals[:accepted]
        if rejected_token is not None:
            new_tokens.append(rejected_token)
        elif accepted == n_draft:
            new_tokens.append(_sample(p_bonus))  # all accepted -> free bonus token

        # A round can commit several tokens at once; stop at the first EOS like
        # autoregressive decoding would, instead of keeping whatever follows it.
        hit_eos = eos_id is not None and eos_id in new_tokens
        if hit_eos:
            new_tokens = new_tokens[: new_tokens.index(eos_id) + 1]

        out.extend(new_tokens)
        # Leave the last committed token unprocessed so it's re-fed (and overwrites any
        # speculative KV) next round.
        t_proc = len(out) - 1
        d_proc = len(out) - 1

        if hit_eos:
            break

    out = out[: len(prompt_tokens) + max_new_tokens]
    return SpecResult(tokens=out, num_accepted=n_acc, num_proposed=n_prop)
