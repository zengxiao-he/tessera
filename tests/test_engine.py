"""End-to-end engine: matches greedy generation and serves many requests under memory."""

import pytest
import torch

from tessera.config import get_preset
from tessera.model import Transformer
from tessera.serve import InferenceEngine, Request, SamplingParams


def _model(seed=0):
    torch.manual_seed(seed)
    return Transformer(get_preset("tessera-debug")).eval()


def test_engine_single_request_matches_generate():
    model = _model()
    prompt = [3, 1, 4, 1, 5]
    ref = model.generate(torch.tensor([prompt]), max_new_tokens=12, temperature=0.0)[0].tolist()

    engine = InferenceEngine(model, max_batch=4, block_size=8, num_blocks=128)
    engine.add_request(Request("a", prompt, SamplingParams(max_new_tokens=12, temperature=0.0)))
    out = {}
    while engine.has_unfinished():
        for o in engine.step():
            out[o.request_id] = o
    assert out["a"].all_tokens == ref


def test_engine_serves_many_requests():
    model = _model()
    reqs = [
        Request(f"r{i}", [i % 7, (i + 2) % 7], SamplingParams(max_new_tokens=5 + i, temperature=0.0))
        for i in range(6)
    ]
    engine = InferenceEngine(model, max_batch=3, block_size=8, num_blocks=256)
    results = engine.run(reqs)
    assert len(results) == 6
    for i, r in enumerate(reqs):
        assert len(results[r.request_id].output_tokens) == 5 + i


def test_engine_completes_under_tight_memory():
    """Few blocks => admission throttles, but every request still finishes."""
    model = _model()
    reqs = [
        Request(f"r{i}", [1, 2, 3], SamplingParams(max_new_tokens=8, temperature=0.0))
        for i in range(5)
    ]
    engine = InferenceEngine(model, max_batch=8, block_size=4, num_blocks=6)
    results = engine.run(reqs)
    assert len(results) == 5
    assert all(len(results[f"r{i}"].output_tokens) == 8 for i in range(5))


def test_engine_rejects_oversized_request_instead_of_hanging():
    engine = InferenceEngine(_model(), max_batch=1, block_size=4, num_blocks=1)
    with pytest.raises(ValueError, match="KV blocks"):
        engine.add_request(Request("too-large", [1, 2, 3, 4]))
    assert not engine.has_unfinished()
