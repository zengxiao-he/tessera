"""Continuous-batching admission, memory budgeting, and preemption."""

import pytest

from tessera.serve.api import Request, SamplingParams
from tessera.serve.paged_kv import BlockAllocator
from tessera.serve.scheduler import Scheduler, Status


def _req(rid, prompt_len, max_new=4):
    return Request(rid, list(range(prompt_len)), SamplingParams(max_new_tokens=max_new))


def test_admission_respects_max_running():
    sched = Scheduler(BlockAllocator(64), block_size=4, max_running=2)
    for i in range(4):
        sched.add(_req(f"r{i}", 4))
    running = sched.schedule()
    assert len(running) == 2
    assert len(sched.waiting) == 2


def test_admission_respects_memory():
    # 2 blocks total, each prompt needs 1 block -> only 2 admit even with running slots free.
    sched = Scheduler(BlockAllocator(2), block_size=4, max_running=8)
    for i in range(4):
        sched.add(_req(f"r{i}", 3))
    running = sched.schedule()
    assert len(running) == 2
    assert sched.allocator.num_free == 0


def test_finish_frees_blocks_and_admits_waiting():
    sched = Scheduler(BlockAllocator(2), block_size=4, max_running=8)
    for i in range(3):
        sched.add(_req(f"r{i}", 3))
    running = sched.schedule()
    assert len(running) == 2
    sched.finish(running[0])  # one completes -> its block returns to the pool
    assert sched.allocator.num_free == 1
    running = sched.schedule()  # the waiting request is admitted on the next step
    assert any(s.request.request_id == "r2" for s in running)


def test_preemption_under_memory_pressure():
    sched = Scheduler(BlockAllocator(2), block_size=4, max_running=8)
    sched.add(_req("a", 3))
    sched.add(_req("b", 3))
    a, b = sched.schedule()  # both admitted, 1 block each, pool exhausted
    assert sched.allocator.num_free == 0

    # Both decode a token; now each needs a second block but the pool is empty.
    a.append(99)
    b.append(99)
    running = sched.schedule()  # one grows by evicting the other
    assert len(running) == 1
    survivor = running[0]
    assert len(survivor.block_table) == 2
    other = b if survivor is a else a
    assert other.status == Status.WAITING
    assert other in sched.waiting


def test_rejects_request_larger_than_pool():
    # 1 block of 4 slots can never hold 4 prompt tokens + 1 new token.
    sched = Scheduler(BlockAllocator(1), block_size=4, max_running=8)
    with pytest.raises(ValueError, match="KV blocks"):
        sched.add(_req("too-large", 4, max_new=1))
    assert not sched.waiting
    sched.add(_req("small", 2, max_new=2))  # still accepts requests that fit
    assert [s.request.request_id for s in sched.schedule()] == ["small"]
