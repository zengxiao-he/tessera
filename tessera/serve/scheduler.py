"""Continuous-batching scheduler (iteration-level scheduling).

The defining idea of continuous batching: the batch is recomposed every decode step instead
of being fixed for the lifetime of a request. Finished sequences release their KV blocks
immediately, and waiting requests are admitted into the freed capacity on the very next
step — so a fast 5-token completion never blocks behind a slow 500-token one.

Contract with the engine: `schedule()` does *all* admission, block allocation, and
preemption, then returns exactly the sequences the engine may run this step. The engine
never mutates scheduler state mid-step (that subtlety — processing a snapshot while
preemption edits the running set — is a classic source of KV-block leaks).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from tessera.serve.api import Request
from tessera.serve.paged_kv import BlockAllocator


class Status(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class Sequence:
    request: Request
    tokens: list[int]
    block_table: list[int] = field(default_factory=list)
    status: Status = Status.WAITING

    @property
    def prompt_len(self) -> int:
        return len(self.request.prompt_tokens)

    @property
    def num_output(self) -> int:
        return len(self.tokens) - self.prompt_len

    @property
    def last_token(self) -> int:
        return self.tokens[-1]

    def append(self, token: int) -> None:
        self.tokens.append(token)

    def is_finished(self) -> bool:
        p = self.request.params
        if self.num_output >= p.max_new_tokens:
            return True
        return p.eos_id is not None and self.num_output > 0 and self.last_token == p.eos_id


class Scheduler:
    def __init__(self, allocator: BlockAllocator, block_size: int, max_running: int = 8):
        self.allocator = allocator
        self.block_size = block_size
        self.max_running = max_running
        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []

    def add(self, request: Request) -> Sequence:
        # A request whose worst case (prompt + every new token) exceeds the whole pool can
        # never finish: queued it blocks the head of the line forever, admitted it ends up
        # preempting itself every step. Reject it up front instead.
        need = self.blocks_needed(len(request.prompt_tokens) + request.params.max_new_tokens)
        if need > self.allocator.num_blocks:
            raise ValueError(
                f"request {request.request_id!r} needs up to {need} KV blocks "
                f"but the pool only has {self.allocator.num_blocks}"
            )
        seq = Sequence(request=request, tokens=list(request.prompt_tokens))
        self.waiting.append(seq)
        return seq

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def blocks_needed(self, length: int) -> int:
        return (length + self.block_size - 1) // self.block_size

    def schedule(self) -> list[Sequence]:
        """Admit, grow, and preempt; return the sequences ready to run this step."""
        self._admit()

        preempted: set[int] = set()
        for seq in self.running:
            if id(seq) in preempted:
                continue
            need = self.blocks_needed(len(seq.tokens) + 1)  # room for the next token
            while len(seq.block_table) < need:
                if self.allocator.num_free > 0:
                    seq.block_table.append(self.allocator.allocate())
                else:
                    victim = self._pick_victim(protect=seq, preempted=preempted)
                    if victim is None:
                        break
                    preempted.add(id(victim))
                    self._release(victim)
            if len(seq.block_table) < need:  # still short -> preempt this one
                preempted.add(id(seq))
                self._release(seq)

        return self._apply_preemptions(preempted)

    def finish(self, seq: Sequence) -> None:
        seq.status = Status.FINISHED
        self._release(seq)
        if seq in self.running:
            self.running.remove(seq)

    # -- internals ------------------------------------------------------------
    def _admit(self) -> None:
        while self.waiting and len(self.running) < self.max_running:
            seq = self.waiting[0]
            need = self.blocks_needed(seq.prompt_len + 1)  # prompt KV + first token
            if need > self.allocator.num_free:
                break
            self.waiting.pop(0)
            seq.block_table = [self.allocator.allocate() for _ in range(need)]
            seq.status = Status.RUNNING
            self.running.append(seq)

    def _pick_victim(self, protect: Sequence, preempted: set[int]) -> Sequence | None:
        """Evict the most-recently-admitted running sequence that holds blocks."""
        for seq in reversed(self.running):
            if seq is protect or id(seq) in preempted or not seq.block_table:
                continue
            return seq
        return None

    def _release(self, seq: Sequence) -> None:
        for block in seq.block_table:
            self.allocator.free(block)
        seq.block_table = []

    def _apply_preemptions(self, preempted: set[int]) -> list[Sequence]:
        still: list[Sequence] = []
        for seq in self.running:
            if id(seq) in preempted:
                seq.status = Status.WAITING
                self.waiting.insert(0, seq)
            else:
                still.append(seq)
        self.running = still
        return list(self.running)
