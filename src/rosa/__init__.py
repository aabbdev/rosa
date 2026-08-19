"""ROSA: differentiable sparse retrieval backed by an exact suffix automaton.

The suffix-automaton topology is discrete and exact. Differentiability is
restricted to symbolic tokenization, candidate scoring, soft match
verification, and retrieved values. This keeps the combinatorial search path
well-defined while allowing end-to-end gradient-based training.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

EXACT_KIND = 0
VIRTUAL_KIND = 1
NULL_KIND = 2

__all__ = [
    "EXACT_KIND",
    "NULL_KIND",
    "ROSA",
    "InferenceOutput",
    "ROSAInferenceState",
    "VIRTUAL_KIND",
    "HardCandidates",
    "PreparedHardCandidates",
    "ROSAOutput",
    "build_hard_candidates",
    "build_virtual_pool_indices",
    "forward_candidates_step",
    "forward_candidates_step_into",
    "forward_step",
    "init_candidate_state",
    "init_candidate_buffers",
    "init_inference_state",
    "prefill",
    "reference_rosa",
]


def _reference_single(tokens: list[int]) -> tuple[list[int], list[int], list[int]]:
    """Return exact ROSA predictions for one sequence using brute force.

    This helper is intentionally quadratic and is provided for tests,
    diagnostics, and small-sequence correctness checks.
    """

    n = len(tokens)
    predicted = [-1] * n
    source = [-1] * n
    match_length = [0] * n

    for i in range(n):
        best_length = 0
        best_source = -1
        for j in range(i):
            length = 0
            while (
                i - length >= 0
                and j - length >= 0
                and tokens[i - length] == tokens[j - length]
            ):
                length += 1
            if length > 0 and (
                length > best_length or (length == best_length and j > best_source)
            ):
                best_length = length
                best_source = j

        if best_source >= 0:
            predicted[i] = tokens[best_source + 1]
            source[i] = best_source
            match_length[i] = best_length

    return predicted, source, match_length


def reference_rosa(tokens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Compute the exact ROSA definition with a brute-force implementation.

    Args:
        tokens: Integer tensor with shape ``[N]`` or ``[B, N]``.

    Returns:
        ``(predicted_tokens, source_index, match_length)``. ``-1`` denotes a
        position for which no previous matching suffix exists.
    """

    squeeze = tokens.ndim == 1
    if squeeze:
        tokens = tokens.unsqueeze(0)
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape [N] or [B, N]")

    device = tokens.device
    predicted_rows: list[list[int]] = []
    source_rows: list[list[int]] = []
    length_rows: list[list[int]] = []
    for row in tokens.detach().cpu().tolist():
        predicted, source, match_length = _reference_single([int(v) for v in row])
        predicted_rows.append(predicted)
        source_rows.append(source)
        length_rows.append(match_length)

    predicted_tensor = torch.tensor(predicted_rows, device=device, dtype=torch.long)
    source_tensor = torch.tensor(source_rows, device=device, dtype=torch.long)
    length_tensor = torch.tensor(length_rows, device=device, dtype=torch.long)
    if squeeze:
        return predicted_tensor[0], source_tensor[0], length_tensor[0]
    return predicted_tensor, source_tensor, length_tensor


@dataclass
class HardCandidates:
    """Hard candidates emitted by the exact online suffix automaton.

    Every non-negative ``source_index[..., c]`` is an end position ``j < i``.
    Retrieval uses the continuation at ``j + 1``.  Candidate slot ordering is
    deterministic: longer suffix states first, then newer occurrences first.
    The first valid slot is therefore the exact ROSA choice.
    """

    source_index: Tensor
    match_length: Tensor
    state_id: Tensor
    frequency: Tensor
    mask: Tensor
    rosa_slot: Tensor
    rosa_source_index: Tensor
    rosa_match_length: Tensor
    rosa_predicted_tokens: Tensor


@dataclass(frozen=True, slots=True)
class PreparedHardCandidates:
    """Immutable, reusable snapshot of exact hard candidate construction.

    Tensor version counters and identities are captured so in-place mutation or
    replacement of either the token snapshot or any candidate field is rejected
    before the object can influence a forward pass.
    """

    hard_tokens: Tensor
    candidates: HardCandidates
    suffix_k: int
    occurrences_r: int
    candidate_backend: CandidateBackend
    shape: tuple[int, int]
    device: torch.device
    tensor_versions: tuple[int | None, ...]
    _tensor_ids: tuple[int, ...] = field(repr=False)
    _tensor_snapshots: tuple[Tensor, ...] = field(repr=False)

    def __deepcopy__(self, memo: dict[int, Any]) -> PreparedHardCandidates:
        snapshot = self.hard_tokens.detach().clone()
        candidates = HardCandidates(
            *(
                getattr(self.candidates, name).detach().clone()
                for name in HardCandidates.__dataclass_fields__
            )
        )
        tensors = (snapshot,) + tuple(
            getattr(candidates, name) for name in HardCandidates.__dataclass_fields__
        )
        copied = PreparedHardCandidates(
            hard_tokens=snapshot,
            candidates=candidates,
            suffix_k=self.suffix_k,
            occurrences_r=self.occurrences_r,
            candidate_backend=self.candidate_backend,
            shape=self.shape,
            device=self.device,
            tensor_versions=tuple(_tensor_version(tensor) for tensor in tensors),
            _tensor_ids=tuple(id(tensor) for tensor in tensors),
            _tensor_snapshots=tuple(tensor.detach().clone() for tensor in tensors),
        )
        memo[id(self)] = copied
        return copied


def _tensor_version(tensor: Tensor) -> int | None:
    """Return a mutation counter, or ``None`` for inference tensors."""

    try:
        return tensor._version
    except RuntimeError:
        return None


@dataclass
class ROSAOutput:
    """Complete ROSA forward result."""

    updated: Tensor
    retrieved: Tensor
    hard_tokens: Tensor
    code_soft: tuple[Tensor, Tensor]
    code_st: tuple[Tensor, Tensor]
    candidate_source_index: Tensor
    candidate_kind: Tensor
    candidate_mask: Tensor
    candidate_scores: Tensor
    soft_weights: Tensor
    hard_weights: Tensor
    chosen_candidate: Tensor
    chosen_source_index: Tensor
    chosen_token: Tensor
    chosen_match_length: Tensor
    chosen_is_virtual: Tensor
    hard_rosa_source_index: Tensor
    hard_rosa_predicted_tokens: Tensor
    hard_rosa_match_length: Tensor
    soft_match_score: Tensor
    read_gate: Tensor
    value_gate: Tensor
    aux_losses: dict[str, Tensor]


class _OnlineSuffixAutomaton:
    """Exact integer SAM plus a bounded occurrence cache per state.

    The cache is updated on the full suffix-link chain after every read.  This
    implementation keeps the discrete state machine isolated from the
    differentiable path so it can be replaced by a native kernel without
    changing the public module interface.
    """

    def __init__(self, max_positions: int, occurrences_r: int) -> None:
        max_states = 2 * max_positions + 1
        self.next: list[dict[int, int] | None] = [None] * max_states
        self.link = [-1] * max_states
        self.length = [0] * max_states
        self.occurrences: list[list[int]] = [[] for _ in range(max_states)]
        self.frequency = [0] * max_states
        self.next[0] = {}
        self.last = 0
        self.size = 1
        self.occurrences_r = occurrences_r

    def _transitions(self, state: int) -> dict[int, int]:
        return cast(dict[int, int], self.next[state])

    def extend(self, token: int) -> int:
        cur = self.size
        self.size += 1
        self.next[cur] = {}
        self.length[cur] = self.length[self.last] + 1
        p = self.last

        while p != -1 and token not in self._transitions(p):
            self._transitions(p)[token] = cur
            p = self.link[p]

        if p == -1:
            self.link[cur] = 0
        else:
            q = self._transitions(p)[token]
            if self.length[p] + 1 == self.length[q]:
                self.link[cur] = q
            else:
                clone = self.size
                self.size += 1
                self.next[clone] = self._transitions(q).copy()
                self.length[clone] = self.length[p] + 1
                self.link[clone] = self.link[q]
                self.occurrences[clone] = list(self.occurrences[q])
                self.frequency[clone] = self.frequency[q]

                while p != -1 and self._transitions(p).get(token) == q:
                    self._transitions(p)[token] = clone
                    p = self.link[p]

                self.link[q] = clone
                self.link[cur] = clone

        self.last = cur
        return cur

    def read_candidates(
        self,
        suffix_k: int,
        occurrences_r: int,
    ) -> list[tuple[int, int, int, int]]:
        out: list[tuple[int, int, int, int]] = []
        seen_positions: set[int] = set()
        v = self.last
        states_with_history = 0
        while v != -1 and states_with_history < suffix_k:
            if self.length[v] > 0 and self.occurrences[v]:
                states_with_history += 1
                for j in reversed(self.occurrences[v][-occurrences_r:]):
                    if j not in seen_positions:
                        out.append((j, self.length[v], v, self.frequency[v]))
                        seen_positions.add(j)
            v = self.link[v]
        return out

    def write_current_end(self, end_position: int) -> None:
        v = self.last
        while v != -1:
            self.frequency[v] += 1
            bucket = self.occurrences[v]
            bucket.append(end_position)
            if len(bucket) > self.occurrences_r:
                del bucket[0]
            v = self.link[v]


@dataclass
class _PythonInferenceState:
    batch_size: int
    max_length: int
    position: int
    automata: list[_OnlineSuffixAutomaton]
    history: list[list[int]]


def _init_python_inference_state(
    batch_size: int,
    max_length: int,
) -> _PythonInferenceState:
    return _PythonInferenceState(
        batch_size=batch_size,
        max_length=max_length,
        position=0,
        automata=[_OnlineSuffixAutomaton(max_length, 1) for _ in range(batch_size)],
        history=[[] for _ in range(batch_size)],
    )


def _python_forward_step(state: _PythonInferenceState, tokens: Tensor) -> Tensor:
    if state.position >= state.max_length:
        raise RuntimeError("inference state capacity exceeded")
    device = tokens.device
    predictions: list[int] = []
    for batch_index, token in enumerate(tokens.detach().cpu().tolist()):
        value = int(token)
        history = state.history[batch_index]
        automaton = state.automata[batch_index]
        history.append(value)
        automaton.extend(value)
        candidates = automaton.read_candidates(1, 1)
        predictions.append(history[candidates[0][0] + 1] if candidates else -1)
        automaton.write_current_end(state.position)
    state.position += 1
    return torch.tensor(predictions, dtype=torch.long, device=device)


def _build_single_hard_candidates(
    tokens: list[int],
    suffix_k: int,
    occurrences_r: int,
) -> tuple[list[list[tuple[int, int, int, int]]], list[int], list[int], list[int]]:
    sam = _OnlineSuffixAutomaton(len(tokens), occurrences_r)
    all_candidates: list[list[tuple[int, int, int, int]]] = []
    rosa_src = [-1] * len(tokens)
    rosa_len = [0] * len(tokens)
    rosa_pred = [-1] * len(tokens)

    for i, token in enumerate(tokens):
        sam.extend(token)
        candidates = sam.read_candidates(suffix_k, occurrences_r)
        all_candidates.append(candidates)
        if candidates:
            j, mlen, _, _ = candidates[0]
            rosa_src[i] = j
            rosa_len[i] = mlen
            rosa_pred[i] = tokens[j + 1]
        sam.write_current_end(i)

    return all_candidates, rosa_src, rosa_len, rosa_pred


def build_hard_candidates(
    tokens: Tensor,
    suffix_k: int = 16,
    occurrences_r: int = 4,
) -> HardCandidates:
    """Build exact hard SAM candidates for ROSA.

    ``suffix_k`` bounds suffix states examined at read time and
    ``occurrences_r`` bounds historical end positions retained per state.
    Slot 0, when valid, is the exact standard ROSA retrieval source.
    """

    if suffix_k <= 0:
        raise ValueError("suffix_k must be > 0")
    if occurrences_r <= 0:
        raise ValueError("occurrences_r must be > 0")

    squeeze = tokens.ndim == 1
    if squeeze:
        tokens = tokens.unsqueeze(0)
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape [N] or [B, N]")

    bsz, n = tokens.shape
    slots = suffix_k * occurrences_r
    device = tokens.device
    source = torch.full((bsz, n, slots), -1, dtype=torch.long, device=device)
    mlen = torch.zeros((bsz, n, slots), dtype=torch.long, device=device)
    state = torch.full((bsz, n, slots), -1, dtype=torch.long, device=device)
    freq = torch.zeros((bsz, n, slots), dtype=torch.long, device=device)
    mask = torch.zeros((bsz, n, slots), dtype=torch.bool, device=device)
    rosa_slot = torch.full((bsz, n), -1, dtype=torch.long, device=device)
    rosa_src = torch.full((bsz, n), -1, dtype=torch.long, device=device)
    rosa_len = torch.zeros((bsz, n), dtype=torch.long, device=device)
    rosa_pred = torch.full((bsz, n), -1, dtype=torch.long, device=device)

    for b, row in enumerate(tokens.detach().cpu().tolist()):
        candidates, rs, rl, rp = _build_single_hard_candidates(
            [int(x) for x in row], suffix_k, occurrences_r
        )
        rosa_src[b] = torch.tensor(rs, dtype=torch.long, device=device)
        rosa_len[b] = torch.tensor(rl, dtype=torch.long, device=device)
        rosa_pred[b] = torch.tensor(rp, dtype=torch.long, device=device)
        for i, row_candidates in enumerate(candidates):
            k = min(slots, len(row_candidates))
            if k == 0:
                continue
            vals = row_candidates[:k]
            source[b, i, :k] = torch.tensor([x[0] for x in vals], device=device)
            mlen[b, i, :k] = torch.tensor([x[1] for x in vals], device=device)
            state[b, i, :k] = torch.tensor([x[2] for x in vals], device=device)
            freq[b, i, :k] = torch.tensor([x[3] for x in vals], device=device)
            mask[b, i, :k] = True
            rosa_slot[b, i] = 0

    result = HardCandidates(
        source,
        mlen,
        state,
        freq,
        mask,
        rosa_slot,
        rosa_src,
        rosa_len,
        rosa_pred,
    )
    if not squeeze:
        return result
    return HardCandidates(
        *(getattr(result, name)[0] for name in result.__dataclass_fields__)
    )


CandidateBackend = Literal["auto", "python", "stateful"]


def _build_stateful_hard_candidates(
    tokens: Tensor,
    suffix_k: int,
    occurrences_r: int,
) -> HardCandidates:
    """Prefill a full sequence through the exact bounded stateful backend."""

    from ._stateful_candidates_numba import init_candidate_state as initialize
    from ._stateful_candidates_numba import prefill_candidates

    squeeze = tokens.ndim == 1
    if squeeze:
        tokens = tokens.unsqueeze(0)
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape [N] or [B, N]")

    batch_size, sequence_length = tokens.shape
    state = initialize(
        batch_size,
        sequence_length,
        suffix_k=suffix_k,
        occurrences_r=occurrences_r,
    )
    try:
        candidates = prefill_candidates(state, tokens)
        result = HardCandidates(
            *(getattr(candidates, name) for name in HardCandidates.__dataclass_fields__)
        )
        if not squeeze:
            return result
        return HardCandidates(
            *(getattr(result, name)[0] for name in result.__dataclass_fields__)
        )
    finally:
        state.close()


def _build_forward_hard_candidates(
    tokens: Tensor,
    suffix_k: int,
    occurrences_r: int,
    backend: CandidateBackend,
) -> HardCandidates:
    """Dispatch ROSA forward candidates without weakening backend failures."""

    if backend == "python":
        return build_hard_candidates(tokens, suffix_k, occurrences_r)
    try:
        return _build_stateful_hard_candidates(tokens, suffix_k, occurrences_r)
    except ModuleNotFoundError as error:
        if error.name not in {"numba", "numpy"}:
            raise
        if backend == "stateful":
            raise RuntimeError(
                "stateful candidate backend requires the 'numba' extra"
            ) from error
        return build_hard_candidates(tokens, suffix_k, occurrences_r)


def _virtual_pool_single(i: int, pool_size: int) -> list[int]:
    """Causal bounded pool: half recent positions, half history anchors."""

    if i <= 0:
        return []
    recent_n = min(i, (pool_size + 1) // 2)
    recent = list(range(i - recent_n, i))
    remaining = pool_size - len(recent)
    older_end = i - recent_n
    older: list[int] = []
    if remaining > 0 and older_end > 0:
        if remaining == 1:
            older = [0]
        else:
            older = [
                round(k * (older_end - 1) / (remaining - 1)) for k in range(remaining)
            ]
    return list(dict.fromkeys(older + recent))[-pool_size:]


def build_virtual_pool_indices(
    batch_size: int,
    sequence_length: int,
    pool_size: int,
    device: torch.device,
) -> Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be > 0")
    if pool_size <= 0:
        raise ValueError("pool_size must be > 0")
    out = torch.full(
        (batch_size, sequence_length, pool_size),
        -1,
        dtype=torch.long,
        device=device,
    )
    for i in range(sequence_length):
        pool = _virtual_pool_single(i, pool_size)
        if pool:
            out[:, i, : len(pool)] = torch.tensor(pool, dtype=torch.long, device=device)
    return out


def _gather_sequence(x: Tensor, index: Tensor) -> Tensor:
    """Gather x[b, index[b,...]] while preserving arbitrary index shape."""

    if x.ndim != 3:
        raise ValueError("x must have shape [B, N, D]")
    if index.ndim < 2 or index.shape[0] != x.shape[0]:
        raise ValueError("index must start with the same batch dimension as x")
    safe = index.clamp(min=0, max=x.shape[1] - 1)
    batch_shape = [x.shape[0]] + [1] * (index.ndim - 1)
    batch = (
        torch.arange(x.shape[0], device=x.device).reshape(batch_shape).expand_as(index)
    )
    return x[batch, safe]


def _soft_match_torch(
    st1: Tensor,
    st2: Tensor,
    source_index: Tensor,
    candidate_mask: Tensor,
    window: int,
) -> Tensor:
    """Pure-Torch soft suffix verification with a statically bounded window."""

    bsz, n, candidates = source_index.shape
    positions = torch.arange(n, device=source_index.device).view(1, n, 1)
    positions = positions.expand(bsz, n, candidates)
    survival = torch.ones((bsz, n, candidates), dtype=st1.dtype, device=st1.device)
    score = torch.zeros_like(survival)
    for r in range(window):
        left_idx = positions - r
        right_idx = source_index - r
        valid = candidate_mask & (left_idx >= 0) & (right_idx >= 0)
        left1 = _gather_sequence(st1, left_idx)
        right1 = _gather_sequence(st1, right_idx)
        left2 = _gather_sequence(st2, left_idx)
        right2 = _gather_sequence(st2, right_idx)
        eq = (left1 * right1).sum(-1) * (left2 * right2).sum(-1)
        survival = survival * eq * valid.to(eq.dtype)
        score = score + survival
    return score


def _soft_match_at_positions_torch(
    st1: Tensor,
    st2: Tensor,
    query_positions: Tensor,
    source_index: Tensor,
    candidate_mask: Tensor,
    window: int,
) -> Tensor:
    """Soft suffix verification for selected real sequence positions."""

    bsz, queries, candidates = source_index.shape
    positions = query_positions.unsqueeze(-1).expand(bsz, queries, candidates)
    survival = torch.ones(
        (bsz, queries, candidates), dtype=st1.dtype, device=st1.device
    )
    score = torch.zeros_like(survival)
    for r in range(window):
        left_idx = positions - r
        right_idx = source_index - r
        valid = candidate_mask & (left_idx >= 0) & (right_idx >= 0)
        left1 = _gather_sequence(st1, left_idx)
        right1 = _gather_sequence(st1, right_idx)
        left2 = _gather_sequence(st2, left_idx)
        right2 = _gather_sequence(st2, right_idx)
        eq = (left1 * right1).sum(-1) * (left2 * right2).sum(-1)
        survival = survival * eq * valid.to(eq.dtype)
        score = score + survival
    return score


# A callable owns shape specializations while the outer cache keeps exactly one
# full-graph compiler island per static verification window. Specializations
# are initialized once per input signature; already-ready calls do not take the
# signature lock.
_SOFT_MATCH_COMPILE_ENABLED = True
_SOFT_MATCH_COMPILED: dict[int, Any] = {}
_SoftMatchTensorSignature = tuple[
    str,
    int | None,
    torch.dtype,
    torch.layout,
    tuple[int, ...],
    tuple[int, ...],
    int,
    bool,
]
_SoftMatchSignature = tuple[
    int,
    bool,
    bool,
    tuple[
        _SoftMatchTensorSignature,
        _SoftMatchTensorSignature,
        _SoftMatchTensorSignature,
        _SoftMatchTensorSignature,
    ],
]
_SOFT_MATCH_COMPILE_FAILURES: set[_SoftMatchSignature] = set()
_SOFT_MATCH_COMPILE_READY: set[_SoftMatchSignature] = set()
_SOFT_MATCH_SIGNATURE_LOCKS: dict[_SoftMatchSignature, threading.Lock] = {}
_SOFT_MATCH_WINDOW_LOCKS: dict[int, threading.Lock] = {}
_SOFT_MATCH_COMPILE_MASTER_LOCK = threading.Lock()


def _soft_match_signature(
    st1: Tensor,
    st2: Tensor,
    source_index: Tensor,
    candidate_mask: Tensor,
    window: int,
) -> _SoftMatchSignature:
    def tensor_signature(tensor: Tensor) -> _SoftMatchTensorSignature:
        return (
            tensor.device.type,
            tensor.device.index,
            tensor.dtype,
            tensor.layout,
            tuple(tensor.shape),
            tuple(tensor.stride()),
            int(tensor.storage_offset()),
            tensor.requires_grad,
        )

    tensors = (st1, st2, source_index, candidate_mask)
    return (
        window,
        torch.is_grad_enabled(),
        torch.is_inference_mode_enabled(),
        tuple(tensor_signature(tensor) for tensor in tensors),  # type: ignore[return-value]
    )


def _clear_soft_match_compile_cache() -> None:
    """Clear all compiled soft-match process state, primarily for tests."""

    with _SOFT_MATCH_COMPILE_MASTER_LOCK:
        _SOFT_MATCH_COMPILED.clear()
        _SOFT_MATCH_COMPILE_FAILURES.clear()
        _SOFT_MATCH_COMPILE_READY.clear()
        _SOFT_MATCH_SIGNATURE_LOCKS.clear()
        _SOFT_MATCH_WINDOW_LOCKS.clear()


def _soft_match(
    st1: Tensor,
    st2: Tensor,
    source_index: Tensor,
    candidate_mask: Tensor,
    window: int,
) -> Tensor:
    """Use a static compiled island with eager fallback for forward failures."""

    if not _SOFT_MATCH_COMPILE_ENABLED:
        return _soft_match_torch(st1, st2, source_index, candidate_mask, window)

    signature = _soft_match_signature(st1, st2, source_index, candidate_mask, window)
    with _SOFT_MATCH_COMPILE_MASTER_LOCK:
        signature_lock = _SOFT_MATCH_SIGNATURE_LOCKS.setdefault(
            signature, threading.Lock()
        )
        window_lock = _SOFT_MATCH_WINDOW_LOCKS.setdefault(window, threading.Lock())
        if signature in _SOFT_MATCH_COMPILE_FAILURES:
            failed = True
            ready = False
            compiled = None
        else:
            failed = False
            ready = signature in _SOFT_MATCH_COMPILE_READY
            compiled = _SOFT_MATCH_COMPILED.get(window)
    if failed:
        return _soft_match_torch(st1, st2, source_index, candidate_mask, window)
    if ready:
        assert compiled is not None
        try:
            return compiled(st1, st2, source_index, candidate_mask)
        except Exception:
            with _SOFT_MATCH_COMPILE_MASTER_LOCK:
                _SOFT_MATCH_COMPILE_FAILURES.add(signature)
            return _soft_match_torch(st1, st2, source_index, candidate_mask, window)

    with signature_lock:
        with _SOFT_MATCH_COMPILE_MASTER_LOCK:
            if signature in _SOFT_MATCH_COMPILE_FAILURES:  # pragma: no cover - race
                failed = True
                ready = False
                compiled = None
            else:
                failed = False
                ready = signature in _SOFT_MATCH_COMPILE_READY
                compiled = _SOFT_MATCH_COMPILED.get(window)
        if not failed and not ready and compiled is None:
            with window_lock:
                with _SOFT_MATCH_COMPILE_MASTER_LOCK:
                    compiled = _SOFT_MATCH_COMPILED.get(window)
                if compiled is None:  # pragma: no branch - concurrent recheck
                    try:
                        compiled = torch.compile(
                            lambda a, b, source, mask: _soft_match_torch(
                                a, b, source, mask, window
                            ),
                            fullgraph=True,
                            dynamic=False,
                        )
                    except Exception:
                        with _SOFT_MATCH_COMPILE_MASTER_LOCK:
                            _SOFT_MATCH_COMPILE_FAILURES.add(signature)
                        return _soft_match_torch(
                            st1, st2, source_index, candidate_mask, window
                        )
                    with _SOFT_MATCH_COMPILE_MASTER_LOCK:
                        _SOFT_MATCH_COMPILED[window] = compiled

        if not failed and not ready:
            assert compiled is not None
            try:
                result = compiled(st1, st2, source_index, candidate_mask)
            except Exception:
                # Keep the window callable: an unsupported shape, dtype, or
                # device must not disable its other specializations.
                with _SOFT_MATCH_COMPILE_MASTER_LOCK:
                    _SOFT_MATCH_COMPILE_FAILURES.add(signature)
                return _soft_match_torch(st1, st2, source_index, candidate_mask, window)

            # torch.compile is lazy, so the signature becomes ready only after
            # its first forward execution succeeds. Deferred AOT backward
            # errors are intentionally propagated to the caller.
            with _SOFT_MATCH_COMPILE_MASTER_LOCK:
                _SOFT_MATCH_COMPILE_READY.add(signature)
            return result

    if failed:  # pragma: no cover - concurrent recheck
        return _soft_match_torch(st1, st2, source_index, candidate_mask, window)
    assert ready and compiled is not None
    try:
        return compiled(st1, st2, source_index, candidate_mask)
    except Exception:  # pragma: no cover - concurrent runtime failure
        with _SOFT_MATCH_COMPILE_MASTER_LOCK:
            _SOFT_MATCH_COMPILE_FAILURES.add(signature)
        return _soft_match_torch(st1, st2, source_index, candidate_mask, window)


def _st_categorical(
    logits: Tensor, temperature: float
) -> tuple[Tensor, Tensor, Tensor]:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    soft = F.softmax(logits / temperature, dim=-1)
    ids = logits.argmax(dim=-1)
    hard = F.one_hot(ids, num_classes=logits.shape[-1]).to(logits.dtype)
    st = hard + (soft - soft.detach())
    return soft, st, ids


def _balance_kl(probs: Tensor) -> Tensor:
    mean = probs.mean(dim=(0, 1)).clamp_min(1e-9)
    uniform_log_prob = -math.log(probs.shape[-1])
    return torch.sum(mean * (torch.log(mean) - uniform_log_prob))


class ROSA(nn.Module):
    """Differentiable ROSA retrieval with an exact suffix-automaton backbone.

    The module keeps suffix-automaton topology discrete and exact while making
    tokenization, candidate ranking, and value retrieval differentiable. Exact
    suffix candidates are augmented with bounded multi-occurrence history, a
    causal sparse virtual-candidate router, and an explicit NULL candidate.

    Candidate scoring uses an exact ROSA prior plus a learned residual::

        score = rosa_prior + learned_residual_scale * learned_score

    Setting ``learned_residual_scale=0`` recovers standard ROSA selection.
    No trainable ``S x V x S`` transition tensor is allocated.
    """

    learned_residual_scale: Tensor
    virtual_scale: Tensor
    neural_value_scale: Tensor

    def __init__(
        self,
        d_model: int,
        codebook_sizes: Sequence[int] = (16, 16),
        suffix_k: int = 16,
        occurrences_r: int = 4,
        soft_verify_window: int = 32,
        virtual_candidates: int = 4,
        virtual_pool_size: int = 64,
        dense_recent_candidates: int = 0,
        sparse_old_candidates: int = 0,
        sparse_old_pool_size: int = 64,
        soft_candidates_forward: bool = False,
        selector_dim: int = 128,
        token_temperature: float = 1.0,
        retrieval_temperature: float = 1.0,
        tie_break_scale: float = 0.25,
        virtual_prior_bias: float = -0.5,
        null_prior_bias: float = 0.0,
        read_gate_bias: float = -2.0,
        value_gate_bias: float = -3.0,
        learned_residual_scale: float = 0.0,
        virtual_scale: float = 0.0,
        neural_value_scale: float = 0.0,
        candidate_backend: CandidateBackend = "auto",
        compile_soft_match: bool = False,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if len(codebook_sizes) != 2 or any(int(v) <= 1 for v in codebook_sizes):
            raise ValueError("codebook_sizes must contain exactly two integers > 1")
        if suffix_k <= 0 or occurrences_r <= 0 or soft_verify_window <= 0:
            raise ValueError(
                "suffix_k, occurrences_r and soft_verify_window must be > 0"
            )
        if virtual_candidates < 0 or virtual_pool_size < virtual_candidates:
            raise ValueError("virtual_pool_size must be >= virtual_candidates >= 0")
        if dense_recent_candidates < 0:
            raise ValueError("dense_recent_candidates must be >= 0")
        if sparse_old_candidates < 0 or sparse_old_pool_size < sparse_old_candidates:
            raise ValueError(
                "sparse_old_pool_size must be >= sparse_old_candidates >= 0"
            )
        if not isinstance(soft_candidates_forward, bool):
            raise TypeError("soft_candidates_forward must be a bool")
        if not isinstance(compile_soft_match, bool):
            raise TypeError("compile_soft_match must be a bool")
        if selector_dim <= 0:
            raise ValueError("selector_dim must be > 0")
        if token_temperature <= 0 or retrieval_temperature <= 0:
            raise ValueError("temperatures must be > 0")
        if not 0.0 <= tie_break_scale < 1.0:
            raise ValueError("tie_break_scale must be in [0, 1)")
        if virtual_prior_bias > null_prior_bias or null_prior_bias >= 1.0:
            raise ValueError("require virtual_prior_bias <= null_prior_bias < 1")
        if not 0.0 <= learned_residual_scale <= 1.0:
            raise ValueError("learned_residual_scale must be in [0, 1]")
        if not 0.0 <= virtual_scale <= 1.0:
            raise ValueError("virtual_scale must be in [0, 1]")
        if not 0.0 <= neural_value_scale <= 1.0:
            raise ValueError("neural_value_scale must be in [0, 1]")
        if candidate_backend not in {"auto", "python", "stateful"}:
            raise ValueError(
                "candidate_backend must be 'auto', 'python', or 'stateful'"
            )

        self.d_model = d_model
        self.codebook_sizes = (int(codebook_sizes[0]), int(codebook_sizes[1]))
        self.suffix_k = suffix_k
        self.occurrences_r = occurrences_r
        self.soft_verify_window = soft_verify_window
        self.virtual_candidates = virtual_candidates
        self.virtual_pool_size = virtual_pool_size
        self.dense_recent_candidates = dense_recent_candidates
        self.sparse_old_candidates = sparse_old_candidates
        self.sparse_old_pool_size = sparse_old_pool_size
        self.soft_candidates_forward = soft_candidates_forward
        self.compile_soft_match = compile_soft_match
        self.candidate_backend: CandidateBackend = candidate_backend
        self.selector_dim = selector_dim
        self.token_temperature = token_temperature
        self.retrieval_temperature = retrieval_temperature
        self.tie_break_scale = tie_break_scale
        self.virtual_prior_bias = virtual_prior_bias
        self.null_prior_bias = null_prior_bias

        c1, c2 = self.codebook_sizes
        self.code_head_1 = nn.Linear(d_model, c1)
        self.code_head_2 = nn.Linear(d_model, c2)
        self.symbol_embedding_1 = nn.Embedding(c1, d_model)
        self.symbol_embedding_2 = nn.Embedding(c2, d_model)

        self.selector_query = nn.Linear(d_model, selector_dim, bias=False)
        self.selector_key = nn.Linear(d_model, selector_dim, bias=False)
        self.virtual_query = nn.Linear(d_model, selector_dim, bias=False)
        self.virtual_key = nn.Linear(d_model, selector_dim, bias=False)

        # [log_len, log_age, log_freq, soft_match, exact, virtual,
        #  null, rosa_exact, virtual_router_score]
        self.feature_mlp = nn.Sequential(
            nn.Linear(9, selector_dim),
            nn.SiLU(),
            nn.Linear(selector_dim, 1),
        )
        self.kind_bias = nn.Parameter(torch.zeros(3))
        self.null_head = nn.Linear(d_model, 1)

        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_gate_head = nn.Linear(selector_dim * 2, 1)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.read_gate_head = nn.Linear(d_model, 1)
        nn.init.constant_(self.read_gate_head.bias, read_gate_bias)
        nn.init.constant_(self.value_gate_head.bias, value_gate_bias)

        self.register_buffer(
            "learned_residual_scale",
            torch.tensor(float(learned_residual_scale)),
        )
        self.register_buffer("virtual_scale", torch.tensor(float(virtual_scale)))
        self.register_buffer(
            "neural_value_scale",
            torch.tensor(float(neural_value_scale)),
        )

    @property
    def vocab_size(self) -> int:
        return self.codebook_sizes[0] * self.codebook_sizes[1]

    def set_learned_residual_scale(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("value must be in [0, 1]")
        self.learned_residual_scale.fill_(float(value))

    def set_virtual_scale(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("value must be in [0, 1]")
        self.virtual_scale.fill_(float(value))

    def set_neural_value_scale(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("value must be in [0, 1]")
        self.neural_value_scale.fill_(float(value))

    def encode(
        self,
        z_a: Tensor,
        code_logits: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], Tensor]:
        if z_a.ndim != 3 or z_a.shape[-1] != self.d_model:
            raise ValueError("z_a must have shape [B, N, d_model]")
        if code_logits is None:
            l1 = self.code_head_1(z_a)
            l2 = self.code_head_2(z_a)
        else:
            if len(code_logits) != 2:
                raise ValueError("code_logits must be a pair")
            l1, l2 = code_logits
            expected1 = (*z_a.shape[:2], self.codebook_sizes[0])
            expected2 = (*z_a.shape[:2], self.codebook_sizes[1])
            if tuple(l1.shape) != expected1 or tuple(l2.shape) != expected2:
                raise ValueError(
                    f"code_logits shapes must be {expected1} and {expected2}"
                )
        p1, st1, id1 = _st_categorical(l1, self.token_temperature)
        p2, st2, id2 = _st_categorical(l2, self.token_temperature)
        hard_tokens = id1 * self.codebook_sizes[1] + id2
        return (p1, p2), (st1, st2), hard_tokens

    @staticmethod
    def _prepared_tensors(prepared: PreparedHardCandidates) -> tuple[Tensor, ...]:
        hard = prepared.candidates
        return (
            prepared.hard_tokens,
            hard.source_index,
            hard.match_length,
            hard.state_id,
            hard.frequency,
            hard.mask,
            hard.rosa_slot,
            hard.rosa_source_index,
            hard.rosa_match_length,
            hard.rosa_predicted_tokens,
        )

    def prepare_hard_candidates(self, hard_tokens: Tensor) -> PreparedHardCandidates:
        """Build exact candidates once for reuse by compatible ROSA modules."""

        if not isinstance(hard_tokens, Tensor):
            raise TypeError("hard_tokens must be a Tensor")
        if hard_tokens.dtype != torch.long:
            raise TypeError("hard_tokens must have dtype torch.long")
        if hard_tokens.ndim != 2:
            raise ValueError("hard_tokens must have shape [B, N]")
        if hard_tokens.shape[0] <= 0 or hard_tokens.shape[1] <= 0:
            raise ValueError("hard_tokens dimensions must be > 0")
        if hard_tokens.device != self.learned_residual_scale.device:
            raise ValueError("hard_tokens must be on the same device as ROSA")

        snapshot = hard_tokens.detach().clone()
        candidates = _build_forward_hard_candidates(
            snapshot,
            self.suffix_k,
            self.occurrences_r,
            self.candidate_backend,
        )
        tensors = (snapshot,) + tuple(
            getattr(candidates, name) for name in HardCandidates.__dataclass_fields__
        )
        return PreparedHardCandidates(
            hard_tokens=snapshot,
            candidates=candidates,
            suffix_k=self.suffix_k,
            occurrences_r=self.occurrences_r,
            candidate_backend=self.candidate_backend,
            shape=(int(snapshot.shape[0]), int(snapshot.shape[1])),
            device=snapshot.device,
            tensor_versions=tuple(_tensor_version(tensor) for tensor in tensors),
            _tensor_ids=tuple(id(tensor) for tensor in tensors),
            _tensor_snapshots=tuple(tensor.detach().clone() for tensor in tensors),
        )

    def _validate_prepared_hard_candidates(
        self,
        prepared: PreparedHardCandidates,
        hard_tokens: Tensor,
    ) -> HardCandidates:
        if not isinstance(prepared, PreparedHardCandidates):
            raise TypeError("hard_candidates must be a PreparedHardCandidates")
        if prepared.suffix_k != self.suffix_k:
            raise ValueError("hard_candidates suffix_k does not match ROSA")
        if prepared.occurrences_r != self.occurrences_r:
            raise ValueError("hard_candidates occurrences_r does not match ROSA")
        if prepared.candidate_backend != self.candidate_backend:
            raise ValueError("hard_candidates candidate_backend does not match ROSA")
        expected_shape = (int(hard_tokens.shape[0]), int(hard_tokens.shape[1]))
        if prepared.shape != expected_shape:
            raise ValueError("hard_candidates B/N shape does not match forward")
        if prepared.device != hard_tokens.device:
            raise ValueError("hard_candidates device does not match forward")

        tensors = self._prepared_tensors(prepared)
        if not all(isinstance(tensor, Tensor) for tensor in tensors):
            raise TypeError("hard_candidates fields must all be Tensors")
        if (
            len(prepared.tensor_versions) != len(tensors)
            or len(prepared._tensor_ids) != len(tensors)
            or len(prepared._tensor_snapshots) != len(tensors)
        ):
            raise ValueError("hard_candidates tensor metadata is invalid")
        if (
            prepared.hard_tokens.dtype != torch.long
            or tuple(prepared.hard_tokens.shape) != expected_shape
            or prepared.hard_tokens.device != prepared.device
        ):
            raise ValueError("hard_candidates hard_tokens metadata is invalid")
        slots = self.suffix_k * self.occurrences_r
        slot_shape = (*expected_shape, slots)
        row_shape = expected_shape
        expected = {
            "source_index": (slot_shape, torch.long),
            "match_length": (slot_shape, torch.long),
            "state_id": (slot_shape, torch.long),
            "frequency": (slot_shape, torch.long),
            "mask": (slot_shape, torch.bool),
            "rosa_slot": (row_shape, torch.long),
            "rosa_source_index": (row_shape, torch.long),
            "rosa_match_length": (row_shape, torch.long),
            "rosa_predicted_tokens": (row_shape, torch.long),
        }
        for name, (shape, dtype) in expected.items():
            tensor = getattr(prepared.candidates, name)
            if not isinstance(tensor, Tensor):
                raise TypeError(f"hard_candidates.{name} must be a Tensor")
            if tuple(tensor.shape) != shape:
                raise ValueError(f"hard_candidates.{name} has invalid shape")
            if tensor.dtype != dtype:
                raise TypeError(f"hard_candidates.{name} has invalid dtype")
            if tensor.device != prepared.device:
                raise ValueError(f"hard_candidates.{name} has invalid device")

        for tensor, identity, version, snapshot in zip(
            tensors,
            prepared._tensor_ids,
            prepared.tensor_versions,
            prepared._tensor_snapshots,
            strict=True,
        ):
            if not isinstance(snapshot, Tensor):
                raise TypeError("hard_candidates integrity snapshots must be Tensors")
            if (
                snapshot.shape != tensor.shape
                or snapshot.dtype != tensor.dtype
                or snapshot.device != tensor.device
            ):
                raise ValueError("hard_candidates tensor metadata is invalid")
            if (
                id(tensor) != identity
                or _tensor_version(tensor) != version
                or not torch.equal(tensor, snapshot)
            ):
                raise ValueError("hard_candidates was mutated")
        if not torch.equal(prepared.hard_tokens, hard_tokens):
            raise ValueError("hard_candidates is stale for encoded hard_tokens")
        return prepared.candidates

    def _soft_match(
        self,
        st1: Tensor,
        st2: Tensor,
        source_index: Tensor,
        candidate_mask: Tensor,
    ) -> Tensor:
        implementation = _soft_match if self.compile_soft_match else _soft_match_torch
        return implementation(
            st1,
            st2,
            source_index,
            candidate_mask,
            self.soft_verify_window,
        )

    def _virtual_candidates(
        self,
        z_a: Tensor,
        exact_source: Tensor,
        exact_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        bsz, n, _ = z_a.shape
        if self.virtual_candidates == 0:
            shape = (bsz, n, 0)
            return (
                torch.empty(shape, dtype=torch.long, device=z_a.device),
                torch.empty(shape, dtype=torch.bool, device=z_a.device),
                torch.empty(shape, dtype=z_a.dtype, device=z_a.device),
            )
        pool = build_virtual_pool_indices(bsz, n, self.virtual_pool_size, z_a.device)
        pool_mask = pool >= 0
        duplicate = (
            pool.unsqueeze(-1) == exact_source.unsqueeze(-2)
        ) & exact_mask.unsqueeze(-2)
        pool_mask = pool_mask & ~duplicate.any(dim=-1)

        q = self.virtual_query(z_a).unsqueeze(-2)
        k = self._virtual_pool_keys(z_a, pool)
        router_score = (q * k).sum(-1) / math.sqrt(self.selector_dim)
        masked = router_score.masked_fill(~pool_mask, -1e9)
        _, top_idx = torch.topk(masked, k=self.virtual_candidates, dim=-1)
        selected_source = pool.gather(-1, top_idx)
        selected_mask = pool_mask.gather(-1, top_idx)
        selected_router = router_score.gather(-1, top_idx)
        return selected_source, selected_mask, selected_router

    def _virtual_pool_keys(self, z_a: Tensor, pool: Tensor) -> Tensor:
        return _gather_sequence(self.virtual_key(z_a), pool)

    def _candidate_selector_keys(self, z_a: Tensor, source: Tensor) -> Tensor:
        return _gather_sequence(self.selector_key(z_a), source)

    def _candidate_symbolic_values(
        self,
        st1: Tensor,
        st2: Tensor,
        next_position: Tensor,
        mask: Tensor,
    ) -> Tensor:
        symbol_sequence = (
            st1 @ self.symbol_embedding_1.weight + st2 @ self.symbol_embedding_2.weight
        )
        symbolic_value = _gather_sequence(symbol_sequence, next_position)
        return symbolic_value * mask.unsqueeze(-1).to(symbolic_value.dtype)

    def _candidate_neural_values(self, z_a: Tensor, next_position: Tensor) -> Tensor:
        return _gather_sequence(self.value_proj(z_a), next_position)

    def _hybrid_soft_candidates(
        self,
        st1: Tensor,
        st2: Tensor,
        exact_source: Tensor,
        exact_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Build separately budgeted recent and old soft-only candidates."""

        bsz, n, _ = exact_source.shape
        device = exact_source.device
        dense_count = self.dense_recent_candidates
        sparse_count = self.sparse_old_candidates

        positions = torch.arange(n, device=device).view(1, n, 1)
        if dense_count:
            offsets = torch.arange(1, dense_count + 1, device=device).view(1, 1, -1)
            dense_source = (positions - offsets).expand(bsz, -1, -1)
            dense_mask = dense_source >= 0
            dense_duplicate = (
                dense_source.unsqueeze(-1) == exact_source.unsqueeze(-2)
            ) & exact_mask.unsqueeze(-2)
            dense_mask = dense_mask & ~dense_duplicate.any(dim=-1)
        else:
            dense_source = torch.empty((bsz, n, 0), dtype=torch.long, device=device)
            dense_mask = torch.empty((bsz, n, 0), dtype=torch.bool, device=device)

        if not sparse_count:
            sparse_source = torch.empty((bsz, n, 0), dtype=torch.long, device=device)
            sparse_mask = torch.empty((bsz, n, 0), dtype=torch.bool, device=device)
            return dense_source, dense_mask, sparse_source, sparse_mask

        pool_size = self.sparse_old_pool_size
        # H is constant: evenly spaced anchors cover the admissible old range.
        # Build every position in one tensor operation; a Python loop here
        # would launch O(N) tiny CUDA kernels. The strict upper bound keeps old
        # and recent quotas disjoint.
        old_count = (torch.arange(n, device=device) - dense_count).clamp_min(0)
        anchor_rank = torch.arange(pool_size, device=device).view(1, -1)
        valid_anchors = anchor_rank < old_count.clamp_max(pool_size).view(-1, 1)
        if pool_size == 1:
            position_anchors = torch.zeros((n, 1), dtype=torch.long, device=device)
        else:
            spread = torch.round(
                anchor_rank * (old_count - 1).clamp_min(0).view(-1, 1) / (pool_size - 1)
            ).to(torch.long)
            position_anchors = torch.where(
                old_count.view(-1, 1) <= pool_size,
                anchor_rank.expand(n, -1),
                spread,
            )
        pool = position_anchors.unsqueeze(0).expand(bsz, -1, -1)
        pool_mask = valid_anchors.unsqueeze(0).expand(bsz, -1, -1)
        hard_duplicate = (
            pool.unsqueeze(-1) == exact_source.unsqueeze(-2)
        ) & exact_mask.unsqueeze(-2)
        dense_duplicate = (
            pool.unsqueeze(-1) == dense_source.unsqueeze(-2)
        ) & dense_mask.unsqueeze(-2)
        pool_mask = (
            pool_mask & ~hard_duplicate.any(dim=-1) & ~dense_duplicate.any(dim=-1)
        )
        pool_score = self._soft_match(st1, st2, pool, pool_mask).masked_fill(
            ~pool_mask, -1e9
        )

        # Stable lexicographic order: score descending, then source descending.
        recency_order = torch.argsort(pool, dim=-1, descending=True, stable=True)
        ordered_score = pool_score.gather(-1, recency_order)
        score_order = torch.argsort(ordered_score, dim=-1, descending=True, stable=True)
        selected = recency_order.gather(-1, score_order)[..., :sparse_count]
        sparse_source = pool.gather(-1, selected)
        sparse_mask = pool_mask.gather(-1, selected)
        return dense_source, dense_mask, sparse_source, sparse_mask

    @staticmethod
    def _validate_query_positions(query_positions: Tensor, z_a: Tensor) -> None:
        if not isinstance(query_positions, Tensor):
            raise TypeError("query_positions must be a Tensor")
        if query_positions.dtype != torch.long:
            raise TypeError("query_positions must have dtype torch.long")
        if query_positions.ndim != 2 or query_positions.shape[0] != z_a.shape[0]:
            raise ValueError("query_positions must have shape [B, Q]")
        if query_positions.shape[1] <= 0:
            raise ValueError("query_positions must contain at least one query")
        if query_positions.device != z_a.device:
            raise ValueError("query_positions must be on the same device as z_a")
        if bool(((query_positions < 0) | (query_positions >= z_a.shape[1])).any()):
            raise ValueError("query_positions values must be in [0, N)")
        ordered = query_positions.sort(dim=1).values
        if ordered.shape[1] > 1 and bool((ordered[:, 1:] == ordered[:, :-1]).any()):
            raise ValueError("query_positions must be unique within each batch row")

    @staticmethod
    def _scatter_queries(
        values: Tensor,
        query_positions: Tensor,
        sequence_length: int,
        fill_value: float | int | bool,
    ) -> Tensor:
        shape = (values.shape[0], sequence_length, *values.shape[2:])
        result = torch.full(shape, fill_value, dtype=values.dtype, device=values.device)
        index = query_positions.reshape(
            values.shape[0], values.shape[1], *([1] * (values.ndim - 2))
        ).expand_as(values)
        return result.scatter(1, index, values)

    def _attach_skipped_virtual_gradients(
        self, retrieved: Tensor, updated: Tensor
    ) -> tuple[Tensor, Tensor]:
        if self.virtual_candidates or not torch.is_grad_enabled():
            return retrieved, updated
        virtual_zero = retrieved.new_zeros(())
        for module in (self.virtual_query, self.virtual_key):
            for parameter in module.parameters():
                virtual_zero = virtual_zero + parameter.sum() * 0
        return retrieved + virtual_zero, updated + virtual_zero

    def _forward_query_only(
        self,
        z_a: Tensor,
        z_b: Tensor,
        code_logits: tuple[Tensor, Tensor] | None,
        query_positions: Tensor,
        hard_candidates: PreparedHardCandidates | None,
    ) -> ROSAOutput:
        """Run candidate activations only at selected sequence positions."""

        (soft1, soft2), (st1, st2), hard_tokens = self.encode(z_a, code_logits)
        hard = (
            _build_forward_hard_candidates(
                hard_tokens,
                self.suffix_k,
                self.occurrences_r,
                self.candidate_backend,
            )
            if hard_candidates is None
            else self._validate_prepared_hard_candidates(hard_candidates, hard_tokens)
        )
        bsz, n, _ = z_a.shape
        queries = query_positions.shape[1]
        z_query = _gather_sequence(z_a, query_positions)
        exact_source = _gather_sequence(hard.source_index, query_positions)
        exact_mask = _gather_sequence(hard.mask, query_positions)
        exact_match_length = _gather_sequence(hard.match_length, query_positions)
        exact_frequency = _gather_sequence(hard.frequency, query_positions)
        exact_slots = exact_source.shape[-1]

        if self.virtual_candidates:
            # The pool is discrete and cheap to build for N; floating candidate
            # projections below are restricted to [B, Q, C, *].
            virtual_pool = _gather_sequence(
                build_virtual_pool_indices(bsz, n, self.virtual_pool_size, z_a.device),
                query_positions,
            )
            virtual_pool_mask = virtual_pool >= 0
            virtual_duplicate = (
                virtual_pool.unsqueeze(-1) == exact_source.unsqueeze(-2)
            ) & exact_mask.unsqueeze(-2)
            virtual_pool_mask = virtual_pool_mask & ~virtual_duplicate.any(dim=-1)
            virtual_query = self.virtual_query(z_query).unsqueeze(-2)
            virtual_keys = self.virtual_key(_gather_sequence(z_a, virtual_pool))
            virtual_router_all = (virtual_query * virtual_keys).sum(-1) / math.sqrt(
                self.selector_dim
            )
            virtual_masked = virtual_router_all.masked_fill(~virtual_pool_mask, -1e9)
            _, virtual_top = torch.topk(
                virtual_masked, k=self.virtual_candidates, dim=-1
            )
            virtual_source = virtual_pool.gather(-1, virtual_top)
            virtual_mask = virtual_pool_mask.gather(-1, virtual_top)
            virtual_router = virtual_router_all.gather(-1, virtual_top)
            virtual_mask = virtual_mask & (self.virtual_scale > 0)
            virtual_router = virtual_router * self.virtual_scale
        else:
            shape = (bsz, queries, 0)
            virtual_source = torch.empty(shape, dtype=torch.long, device=z_a.device)
            virtual_mask = torch.empty(shape, dtype=torch.bool, device=z_a.device)
            virtual_router = torch.empty(shape, dtype=z_a.dtype, device=z_a.device)

        dense_count = self.dense_recent_candidates
        sparse_count = self.sparse_old_candidates
        if dense_count:
            offsets = torch.arange(1, dense_count + 1, device=z_a.device).view(1, 1, -1)
            dense_source = query_positions.unsqueeze(-1) - offsets
            dense_mask = dense_source >= 0
            dense_duplicate = (
                dense_source.unsqueeze(-1) == exact_source.unsqueeze(-2)
            ) & exact_mask.unsqueeze(-2)
            dense_mask = dense_mask & ~dense_duplicate.any(dim=-1)
        else:
            dense_source = torch.empty(
                (bsz, queries, 0), dtype=torch.long, device=z_a.device
            )
            dense_mask = torch.empty(
                (bsz, queries, 0), dtype=torch.bool, device=z_a.device
            )

        if sparse_count:
            pool_size = self.sparse_old_pool_size
            old_count = (query_positions - dense_count).clamp_min(0)
            anchor_rank = torch.arange(pool_size, device=z_a.device).view(1, 1, -1)
            sparse_pool_mask = anchor_rank < old_count.clamp_max(pool_size).unsqueeze(
                -1
            )
            if pool_size == 1:
                sparse_pool = torch.zeros(
                    (bsz, queries, 1), dtype=torch.long, device=z_a.device
                )
            else:
                spread = torch.round(
                    anchor_rank
                    * (old_count - 1).clamp_min(0).unsqueeze(-1)
                    / (pool_size - 1)
                ).to(torch.long)
                sparse_pool = torch.where(
                    old_count.unsqueeze(-1) <= pool_size,
                    anchor_rank.expand(bsz, queries, -1),
                    spread,
                )
            hard_duplicate = (
                sparse_pool.unsqueeze(-1) == exact_source.unsqueeze(-2)
            ) & exact_mask.unsqueeze(-2)
            dense_duplicate = (
                sparse_pool.unsqueeze(-1) == dense_source.unsqueeze(-2)
            ) & dense_mask.unsqueeze(-2)
            sparse_pool_mask = (
                sparse_pool_mask
                & ~hard_duplicate.any(dim=-1)
                & ~dense_duplicate.any(dim=-1)
            )
            sparse_pool_score = _soft_match_at_positions_torch(
                st1,
                st2,
                query_positions,
                sparse_pool,
                sparse_pool_mask,
                self.soft_verify_window,
            ).masked_fill(~sparse_pool_mask, -1e9)
            recency_order = torch.argsort(
                sparse_pool, dim=-1, descending=True, stable=True
            )
            ordered_score = sparse_pool_score.gather(-1, recency_order)
            score_order = torch.argsort(
                ordered_score, dim=-1, descending=True, stable=True
            )
            sparse_selected = recency_order.gather(-1, score_order)[..., :sparse_count]
            sparse_source = sparse_pool.gather(-1, sparse_selected)
            sparse_mask = sparse_pool_mask.gather(-1, sparse_selected)
        else:
            sparse_source = torch.empty(
                (bsz, queries, 0), dtype=torch.long, device=z_a.device
            )
            sparse_mask = torch.empty(
                (bsz, queries, 0), dtype=torch.bool, device=z_a.device
            )

        null_source = torch.full(
            (bsz, queries, 1), -1, dtype=torch.long, device=z_a.device
        )
        null_mask = torch.ones((bsz, queries, 1), dtype=torch.bool, device=z_a.device)
        source = torch.cat(
            [exact_source, virtual_source, dense_source, sparse_source, null_source],
            dim=-1,
        )
        mask = torch.cat(
            [exact_mask, virtual_mask, dense_mask, sparse_mask, null_mask], dim=-1
        )
        kind = torch.cat(
            [
                torch.full_like(exact_source, EXACT_KIND),
                torch.full_like(virtual_source, VIRTUAL_KIND),
                torch.full_like(dense_source, VIRTUAL_KIND),
                torch.full_like(sparse_source, VIRTUAL_KIND),
                torch.full_like(null_source, NULL_KIND),
            ],
            dim=-1,
        )
        hard_match_length = torch.cat(
            [
                exact_match_length,
                torch.zeros_like(virtual_source),
                torch.zeros_like(dense_source),
                torch.zeros_like(sparse_source),
                torch.zeros_like(null_source),
            ],
            dim=-1,
        )
        frequency = torch.cat(
            [
                exact_frequency,
                torch.ones_like(virtual_source),
                torch.ones_like(dense_source),
                torch.ones_like(sparse_source),
                torch.zeros_like(null_source),
            ],
            dim=-1,
        )
        non_null_mask = mask & (kind != NULL_KIND)
        soft_match = _soft_match_at_positions_torch(
            st1,
            st2,
            query_positions,
            source,
            non_null_mask,
            self.soft_verify_window,
        )

        safe_source = source.clamp_min(0)
        age = (query_positions.unsqueeze(-1) - safe_source).clamp_min(0)
        log_len = torch.log1p(hard_match_length.to(z_a.dtype))
        log_age = torch.log1p(age.to(z_a.dtype)) / max(math.log1p(n), 1.0)
        log_freq = torch.log1p(frequency.to(z_a.dtype))
        is_exact = (kind == EXACT_KIND).to(z_a.dtype)
        is_virtual = (kind == VIRTUAL_KIND).to(z_a.dtype)
        is_null = (kind == NULL_KIND).to(z_a.dtype)
        candidate_number = torch.arange(source.shape[-1], device=z_a.device).view(
            1, 1, -1
        )
        exact_rosa_slot = hard.rosa_slot.gather(1, query_positions).unsqueeze(-1)
        is_rosa = ((candidate_number == exact_rosa_slot) & (exact_rosa_slot >= 0)).to(
            z_a.dtype
        )
        router_feature = torch.zeros_like(source, dtype=z_a.dtype)
        router_feature[..., exact_slots : exact_slots + self.virtual_candidates] = (
            virtual_router
        )
        features = torch.stack(
            [
                log_len,
                log_age,
                log_freq,
                soft_match / float(self.soft_verify_window),
                is_exact,
                is_virtual,
                is_null,
                is_rosa,
                router_feature,
            ],
            dim=-1,
        )

        query = self.selector_query(z_query).unsqueeze(-2)
        candidate_z = _gather_sequence(z_a, source)
        cand_key = self.selector_key(candidate_z)
        semantic = (query * cand_key).sum(-1) / math.sqrt(self.selector_dim)
        learned = semantic + self.feature_mlp(features).squeeze(-1)
        learned = learned + self.kind_bias[kind]
        learned[..., -1] = learned[..., -1] + self.null_head(z_query).squeeze(-1)
        tie = self.tie_break_scale * (safe_source.to(z_a.dtype) + 1.0) / (n + 1.0)
        rosa_prior = torch.where(
            kind == EXACT_KIND,
            hard_match_length.to(z_a.dtype) + tie,
            torch.full_like(learned, self.virtual_prior_bias),
        )
        rosa_prior = torch.where(
            kind == NULL_KIND,
            torch.full_like(rosa_prior, self.null_prior_bias),
            rosa_prior,
        )
        virtual_log_gate = torch.log(self.virtual_scale.clamp_min(1e-6))
        legacy_virtual = torch.zeros_like(is_virtual)
        legacy_virtual[..., exact_slots : exact_slots + self.virtual_candidates] = 1.0
        rosa_prior = rosa_prior + legacy_virtual * virtual_log_gate
        scores = rosa_prior + self.learned_residual_scale * learned
        scores = scores.masked_fill(~mask, -1e9)

        soft_weights = F.softmax(scores / self.retrieval_temperature, dim=-1)
        hard_scores = scores
        if not self.soft_candidates_forward:
            soft_start = exact_slots + self.virtual_candidates
            soft_end = soft_start + dense_count + sparse_count
            soft_only = torch.zeros_like(mask)
            soft_only[..., soft_start:soft_end] = True
            hard_scores = scores.masked_fill(soft_only, -1e9)
        chosen = hard_scores.argmax(dim=-1)
        hard_weights = F.one_hot(chosen, num_classes=scores.shape[-1]).to(z_a.dtype)
        st_weights = hard_weights + (soft_weights - soft_weights.detach())

        next_position = torch.where(non_null_mask, source + 1, torch.zeros_like(source))
        next_st1 = _gather_sequence(st1, next_position)
        next_st2 = _gather_sequence(st2, next_position)
        symbolic_value = (
            next_st1 @ self.symbol_embedding_1.weight
            + next_st2 @ self.symbol_embedding_2.weight
        ) * non_null_mask.unsqueeze(-1).to(z_a.dtype)
        query_expanded = query.expand_as(cand_key)
        value_gate = torch.sigmoid(
            self.value_gate_head(torch.cat([query_expanded, cand_key], dim=-1))
        ).squeeze(-1)
        value_gate = value_gate * non_null_mask.to(z_a.dtype)
        neural_value = self.value_proj(_gather_sequence(z_a, next_position))
        neural_value = neural_value * value_gate.unsqueeze(-1)
        candidate_value = symbolic_value + self.neural_value_scale * neural_value
        if (dense_count or sparse_count) and not self.soft_candidates_forward:
            historical_end = exact_slots + self.virtual_candidates
            historical_scores = torch.cat(
                [scores[..., :historical_end], scores[..., -1:]], dim=-1
            )
            historical_values = torch.cat(
                [
                    candidate_value[..., :historical_end, :],
                    candidate_value[..., -1:, :],
                ],
                dim=-2,
            )
            historical_soft = F.softmax(
                historical_scores / self.retrieval_temperature, dim=-1
            )
            historical_chosen = historical_scores.argmax(dim=-1)
            historical_hard = F.one_hot(
                historical_chosen, num_classes=historical_scores.shape[-1]
            ).to(z_a.dtype)
            historical_st = historical_hard + (
                historical_soft - historical_soft.detach()
            )
            historical_retrieved = torch.sum(
                historical_st.unsqueeze(-1) * historical_values, dim=-2
            )
            union_soft_retrieved = torch.sum(
                soft_weights.unsqueeze(-1) * candidate_value, dim=-2
            )
            historical_soft_retrieved = torch.sum(
                historical_soft.unsqueeze(-1) * historical_values, dim=-2
            )
            backward_delta = union_soft_retrieved - historical_soft_retrieved
            retrieved = historical_retrieved + (
                backward_delta - backward_delta.detach()
            )
        else:
            retrieved = torch.sum(st_weights.unsqueeze(-1) * candidate_value, dim=-2)

        read_gate = torch.sigmoid(self.read_gate_head(z_query))
        updated_query = _gather_sequence(
            z_b, query_positions
        ) + read_gate * self.out_proj(retrieved)
        update_index = query_positions.unsqueeze(-1).expand_as(updated_query)
        updated = z_b.scatter(1, update_index, updated_query)
        retrieved, updated = self._attach_skipped_virtual_gradients(retrieved, updated)
        chosen_source = source.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
        chosen_kind = kind.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
        chosen_match = hard_match_length.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
        chosen_next = (chosen_source + 1).clamp(min=0, max=n - 1)
        chosen_id1 = hard_tokens // self.codebook_sizes[1]
        chosen_id2 = hard_tokens % self.codebook_sizes[1]
        c1 = _gather_sequence(chosen_id1.unsqueeze(-1), chosen_next).squeeze(-1)
        c2 = _gather_sequence(chosen_id2.unsqueeze(-1), chosen_next).squeeze(-1)
        chosen_token = c1 * self.codebook_sizes[1] + c2
        chosen_token = torch.where(
            chosen_kind == NULL_KIND, -torch.ones_like(chosen_token), chosen_token
        )

        eps = 1e-9
        rosa_target = torch.where(
            exact_rosa_slot.squeeze(-1) >= 0,
            exact_rosa_slot.squeeze(-1),
            torch.full_like(chosen, scores.shape[-1] - 1),
        )
        rosa_prob = soft_weights.gather(-1, rosa_target.unsqueeze(-1)).squeeze(-1)
        hard_prob = soft_weights.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
        virtual_slice = soft_weights[..., exact_slots:-1]
        aux_losses = {
            "rosa_distillation": -torch.log(rosa_prob.clamp_min(eps)).mean(),
            "hard_soft_consistency": -torch.log(hard_prob.clamp_min(eps)).mean(),
            "code_balance": 0.5 * (_balance_kl(soft1) + _balance_kl(soft2)),
            "virtual_usage": virtual_slice.sum(-1).mean(),
        }

        scatter = self._scatter_queries
        return ROSAOutput(
            updated=updated,
            retrieved=scatter(retrieved, query_positions, n, 0),
            hard_tokens=hard_tokens,
            code_soft=(soft1, soft2),
            code_st=(st1, st2),
            candidate_source_index=scatter(source, query_positions, n, -1),
            candidate_kind=scatter(kind, query_positions, n, -1),
            candidate_mask=scatter(mask, query_positions, n, False),
            candidate_scores=scatter(scores, query_positions, n, -torch.inf),
            soft_weights=scatter(soft_weights, query_positions, n, 0),
            hard_weights=scatter(hard_weights, query_positions, n, 0),
            chosen_candidate=scatter(chosen, query_positions, n, -1),
            chosen_source_index=scatter(chosen_source, query_positions, n, -1),
            chosen_token=scatter(chosen_token, query_positions, n, -1),
            chosen_match_length=scatter(chosen_match, query_positions, n, 0),
            chosen_is_virtual=scatter(
                chosen_kind == VIRTUAL_KIND, query_positions, n, False
            ),
            hard_rosa_source_index=scatter(
                hard.rosa_source_index.gather(1, query_positions),
                query_positions,
                n,
                -1,
            ),
            hard_rosa_predicted_tokens=scatter(
                hard.rosa_predicted_tokens.gather(1, query_positions),
                query_positions,
                n,
                -1,
            ),
            hard_rosa_match_length=scatter(
                hard.rosa_match_length.gather(1, query_positions),
                query_positions,
                n,
                0,
            ),
            soft_match_score=scatter(soft_match, query_positions, n, 0),
            read_gate=scatter(read_gate, query_positions, n, 0),
            value_gate=scatter(value_gate, query_positions, n, 0),
            aux_losses=aux_losses,
        )

    def forward(
        self,
        z_a: Tensor,
        z_b: Tensor | None = None,
        code_logits: tuple[Tensor, Tensor] | None = None,
        *,
        query_positions: Tensor | None = None,
        hard_candidates: PreparedHardCandidates | None = None,
    ) -> ROSAOutput:
        if z_a.ndim != 3 or z_a.shape[-1] != self.d_model:
            raise ValueError("z_a must have shape [B, N, d_model]")
        if z_a.shape[1] <= 0:
            raise ValueError("sequence length must be > 0")
        if z_b is None:
            z_b = z_a
        elif z_b.shape != z_a.shape:
            raise ValueError("z_b must have the same shape as z_a")

        if query_positions is not None:
            self._validate_query_positions(query_positions, z_a)
            n = z_a.shape[1]
            if query_positions.shape[1] == n:
                full_positions = torch.arange(n, device=z_a.device).expand(
                    z_a.shape[0], -1
                )
                if torch.equal(query_positions, full_positions):
                    # Route the canonical full query set through the untouched
                    # historical implementation, including operation order.
                    return self.forward(
                        z_a, z_b, code_logits, hard_candidates=hard_candidates
                    )
            return self._forward_query_only(
                z_a, z_b, code_logits, query_positions, hard_candidates
            )

        (soft1, soft2), (st1, st2), hard_tokens = self.encode(z_a, code_logits)
        # Keep the exact, non-differentiable automaton on CPU as proposed for
        # RWKV-8 ROSA. Accelerator backends may optimize the tensor path around
        # it, but must not silently replace this exact discrete control path.
        hard = (
            _build_forward_hard_candidates(
                hard_tokens,
                self.suffix_k,
                self.occurrences_r,
                self.candidate_backend,
            )
            if hard_candidates is None
            else self._validate_prepared_hard_candidates(hard_candidates, hard_tokens)
        )
        exact_source = hard.source_index
        exact_mask = hard.mask
        exact_slots = exact_source.shape[-1]

        if self.virtual_candidates:
            virtual_source, virtual_mask, virtual_router = self._virtual_candidates(
                z_a, exact_source, exact_mask
            )
            virtual_mask = virtual_mask & (self.virtual_scale > 0)
            virtual_router = virtual_router * self.virtual_scale
        else:
            shape = (*exact_source.shape[:2], 0)
            virtual_source = torch.empty(shape, dtype=torch.long, device=z_a.device)
            virtual_mask = torch.empty(shape, dtype=torch.bool, device=z_a.device)
            virtual_router = torch.empty(shape, dtype=z_a.dtype, device=z_a.device)

        dense_source, dense_mask, sparse_source, sparse_mask = (
            self._hybrid_soft_candidates(st1, st2, exact_source, exact_mask)
        )

        bsz, n, _ = z_a.shape
        null_source = torch.full((bsz, n, 1), -1, dtype=torch.long, device=z_a.device)
        null_mask = torch.ones((bsz, n, 1), dtype=torch.bool, device=z_a.device)
        source = torch.cat(
            [
                exact_source,
                virtual_source,
                dense_source,
                sparse_source,
                null_source,
            ],
            dim=-1,
        )
        mask = torch.cat(
            [exact_mask, virtual_mask, dense_mask, sparse_mask, null_mask], dim=-1
        )

        exact_kind = torch.full_like(exact_source, EXACT_KIND)
        virtual_kind = torch.full_like(virtual_source, VIRTUAL_KIND)
        dense_kind = torch.full_like(dense_source, VIRTUAL_KIND)
        sparse_kind = torch.full_like(sparse_source, VIRTUAL_KIND)
        null_kind = torch.full_like(null_source, NULL_KIND)
        kind = torch.cat(
            [exact_kind, virtual_kind, dense_kind, sparse_kind, null_kind], dim=-1
        )

        zeros_virtual = torch.zeros_like(virtual_source)
        zeros_dense = torch.zeros_like(dense_source)
        zeros_sparse = torch.zeros_like(sparse_source)
        zeros_null = torch.zeros_like(null_source)
        hard_match_length = torch.cat(
            [
                hard.match_length,
                zeros_virtual,
                zeros_dense,
                zeros_sparse,
                zeros_null,
            ],
            dim=-1,
        )
        frequency = torch.cat(
            [
                hard.frequency,
                torch.ones_like(virtual_source),
                torch.ones_like(dense_source),
                torch.ones_like(sparse_source),
                zeros_null,
            ],
            dim=-1,
        )

        non_null_mask = mask & (kind != NULL_KIND)
        soft_match = self._soft_match(st1, st2, source, non_null_mask)

        positions = torch.arange(n, device=z_a.device).view(1, n, 1).expand_as(source)
        safe_source = source.clamp_min(0)
        age = (positions - safe_source).clamp_min(0)
        log_len = torch.log1p(hard_match_length.to(z_a.dtype))
        log_age = torch.log1p(age.to(z_a.dtype)) / max(math.log1p(n), 1.0)
        log_freq = torch.log1p(frequency.to(z_a.dtype))
        soft_match_norm = soft_match / float(self.soft_verify_window)

        is_exact = (kind == EXACT_KIND).to(z_a.dtype)
        is_virtual = (kind == VIRTUAL_KIND).to(z_a.dtype)
        is_null = (kind == NULL_KIND).to(z_a.dtype)
        candidate_number = torch.arange(source.shape[-1], device=z_a.device).view(
            1, 1, -1
        )
        rosa_slot = hard.rosa_slot.unsqueeze(-1)
        is_rosa = ((candidate_number == rosa_slot) & (rosa_slot >= 0)).to(z_a.dtype)
        router_feature = torch.zeros_like(source, dtype=z_a.dtype)
        router_feature[..., exact_slots : exact_slots + self.virtual_candidates] = (
            virtual_router
        )

        features = torch.stack(
            [
                log_len,
                log_age,
                log_freq,
                soft_match_norm,
                is_exact,
                is_virtual,
                is_null,
                is_rosa,
                router_feature,
            ],
            dim=-1,
        )

        query = self.selector_query(z_a).unsqueeze(-2)
        cand_key = self._candidate_selector_keys(z_a, source)
        semantic = (query * cand_key).sum(-1) / math.sqrt(self.selector_dim)
        learned = semantic + self.feature_mlp(features).squeeze(-1)
        learned = learned + self.kind_bias[kind]
        learned[..., -1] = learned[..., -1] + self.null_head(z_a).squeeze(-1)

        # Exact ROSA prior plus a learned residual. Integer match length
        # dominates recency tie-breaking because tie_break_scale < 1.
        tie = self.tie_break_scale * (safe_source.to(z_a.dtype) + 1.0) / (n + 1.0)
        rosa_prior = torch.where(
            kind == EXACT_KIND,
            hard_match_length.to(z_a.dtype) + tie,
            torch.full_like(learned, self.virtual_prior_bias),
        )
        rosa_prior = torch.where(
            kind == NULL_KIND,
            torch.full_like(rosa_prior, self.null_prior_bias),
            rosa_prior,
        )
        # Curriculum gate for virtual candidates. At zero, the virtual branch
        # is effectively disabled while exact suffix and NULL candidates remain.
        virtual_log_gate = torch.log(self.virtual_scale.clamp_min(1e-6))
        legacy_virtual = torch.zeros_like(is_virtual)
        legacy_virtual[..., exact_slots : exact_slots + self.virtual_candidates] = 1.0
        rosa_prior = rosa_prior + legacy_virtual * virtual_log_gate

        scores = rosa_prior + self.learned_residual_scale * learned
        scores = scores.masked_fill(~mask, -1e9)

        soft_weights = F.softmax(scores / self.retrieval_temperature, dim=-1)
        hard_scores = scores
        if not self.soft_candidates_forward:
            soft_start = exact_slots + self.virtual_candidates
            soft_end = (
                soft_start + self.dense_recent_candidates + self.sparse_old_candidates
            )
            soft_only = torch.zeros_like(mask)
            soft_only[..., soft_start:soft_end] = True
            hard_scores = scores.masked_fill(soft_only, -1e9)
        chosen = hard_scores.argmax(dim=-1)
        hard_weights = F.one_hot(chosen, num_classes=scores.shape[-1]).to(z_a.dtype)
        # Parenthesizing the zero-valued correction makes the forward exactly
        # one-hot while retaining the full-union softmax backward.
        st_weights = hard_weights + (soft_weights - soft_weights.detach())

        next_position = torch.where(non_null_mask, source + 1, torch.zeros_like(source))
        symbolic_value = self._candidate_symbolic_values(
            st1, st2, next_position, non_null_mask
        )
        query_expanded = query.expand_as(cand_key)
        value_gate = torch.sigmoid(
            self.value_gate_head(torch.cat([query_expanded, cand_key], dim=-1))
        ).squeeze(-1)
        value_gate = value_gate * non_null_mask.to(z_a.dtype)
        neural_value = self._candidate_neural_values(z_a, next_position)
        neural_value = neural_value * value_gate.unsqueeze(-1)
        candidate_value = symbolic_value + self.neural_value_scale * neural_value
        hybrid_enabled = bool(
            self.dense_recent_candidates or self.sparse_old_candidates
        )
        if hybrid_enabled and not self.soft_candidates_forward:
            # Recreate the historical [hard, legacy virtual, NULL] reduction
            # exactly, then replace only its soft backward correction with the
            # full-union correction. The parenthesized delta is exactly zero
            # in forward arithmetic.
            historical_end = exact_slots + self.virtual_candidates
            historical_scores = torch.cat(
                [scores[..., :historical_end], scores[..., -1:]], dim=-1
            )
            historical_values = torch.cat(
                [
                    candidate_value[..., :historical_end, :],
                    candidate_value[..., -1:, :],
                ],
                dim=-2,
            )
            historical_soft = F.softmax(
                historical_scores / self.retrieval_temperature, dim=-1
            )
            historical_chosen = historical_scores.argmax(dim=-1)
            historical_hard = F.one_hot(
                historical_chosen, num_classes=historical_scores.shape[-1]
            ).to(z_a.dtype)
            historical_st = historical_hard + (
                historical_soft - historical_soft.detach()
            )
            historical_retrieved = torch.sum(
                historical_st.unsqueeze(-1) * historical_values, dim=-2
            )
            union_soft_retrieved = torch.sum(
                soft_weights.unsqueeze(-1) * candidate_value, dim=-2
            )
            historical_soft_retrieved = torch.sum(
                historical_soft.unsqueeze(-1) * historical_values, dim=-2
            )
            backward_delta = union_soft_retrieved - historical_soft_retrieved
            retrieved = historical_retrieved + (
                backward_delta - backward_delta.detach()
            )
        else:
            retrieved = torch.sum(st_weights.unsqueeze(-1) * candidate_value, dim=-2)

        read_gate = torch.sigmoid(self.read_gate_head(z_a))
        updated = z_b + read_gate * self.out_proj(retrieved)
        retrieved, updated = self._attach_skipped_virtual_gradients(retrieved, updated)

        chosen_source = source.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
        chosen_kind = kind.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
        chosen_match = hard_match_length.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
        chosen_next = (chosen_source + 1).clamp(min=0, max=n - 1)
        chosen_id1 = hard_tokens // self.codebook_sizes[1]
        chosen_id2 = hard_tokens % self.codebook_sizes[1]
        # The IDs are already discrete. Gather them directly instead of
        # allocating two one-hot sequences only to immediately argmax them.
        c1 = _gather_sequence(chosen_id1.unsqueeze(-1), chosen_next).squeeze(-1)
        c2 = _gather_sequence(chosen_id2.unsqueeze(-1), chosen_next).squeeze(-1)
        chosen_token = c1 * self.codebook_sizes[1] + c2
        chosen_token = torch.where(
            chosen_kind == NULL_KIND, -torch.ones_like(chosen_token), chosen_token
        )

        eps = 1e-9
        rosa_target = torch.where(
            hard.rosa_slot >= 0,
            hard.rosa_slot,
            torch.full_like(hard.rosa_slot, scores.shape[-1] - 1),
        )
        rosa_prob = soft_weights.gather(-1, rosa_target.unsqueeze(-1)).squeeze(-1)
        hard_prob = soft_weights.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
        virtual_slice = soft_weights[..., exact_slots:-1]
        aux_losses = {
            "rosa_distillation": -torch.log(rosa_prob.clamp_min(eps)).mean(),
            "hard_soft_consistency": -torch.log(hard_prob.clamp_min(eps)).mean(),
            "code_balance": 0.5 * (_balance_kl(soft1) + _balance_kl(soft2)),
            "virtual_usage": virtual_slice.sum(-1).mean(),
        }

        return ROSAOutput(
            updated=updated,
            retrieved=retrieved,
            hard_tokens=hard_tokens,
            code_soft=(soft1, soft2),
            code_st=(st1, st2),
            candidate_source_index=source,
            candidate_kind=kind,
            candidate_mask=mask,
            candidate_scores=scores,
            soft_weights=soft_weights,
            hard_weights=hard_weights,
            chosen_candidate=chosen,
            chosen_source_index=chosen_source,
            chosen_token=chosen_token,
            chosen_match_length=chosen_match,
            chosen_is_virtual=chosen_kind == VIRTUAL_KIND,
            hard_rosa_source_index=hard.rosa_source_index,
            hard_rosa_predicted_tokens=hard.rosa_predicted_tokens,
            hard_rosa_match_length=hard.rosa_match_length,
            soft_match_score=soft_match,
            read_gate=read_gate,
            value_gate=value_gate,
            aux_losses=aux_losses,
        )

    @staticmethod
    def combine_losses(
        lm_loss: Tensor,
        aux_losses: dict[str, Tensor],
        rosa_weight: float = 0.0,
        consistency_weight: float = 0.0,
        balance_weight: float = 0.0,
        virtual_weight: float = 0.0,
    ) -> Tensor:
        required = {
            "rosa_distillation",
            "hard_soft_consistency",
            "code_balance",
            "virtual_usage",
        }
        if set(aux_losses) != required:
            raise ValueError(
                f"aux_losses must have exactly the keys {sorted(required)}"
            )
        return (
            lm_loss
            + rosa_weight * aux_losses["rosa_distillation"]
            + consistency_weight * aux_losses["hard_soft_consistency"]
            + balance_weight * aux_losses["code_balance"]
            + virtual_weight * aux_losses["virtual_usage"]
        )


InferenceBackend = Literal[
    "auto",
    "python",
    "numba",
    "rlbwt",
    "rlbwt_native",
    "rlbwt_compact256",
    "rlbwt_mc128",
    "rlbwt_mc192",
]
InferenceMode = Literal["top1", "rich"]


@dataclass(frozen=True, slots=True)
class InferenceOutput:
    """Result returned by the unified inference-state methods.

    ``candidates`` is ``None`` in top-1 mode, a ``CandidateStep`` for one
    decoding step, and ``HardCandidates`` for a prefill in rich mode.
    """

    predicted_tokens: Tensor
    candidates: object | None


@dataclass(slots=True)
class ROSAInferenceState:
    """Mutable, fixed-capacity state for exact autoregressive ROSA inference.

    Create states with :func:`init_inference_state`. Each state is independent,
    CPU-resident, and intended to be mutated by one decoding stream at a time.
    """

    batch_size: int
    max_length: int
    backend: Literal[
        "python",
        "numba",
        "rlbwt",
        "rlbwt_native",
        "rlbwt_compact256",
        "rlbwt_mc128",
        "rlbwt_mc192",
    ]
    mode: InferenceMode
    ragged: bool
    suffix_k: int
    occurrences_r: int
    _impl: object | None = field(repr=False)

    def _require_impl(self) -> object:
        impl = self._impl
        if impl is None:
            raise RuntimeError("state is closed")
        return impl

    @property
    def position(self) -> int:
        """Number of tokens consumed by every batch row."""

        impl = self._require_impl()
        if self.ragged:
            raise AttributeError(
                "position is undefined for ragged states; use positions"
            )
        return int(cast(Any, impl).position)

    @property
    def positions(self) -> Tensor:
        """Consumed-token counts for every row, always returned as a copy."""

        impl = self._require_impl()
        if self.ragged:
            if self.mode == "top1":
                return cast(Any, impl).positions.clone()
            return torch.from_numpy(cast(Any, impl).positions.copy())
        return torch.full((self.batch_size,), self.position, dtype=torch.long)

    def step(
        self,
        tokens: Tensor,
        active: Tensor | None = None,
        reset: Tensor | None = None,
    ) -> InferenceOutput:
        """Consume one token per selected row using the configured mode."""

        self._require_impl()
        return _inference_step(self, tokens, active=active, reset=reset)

    def step_into(self, tokens: Tensor, buffers: object) -> InferenceOutput:
        """Consume one rich uniform step into opt-in reusable buffers.

        Candidate tensors in the result alias ``buffers`` until its next use.
        """

        self._require_impl()
        return _inference_step_into(self, tokens, buffers)

    def prefill(self, tokens: Tensor) -> InferenceOutput:
        """Consume an initial dense context and return every step result."""

        self._require_impl()
        return _inference_prefill(self, tokens)

    def close(self) -> None:
        """Release persistent backend storage; repeated calls are safe."""

        impl = self._impl
        if impl is None:
            return
        self._impl = None
        close = getattr(impl, "close", None)
        if close is not None:
            close()

    def __enter__(self) -> ROSAInferenceState:
        self._require_impl()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def reset(self) -> None:
        """Replace storage with a fresh open state, reopening a closed state."""

        self.close()
        self._impl = _make_inference_impl(
            self.batch_size,
            self.max_length,
            self.backend,
            self.mode,
            self.ragged,
            self.suffix_k,
            self.occurrences_r,
        )


def _make_inference_impl(
    batch_size: int,
    max_length: int,
    backend: Literal[
        "python",
        "numba",
        "rlbwt",
        "rlbwt_native",
        "rlbwt_compact256",
        "rlbwt_mc128",
        "rlbwt_mc192",
    ],
    mode: InferenceMode = "top1",
    ragged: bool = False,
    suffix_k: int = 16,
    occurrences_r: int = 4,
) -> object:
    if mode == "rich":
        if backend != "numba":
            raise ValueError(  # pragma: no cover - validated by public initializer
                "rich inference requires backend='auto' or 'numba'"
            )
        if ragged:
            return init_candidate_state(
                batch_size,
                max_length,
                suffix_k=suffix_k,
                occurrences_r=occurrences_r,
                ragged=True,
            )
        return init_candidate_state(
            batch_size,
            max_length,
            suffix_k=suffix_k,
            occurrences_r=occurrences_r,
        )
    if ragged:
        if backend != "numba":
            raise ValueError(  # pragma: no cover - validated by public initializer
                "ragged inference requires backend='auto' or 'numba'"
            )
        from .ragged import init_ragged_state

        return init_ragged_state(batch_size, max_length)
    if backend == "python":
        return _init_python_inference_state(batch_size, max_length)
    if backend == "rlbwt":
        from ._rlbwt_backend import _init_rlbwt_state

        return _init_rlbwt_state(batch_size, max_length)
    if backend == "rlbwt_native":
        from ._rlbwt_backend import _init_native_rlbwt_state

        return _init_native_rlbwt_state(batch_size, max_length)
    if backend == "rlbwt_compact256":
        from ._rlbwt_backend import _init_native_rlbwt_compact_state

        return _init_native_rlbwt_compact_state(batch_size, max_length)
    if backend in {"rlbwt_mc128", "rlbwt_mc192"}:
        from ._rlbwt_backend import _init_native_rlbwt_mc_state

        return _init_native_rlbwt_mc_state(
            batch_size,
            max_length,
            2 if backend == "rlbwt_mc128" else 3,
        )
    try:
        from ._stateful_numba import _init_inference_state
    except ModuleNotFoundError as error:
        if error.name not in {"numba", "numpy"}:
            raise
        raise ImportError(
            "Numba inference requires `pip install rosa-torch[numba]`"
        ) from error
    return _init_inference_state(batch_size, max_length)


def init_candidate_state(
    batch_size: int,
    max_length: int,
    *,
    suffix_k: int = 16,
    occurrences_r: int = 4,
    ragged: bool = False,
) -> Any:
    """Allocate the exact bounded rich-candidate inference state.

    This optional path requires the ``numba`` extra. It tracks K suffix states,
    top-R occurrences and unbounded frequencies without eager suffix-chain
    propagation.
    """

    try:
        from ._stateful_candidates_numba import init_candidate_state as initialize
    except ModuleNotFoundError as error:
        if error.name == "numba":
            raise RuntimeError(
                "rich stateful candidates require the 'numba' extra"
            ) from error
        raise  # pragma: no cover - unrelated optional import failure
    if ragged:
        return initialize(
            batch_size,
            max_length,
            suffix_k=suffix_k,
            occurrences_r=occurrences_r,
            ragged=True,
        )
    return initialize(
        batch_size,
        max_length,
        suffix_k=suffix_k,
        occurrences_r=occurrences_r,
    )


def init_candidate_buffers(state: object, *, sequence_length: int | None = None) -> Any:
    """Allocate reusable packed CPU outputs for rich ``*_into`` calls."""

    if isinstance(state, ROSAInferenceState):
        if state.mode != "rich":
            raise ValueError("candidate buffers require a rich state")
        state = state._impl
    from ._stateful_candidates_numba import init_candidate_buffers as initialize

    return initialize(cast(Any, state), sequence_length=sequence_length)


def forward_candidates_step(state: object, tokens: Tensor) -> Any:
    """Consume one token and return exact bounded hard candidates."""

    if isinstance(state, ROSAInferenceState):
        if state.mode != "rich":
            raise ValueError("forward_candidates_step requires a rich state")
        candidates = state.step(tokens).candidates
        if candidates is None:  # pragma: no cover - guarded by mode
            raise RuntimeError("rich inference did not return candidates")
        return candidates

    try:
        from ._stateful_candidates_numba import forward_candidates_step as step
    except ModuleNotFoundError as error:
        if error.name == "numba":
            raise RuntimeError(
                "rich stateful candidates require the 'numba' extra"
            ) from error
        raise  # pragma: no cover - unrelated optional import failure
    return step(cast(Any, state), tokens)


def forward_candidates_step_into(state: object, tokens: Tensor, buffers: object) -> Any:
    """Consume one rich step into explicitly reusable caller-owned buffers."""

    if isinstance(state, ROSAInferenceState):
        candidates = state.step_into(tokens, buffers).candidates
        if candidates is None:  # pragma: no cover - guarded by step_into
            raise RuntimeError("rich inference did not return candidates")
        return candidates
    from ._stateful_candidates_numba import forward_candidates_step_into as step

    return step(cast(Any, state), tokens, cast(Any, buffers))


def init_inference_state(
    batch_size: int,
    max_length: int = 8192,
    *,
    backend: InferenceBackend = "auto",
    mode: InferenceMode = "top1",
    ragged: bool = False,
    suffix_k: int = 16,
    occurrences_r: int = 4,
) -> ROSAInferenceState:
    """Create an exact top-1 or rich autoregressive ROSA inference state.

    ``backend="auto"`` selects the Link-Cut Tree Numba backend when the
    optional dependency is installed and otherwise uses the exact Python
    fallback for uniform top-1 inference. Rich and ragged modes require the
    Numba extra. Capacity is fixed so memory use remains predictable.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if max_length <= 0:
        raise ValueError("max_length must be > 0")
    if backend not in {
        "auto",
        "python",
        "numba",
        "rlbwt",
        "rlbwt_native",
        "rlbwt_compact256",
        "rlbwt_mc128",
        "rlbwt_mc192",
    }:
        raise ValueError(
            "backend must be 'auto', 'python', 'numba', 'rlbwt', "
            "'rlbwt_native', 'rlbwt_compact256', 'rlbwt_mc128', or 'rlbwt_mc192'"
        )
    if mode not in {"top1", "rich"}:
        raise ValueError("mode must be 'top1' or 'rich'")
    if suffix_k <= 0:
        raise ValueError("suffix_k must be > 0")
    if occurrences_r <= 0:
        raise ValueError("occurrences_r must be > 0")
    if (mode == "rich" or ragged) and backend in {
        "python",
        "rlbwt",
        "rlbwt_native",
        "rlbwt_compact256",
        "rlbwt_mc128",
        "rlbwt_mc192",
    }:
        feature = "rich" if mode == "rich" else "ragged"
        raise ValueError(f"{feature} inference does not support backend={backend!r}")

    selected: Literal[
        "python",
        "numba",
        "rlbwt",
        "rlbwt_native",
        "rlbwt_compact256",
        "rlbwt_mc128",
        "rlbwt_mc192",
    ]
    if backend == "auto":
        if mode == "rich" or ragged:
            selected = "numba"
            impl = _make_inference_impl(
                batch_size,
                max_length,
                selected,
                mode,
                ragged,
                suffix_k,
                occurrences_r,
            )
            return ROSAInferenceState(
                batch_size,
                max_length,
                selected,
                mode,
                ragged,
                suffix_k,
                occurrences_r,
                impl,
            )
        try:
            impl = _make_inference_impl(batch_size, max_length, "numba")
            selected = "numba"
        except ImportError:
            impl = _make_inference_impl(batch_size, max_length, "python")
            selected = "python"
    else:
        selected = backend
        impl = _make_inference_impl(
            batch_size,
            max_length,
            selected,
            mode,
            ragged,
            suffix_k,
            occurrences_r,
        )
    return ROSAInferenceState(
        batch_size,
        max_length,
        selected,
        mode,
        ragged,
        suffix_k,
        occurrences_r,
        impl,
    )


def _normalize_step_tokens(
    state: ROSAInferenceState, token: Tensor
) -> tuple[Tensor, bool]:
    if not isinstance(token, Tensor):
        raise TypeError("token must be a torch.Tensor")
    squeeze = token.ndim == 0
    if squeeze and state.batch_size == 1:
        token = token.unsqueeze(0)
    if token.ndim != 1 or token.shape[0] != state.batch_size:
        raise ValueError("token must have shape [batch_size]")
    if token.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError("token must use an integer dtype")
    return token, squeeze


def _rich_ragged_step(
    state: ROSAInferenceState,
    tokens: Tensor,
    active: Tensor | None,
    reset: Tensor | None,
) -> Any:
    from ._stateful_candidates_numba import forward_candidates_step_masked

    active_mask = (
        torch.ones(state.batch_size, dtype=torch.bool) if active is None else active
    )
    return forward_candidates_step_masked(
        cast(Any, state._impl), tokens, active_mask, reset
    )


def _inference_step(
    state: ROSAInferenceState,
    token: Tensor,
    *,
    active: Tensor | None = None,
    reset: Tensor | None = None,
) -> InferenceOutput:
    if not isinstance(state, ROSAInferenceState):  # pragma: no cover - method only
        raise TypeError("state must be a ROSAInferenceState")
    token, squeeze = _normalize_step_tokens(state, token)
    if not state.ragged and (active is not None or reset is not None):
        raise ValueError("active and reset are only valid for ragged states")

    candidates: object | None = None
    if state.mode == "rich":
        if state.ragged:
            candidates = _rich_ragged_step(state, token, active, reset)
        else:
            candidates = forward_candidates_step(state._impl, token)
        output = cast(Any, candidates).rosa_predicted_tokens
    elif state.ragged:
        output = cast(Any, state._impl).step(token, active=active, reset=reset)
    elif state.backend == "python":
        output = _python_forward_step(cast(_PythonInferenceState, state._impl), token)
    elif state.backend == "rlbwt":
        from ._rlbwt_backend import _forward_step as rlbwt_step

        output = rlbwt_step(cast(Any, state._impl), token)
    elif state.backend == "rlbwt_compact256":
        from ._rlbwt_backend import _compact_forward_step

        output = _compact_forward_step(cast(Any, state._impl), token)
    elif state.backend in {"rlbwt_native", "rlbwt_mc128", "rlbwt_mc192"}:
        from ._rlbwt_backend import _native_forward_step

        output = _native_forward_step(cast(Any, state._impl), token)
    else:
        from ._stateful_numba import _forward_step

        output = _forward_step(cast(Any, state._impl), token)
    if squeeze:
        output = output[0]
        if candidates is not None:
            from ._stateful_candidates_numba import CandidateStep

            candidate_data = cast(Any, candidates)
            candidates = CandidateStep(
                *(
                    getattr(candidate_data, name)[0]
                    for name in candidate_data.__dataclass_fields__
                )
            )
    return InferenceOutput(output, candidates)


def _inference_step_into(
    state: ROSAInferenceState, token: Tensor, buffers: object
) -> InferenceOutput:
    if state.mode != "rich" or state.ragged:
        raise ValueError("step_into requires a uniform rich inference state")
    token, squeeze = _normalize_step_tokens(state, token)
    from ._stateful_candidates_numba import forward_candidates_step_into as step

    candidates = step(cast(Any, state._impl), token, cast(Any, buffers))
    output = candidates.rosa_predicted_tokens
    if squeeze:
        output = output[0]
        from ._stateful_candidates_numba import CandidateStep

        candidates = CandidateStep(
            *(getattr(candidates, name)[0] for name in candidates.__dataclass_fields__)
        )
    return InferenceOutput(output, candidates)


def forward_step(state: ROSAInferenceState, token: Tensor) -> Tensor:
    """Consume one token per batch row and return exact ROSA predictions.

    ``token`` must be an integer tensor shaped ``[batch_size]``. A scalar is
    also accepted when ``batch_size == 1`` and produces a scalar prediction.
    The state remains on CPU; the prediction is returned on the token device.
    """

    if not isinstance(state, ROSAInferenceState):
        raise TypeError("state must be a ROSAInferenceState")
    return state.step(token).predicted_tokens


def _empty_rich_candidates(
    state: ROSAInferenceState, device: torch.device
) -> HardCandidates:
    slots = state.suffix_k * state.occurrences_r
    candidate_shape = (state.batch_size, 0, slots)
    rosa_shape = (state.batch_size, 0)
    source = torch.empty(candidate_shape, dtype=torch.long, device=device)
    zeros = torch.empty(candidate_shape, dtype=torch.long, device=device)
    return HardCandidates(
        source,
        zeros.clone(),
        zeros.clone(),
        zeros.clone(),
        torch.empty(candidate_shape, dtype=torch.bool, device=device),
        torch.empty(rosa_shape, dtype=torch.long, device=device),
        torch.empty(rosa_shape, dtype=torch.long, device=device),
        torch.empty(rosa_shape, dtype=torch.long, device=device),
        torch.empty(rosa_shape, dtype=torch.long, device=device),
    )


def _stack_rich_steps(steps: list[Any], dim: int) -> HardCandidates:
    return HardCandidates(
        *(
            torch.stack([getattr(step, name) for step in steps], dim=dim)
            for name in HardCandidates.__dataclass_fields__
        )
    )


def _inference_prefill(state: ROSAInferenceState, tokens: Tensor) -> InferenceOutput:
    if not isinstance(state, ROSAInferenceState):  # pragma: no cover - method only
        raise TypeError("state must be a ROSAInferenceState")
    if not isinstance(tokens, Tensor):
        raise TypeError("tokens must be a torch.Tensor")
    squeeze = tokens.ndim == 1
    if squeeze and state.batch_size == 1:
        tokens = tokens.unsqueeze(0)
    if tokens.ndim != 2 or tokens.shape[0] != state.batch_size:
        raise ValueError("tokens must have shape [batch_size, sequence_length]")
    if tokens.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError("tokens must use an integer dtype")
    if bool(torch.any(state.positions != 0)):
        raise RuntimeError("prefill requires an empty inference state")
    if tokens.shape[1] > state.max_length:
        raise RuntimeError("inference state capacity exceeded")

    candidates: HardCandidates | None = None
    if state.mode == "top1" and not state.ragged and state.backend == "python":
        backend_state = cast(_PythonInferenceState, state._impl)
        if tokens.shape[1] == 0:
            output = torch.empty(tokens.shape, dtype=torch.long, device=tokens.device)
        else:
            output = torch.stack(
                [
                    _python_forward_step(backend_state, tokens[:, position])
                    for position in range(tokens.shape[1])
                ],
                dim=1,
            )
    elif state.mode == "top1" and not state.ragged and state.backend == "rlbwt":
        from ._rlbwt_backend import _prefill as rlbwt_prefill

        output = rlbwt_prefill(cast(Any, state._impl), tokens)
    elif (
        state.mode == "top1"
        and not state.ragged
        and state.backend == "rlbwt_compact256"
    ):
        from ._rlbwt_backend import _compact_prefill

        output = _compact_prefill(cast(Any, state._impl), tokens)
    elif (
        state.mode == "top1"
        and not state.ragged
        and state.backend in {"rlbwt_native", "rlbwt_mc128", "rlbwt_mc192"}
    ):
        from ._rlbwt_backend import _native_prefill

        output = _native_prefill(cast(Any, state._impl), tokens)
    elif state.mode == "top1" and not state.ragged:
        from ._stateful_numba import _prefill

        output = _prefill(cast(Any, state._impl), tokens)
    elif state.mode == "rich" and not state.ragged:
        from ._stateful_candidates_numba import prefill_candidates

        rich_output = prefill_candidates(cast(Any, state._impl), tokens)
        output = rich_output.rosa_predicted_tokens
        candidates = HardCandidates(
            *(
                getattr(rich_output, name)
                for name in HardCandidates.__dataclass_fields__
            )
        )
    elif tokens.shape[1] == 0:
        output = torch.empty(tokens.shape, dtype=torch.long, device=tokens.device)
        if state.mode == "rich":
            candidates = _empty_rich_candidates(state, tokens.device)
    else:
        step_outputs = [
            state.step(tokens[:, position]) for position in range(tokens.shape[1])
        ]
        output = torch.stack([item.predicted_tokens for item in step_outputs], dim=1)
        if state.mode == "rich":
            candidates = _stack_rich_steps(
                [cast(Any, item.candidates) for item in step_outputs], 1
            )
    if squeeze:
        output = output[0]
        if candidates is not None:
            candidates = HardCandidates(
                *(
                    getattr(candidates, name)[0]
                    for name in candidates.__dataclass_fields__
                )
            )
    return InferenceOutput(output, candidates)


def prefill(state: ROSAInferenceState, tokens: Tensor) -> Tensor:
    """Consume an initial context and return exact ROSA predictions.

    The state must be empty. ``tokens`` uses shape ``[batch_size, N]``; a
    one-dimensional context is accepted when ``batch_size == 1``. The Numba
    backend fuses the complete replay into one compiled call.
    """

    if not isinstance(state, ROSAInferenceState):
        raise TypeError("state must be a ROSAInferenceState")
    return state.prefill(tokens).predicted_tokens
