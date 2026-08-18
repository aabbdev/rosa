"""Exact CPU prototype based on the online RLBWT of reversed prefixes.

This module intentionally favours a direct implementation of the definitions
over asymptotic performance.  The BWT is genuinely stored as maximal runs,
while PA and LCS are stored explicitly.  In particular, this is not the
sampled/compressed PA/LCS data structure from the paper.

The state is uniform: every batch row has consumed ``state.position`` tokens,
and every row has the same immutable ``max_length`` capacity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


class _Sentinel:
    """Type of the unique out-of-band BWT sentinel."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "$"


_SENTINEL = _Sentinel()
"""The sentinel object returned by :func:`_reconstruct_rlbwt`."""


@dataclass
class _RLBWTRun:
    """One maximal BWT run.

    ``symbol`` is meaningful only for an ordinary run.  Keeping the sentinel
    flag separate is essential because every signed int64 value is a valid
    input token.
    """

    symbol: int = 0
    length: int = 1
    is_sentinel: bool = False


@dataclass
class _RLBWTRowState:
    history: list[int]
    pa: list[int]
    lcs: list[int]
    runs: list[_RLBWTRun] = field(default_factory=lambda: [_RLBWTRun(is_sentinel=True)])
    source: int = -1
    lrs: int = 0


@dataclass
class _RLBWTState:
    """Fixed-capacity, uniformly positioned batch state."""

    batch_size: int
    max_length: int
    position: int
    rows: list[_RLBWTRowState]

    @property
    def history(self) -> list[list[int]]:
        """Preallocated token histories, exposed for diagnostics/tests."""

        return [row.history for row in self.rows]

    @property
    def pa(self) -> list[list[int]]:
        """Explicit PA storage; only ``position + 1`` entries are live."""

        return [row.pa for row in self.rows]

    @property
    def lcs(self) -> list[list[int]]:
        """Explicit LCS storage; only ``position + 1`` entries are live."""

        return [row.lcs for row in self.rows]

    @property
    def sources(self) -> list[int]:
        return [row.source for row in self.rows]

    @property
    def lrs_lengths(self) -> list[int]:
        return [row.lrs for row in self.rows]


def _init_rlbwt_state(batch_size: int, max_length: int) -> _RLBWTState:
    """Allocate an empty uniform RLBWT state with fixed sequence capacity."""

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if max_length <= 0:
        raise ValueError("max_length must be > 0")

    rows = [
        _RLBWTRowState(
            history=[0] * max_length,
            # PA includes the initial prefix consisting only of the conceptual
            # leading sentinel, hence the extra slot.
            pa=[0] * (max_length + 1),
            lcs=[0] * (max_length + 1),
        )
        for _ in range(batch_size)
    ]
    return _RLBWTState(batch_size, max_length, 0, rows)


def _init_native_rlbwt_state(batch_size: int, max_length: int) -> object:
    """Create the optional native companion state after capability checks."""

    try:
        import rosa_native_step  # type: ignore[reportMissingImports]
    except ModuleNotFoundError as error:
        if error.name != "rosa_native_step":
            raise
        raise ImportError(
            "native RLBWT inference requires a compatible rosa-torch-native wheel"
        ) from error
    if (
        getattr(rosa_native_step, "rlbwt_abi_version", None) != 1
        or getattr(rosa_native_step, "NativeRLBWTState", None) is None
    ):
        raise ImportError("installed rosa-torch-native lacks RLBWT ABI 1")
    return rosa_native_step.NativeRLBWTState(batch_size, max_length)


def _init_native_rlbwt_compact_state(batch_size: int, max_length: int) -> object:
    """Create the explicit native identity-coded uint8-vocabulary state."""

    try:
        import rosa_native_step  # type: ignore[reportMissingImports]
    except ModuleNotFoundError as error:
        if error.name != "rosa_native_step":
            raise
        raise ImportError(
            "compact RLBWT inference requires a compatible rosa-torch-native wheel"
        ) from error
    if (
        getattr(rosa_native_step, "rlbwt_compact_abi_version", None) != 1
        or getattr(rosa_native_step, "NativeRLBWTCompactState", None) is None
    ):
        raise ImportError("installed rosa-torch-native lacks compact RLBWT ABI 1")
    return rosa_native_step.NativeRLBWTCompactState(batch_size, max_length, 256)


def _init_native_rlbwt_mc_state(
    batch_size: int, max_length: int, lanes: int, seed: int = 20260811
) -> object:
    """Create an explicit Monte-Carlo native RLBWT state.

    Long LCE queries trust ``lanes`` independent rolling hashes modulo 2^64;
    collisions are possible by contract. LCEs of at most 64 tokens are exact.
    """

    try:
        import rosa_native_step  # type: ignore[reportMissingImports]
    except ModuleNotFoundError as error:
        if error.name != "rosa_native_step":
            raise
        raise ImportError(
            "Monte-Carlo RLBWT inference requires a compatible rosa-torch-native wheel"
        ) from error
    if (
        getattr(rosa_native_step, "rlbwt_mc_abi_version", None) != 1
        or getattr(rosa_native_step, "NativeRLBWTStateMC", None) is None
    ):
        raise ImportError("installed rosa-torch-native lacks RLBWT MC ABI 1")
    return rosa_native_step.NativeRLBWTStateMC(batch_size, max_length, lanes, seed)


def _same_run(run: _RLBWTRun, symbol: int) -> bool:
    return not run.is_sentinel and run.symbol == symbol


def _merge_around(runs: list[_RLBWTRun], index: int) -> None:
    """Restore maximality after changing one ordinary run."""

    if index > 0 and _same_run(runs[index - 1], runs[index].symbol):
        runs[index - 1].length += runs[index].length
        del runs[index]
        index -= 1
    if index + 1 < len(runs) and _same_run(runs[index + 1], runs[index].symbol):
        runs[index].length += runs[index + 1].length
        del runs[index + 1]


def _sentinel_location(runs: list[_RLBWTRun]) -> tuple[int, int]:
    """Return ``(run_index, expanded_position)`` of the unique sentinel."""

    position = 0
    found = -1
    sentinel_position = -1
    for run_index, run in enumerate(runs):
        if run.is_sentinel:
            if found != -1 or run.length != 1:
                raise RuntimeError("invalid RLBWT sentinel representation")
            found = run_index
            sentinel_position = position
        position += run.length
    if found == -1:
        raise RuntimeError("RLBWT sentinel is missing")
    return found, sentinel_position


def _insertion_rank(runs: list[_RLBWTRun], symbol: int) -> int:
    """Compute the zero-based insertion index from the paper's equation.

    The paper uses one-based ``ell = C[c] + rank_c(BWT, s) + 1``.  Therefore
    the returned zero-based index is ``C[c] + rank_c(BWT, s)``.  The sentinel
    contributes one to ``C[c]`` because it is ordered below every int64 token.
    """

    _, sentinel_position = _sentinel_location(runs)
    less = 0
    rank_through_sentinel = 0
    expanded_position = 0
    for run in runs:
        if run.is_sentinel:
            less += 1
        else:
            if run.symbol < symbol:
                less += run.length
            if run.symbol == symbol and expanded_position <= sentinel_position:
                rank_through_sentinel += min(
                    run.length, sentinel_position - expanded_position + 1
                )
        expanded_position += run.length
    return less + rank_through_sentinel


def _replace_sentinel(runs: list[_RLBWTRun], symbol: int) -> None:
    run_index, _ = _sentinel_location(runs)
    runs[run_index] = _RLBWTRun(symbol=symbol)
    _merge_around(runs, run_index)


def _insert_sentinel(runs: list[_RLBWTRun], position: int) -> None:
    """Insert the sentinel before expanded BWT position ``position``."""

    total = sum(run.length for run in runs)
    if position < 0 or position > total:
        raise RuntimeError("RLBWT insertion position is out of range")
    if position == total:
        runs.append(_RLBWTRun(is_sentinel=True))
        return

    start = 0
    for run_index, run in enumerate(runs):
        end = start + run.length
        if position == start:
            runs.insert(run_index, _RLBWTRun(is_sentinel=True))
            return
        if start < position < end:
            if run.is_sentinel:
                raise RuntimeError("cannot split the RLBWT sentinel")
            left_length = position - start
            right_length = end - position
            runs[run_index : run_index + 1] = [
                _RLBWTRun(run.symbol, left_length),
                _RLBWTRun(is_sentinel=True),
                _RLBWTRun(run.symbol, right_length),
            ]
            return
        start = end
    raise RuntimeError(  # pragma: no cover - malformed non-partitioning run list
        "failed to insert the RLBWT sentinel"
    )


def _common_suffix(row: _RLBWTRowState, left: int, right: int) -> int:
    """Return LCS of prefixes ending at token-count endpoints left/right."""

    length = 0
    while (
        length < left
        and length < right
        and row.history[left - length - 1] == row.history[right - length - 1]
    ):
        length += 1
    return length


def _update_pa_lcs(
    row: _RLBWTRowState, old_length: int, insertion_index: int
) -> tuple[int, int, int]:
    """Insert the new prefix and return ``(new_rank, x, y)``."""

    old_size = old_length + 1
    new_endpoint = old_length + 1
    if insertion_index < 0 or insertion_index > old_size:
        raise RuntimeError("PA insertion position is out of range")

    predecessor = row.pa[insertion_index - 1] if insertion_index > 0 else None
    successor = row.pa[insertion_index] if insertion_index < old_size else None
    x = _common_suffix(row, new_endpoint, predecessor) if predecessor is not None else 0
    y = _common_suffix(row, new_endpoint, successor) if successor is not None else 0

    # Shift fixed-capacity storage in place.  LCS[i] is the LCS of PA[i-1]
    # and PA[i], with LCS[0] conventionally zero.
    for index in range(old_size, insertion_index, -1):
        row.pa[index] = row.pa[index - 1]
        row.lcs[index] = row.lcs[index - 1]
    row.pa[insertion_index] = new_endpoint
    row.lcs[insertion_index] = x if insertion_index > 0 else 0
    if successor is not None:
        row.lcs[insertion_index + 1] = y
    return insertion_index, x, y


def _select_rosa_source(
    row: _RLBWTRowState, new_rank: int, new_size: int, lrs: int
) -> int:
    """Select the newest old endpoint in the maximal PA interval."""

    if lrs == 0:
        return -1
    left = new_rank
    while left > 0 and row.lcs[left] >= lrs:
        left -= 1
    right = new_rank
    while right + 1 < new_size and row.lcs[right + 1] >= lrs:
        right += 1

    new_endpoint = new_size - 1
    previous_endpoint = max(
        row.pa[index]
        for index in range(left, right + 1)
        if row.pa[index] != new_endpoint
    )
    return previous_endpoint - 1


def _step_row(row: _RLBWTRowState, old_length: int, token: int) -> int:
    row.history[old_length] = token

    # Compute ell against the old BWT, then perform exactly the standard
    # replace-terminator/insert-terminator update.
    insertion_index = _insertion_rank(row.runs, token)
    _replace_sentinel(row.runs, token)
    _insert_sentinel(row.runs, insertion_index)

    new_rank, x, y = _update_pa_lcs(row, old_length, insertion_index)
    row.lrs = max(x, y)
    row.source = _select_rosa_source(row, new_rank, old_length + 2, row.lrs)
    if row.source < 0:
        return -1
    # source is the previous occurrence's inclusive zero-based endpoint.
    return row.history[row.source + 1]


def _forward_step(state: _RLBWTState, tokens: Tensor) -> Tensor:
    """Consume one int64-compatible token per row and return ROSA top-1."""

    if not isinstance(tokens, Tensor):
        raise TypeError("tokens must be a torch.Tensor")
    if tokens.ndim == 0 and state.batch_size == 1:
        tokens = tokens.unsqueeze(0)
    if tokens.ndim != 1 or tokens.shape[0] != state.batch_size:
        raise ValueError("tokens must have shape [batch_size]")
    if state.position >= state.max_length:
        raise RuntimeError("inference state capacity exceeded")

    device = tokens.device
    cpu_tokens = tokens.detach().to(device="cpu", dtype=torch.long).contiguous()
    predictions = [
        _step_row(row, state.position, int(token))
        for row, token in zip(state.rows, cpu_tokens.tolist(), strict=True)
    ]
    state.position += 1
    return torch.tensor(predictions, dtype=torch.long, device=device)


def _prefill(state: _RLBWTState, tokens: Tensor) -> Tensor:
    """Replay a dense initial context through the incremental RLBWT update."""

    if state.position != 0:
        raise RuntimeError("prefill requires an empty inference state")
    if not isinstance(tokens, Tensor):
        raise TypeError("tokens must be a torch.Tensor")
    if tokens.ndim != 2 or tokens.shape[0] != state.batch_size:
        raise ValueError("tokens must have shape [batch_size, sequence_length]")
    if tokens.shape[1] > state.max_length:
        raise RuntimeError("inference state capacity exceeded")
    if tokens.shape[1] == 0:
        return torch.empty(tokens.shape, dtype=torch.long, device=tokens.device)

    output = torch.empty(tokens.shape, dtype=torch.long, device=tokens.device)
    for position in range(tokens.shape[1]):
        output[:, position] = _forward_step(state, tokens[:, position])
    return output


def _native_forward_step(state: object, tokens: Tensor) -> Tensor:
    """Dispatch one batch token vector through the optional native state."""

    device = tokens.device
    cpu_tokens = tokens.detach().to(device="cpu", dtype=torch.long).contiguous()
    output = state.step(cpu_tokens.numpy())  # type: ignore[attr-defined]
    return torch.from_numpy(output).to(device)


def _native_prefill(state: object, tokens: Tensor) -> Tensor:
    """Dispatch a dense context through one fused optional native call."""

    device = tokens.device
    if tokens.shape[1] == 0:
        return torch.empty(tokens.shape, dtype=torch.long, device=device)
    cpu_tokens = tokens.detach().to(device="cpu", dtype=torch.long).contiguous()
    output = state.prefill(cpu_tokens.numpy())  # type: ignore[attr-defined]
    return torch.from_numpy(output).to(device)


def _validate_compact_tokens(tokens: Tensor) -> None:
    if bool(torch.any((tokens < 0) | (tokens > 255))):
        raise ValueError("rlbwt_compact256 tokens must be in [0, 255]")


def _compact_forward_step(state: object, tokens: Tensor) -> Tensor:
    _validate_compact_tokens(tokens)
    return _native_forward_step(state, tokens)


def _compact_prefill(state: object, tokens: Tensor) -> Tensor:
    _validate_compact_tokens(tokens)
    return _native_prefill(state, tokens)


def _reconstruct_rlbwt(
    state: _RLBWTState, batch_index: int = 0
) -> list[int | _Sentinel]:
    """Expand one row's RLBWT for tests, preserving the sentinel identity."""

    if batch_index < 0 or batch_index >= state.batch_size:
        raise IndexError("batch_index is out of range")
    result: list[int | _Sentinel] = []
    for run in state.rows[batch_index].runs:
        value: int | _Sentinel = _SENTINEL if run.is_sentinel else run.symbol
        result.extend([value] * run.length)
    return result


# A descriptive alias is convenient for diagnostics that do not rely on the
# private helper name.
reconstruct_rlbwt = _reconstruct_rlbwt


__all__ = [
    "_RLBWTRun",
    "_RLBWTState",
    "_SENTINEL",
    "_forward_step",
    "_init_rlbwt_state",
    "_init_native_rlbwt_state",
    "_init_native_rlbwt_compact_state",
    "_init_native_rlbwt_mc_state",
    "_native_forward_step",
    "_native_prefill",
    "_compact_forward_step",
    "_compact_prefill",
    "_prefill",
    "_reconstruct_rlbwt",
    "reconstruct_rlbwt",
]
