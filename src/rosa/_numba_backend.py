"""Optional Numba fast path for exact CPU ROSA inference."""

from __future__ import annotations

import numpy as np
import torch
from numba import njit, prange
from torch import Tensor


@njit(cache=True, nogil=True, inline="always")
def _find_transition(  # pragma: no cover - executed as compiled Numba code
    head: np.ndarray,
    edge_token: np.ndarray,
    edge_target: np.ndarray,
    edge_next: np.ndarray,
    state: int,
    token: int,
) -> int:
    edge = head[state]
    while edge != -1:
        if edge_token[edge] == token:
            return int(edge_target[edge])
        edge = edge_next[edge]
    return -1


@njit(cache=True, nogil=True, inline="always")
def _add_transition(  # pragma: no cover - executed as compiled Numba code
    head: np.ndarray,
    edge_token: np.ndarray,
    edge_target: np.ndarray,
    edge_next: np.ndarray,
    edge_count: int,
    state: int,
    token: int,
    target: int,
) -> int:
    if edge_count >= edge_token.shape[0]:
        raise RuntimeError("suffix automaton transition capacity exceeded")
    edge_token[edge_count] = token
    edge_target[edge_count] = target
    edge_next[edge_count] = head[state]
    head[state] = edge_count
    return edge_count + 1


@njit(cache=True, nogil=True, inline="always")
def _replace_transition(  # pragma: no cover - executed as compiled Numba code
    head: np.ndarray,
    edge_token: np.ndarray,
    edge_target: np.ndarray,
    edge_next: np.ndarray,
    state: int,
    token: int,
    target: int,
) -> None:
    edge = head[state]
    while edge != -1:
        if edge_token[edge] == token:
            edge_target[edge] = target
            return
        edge = edge_next[edge]
    raise RuntimeError("suffix automaton transition not found")


@njit(cache=True, nogil=True)
def _predict_row(  # pragma: no cover - executed as compiled Numba code
    tokens: np.ndarray,
) -> np.ndarray:
    n = tokens.shape[0]
    max_states = 2 * n + 1
    max_edges = 4 * n + 1
    head = np.full(max_states, -1, dtype=np.int32)
    edge_token = np.empty(max_edges, dtype=np.int64)
    edge_target = np.empty(max_edges, dtype=np.int32)
    edge_next = np.empty(max_edges, dtype=np.int32)
    suffix_link = np.full(max_states, -1, dtype=np.int32)
    length = np.zeros(max_states, dtype=np.int32)
    latest_end = np.full(max_states, -1, dtype=np.int32)
    predicted = np.full(n, -1, dtype=np.int64)
    last = 0
    size = 1
    edge_count = 0

    for i in range(n):
        token = int(tokens[i])
        current = size
        size += 1
        length[current] = length[last] + 1
        state = last

        while (
            state != -1
            and _find_transition(head, edge_token, edge_target, edge_next, state, token)
            == -1
        ):
            edge_count = _add_transition(
                head,
                edge_token,
                edge_target,
                edge_next,
                edge_count,
                state,
                token,
                current,
            )
            state = suffix_link[state]

        if state == -1:
            suffix_link[current] = 0
        else:
            target = _find_transition(
                head, edge_token, edge_target, edge_next, state, token
            )
            if length[state] + 1 == length[target]:
                suffix_link[current] = target
            else:
                clone = size
                size += 1
                length[clone] = length[state] + 1
                suffix_link[clone] = suffix_link[target]
                latest_end[clone] = latest_end[target]
                edge = head[target]
                while edge != -1:
                    edge_count = _add_transition(
                        head,
                        edge_token,
                        edge_target,
                        edge_next,
                        edge_count,
                        clone,
                        edge_token[edge],
                        edge_target[edge],
                    )
                    edge = edge_next[edge]
                while (
                    state != -1
                    and _find_transition(
                        head, edge_token, edge_target, edge_next, state, token
                    )
                    == target
                ):
                    _replace_transition(
                        head,
                        edge_token,
                        edge_target,
                        edge_next,
                        state,
                        token,
                        clone,
                    )
                    state = suffix_link[state]
                suffix_link[target] = clone
                suffix_link[current] = clone

        last = current
        state = last
        while state != -1:
            if length[state] > 0 and latest_end[state] >= 0:
                predicted[i] = tokens[latest_end[state] + 1]
                break
            state = suffix_link[state]

        state = last
        while state != -1:
            latest_end[state] = i
            state = suffix_link[state]

    return predicted


@njit(cache=True, nogil=True, parallel=True)
def _predict_batch(  # pragma: no cover - executed as compiled Numba code
    tokens: np.ndarray,
) -> np.ndarray:
    output = np.empty(tokens.shape, dtype=np.int64)
    for batch_index in prange(tokens.shape[0]):
        output[batch_index] = _predict_row(tokens[batch_index])
    return output


@njit(cache=True, nogil=True)
def _predict_serial_batch(  # pragma: no cover - executed as compiled Numba code
    tokens: np.ndarray,
) -> np.ndarray:
    output = np.empty(tokens.shape, dtype=np.int64)
    for batch_index in range(tokens.shape[0]):
        output[batch_index] = _predict_row(tokens[batch_index])
    return output


def predict_exact(tokens: Tensor) -> Tensor:
    """Return exact ROSA predictions using an optional CPU Numba backend."""

    squeeze = tokens.ndim == 1
    if squeeze:
        tokens = tokens.unsqueeze(0)
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape [N] or [B, N]")
    device = tokens.device
    if tokens.shape[1] == 0:
        output = torch.empty(tokens.shape, dtype=torch.long, device=device)
        return output[0] if squeeze else output
    cpu_tokens = tokens.detach().to(device="cpu", dtype=torch.long).contiguous()
    cpu_array = cpu_tokens.numpy()
    if cpu_array.shape[0] == 1:
        output_array = _predict_row(cpu_array[0]).reshape(1, -1)
    elif cpu_array.size <= 4096:
        output_array = _predict_serial_batch(cpu_array)
    else:
        output_array = _predict_batch(cpu_array)
    output = torch.from_numpy(output_array).to(device)
    return output[0] if squeeze else output
