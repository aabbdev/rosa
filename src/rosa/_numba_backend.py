"""Optional Numba fast path for exact CPU ROSA inference."""

from __future__ import annotations

import numpy as np
import torch
from numba import njit, prange
from torch import Tensor


@njit(cache=True, nogil=True)
def _predict_row(tokens: np.ndarray) -> np.ndarray:
    n = tokens.shape[0]
    vocabulary_size = int(tokens.max()) + 1
    max_states = 2 * n + 1
    transitions = np.full((max_states, vocabulary_size), -1, dtype=np.int32)
    suffix_link = np.full(max_states, -1, dtype=np.int32)
    length = np.zeros(max_states, dtype=np.int32)
    latest_end = np.full(max_states, -1, dtype=np.int32)
    predicted = np.full(n, -1, dtype=np.int64)
    last = 0
    size = 1

    for i in range(n):
        token = int(tokens[i])
        current = size
        size += 1
        length[current] = length[last] + 1
        state = last

        while state != -1 and transitions[state, token] == -1:
            transitions[state, token] = current
            state = suffix_link[state]

        if state == -1:
            suffix_link[current] = 0
        else:
            target = transitions[state, token]
            if length[state] + 1 == length[target]:
                suffix_link[current] = target
            else:
                clone = size
                size += 1
                transitions[clone, :] = transitions[target, :]
                length[clone] = length[state] + 1
                suffix_link[clone] = suffix_link[target]
                latest_end[clone] = latest_end[target]
                while state != -1 and transitions[state, token] == target:
                    transitions[state, token] = clone
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
def _predict_batch(tokens: np.ndarray) -> np.ndarray:
    output = np.empty(tokens.shape, dtype=np.int64)
    for batch_index in prange(tokens.shape[0]):
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
    cpu_tokens = tokens.detach().to(device="cpu", dtype=torch.long).contiguous()
    output = torch.from_numpy(_predict_batch(cpu_tokens.numpy())).to(device)
    return output[0] if squeeze else output
