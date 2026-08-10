from __future__ import annotations

import statistics
import time

import numpy as np
import rosa_native_step
import torch

from rosa._stateful_candidates_numba import (
    forward_candidates_step,
    init_candidate_state,
)


def measure_native(tokens: torch.Tensor, suffix_k: int, occurrences_r: int) -> float:
    started = time.perf_counter_ns()
    state = init_candidate_state(
        tokens.shape[0],
        tokens.shape[1],
        suffix_k=suffix_k,
        occurrences_r=occurrences_r,
    )
    native = rosa_native_step.NativeCandidateState(state)
    for position in range(tokens.shape[1]):
        native.step(np.ascontiguousarray(tokens[:, position].numpy()))
    return (time.perf_counter_ns() - started) / 1e6


def measure_numba(tokens: torch.Tensor, suffix_k: int, occurrences_r: int) -> float:
    started = time.perf_counter_ns()
    state = init_candidate_state(
        tokens.shape[0],
        tokens.shape[1],
        suffix_k=suffix_k,
        occurrences_r=occurrences_r,
    )
    for position in range(tokens.shape[1]):
        forward_candidates_step(state, tokens[:, position])
    return (time.perf_counter_ns() - started) / 1e6


def main() -> None:
    batch_size, length, suffix_k, occurrences_r = 8, 4096, 16, 4
    tokens = torch.randint(
        128,
        (batch_size, length),
        generator=torch.Generator().manual_seed(20260811),
    )
    warm = init_candidate_state(1, 2, suffix_k=1, occurrences_r=1)
    forward_candidates_step(warm, torch.tensor([0]))
    native_ms = [measure_native(tokens, suffix_k, occurrences_r) for _ in range(5)]
    numba_ms = [measure_numba(tokens, suffix_k, occurrences_r) for _ in range(5)]
    native = statistics.median(native_ms)
    numba = statistics.median(numba_ms)
    print(f"native_ms={native:.6f}")
    print(f"numba_ms={numba:.6f}")
    print(f"native_vs_numba={numba / native:.6f}x")


if __name__ == "__main__":
    main()
