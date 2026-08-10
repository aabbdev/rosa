from __future__ import annotations

import gc
import weakref
from itertools import product

import numpy as np
import rosa_native_step
import torch

from rosa._stateful_candidates_numba import (
    CandidateStep,
    forward_candidates_step,
    init_candidate_state,
)


def assert_step_equal(actual: tuple[np.ndarray, ...], expected: CandidateStep) -> None:
    expected_arrays = (
        expected.source_index.numpy(),
        expected.match_length.numpy(),
        expected.state_id.numpy(),
        expected.frequency.numpy(),
    )
    for candidate, oracle in zip(actual[:4], expected_arrays, strict=True):
        assert np.array_equal(candidate, oracle)
    assert np.array_equal(actual[4], expected.mask.sum(dim=1).numpy().astype(np.int32))


def compare(tokens: torch.Tensor, suffix_k: int, occurrences_r: int) -> None:
    oracle = init_candidate_state(
        tokens.shape[0],
        tokens.shape[1],
        suffix_k=suffix_k,
        occurrences_r=occurrences_r,
    )
    candidate = init_candidate_state(
        tokens.shape[0],
        tokens.shape[1],
        suffix_k=suffix_k,
        occurrences_r=occurrences_r,
    )
    native = rosa_native_step.NativeCandidateState(candidate)
    for position in range(tokens.shape[1]):
        expected = forward_candidates_step(oracle, tokens[:, position])
        column = np.ascontiguousarray(tokens[:, position].numpy())
        assert_step_equal(native.step(column), expected)
    assert native.position == candidate.position == tokens.shape[1]

    native.reset()
    assert native.position == candidate.position == 0
    replay_oracle = init_candidate_state(
        tokens.shape[0],
        tokens.shape[1],
        suffix_k=suffix_k,
        occurrences_r=occurrences_r,
    )
    for position in range(tokens.shape[1]):
        expected = forward_candidates_step(replay_oracle, tokens[:, position])
        column = np.ascontiguousarray(tokens[:, position].numpy())
        assert_step_equal(native.step(column), expected)


def main() -> None:
    assert rosa_native_step.candidate_abi_version == 1
    binary = torch.tensor(list(product(range(2), repeat=9)), dtype=torch.long)
    for suffix_k, occurrences_r in ((1, 1), (2, 3), (4, 2), (5, 4)):
        compare(binary, suffix_k, occurrences_r)

    generator = torch.Generator().manual_seed(20260811)
    random_tokens = torch.randint(-3, 9, (7, 193), generator=generator)
    compare(random_tokens, 7, 5)

    # A native wrapper may take over an already-mutated Numba state.
    oracle = init_candidate_state(7, 193, suffix_k=7, occurrences_r=5)
    candidate = init_candidate_state(7, 193, suffix_k=7, occurrences_r=5)
    for position in range(83):
        forward_candidates_step(oracle, random_tokens[:, position])
        forward_candidates_step(candidate, random_tokens[:, position])
    continuation = rosa_native_step.NativeCandidateState(candidate)
    for position in range(83, random_tokens.shape[1]):
        expected = forward_candidates_step(oracle, random_tokens[:, position])
        column = np.ascontiguousarray(random_tokens[:, position].numpy())
        assert_step_equal(continuation.step(column), expected)

    state = init_candidate_state(2, 4, suffix_k=3, occurrences_r=2)
    state_ref = weakref.ref(state)
    native = rosa_native_step.NativeCandidateState(state)
    del state
    gc.collect()
    assert state_ref() is not None
    native.step(np.array([1, 2], dtype=np.int64))

    for invalid in (
        np.zeros(2, dtype=np.int32),
        np.zeros((2, 1), dtype=np.int64),
        np.zeros(4, dtype=np.int64)[::2],
    ):
        try:
            native.step(invalid)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid candidate-step input was accepted")

    malformed = init_candidate_state(2, 4)
    malformed.native_candidate_abi_version = 2
    try:
        rosa_native_step.NativeCandidateState(malformed)
    except ValueError as error:
        assert "ABI" in str(error)
    else:
        raise AssertionError("unsupported candidate ABI was accepted")

    malformed = init_candidate_state(2, 4)
    malformed.occurrences = np.empty((2, 1, 4), dtype=np.int64)
    try:
        rosa_native_step.NativeCandidateState(malformed)
    except ValueError as error:
        assert "layout" in str(error)
    else:
        raise AssertionError("malformed candidate layout was accepted")

    print("rosa_native_step candidate smoke: ok")


if __name__ == "__main__":
    main()
