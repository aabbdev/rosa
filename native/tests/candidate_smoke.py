from __future__ import annotations

import gc
import os
import threading
import types
import weakref
from concurrent.futures import ThreadPoolExecutor
from itertools import product

import numpy as np
import rosa_native_step
import torch

from rosa._stateful_candidates_numba import (
    CandidateStep,
    forward_candidates_step,
    forward_candidates_step_masked,
    init_candidate_state,
    prefill_candidates,
    reset_candidates_masked,
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

    into_state = init_candidate_state(2, 3, suffix_k=2, occurrences_r=2)
    into_native = rosa_native_step.NativeCandidateState(into_state)
    into_arrays = (
        np.empty((2, 4), dtype=np.int64),
        np.empty((2, 4), dtype=np.int64),
        np.empty((2, 4), dtype=np.int64),
        np.empty((2, 4), dtype=np.int64),
        np.empty(2, dtype=np.int32),
    )
    first_tokens = np.array([0, 3], dtype=np.int64)
    into_native.step_into(first_tokens, *into_arrays)
    allocating_state = init_candidate_state(2, 3, suffix_k=2, occurrences_r=2)
    allocating_native = rosa_native_step.NativeCandidateState(allocating_state)
    allocated = allocating_native.step(first_tokens)
    assert all(
        np.array_equal(actual, expected)
        for actual, expected in zip(into_arrays, allocated, strict=True)
    )
    try:
        overlap_state = rosa_native_step.NativeCandidateState(
            init_candidate_state(2, 1, suffix_k=2, occurrences_r=2)
        )
        overlap_state.step_into(
            first_tokens,
            into_arrays[0],
            into_arrays[0],
            into_arrays[2],
            into_arrays[3],
            into_arrays[4],
        )
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("overlapping outputs were accepted")

    prefix = np.array([[0, 1, 0], [3, 3, 4]], dtype=np.int64)
    prefill_state = rosa_native_step.NativeCandidateState(
        init_candidate_state(2, 3, suffix_k=2, occurrences_r=2)
    )
    prefill_arrays = (
        np.empty((2, 3, 4), dtype=np.int64),
        np.empty((2, 3, 4), dtype=np.int64),
        np.empty((2, 3, 4), dtype=np.int64),
        np.empty((2, 3, 4), dtype=np.int64),
        np.empty((2, 3), dtype=np.int32),
    )
    prefill_state.prefill_into(prefix, *prefill_arrays)
    prefill_allocating = rosa_native_step.NativeCandidateState(
        init_candidate_state(2, 3, suffix_k=2, occurrences_r=2)
    ).prefill(prefix)
    assert all(
        np.array_equal(actual, expected)
        for actual, expected in zip(prefill_arrays, prefill_allocating, strict=True)
    )

    # Pools are lazy, never useful below the prefill threshold, and invalid
    # thread limits (including signed strings) select the serial fallback.
    small_pool = rosa_native_step.NativeCandidateState(init_candidate_state(3, 4))
    assert small_pool.worker_count == 0
    small_pool.prefill(np.zeros((3, 4), dtype=np.int64))
    assert small_pool.worker_count == 0

    previous_threads = os.environ.get("ROSA_NATIVE_THREADS")
    os.environ["ROSA_NATIVE_THREADS"] = "-2"
    try:
        invalid_limit_pool = rosa_native_step.NativeCandidateState(
            init_candidate_state(16, 4)
        )
        assert invalid_limit_pool.worker_count == 0
        invalid_limit_pool.prefill(np.zeros((16, 4), dtype=np.int64))
        assert invalid_limit_pool.worker_count == 0
    finally:
        if previous_threads is None:
            os.environ.pop("ROSA_NATIVE_THREADS", None)
        else:
            os.environ["ROSA_NATIVE_THREADS"] = previous_threads

    ragged_pool = rosa_native_step.NativeCandidateState(
        init_candidate_state(16, 4, ragged=True)
    )
    ragged_pool.step_masked(
        np.zeros(16, dtype=np.int64),
        np.ones(16, dtype=np.bool_),
        np.zeros(16, dtype=np.bool_),
    )
    assert ragged_pool.worker_count == 0

    # Concurrent mutators serialize while the GIL is released; publication of
    # position remains atomic from Python's perspective.
    concurrent_state = init_candidate_state(4, 8)
    concurrent_native = rosa_native_step.NativeCandidateState(concurrent_state)
    repeated = np.arange(4, dtype=np.int64)
    sequential_state = init_candidate_state(4, 8)
    sequential_native = rosa_native_step.NativeCandidateState(sequential_state)
    expected_steps = [sequential_native.step(repeated) for _ in range(2)]
    started = threading.Barrier(3)

    def concurrent_step() -> tuple[np.ndarray, ...]:
        started.wait()
        return concurrent_native.step(repeated)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(concurrent_step) for _ in range(2)]
        started.wait()
        actual_steps = [future.result() for future in futures]
    assert concurrent_native.position == concurrent_state.position == 2
    assert all(
        any(
            all(
                np.array_equal(actual, expected)
                for actual, expected in zip(step, candidate, strict=True)
            )
            for candidate in expected_steps
        )
        for step in actual_steps
    )

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

    # One native prefill call emits every historical field and leaves a
    # continuation-compatible state, including clone-heavy/random inputs.
    prefix_length = 137
    prefill_oracle = init_candidate_state(7, 193, suffix_k=7, occurrences_r=5)
    prefill_oracle.native_state = False
    expected_prefill = prefill_candidates(
        prefill_oracle, random_tokens[:, :prefix_length]
    )
    prefill_state = init_candidate_state(7, 193, suffix_k=7, occurrences_r=5)
    native_prefill = rosa_native_step.NativeCandidateState(prefill_state)
    actual_prefill = native_prefill.prefill(
        np.ascontiguousarray(random_tokens[:, :prefix_length].numpy())
    )
    expected_arrays = (
        expected_prefill.source_index.numpy(),
        expected_prefill.match_length.numpy(),
        expected_prefill.state_id.numpy(),
        expected_prefill.frequency.numpy(),
        expected_prefill.mask.sum(dim=2).numpy().astype(np.int32),
    )
    for actual, expected in zip(actual_prefill, expected_arrays, strict=True):
        assert np.array_equal(actual, expected)
    assert native_prefill.position == prefill_state.position == prefix_length
    assert np.array_equal(native_prefill.positions, np.full(7, prefix_length))
    for position in range(prefix_length, random_tokens.shape[1]):
        expected = forward_candidates_step(prefill_oracle, random_tokens[:, position])
        assert_step_equal(
            native_prefill.step(
                np.ascontiguousarray(random_tokens[:, position].numpy())
            ),
            expected,
        )

    # Ragged active/reset/recycle follows independent per-row positions.  The
    # Numba fallback is the exact oracle and inactive rows emit empty outputs.
    ragged_oracle = init_candidate_state(
        5, 17, suffix_k=5, occurrences_r=3, ragged=True
    )
    ragged_oracle.native_state = False
    ragged_state = init_candidate_state(5, 17, suffix_k=5, occurrences_r=3, ragged=True)
    native_ragged = rosa_native_step.NativeCandidateState(ragged_state)
    ragged_generator = np.random.default_rng(1804)
    for iteration in range(73):
        token_values = ragged_generator.integers(-2, 7, size=5, dtype=np.int64)
        active = ragged_generator.random(5) < 0.72
        reset = np.logical_and(active, ragged_generator.random(5) < 0.13)
        full = np.logical_and(active, ragged_oracle.positions >= 17)
        reset = np.logical_or(reset, full)
        expected = forward_candidates_step_masked(
            ragged_oracle,
            torch.from_numpy(token_values),
            torch.from_numpy(active),
            torch.from_numpy(reset),
        )
        actual = native_ragged.step_masked(token_values, active, reset)
        assert_step_equal(actual, expected)
        assert np.array_equal(native_ragged.positions, ragged_oracle.positions)
    reset_rows = np.array([True, False, True, False, True])
    reset_candidates_masked(ragged_oracle, torch.from_numpy(reset_rows))
    native_ragged.reset_masked(reset_rows)
    assert np.array_equal(native_ragged.positions, ragged_oracle.positions)
    assert np.array_equal(ragged_state.size, ragged_oracle.size)

    for action in (
        lambda: native_ragged.step(np.zeros(5, dtype=np.int64)),
        lambda: native_ragged.prefill(np.zeros((5, 1), dtype=np.int64)),
        lambda: rosa_native_step.NativeCandidateState(
            init_candidate_state(1, 2)
        ).step_masked(
            np.zeros(1, dtype=np.int64),
            np.ones(1, dtype=np.bool_),
            np.zeros(1, dtype=np.bool_),
        ),
    ):
        try:
            action()
        except RuntimeError:
            pass
        else:
            raise AssertionError("uniform/ragged candidate modes were mixed")

    state = init_candidate_state(2, 4, suffix_k=3, occurrences_r=2)
    state_ref = weakref.ref(state)
    history_ref = weakref.ref(state.history)
    native = rosa_native_step.NativeCandidateState(state)
    state.position = 3
    native.step(np.array([1, 2], dtype=np.int64))
    assert state.position == native.position == 1
    del state
    gc.collect()
    assert state_ref() is None
    assert history_ref() is not None
    native.step(np.array([3, 4], dtype=np.int64))
    assert native.position == 2
    assert history_ref()[:, :2].tolist() == [[1, 3], [2, 4]]

    cyclic_owner = init_candidate_state(1, 2)
    cyclic_wrapper = rosa_native_step.NativeCandidateState(cyclic_owner)
    cyclic_owner.native_state = cyclic_wrapper
    cyclic_owner_ref = weakref.ref(cyclic_owner)
    cyclic_wrapper_ref = weakref.ref(cyclic_wrapper)
    del cyclic_owner, cyclic_wrapper
    gc.collect()
    assert cyclic_owner_ref() is None
    assert cyclic_wrapper_ref() is None

    nonweak_owner = types.SimpleNamespace(**vars(init_candidate_state(1, 2)))
    nonweak = rosa_native_step.NativeCandidateState(nonweak_owner)
    nonweak.step(np.array([11], dtype=np.int64))
    assert nonweak.position == 1

    # ABI 1 compatibility: pre-positions uniform states remain accepted.
    legacy = init_candidate_state(1, 2)
    del legacy.positions
    legacy_native = rosa_native_step.NativeCandidateState(legacy)
    legacy_native.step(np.array([1], dtype=np.int64))

    invalid_position = init_candidate_state(1, 2, ragged=True)
    invalid_position.positions[0] = -1
    try:
        rosa_native_step.NativeCandidateState(invalid_position)
    except ValueError as error:
        assert "positions" in str(error)
    else:
        raise AssertionError("negative candidate position was accepted")

    runtime_position = init_candidate_state(1, 2, ragged=True)
    runtime_native = rosa_native_step.NativeCandidateState(runtime_position)
    runtime_position.positions[0] = -1
    try:
        runtime_native.step_masked(
            np.array([1], dtype=np.int64),
            np.array([True]),
            np.array([False]),
        )
    except RuntimeError as error:
        assert "non-negative" in str(error)
    else:
        raise AssertionError("mutated negative candidate position was consumed")

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
