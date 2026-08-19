from __future__ import annotations

import gc
import sys
import unittest
import weakref
from itertools import product
from unittest.mock import patch

import numpy as np
import torch

from rosa import (
    build_hard_candidates,
    forward_candidates_step,
    forward_candidates_step_into,
    init_candidate_buffers,
    init_candidate_state,
)
from rosa._stateful_candidates_numba import (
    CandidateState,
    CandidateStep,
    forward_candidates_step_masked,
    prefill_candidates,
    prefill_candidates_into,
    prefill_candidates_selected,
    reset_candidates_masked,
)
from rosa._stateful_candidates_numba import (
    init_candidate_state as init_candidate_state_internal,
)

_FIELDS = (
    "source_index",
    "match_length",
    "state_id",
    "frequency",
    "mask",
    "rosa_slot",
    "rosa_source_index",
    "rosa_match_length",
    "rosa_predicted_tokens",
)


class TestStatefulCandidates(unittest.TestCase):
    def assert_step_equal(self, actual: CandidateStep, expected: CandidateStep) -> None:
        for name in _FIELDS:
            self.assertTrue(
                torch.equal(getattr(actual, name), getattr(expected, name)), name
            )

    def test_close_is_idempotent_breaks_cycles_and_rejects_use(self) -> None:
        state = init_candidate_state_internal(1, 2, suffix_k=2, occurrences_r=2)
        buffers = init_candidate_buffers(state)

        class NativeOwner:
            def __init__(self, candidate_state: CandidateState) -> None:
                self.candidate_state = candidate_state

        state.native_state = NativeOwner(state)
        native_ref = weakref.ref(state.native_state)
        state.close()
        state.close()
        self.assertIsNone(native_ref())
        with self.assertRaisesRegex(RuntimeError, "^state is closed$"):
            _ = state.position
        with self.assertRaisesRegex(RuntimeError, "^state is closed$"):
            _ = state.positions
        with self.assertRaisesRegex(RuntimeError, "^state is closed$"):
            forward_candidates_step(state, torch.tensor([0]))
        with self.assertRaisesRegex(RuntimeError, "^state is closed$"):
            forward_candidates_step_into(state, torch.tensor([0]), buffers)
        with self.assertRaisesRegex(RuntimeError, "^state is closed$"):
            prefill_candidates(state, torch.tensor([[0]]))

        state_ref = weakref.ref(state)
        del state
        gc.collect()
        self.assertIsNone(state_ref())

    def assert_matches_oracle(
        self,
        tokens: torch.Tensor,
        *,
        suffix_k: int,
        occurrences_r: int,
        split_at: int | None = None,
    ) -> CandidateState:
        state = init_candidate_state(
            tokens.shape[0],
            tokens.shape[1],
            suffix_k=suffix_k,
            occurrences_r=occurrences_r,
        )
        state.native_state = False
        steps: list[CandidateStep] = []
        boundary = tokens.shape[1] if split_at is None else split_at
        for position in range(boundary):
            steps.append(forward_candidates_step(state, tokens[:, position]))
        # Deliberately retain and continue the same mutable state across calls.
        for position in range(boundary, tokens.shape[1]):
            steps.append(forward_candidates_step(state, tokens[:, position]))

        expected = build_hard_candidates(
            tokens,
            suffix_k=suffix_k,
            occurrences_r=occurrences_r,
        )
        for field in _FIELDS:
            with self.subTest(field=field, suffix_k=suffix_k, r=occurrences_r):
                actual = torch.stack([getattr(step, field) for step in steps], dim=1)
                self.assertTrue(torch.equal(actual, getattr(expected, field)))
        self.assertEqual(state.position, tokens.shape[1])
        return state

    def test_exhaustive_binary_clones_for_multiple_k_and_r(self) -> None:
        rows = list(product(range(2), repeat=9))
        tokens = torch.tensor(rows, dtype=torch.long)
        for suffix_k, occurrences_r in ((1, 1), (2, 3), (4, 2), (5, 4)):
            state = self.assert_matches_oracle(
                tokens,
                suffix_k=suffix_k,
                occurrences_r=occurrences_r,
            )
            # Binary strings exercise SAM clone creation in aggregate.
            self.assertTrue(torch.from_numpy(state.size > tokens.shape[1] + 1).any())

    def test_newest_first_frequency_beyond_r_and_deduplication(self) -> None:
        tokens = torch.tensor([[0, 1, 0, 2, 0, 3, 0, 4, 0]], dtype=torch.long)
        state = init_candidate_state(1, 9, suffix_k=6, occurrences_r=3)
        result: CandidateStep | None = None
        for position in range(tokens.shape[1]):
            result = forward_candidates_step(state, tokens[:, position])
        assert result is not None

        self.assertEqual(result.source_index[0, :3].tolist(), [6, 4, 2])
        self.assertEqual(result.frequency[0, :3].tolist(), [4, 4, 4])
        valid_sources = result.source_index[0][result.mask[0]].tolist()
        self.assertEqual(len(valid_sources), len(set(valid_sources)))
        self.assert_matches_oracle(tokens, suffix_k=6, occurrences_r=3)

    def test_batched_continuation_matches_offline_oracle(self) -> None:
        generator = torch.Generator().manual_seed(20260811)
        random_tokens = torch.randint(7, (4, 73), generator=generator)
        repetitive = torch.arange(73).remainder(3).unsqueeze(0)
        tokens = torch.cat((random_tokens, repetitive), dim=0)
        self.assert_matches_oracle(
            tokens,
            suffix_k=7,
            occurrences_r=5,
            split_at=41,
        )

    def test_prefill_emits_every_position_and_continues(self) -> None:
        generator = torch.Generator().manual_seed(1804)
        prefix = torch.randint(5, (5, 61), generator=generator)
        continuation = torch.randint(5, (5, 17), generator=generator)
        state = init_candidate_state(5, 78, suffix_k=6, occurrences_r=4)
        state.native_state = False
        actual = prefill_candidates(state, prefix)
        expected = build_hard_candidates(prefix, suffix_k=6, occurrences_r=4)
        for field in _FIELDS:
            self.assertTrue(
                torch.equal(getattr(actual, field), getattr(expected, field))
            )
        self.assertEqual(state.position, prefix.shape[1])
        self.assertEqual(state.positions.tolist(), [prefix.shape[1]] * 5)

        steps = [
            forward_candidates_step(state, continuation[:, position])
            for position in range(continuation.shape[1])
        ]
        full = torch.cat((prefix, continuation), dim=1)
        expected_full = build_hard_candidates(full, suffix_k=6, occurrences_r=4)
        for field in _FIELDS:
            actual_tail = torch.stack([getattr(step, field) for step in steps], dim=1)
            self.assertTrue(
                torch.equal(
                    actual_tail, getattr(expected_full, field)[:, prefix.shape[1] :]
                )
            )

    def test_selected_prefill_is_q_only_ordered_and_continuable(self) -> None:
        tokens_cpu = torch.tensor(
            [[0, 1, 0, 2, 0, 1, 0, 3, 0], [3, 3, 4, 3, 5, 3, 4, 3, 3]]
        )
        query_cases = (
            torch.tensor([[8], [0]]),
            torch.tensor([[8, 0, 4], [1, 8, 0]]),
            torch.tensor([list(reversed(range(9))), list(range(9))]),
        )
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        backends = ["numba"]
        try:
            import rosa_native_step
        except ModuleNotFoundError:
            rosa_native_step = None
        if rosa_native_step is not None and hasattr(
            rosa_native_step.NativeCandidateState, "prefill_selected"
        ):
            backends.append("native")

        for device, backend, queries_cpu in product(devices, backends, query_cases):
            with self.subTest(device=device, backend=backend, queries=queries_cpu):
                tokens = tokens_cpu.to(device)
                queries = queries_cpu.to(device)
                full_state = init_candidate_state_internal(
                    2, 10, suffix_k=3, occurrences_r=2
                )
                selected_state = init_candidate_state_internal(
                    2, 10, suffix_k=3, occurrences_r=2
                )
                if backend == "numba":
                    full_state.native_state = False
                    selected_state.native_state = False
                else:
                    assert rosa_native_step is not None
                    full_state.native_state = rosa_native_step.NativeCandidateState(
                        full_state
                    )
                    selected_state.native_state = rosa_native_step.NativeCandidateState(
                        selected_state
                    )
                full = prefill_candidates(full_state, tokens)
                selected = prefill_candidates_selected(selected_state, tokens, queries)
                batch = torch.arange(2, device=device).unsqueeze(1)
                expected = CandidateStep(
                    *(getattr(full, field)[batch, queries] for field in _FIELDS)
                )
                self.assert_step_equal(selected, expected)
                self.assertEqual(selected.source_index.shape[:2], queries.shape)
                self.assertEqual(selected_state.position, 9)
                self.assertEqual(selected_state.positions.tolist(), [9, 9])

                continuation = torch.tensor([2, 4], device=device)
                self.assert_step_equal(
                    forward_candidates_step(selected_state, continuation),
                    forward_candidates_step(full_state, continuation),
                )

    def test_selected_prefill_validation_and_old_wheel_fallback(self) -> None:
        tokens = torch.tensor([[0, 1, 0], [2, 2, 3]])

        class OldWheel:
            def prefill(self, _tokens: np.ndarray) -> tuple[np.ndarray, ...]:
                raise AssertionError("full prefill fallback must not be called")

        old_state = init_candidate_state_internal(2, 3, suffix_k=2, occurrences_r=2)
        old_state.native_state = OldWheel()
        expected_state = init_candidate_state_internal(
            2, 3, suffix_k=2, occurrences_r=2
        )
        expected_state.native_state = False
        queries = torch.tensor([[2, 0], [1, 2]])
        actual = prefill_candidates_selected(old_state, tokens, queries)
        full = prefill_candidates(expected_state, tokens)
        batch = torch.arange(2).unsqueeze(1)
        self.assert_step_equal(
            actual,
            CandidateStep(*(getattr(full, field)[batch, queries] for field in _FIELDS)),
        )
        self.assertIs(old_state.native_state, False)

        invalid = (
            ([[0], [1]], TypeError, "must be a Tensor"),
            (torch.zeros(2, 1), TypeError, "dtype torch.long"),
            (torch.zeros(2, dtype=torch.long), ValueError, r"\[B, Q\]"),
            (torch.zeros(1, 1, dtype=torch.long), ValueError, r"\[B, Q\]"),
            (torch.empty(2, 0, dtype=torch.long), ValueError, "at least one"),
            (torch.tensor([[0, 3], [1, 2]]), ValueError, r"\[0, N\)"),
            (torch.tensor([[1, 1], [0, 2]]), ValueError, "unique"),
        )
        for positions, error_type, message in invalid:
            state = init_candidate_state_internal(2, 3)
            state.native_state = False
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(error_type, message),
            ):
                prefill_candidates_selected(state, tokens, positions)  # type: ignore[arg-type]
            self.assertEqual(state.position, 0)
            self.assertEqual(state.size.tolist(), [1, 1])

        nonempty = init_candidate_state_internal(2, 4)
        nonempty.native_state = False
        forward_candidates_step(nonempty, tokens[:, 0])
        with self.assertRaisesRegex(RuntimeError, "empty candidate state"):
            prefill_candidates_selected(nonempty, tokens, queries)

        if torch.cuda.is_available():
            device_state = init_candidate_state_internal(2, 3)
            with self.assertRaisesRegex(ValueError, "same device"):
                prefill_candidates_selected(device_state, tokens.cuda(), queries)

    def test_allocating_native_dispatch_paths(self) -> None:
        def outputs(
            state: CandidateState, sequence_length: int | None = None
        ) -> tuple[np.ndarray, ...]:
            slots = state.suffix_k * state.occurrences_r
            prefix = (
                (state.batch_size,)
                if sequence_length is None
                else (state.batch_size, sequence_length)
            )
            shape = (*prefix, slots)
            return (
                np.full(shape, -1, dtype=np.int64),
                np.zeros(shape, dtype=np.int64),
                np.full(shape, -1, dtype=np.int64),
                np.zeros(shape, dtype=np.int64),
                np.zeros(prefix, dtype=np.int32),
            )

        uniform = init_candidate_state(2, 2, suffix_k=2, occurrences_r=2)

        def native_step(
            state: CandidateState, tokens: torch.Tensor
        ) -> tuple[np.ndarray, ...]:
            state.position += 1
            return outputs(state)

        with patch(
            "rosa._stateful_candidates_numba._native_candidate_step",
            side_effect=native_step,
        ):
            result = forward_candidates_step(uniform, torch.tensor([1, 2]))
        self.assertFalse(bool(result.mask.any()))
        self.assertEqual(uniform.positions.tolist(), [1, 1])

        ragged = init_candidate_state_internal(
            2, 2, suffix_k=2, occurrences_r=2, ragged=True
        )

        def native_masked(
            state: CandidateState, method: str, *arrays: np.ndarray
        ) -> tuple[np.ndarray, ...] | None:
            self.assertEqual(method, "step_masked")
            active = arrays[1].astype(bool)
            reset = arrays[2].astype(bool)
            state.positions[np.logical_and(active, reset)] = 0
            state.positions[active] += 1
            return outputs(state)

        with patch(
            "rosa._stateful_candidates_numba._native_candidate_call",
            side_effect=native_masked,
        ):
            result = forward_candidates_step_masked(
                ragged,
                torch.tensor([1, 2]),
                torch.tensor([True, False]),
                torch.tensor([True, False]),
            )
        self.assertFalse(bool(result.mask.any()))
        self.assertEqual(ragged.positions.tolist(), [1, 0])

        prefilled = init_candidate_state(2, 2, suffix_k=2, occurrences_r=2)

        def native_prefill(
            state: CandidateState, method: str, tokens: np.ndarray
        ) -> tuple[np.ndarray, ...]:
            self.assertEqual(method, "prefill")
            state.position = tokens.shape[1]
            return outputs(state, tokens.shape[1])

        with patch(
            "rosa._stateful_candidates_numba._native_candidate_call",
            side_effect=native_prefill,
        ):
            result = prefill_candidates(prefilled, torch.tensor([[1, 2], [3, 4]]))
        self.assertFalse(bool(result.mask.any()))
        self.assertEqual(prefilled.positions.tolist(), [2, 2])

    def test_caller_owned_step_and_prefill_buffers_are_exact(self) -> None:
        tokens = torch.tensor([[0, 1, 0, 2], [3, 3, 4, 3]])
        state = init_candidate_state(2, 4, suffix_k=3, occurrences_r=2)
        state.native_state = False
        buffers = init_candidate_buffers(state)
        retained = forward_candidates_step_into(state, tokens[:, 0], buffers)
        retained_source = retained.source_index.clone()
        current = forward_candidates_step_into(state, tokens[:, 1], buffers)
        self.assertIs(retained.source_index, current.source_index)
        self.assertIs(retained.mask, current.mask)
        expected = build_hard_candidates(tokens[:, :2], suffix_k=3, occurrences_r=2)
        for field in _FIELDS:
            self.assertTrue(
                torch.equal(getattr(current, field), getattr(expected, field)[:, 1]),
                field,
            )
        self.assertFalse(torch.equal(retained.source_index, retained_source))

        prefill_state = init_candidate_state(2, 4, suffix_k=3, occurrences_r=2)
        prefill_state.native_state = False
        prefill_buffers = init_candidate_buffers(prefill_state, sequence_length=4)
        actual = prefill_candidates_into(prefill_state, tokens, prefill_buffers)
        expected = build_hard_candidates(tokens, suffix_k=3, occurrences_r=2)
        for field in _FIELDS:
            self.assertTrue(
                torch.equal(getattr(actual, field), getattr(expected, field))
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_caller_owned_buffers_reuse_cuda_storage(self) -> None:
        tokens = torch.tensor([[0, 0], [3, 3]], device="cuda")
        state = init_candidate_state(2, 2, suffix_k=2, occurrences_r=2)
        buffers = init_candidate_buffers(state)
        first = forward_candidates_step_into(state, tokens[:, 0], buffers)
        source_pointer = first.source_index.data_ptr()
        second = forward_candidates_step_into(state, tokens[:, 1], buffers)
        self.assertEqual(second.source_index.device.type, "cuda")
        self.assertEqual(second.source_index.data_ptr(), source_pointer)
        expected = build_hard_candidates(tokens.cpu(), suffix_k=2, occurrences_r=2)
        for field in _FIELDS:
            self.assertTrue(
                torch.equal(
                    getattr(second, field).cpu(), getattr(expected, field)[:, 1]
                ),
                field,
            )

    def test_caller_owned_buffer_validation_and_allocating_snapshots(self) -> None:
        state = init_candidate_state(1, 2, suffix_k=2, occurrences_r=2)
        state.native_state = False
        first = forward_candidates_step(state, torch.tensor([0]))
        snapshot = first.source_index.clone()
        forward_candidates_step(state, torch.tensor([0]))
        self.assertTrue(torch.equal(first.source_index, snapshot))

        wrong_state = init_candidate_state(1, 2, suffix_k=2, occurrences_r=2)
        wrong_state.native_state = False
        buffers = init_candidate_buffers(wrong_state)
        buffers.source_index = np.empty((1, 3), dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "shape"):
            forward_candidates_step_into(wrong_state, torch.tensor([0]), buffers)

        buffers = init_candidate_buffers(wrong_state)
        buffers.count.flags.writeable = False
        with self.assertRaisesRegex(ValueError, "writable"):
            forward_candidates_step_into(wrong_state, torch.tensor([0]), buffers)

        for replacement, error in (
            (object(), TypeError),
            (np.empty((1, 4), dtype=np.int32), TypeError),
            (np.empty((1, 8), dtype=np.int64)[:, ::2], ValueError),
        ):
            buffers = init_candidate_buffers(wrong_state)
            buffers.source_index = replacement  # type: ignore[assignment]
            with self.assertRaises(error):
                forward_candidates_step_into(wrong_state, torch.tensor([0]), buffers)

        buffers = init_candidate_buffers(wrong_state)
        buffers.match_length = buffers.source_index
        with self.assertRaisesRegex(ValueError, "overlap"):
            forward_candidates_step_into(wrong_state, torch.tensor([0]), buffers)

        token_overlap_state = init_candidate_state(1, 1, suffix_k=1, occurrences_r=1)
        token_overlap_state.native_state = False
        buffers = init_candidate_buffers(token_overlap_state)
        with self.assertRaisesRegex(ValueError, "tokens"):
            forward_candidates_step_into(
                token_overlap_state,
                torch.from_numpy(buffers.source_index.reshape(-1)),
                buffers,
            )

        buffers = init_candidate_buffers(wrong_state)
        buffers.source_index = wrong_state.lazy_prefix.reshape(1, -1)[:, :4]
        with self.assertRaisesRegex(ValueError, "state storage"):
            forward_candidates_step_into(wrong_state, torch.tensor([0]), buffers)

        with self.assertRaisesRegex(TypeError, "CandidateBuffers"):
            forward_candidates_step_into(
                wrong_state,
                torch.tensor([0]),
                object(),  # type: ignore[arg-type]
            )

        with self.assertRaisesRegex(TypeError, "CandidateState"):
            init_candidate_buffers(object())
        with self.assertRaisesRegex(ValueError, "sequence_length"):
            init_candidate_buffers(wrong_state, sequence_length=-1)

        ragged = init_candidate_state_internal(1, 2, ragged=True)
        with self.assertRaisesRegex(RuntimeError, "ragged"):
            forward_candidates_step_into(
                ragged, torch.tensor([0]), init_candidate_buffers(ragged)
            )
        with self.assertRaisesRegex(RuntimeError, "ragged"):
            prefill_candidates_into(
                ragged,
                torch.tensor([[0]]),
                init_candidate_buffers(ragged, sequence_length=1),
            )

        full = init_candidate_state(1, 1)
        full.native_state = False
        full_buffers = init_candidate_buffers(full)
        forward_candidates_step_into(full, torch.tensor([0]), full_buffers)
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            forward_candidates_step_into(full, torch.tensor([0]), full_buffers)

        nonempty = init_candidate_state(1, 2)
        nonempty.native_state = False
        forward_candidates_step_into(
            nonempty, torch.tensor([0]), init_candidate_buffers(nonempty)
        )
        with self.assertRaisesRegex(RuntimeError, "empty"):
            prefill_candidates_into(
                nonempty,
                torch.tensor([[0]]),
                init_candidate_buffers(nonempty, sequence_length=1),
            )

        short = init_candidate_state(1, 1)
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            prefill_candidates_into(
                short,
                torch.tensor([[0, 1]]),
                init_candidate_buffers(short, sequence_length=2),
            )

        native_like = init_candidate_state(1, 1)
        native_buffers = init_candidate_buffers(native_like, sequence_length=1)

        def fill_like_native(
            candidate_state: CandidateState,
            method: str,
            tokens: np.ndarray,
            arrays: tuple[np.ndarray, ...],
            validate_fallback: object,
        ) -> bool:
            self.assertEqual(method, "prefill_into")
            self.assertTrue(callable(validate_fallback))
            validate_fallback()  # type: ignore[operator]
            validate_fallback()  # type: ignore[operator]
            for array in arrays:
                array.fill(0)
            candidate_state.position = tokens.shape[1]
            return True

        with patch(
            "rosa._stateful_candidates_numba._native_candidate_into",
            side_effect=fill_like_native,
        ):
            prefill_candidates_into(native_like, torch.tensor([[0]]), native_buffers)
        self.assertEqual(native_like.positions.tolist(), [1])

        old_state = init_candidate_state(1, 1, suffix_k=2, occurrences_r=2)

        class AllocatingOnlyNative:
            def step(self, tokens: np.ndarray) -> tuple[np.ndarray, ...]:
                old_state.position += 1
                old_state.positions.fill(old_state.position)
                return (
                    np.full((1, 4), -1, dtype=np.int64),
                    np.zeros((1, 4), dtype=np.int64),
                    np.full((1, 4), -1, dtype=np.int64),
                    np.zeros((1, 4), dtype=np.int64),
                    np.zeros(1, dtype=np.int32),
                )

        old_state.native_state = AllocatingOnlyNative()
        old_result = forward_candidates_step_into(
            old_state, torch.tensor([0]), init_candidate_buffers(old_state)
        )
        self.assertFalse(bool(old_result.mask.any()))

        missing_state = init_candidate_state(1, 1)
        missing_state.native_state = object()
        missing_result = forward_candidates_step_into(
            missing_state, torch.tensor([0]), init_candidate_buffers(missing_state)
        )
        self.assertEqual(missing_result.rosa_slot.tolist(), [-1])

        repeated_validation = init_candidate_state(1, 1)
        repeated_validation.native_state = False

        def miss_after_validating(
            candidate_state: CandidateState,
            method: str,
            tokens: np.ndarray,
            arrays: tuple[np.ndarray, ...],
            validate_fallback: object,
        ) -> bool:
            validate_fallback()  # type: ignore[operator]
            validate_fallback()  # type: ignore[operator]
            return False

        with patch(
            "rosa._stateful_candidates_numba._native_candidate_into",
            side_effect=miss_after_validating,
        ):
            forward_candidates_step_into(
                repeated_validation,
                torch.tensor([0]),
                init_candidate_buffers(repeated_validation),
            )

    def test_ragged_inactive_reset_capacity_and_recycle(self) -> None:
        state = init_candidate_state_internal(
            3, 5, suffix_k=4, occurrences_r=3, ragged=True
        )
        state.native_state = False
        histories: list[list[int]] = [[], [], []]
        schedule = (
            ([1, 8, 3], [1, 1, 0], [0, 0, 0]),
            ([2, 9, 4], [1, 0, 1], [0, 0, 0]),
            ([1, 7, 3], [1, 1, 1], [0, 0, 0]),
            ([2, 6, 5], [1, 1, 0], [0, 1, 0]),
            ([1, 6, 3], [1, 1, 1], [0, 0, 1]),
        )
        for token_values, active_values, reset_values in schedule:
            tokens = torch.tensor(token_values)
            active = torch.tensor(active_values, dtype=torch.bool)
            reset = torch.tensor(reset_values, dtype=torch.bool)
            actual = forward_candidates_step_masked(state, tokens, active, reset)
            for batch_index in range(3):
                if not active_values[batch_index]:
                    self.assertFalse(bool(actual.mask[batch_index].any()))
                    continue
                if reset_values[batch_index]:
                    histories[batch_index].clear()
                histories[batch_index].append(token_values[batch_index])
                oracle = build_hard_candidates(
                    torch.tensor([histories[batch_index]]),
                    suffix_k=4,
                    occurrences_r=3,
                )
                for field in _FIELDS:
                    self.assertTrue(
                        torch.equal(
                            getattr(actual, field)[batch_index],
                            getattr(oracle, field)[0, -1],
                        ),
                        field,
                    )
        self.assertEqual(state.positions.tolist(), [5, 2, 1])
        reset_candidates_masked(state, torch.tensor([True, False, True]))
        self.assertEqual(state.positions.tolist(), [0, 2, 0])
        recycled = forward_candidates_step_masked(
            state,
            torch.tensor([4, 0, 4]),
            torch.tensor([True, False, True]),
        )
        self.assertFalse(bool(recycled.mask.any()))

        full = init_candidate_state_internal(1, 1, ragged=True)
        full.native_state = False
        forward_candidates_step_masked(full, torch.tensor([1]), torch.tensor([True]))
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            forward_candidates_step_masked(
                full, torch.tensor([2]), torch.tensor([True])
            )
        # Reset and consume in one operation must be allowed at full capacity.
        forward_candidates_step_masked(
            full,
            torch.tensor([2]),
            torch.tensor([True]),
            torch.tensor([True]),
        )

    def test_uniform_and_ragged_modes_cannot_mix(self) -> None:
        uniform = init_candidate_state(1, 2)
        ragged = init_candidate_state_internal(1, 2, ragged=True)
        with self.assertRaisesRegex(RuntimeError, "ragged"):
            forward_candidates_step_masked(
                uniform, torch.tensor([1]), torch.tensor([True])
            )
        with self.assertRaisesRegex(RuntimeError, "ragged"):
            forward_candidates_step(ragged, torch.tensor([1]))
        with self.assertRaisesRegex(RuntimeError, "ragged"):
            prefill_candidates(ragged, torch.tensor([[1]]))
        with self.assertRaisesRegex(RuntimeError, "ragged"):
            reset_candidates_masked(uniform, torch.tensor([True]))

    def test_scalar_batch_and_validation(self) -> None:
        state = init_candidate_state(1, 2, suffix_k=2, occurrences_r=2)
        first = forward_candidates_step(state, torch.tensor(7))
        self.assertEqual(tuple(first.source_index.shape), (1, 4))
        forward_candidates_step(state, torch.tensor([7]))
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            forward_candidates_step(state, torch.tensor([7]))

        for kwargs, message in (
            ({"batch_size": 0, "max_length": 1}, "batch_size"),
            ({"batch_size": 1, "max_length": 0}, "max_length"),
            ({"batch_size": 1, "max_length": 1, "suffix_k": 0}, "suffix_k"),
            (
                {"batch_size": 1, "max_length": 1, "occurrences_r": 0},
                "occurrences_r",
            ),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    init_candidate_state(**kwargs)

        shape_state = init_candidate_state(2, 1)
        with self.assertRaisesRegex(ValueError, "shape"):
            forward_candidates_step(shape_state, torch.tensor([1]))
        with self.assertRaisesRegex(TypeError, "integer"):
            forward_candidates_step(shape_state, torch.tensor([1.0, 2.0]))
        with self.assertRaisesRegex(TypeError, "CandidateState"):
            forward_candidates_step(object(), torch.tensor([1]))  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "Tensor"):
            forward_candidates_step(state, object())  # type: ignore[arg-type]

        ragged = init_candidate_state_internal(1, 3, ragged=True)
        ragged.native_state = False
        with self.assertRaisesRegex(TypeError, "CandidateState"):
            reset_candidates_masked(object(), torch.tensor([True]))  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "Tensor"):
            reset_candidates_masked(ragged, object())  # type: ignore[arg-type]
        reset_candidates_masked(ragged, torch.tensor(True))
        with self.assertRaisesRegex(ValueError, "shape"):
            reset_candidates_masked(ragged, torch.tensor([True, False]))
        with self.assertRaisesRegex(TypeError, "bool or uint8"):
            reset_candidates_masked(ragged, torch.tensor([1]))

        with self.assertRaisesRegex(TypeError, "active"):
            forward_candidates_step_masked(
                ragged,
                torch.tensor([1]),
                object(),  # type: ignore[arg-type]
            )
        forward_candidates_step_masked(ragged, torch.tensor([1]), torch.tensor(True))
        with self.assertRaisesRegex(ValueError, "active"):
            forward_candidates_step_masked(
                ragged, torch.tensor([1]), torch.tensor([True, False])
            )
        with self.assertRaisesRegex(TypeError, "active"):
            forward_candidates_step_masked(ragged, torch.tensor([1]), torch.tensor([1]))
        with self.assertRaisesRegex(TypeError, "reset"):
            forward_candidates_step_masked(
                ragged,
                torch.tensor([1]),
                torch.tensor([True]),
                object(),  # type: ignore[arg-type]
            )
        forward_candidates_step_masked(
            ragged,
            torch.tensor([1]),
            torch.tensor([True]),
            torch.tensor(True),
        )
        with self.assertRaisesRegex(ValueError, "reset"):
            forward_candidates_step_masked(
                ragged,
                torch.tensor([1]),
                torch.tensor([True]),
                torch.tensor([True, False]),
            )
        with self.assertRaisesRegex(TypeError, "reset"):
            forward_candidates_step_masked(
                ragged,
                torch.tensor([1]),
                torch.tensor([True]),
                torch.tensor([1]),
            )

        prefill_state = init_candidate_state_internal(1, 1)
        prefill_state.native_state = False
        with self.assertRaisesRegex(ValueError, "shape"):
            prefill_candidates(prefill_state, torch.tensor([1]))
        with self.assertRaisesRegex(TypeError, "integer"):
            prefill_candidates(prefill_state, torch.tensor([[1.0]]))
        prefill_candidates(prefill_state, torch.tensor([[1]]))
        with self.assertRaisesRegex(RuntimeError, "empty"):
            prefill_candidates(prefill_state, torch.tensor([[1]]))
        too_long = init_candidate_state_internal(1, 1)
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            prefill_candidates(too_long, torch.tensor([[1, 2]]))

        negative = init_candidate_state_internal(1, 2, ragged=True)
        negative.native_state = False
        negative.positions[0] = -1
        with self.assertRaisesRegex(RuntimeError, "non-negative"):
            forward_candidates_step_masked(
                negative, torch.tensor([1]), torch.tensor([True])
            )

        old_capability = init_candidate_state_internal(1, 2)
        old_capability.native_state = object()
        prefill_candidates(old_capability, torch.tensor([[0, 1]]))
        self.assertIs(old_capability.native_state, False)

    def test_public_wrapper_reports_missing_numba(self) -> None:
        with patch.dict(sys.modules):
            sys.modules.pop("rosa._stateful_candidates_numba", None)
            sys.modules["numba"] = None
            with self.assertRaisesRegex(RuntimeError, "numba"):
                init_candidate_state(1, 1)
            with self.assertRaisesRegex(RuntimeError, "numba"):
                forward_candidates_step(object(), torch.tensor([1]))


if __name__ == "__main__":
    unittest.main()
