from __future__ import annotations

import gc
import unittest
import weakref
from typing import Any, cast

import torch

from rosa import (
    HardCandidates,
    InferenceOutput,
    build_hard_candidates,
    forward_candidates_step,
    forward_candidates_step_into,
    forward_step,
    init_candidate_buffers,
    init_candidate_state,
    init_inference_state,
    prefill,
    reference_rosa,
)


class TestUnifiedInferenceState(unittest.TestCase):
    def test_close_context_manager_and_reset_lifetime(self) -> None:
        tokens = torch.tensor([[0, 1, 0]], dtype=torch.long)
        expected = reference_rosa(tokens)[0]

        with init_inference_state(1, 3, mode="rich", suffix_k=2) as managed:
            buffers = init_candidate_buffers(managed)
            self.assertTrue(
                torch.equal(managed.prefill(tokens).predicted_tokens, expected)
            )
        for access in (
            lambda: managed.position,
            lambda: managed.positions,
            lambda: managed.step(torch.tensor([0])),
            lambda: managed.step_into(torch.tensor([0]), buffers),
            lambda: managed.prefill(tokens),
        ):
            with (
                self.subTest(access=access),
                self.assertRaisesRegex(RuntimeError, "^state is closed$"),
            ):
                access()
        managed.close()

        for mode in ("top1", "rich"):
            with self.subTest(mode=mode):
                state = init_inference_state(1, 3, mode=mode)  # type: ignore[arg-type]
                old_impl = state._impl
                assert old_impl is not None

                class NativeOwner:
                    def __init__(self, impl: object) -> None:
                        self.impl = impl

                old_impl.native_state = NativeOwner(old_impl)  # type: ignore[attr-defined]
                impl_ref = weakref.ref(old_impl)
                array_ref = weakref.ref(old_impl.history)  # type: ignore[attr-defined]
                del old_impl
                state.reset()
                gc.collect()
                self.assertIsNone(impl_ref())
                self.assertIsNone(array_ref())
                self.assertEqual(state.positions.tolist(), [0])
                self.assertTrue(
                    torch.equal(state.prefill(tokens).predicted_tokens, expected)
                )
                state.close()
                state.reset()
                self.assertEqual(state.position, 0)
                self.assertTrue(
                    torch.equal(state.prefill(tokens).predicted_tokens, expected)
                )

    def test_uniform_mode_matrix_prefill_continuation_and_reset(self) -> None:
        tokens = torch.tensor(
            [[0, 1, 0, 2, 0, 3], [4, 4, 5, 4, 4, 6]], dtype=torch.long
        )
        expected, _, _ = reference_rosa(tokens)
        for mode in ("top1", "rich"):
            with self.subTest(mode=mode):
                state = init_inference_state(
                    2,
                    6,
                    mode=mode,  # type: ignore[arg-type]
                    suffix_k=3,
                    occurrences_r=2,
                )
                initial = state.prefill(tokens[:, :4])
                self.assertIsInstance(initial, InferenceOutput)
                continuation = torch.stack(
                    [state.step(tokens[:, index]).predicted_tokens for index in (4, 5)],
                    dim=1,
                )
                self.assertTrue(
                    torch.equal(
                        torch.cat((initial.predicted_tokens, continuation), dim=1),
                        expected,
                    )
                )
                self.assertEqual(state.positions.tolist(), [6, 6])
                if mode == "top1":
                    self.assertIsNone(initial.candidates)
                else:
                    self.assertIsInstance(initial.candidates, HardCandidates)
                    oracle = build_hard_candidates(
                        tokens[:, :4], suffix_k=3, occurrences_r=2
                    )
                    assert isinstance(initial.candidates, HardCandidates)
                    for name in HardCandidates.__dataclass_fields__:
                        self.assertTrue(
                            torch.equal(
                                getattr(initial.candidates, name), getattr(oracle, name)
                            ),
                            name,
                        )
                state.reset()
                self.assertEqual(state.positions.tolist(), [0, 0])
                self.assertTrue(
                    torch.equal(state.prefill(tokens).predicted_tokens, expected)
                )

    def test_legacy_top1_and_candidate_wrappers_remain_compatible(self) -> None:
        tokens = torch.tensor([[0, 1, 0, 2]], dtype=torch.long)
        top1 = init_inference_state(1, 4, backend="numba")
        self.assertTrue(torch.equal(prefill(top1, tokens), reference_rosa(tokens)[0]))

        rich = init_inference_state(1, 4, mode="rich", suffix_k=2, occurrences_r=2)
        candidate = forward_candidates_step(rich, tokens[:, 0])
        self.assertEqual(tuple(candidate.source_index.shape), (1, 4))
        self.assertEqual(forward_step(rich, tokens[:, 1]).shape, (1,))

        old_state = init_candidate_state(1, 1, suffix_k=2, occurrences_r=2)
        old_output = forward_candidates_step(old_state, torch.tensor([7]))
        self.assertEqual(tuple(old_output.source_index.shape), (1, 4))

    def test_rich_step_into_uses_explicit_ephemeral_storage(self) -> None:
        rich = init_inference_state(2, 2, mode="rich", suffix_k=2, occurrences_r=2)
        buffers = init_candidate_buffers(rich)
        first = rich.step_into(torch.tensor([0, 3]), buffers)
        self.assertIsNotNone(first.candidates)
        second = rich.step_into(torch.tensor([0, 3]), buffers)
        expected = build_hard_candidates(
            torch.tensor([[0, 0], [3, 3]]), suffix_k=2, occurrences_r=2
        )
        assert second.candidates is not None
        for field in HardCandidates.__dataclass_fields__:
            self.assertTrue(
                torch.equal(
                    getattr(second.candidates, field), getattr(expected, field)[:, 1]
                ),
                field,
            )

        wrapper = init_inference_state(1, 1, mode="rich", suffix_k=2, occurrences_r=2)
        wrapped = forward_candidates_step_into(
            wrapper, torch.tensor(0), init_candidate_buffers(wrapper)
        )
        self.assertEqual(wrapped.source_index.ndim, 1)

        top1 = init_inference_state(1, 1)
        with self.assertRaisesRegex(ValueError, "rich"):
            init_candidate_buffers(top1)
        with self.assertRaisesRegex(ValueError, "uniform rich"):
            top1.step_into(torch.tensor([0]), buffers)

        rich_ragged = init_inference_state(1, 1, mode="rich", ragged=True)
        with self.assertRaisesRegex(ValueError, "uniform rich"):
            rich_ragged.step_into(torch.tensor([0]), buffers)

    def test_positions_are_copies_and_uniform_position_is_preserved(self) -> None:
        uniform = init_inference_state(2, 2)
        forward_step(uniform, torch.tensor([1, 2]))
        positions = uniform.positions
        positions[0] = 99
        self.assertEqual(uniform.positions.tolist(), [1, 1])
        self.assertEqual(uniform.position, 1)

        ragged = init_inference_state(2, 2, ragged=True)
        ragged.step(torch.tensor([1, 2]), active=torch.tensor([True, False]))
        ragged_positions = ragged.positions
        ragged_positions[0] = 99
        self.assertEqual(ragged.positions.tolist(), [1, 0])
        with self.assertRaisesRegex(AttributeError, "positions"):
            _ = ragged.position

    def test_top1_does_not_allocate_rich_occurrence_storage(self) -> None:
        top1 = init_inference_state(2, 32, backend="numba")
        rich = init_inference_state(2, 32, mode="rich")
        self.assertFalse(hasattr(top1._impl, "occurrences"))
        self.assertTrue(hasattr(rich._impl, "occurrences"))

    def test_ragged_top1_reset_and_inactive_rows(self) -> None:
        state = init_inference_state(2, 3, ragged=True)
        first = state.step(torch.tensor([5, 7]), active=torch.tensor([True, False]))
        self.assertEqual(first.predicted_tokens.tolist(), [-1, -1])
        second = state.step(
            torch.tensor([5, 7]),
            active=torch.tensor([True, True]),
            reset=torch.tensor([True, False]),
        )
        self.assertEqual(second.predicted_tokens.tolist(), [-1, -1])
        self.assertEqual(state.positions.tolist(), [1, 1])
        state.reset()
        self.assertEqual(state.positions.tolist(), [0, 0])

    def test_rich_ragged_matches_independent_row_states(self) -> None:
        state = init_inference_state(
            2,
            4,
            mode="rich",
            ragged=True,
            suffix_k=2,
            occurrences_r=2,
        )
        row_states = [
            init_candidate_state(1, 4, suffix_k=2, occurrences_r=2) for _ in range(2)
        ]
        schedule = [
            (torch.tensor([0, 4]), torch.tensor([1, 0]), torch.tensor([0, 0])),
            (torch.tensor([1, 4]), torch.tensor([1, 1]), torch.tensor([0, 0])),
            (torch.tensor([0, 5]), torch.tensor([1, 1]), torch.tensor([0, 1])),
        ]
        for tokens, active, reset in schedule:
            actual = state.step(tokens, active=active.bool(), reset=reset.bool())
            assert actual.candidates is not None
            for row in range(2):
                if not bool(active[row]):
                    self.assertEqual(actual.predicted_tokens[row].item(), -1)
                    continue
                if bool(reset[row]):
                    row_states[row] = init_candidate_state(
                        1, 4, suffix_k=2, occurrences_r=2
                    )
                expected = forward_candidates_step(row_states[row], tokens[row])
                for name in expected.__dataclass_fields__:
                    self.assertTrue(
                        torch.equal(
                            getattr(actual.candidates, name)[row],
                            getattr(expected, name)[0],
                        ),
                        name,
                    )
        self.assertEqual(state.positions.tolist(), [3, 1])

    def test_rich_scalar_prefill(self) -> None:
        rich = init_inference_state(1, 3, mode="rich", suffix_k=2, occurrences_r=2)
        scalar_step = rich.step(torch.tensor(0))
        self.assertEqual(scalar_step.predicted_tokens.ndim, 0)
        self.assertEqual(cast(Any, scalar_step.candidates).source_index.ndim, 1)
        rich.reset()
        output = rich.prefill(torch.tensor([0, 1, 0]))
        self.assertEqual(tuple(output.predicted_tokens.shape), (3,))
        assert isinstance(output.candidates, HardCandidates)
        self.assertEqual(tuple(output.candidates.source_index.shape), (3, 4))
        rich.reset()
        empty = rich.prefill(torch.empty(0, dtype=torch.long))
        assert isinstance(empty.candidates, HardCandidates)
        self.assertEqual(tuple(empty.candidates.source_index.shape), (0, 4))

    def test_mode_validation_and_uniform_masks(self) -> None:
        with self.assertRaisesRegex(ValueError, "mode"):
            init_inference_state(1, mode="other")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "suffix_k"):
            init_inference_state(1, mode="rich", suffix_k=0)
        with self.assertRaisesRegex(ValueError, "occurrences_r"):
            init_inference_state(1, mode="rich", occurrences_r=0)
        with self.assertRaisesRegex(ValueError, "does not support"):
            init_inference_state(1, mode="rich", backend="python")
        with self.assertRaisesRegex(ValueError, "ragged"):
            init_inference_state(1, ragged=True, backend="python")
        state = init_inference_state(1)
        with self.assertRaisesRegex(ValueError, "ragged"):
            state.step(torch.tensor([1]), active=torch.tensor([True]))
        with self.assertRaisesRegex(ValueError, "rich"):
            forward_candidates_step(state, torch.tensor([1]))

        rich_ragged = init_inference_state(1, 2, mode="rich", ragged=True)
        with self.assertRaisesRegex(TypeError, "active"):
            rich_ragged.step(torch.tensor([1]), active=object())  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "active"):
            rich_ragged.step(torch.tensor([1]), active=torch.tensor([True, False]))
        with self.assertRaisesRegex(TypeError, "active"):
            rich_ragged.step(torch.tensor([1]), active=torch.tensor([1]))

        empty_ragged = init_inference_state(1, 2, mode="rich", ragged=True)
        empty = empty_ragged.prefill(torch.empty((1, 0), dtype=torch.long))
        self.assertEqual(tuple(empty.predicted_tokens.shape), (1, 0))
        self.assertIsInstance(empty.candidates, HardCandidates)

        rich_prefill = init_inference_state(
            1, 3, mode="rich", ragged=True, suffix_k=2, occurrences_r=2
        ).prefill(torch.tensor([[0, 1, 0]]))
        self.assertEqual(rich_prefill.predicted_tokens.tolist(), [[-1, -1, 1]])
        self.assertIsInstance(rich_prefill.candidates, HardCandidates)

        top1_ragged = init_inference_state(1, 2, ragged=True)
        top1_empty = top1_ragged.prefill(torch.empty((1, 0), dtype=torch.long))
        self.assertEqual(tuple(top1_empty.predicted_tokens.shape), (1, 0))
        self.assertIsNone(top1_empty.candidates)
        top1_full = init_inference_state(1, 2, ragged=True).prefill(
            torch.tensor([[0, 0]])
        )
        self.assertEqual(top1_full.predicted_tokens.tolist(), [[-1, 0]])
        self.assertIsNone(top1_full.candidates)


if __name__ == "__main__":
    unittest.main()
