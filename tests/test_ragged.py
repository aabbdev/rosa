from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import torch

from rosa.ragged import RaggedInferenceState, init_ragged_state


class TestRaggedInference(unittest.TestCase):
    def test_close_is_idempotent_and_rejects_use(self) -> None:
        state = init_ragged_state(1, 2, use_native=False)
        state.close()
        state.close()
        with self.assertRaisesRegex(RuntimeError, "^state is closed$"):
            _ = state.positions
        with self.assertRaisesRegex(RuntimeError, "^state is closed$"):
            state.step(torch.tensor([0]))

    def test_mask_reset_and_recycling_match_single_row_oracles(self) -> None:
        from rosa._stateful_numba import _forward_step, _init_inference_state

        ragged = RaggedInferenceState(4, 40, use_native=False)
        oracles = [_init_inference_state(1, 40) for _ in range(4)]
        for oracle in oracles:
            oracle.native_state = False
        generator = torch.Generator().manual_seed(20260811)
        for tick in range(30):
            tokens = torch.randint(-2, 7, (4,), generator=generator)
            active = torch.tensor([(tick + row) % (row + 2) != 0 for row in range(4)])
            reset = torch.tensor(
                [tick in (9 + row, 20 + row) for row in range(4)],
                dtype=torch.bool,
            )
            before = ragged.positions
            actual = ragged.step_masked(tokens, active, reset)
            expected = torch.full((4,), -1, dtype=torch.long)
            for row in range(4):
                if not active[row]:
                    self.assertEqual(ragged.positions[row], before[row])
                    continue
                if reset[row]:
                    oracles[row] = _init_inference_state(1, 40)
                    oracles[row].native_state = False
                expected[row] = _forward_step(oracles[row], tokens[row]).item()
            self.assertTrue(torch.equal(actual, expected), tick)
            self.assertEqual(ragged.positions.tolist(), [o.position for o in oracles])

    def test_capacity_is_per_row_and_reset_precedes_consumption(self) -> None:
        state = init_ragged_state(2, 2, use_native=False)
        state.step(torch.tensor([1, 4]), torch.tensor([1, 0], dtype=torch.uint8))
        state.step(torch.tensor([2, 5]), torch.tensor([1, 0], dtype=torch.uint8))
        self.assertEqual(state.positions.tolist(), [2, 0])
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            state.step(torch.tensor([3, 6]), torch.tensor([1, 0], dtype=torch.uint8))
        output = state.step(
            torch.tensor([3, 6]),
            torch.tensor([1, 1], dtype=torch.uint8),
            torch.tensor([1, 0], dtype=torch.uint8),
        )
        self.assertEqual(output.tolist(), [-1, -1])
        self.assertEqual(state.positions.tolist(), [1, 1])

    def test_inactive_reset_is_ignored_and_inputs_are_validated(self) -> None:
        state = RaggedInferenceState(2, 3, use_native=False)
        state.step(torch.tensor([1, 2]))
        before = state.positions
        output = state.step_masked(
            torch.tensor([3, 4]),
            torch.tensor([False, True]),
            torch.tensor([True, False]),
        )
        self.assertEqual(output[0], -1)
        self.assertEqual(state.positions[0], before[0])
        with self.assertRaisesRegex(ValueError, "tokens"):
            state.step(torch.zeros((2, 1), dtype=torch.long))
        with self.assertRaisesRegex(TypeError, "Tensor"):
            state.step_masked(object(), torch.ones(2, dtype=torch.bool))  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "integer"):
            state.step(torch.tensor([1.9, 2.1]))
        with self.assertRaisesRegex(ValueError, "active"):
            state.step_masked(torch.zeros(2, dtype=torch.long), torch.ones(3))
        with self.assertRaisesRegex(TypeError, "active"):
            state.step_masked(torch.zeros(2, dtype=torch.long), torch.ones(2))
        with self.assertRaisesRegex(TypeError, "reset"):
            state.step_masked(
                torch.zeros(2, dtype=torch.long),
                torch.ones(2, dtype=torch.bool),
                torch.ones(2, dtype=torch.int64),
            )

        scalar = RaggedInferenceState(1, 1, use_native=False)
        self.assertEqual(
            scalar.step_masked(torch.tensor(1), torch.tensor(True)).ndim, 1
        )

    def test_missing_companion_falls_back_exactly(self) -> None:
        with patch.dict(sys.modules, {"rosa_native_step": None}):
            state = RaggedInferenceState(1, 4)
            actual = torch.stack(
                [state.step(torch.tensor(token)) for token in (0, 1, 0, 2)]
            ).flatten()
        self.assertEqual(actual.tolist(), [-1, -1, 1, -1])
        self.assertFalse(state.using_native)

        class OldNativeState:
            def __init__(self, state: object) -> None:
                self.state = state

        old_module = types.SimpleNamespace(NativeState=OldNativeState)
        with patch.dict(sys.modules, {"rosa_native_step": old_module}):
            old_state = RaggedInferenceState(1, 1)
            self.assertIsNone(old_state._native_state())
            self.assertFalse(old_state.using_native)

        class CurrentNativeState(OldNativeState):
            def step_masked(self) -> None:
                return None

        current_module = types.SimpleNamespace(NativeState=CurrentNativeState)
        with patch.dict(sys.modules, {"rosa_native_step": current_module}):
            current_state = RaggedInferenceState(1, 1)
            selected = current_state._native_state()
            self.assertIsInstance(selected, CurrentNativeState)
            self.assertIs(current_state._native_state(), selected)
            self.assertTrue(current_state.using_native)


if __name__ == "__main__":
    unittest.main()
