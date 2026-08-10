from __future__ import annotations

import unittest
from itertools import product
from typing import Any, Literal, cast
from unittest.mock import patch

import torch

from rosa import (
    ROSAInferenceState,
    forward_step,
    init_inference_state,
    reference_rosa,
)


class TestStatefulInference(unittest.TestCase):
    def collect(
        self,
        tokens: torch.Tensor,
        *,
        backend: str,
    ) -> tuple[ROSAInferenceState, torch.Tensor]:
        selected = cast(Literal["auto", "python", "numba"], backend)
        state = init_inference_state(tokens.shape[0], tokens.shape[1], backend=selected)
        output = torch.stack(
            [
                forward_step(state, tokens[:, position])
                for position in range(tokens.shape[1])
            ],
            dim=1,
        )
        return state, output

    def test_python_and_numba_match_reference_step_by_step(self) -> None:
        generator = torch.Generator().manual_seed(20260811)
        cases = [
            torch.zeros((2, 64), dtype=torch.long),
            torch.arange(128).remainder(2).reshape(2, 64),
            torch.randint(17, (3, 63), generator=generator),
            torch.tensor([[-10_000_000, -1, 2**31, 10**12, -1, 2**31]]),
        ]
        for backend in ("python", "numba"):
            for case_index, tokens in enumerate(cases):
                with self.subTest(backend=backend, case=case_index):
                    state, output = self.collect(tokens, backend=backend)
                    expected, _, _ = reference_rosa(tokens)
                    self.assertTrue(torch.equal(output, expected))
                    self.assertEqual(state.position, tokens.shape[1])

    def test_scalar_reset_isolation_and_capacity(self) -> None:
        first = init_inference_state(1, 3, backend="numba")
        second = init_inference_state(1, 3, backend="numba")
        sequence = [0, 1, 0]
        first_pass = [
            forward_step(first, torch.tensor(token)).item() for token in sequence
        ]
        self.assertEqual(second.position, 0)
        first.reset()
        second_pass = [
            forward_step(first, torch.tensor(token)).item() for token in sequence
        ]
        self.assertEqual(first_pass, second_pass)
        self.assertEqual(first.position, 3)
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            forward_step(first, torch.tensor(2))

        python_state = init_inference_state(1, 1, backend="python")
        forward_step(python_state, torch.tensor(0))
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            forward_step(python_state, torch.tensor(1))

    def test_exhaustive_binary_clone_sequences(self) -> None:
        rows = list(product(range(2), repeat=10))
        tokens = torch.tensor(rows, dtype=torch.long)
        _, output = self.collect(tokens, backend="numba")
        expected, _, _ = reference_rosa(tokens)
        self.assertTrue(torch.equal(output, expected))

    def test_auto_backend_and_validation(self) -> None:
        state = init_inference_state(1, backend="auto")
        self.assertIn(state.backend, {"python", "numba"})
        self.assertIn("backend=", repr(state))

        for batch_size, max_length, message in (
            (0, 1, "batch_size"),
            (1, 0, "max_length"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    init_inference_state(batch_size, max_length)
        with self.assertRaisesRegex(ValueError, "backend"):
            init_inference_state(1, backend=cast(Any, "invalid"))

        with self.assertRaisesRegex(TypeError, "state"):
            forward_step(cast(Any, object()), torch.tensor([1]))
        with self.assertRaisesRegex(TypeError, "Tensor"):
            forward_step(state, cast(Any, 1))
        with self.assertRaisesRegex(ValueError, "shape"):
            forward_step(state, torch.tensor([1, 2]))
        with self.assertRaisesRegex(TypeError, "integer"):
            forward_step(state, torch.tensor([1.0]))

    def test_numba_missing_fallback_and_explicit_error(self) -> None:
        real_import = __import__

        def missing_numba(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.endswith("_stateful_numba"):
                raise ModuleNotFoundError("No module named 'numba'", name="numba")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=missing_numba):
            fallback = init_inference_state(1, backend="auto")
            self.assertEqual(fallback.backend, "python")
            with self.assertRaisesRegex(ImportError, "rosa-torch\\[numba\\]"):
                init_inference_state(1, backend="numba")

        def unexpected_missing(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.endswith("_stateful_numba"):
                raise ModuleNotFoundError("unexpected", name="unexpected")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=unexpected_missing):
            with self.assertRaises(ModuleNotFoundError):
                init_inference_state(1, backend="numba")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_token_returns_cuda_prediction(self) -> None:
        tokens = torch.tensor([[0, 1, 0, 2]], device="cuda")
        _, output = self.collect(tokens, backend="numba")
        expected, _, _ = reference_rosa(tokens)
        self.assertEqual(output.device.type, "cuda")
        self.assertTrue(torch.equal(output, expected))
