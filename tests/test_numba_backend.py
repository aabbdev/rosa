from __future__ import annotations

import random
import unittest

import torch

from rosa import reference_rosa

try:
    from rosa._numba_backend import predict_exact
    from rosa._stateful_numba import (
        _forward_step,
        _init_inference_state,
        predict_exact_stateful,
    )
except ModuleNotFoundError as error:
    if error.name not in {"numba", "numpy"}:
        raise
    predict_exact = None


@unittest.skipIf(predict_exact is None, "rosa-torch[numba] is not installed")
class TestNumbaBackend(unittest.TestCase):
    def assert_matches_reference(self, tokens: torch.Tensor) -> None:
        expected, _, _ = reference_rosa(tokens)
        assert predict_exact is not None
        self.assertTrue(torch.equal(predict_exact(tokens), expected))

    def test_empty_squeezed_and_batched_inputs(self) -> None:
        self.assert_matches_reference(torch.empty(0, dtype=torch.long))
        self.assert_matches_reference(torch.empty((3, 0), dtype=torch.long))

        squeezed = torch.tensor([0, 1, 0, 2, 0], dtype=torch.long)
        self.assert_matches_reference(squeezed)
        self.assertEqual(tuple(predict_exact(squeezed).shape), (5,))

    def test_serial_and_parallel_dispatch_match_reference(self) -> None:
        generator = torch.Generator().manual_seed(20260811)
        for shape in ((1, 128), (8, 128), (8, 512), (8, 513)):
            tokens = torch.randint(256, shape, generator=generator, dtype=torch.long)
            self.assert_matches_reference(tokens)

    def test_random_negative_and_large_token_ids(self) -> None:
        rng = random.Random(20260811)
        alphabet = (-10_000_000, -1, 0, 2**31, 10**12)
        for length in range(1, 40):
            with self.subTest(length=length):
                tokens = torch.tensor(
                    [[rng.choice(alphabet) for _ in range(length)]], dtype=torch.long
                )
                self.assert_matches_reference(tokens)

    def test_shape_validation(self) -> None:
        assert predict_exact is not None
        with self.assertRaisesRegex(ValueError, "shape"):
            predict_exact(torch.zeros(1, 2, 3, dtype=torch.long))

    def test_stateful_private_validation_and_full_replay(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size"):
            _init_inference_state(0, 1)
        with self.assertRaisesRegex(ValueError, "max_length"):
            _init_inference_state(1, 0)

        state = _init_inference_state(1, 1)
        self.assertEqual(_forward_step(state, torch.tensor(0)).shape, (1,))
        with self.assertRaisesRegex(ValueError, "shape"):
            _forward_step(state, torch.tensor([1, 2]))

        empty = torch.empty(0, dtype=torch.long)
        self.assertTrue(torch.equal(predict_exact_stateful(empty), empty))
        empty_batch = torch.empty((2, 0), dtype=torch.long)
        self.assertEqual(tuple(predict_exact_stateful(empty_batch).shape), (2, 0))
        with self.assertRaisesRegex(ValueError, "shape"):
            predict_exact_stateful(torch.zeros(1, 2, 3, dtype=torch.long))

        tokens = torch.tensor([0, 1, 0, 2, 0], dtype=torch.long)
        expected, _, _ = reference_rosa(tokens)
        self.assertTrue(torch.equal(predict_exact_stateful(tokens), expected))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_round_trip_matches_reference(self) -> None:
        tokens = torch.tensor(
            [[0, 1, 0, 2, 0, 1, 0, 3]], dtype=torch.long, device="cuda"
        )
        self.assert_matches_reference(tokens)
