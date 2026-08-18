from __future__ import annotations

import unittest
from functools import cmp_to_key
from itertools import product
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import torch

from rosa import forward_step, init_inference_state, prefill, reference_rosa
from rosa._rlbwt_backend import (
    _SENTINEL,
    _forward_step,
    _init_native_rlbwt_compact_state,
    _init_native_rlbwt_mc_state,
    _init_native_rlbwt_state,
    _init_rlbwt_state,
    _insert_sentinel,
    _prefill,
    _reconstruct_rlbwt,
    _RLBWTRun,
    _sentinel_location,
    _update_pa_lcs,
    reconstruct_rlbwt,
)


def _compare_reversed_prefixes(tokens: list[int], left: int, right: int) -> int:
    left_index = left - 1
    right_index = right - 1
    while left_index >= 0 and right_index >= 0:
        if tokens[left_index] != tokens[right_index]:
            return -1 if tokens[left_index] < tokens[right_index] else 1
        left_index -= 1
        right_index -= 1
    if left_index == right_index:
        return 0
    return -1 if left_index < right_index else 1


def _common_suffix(tokens: list[int], left: int, right: int) -> int:
    length = 0
    while (
        length < left
        and length < right
        and tokens[left - length - 1] == tokens[right - length - 1]
    ):
        length += 1
    return length


def _naive_index(tokens: list[int]) -> tuple[list[int], list[int], list[Any]]:
    pa = sorted(
        range(len(tokens) + 1),
        key=cmp_to_key(
            lambda left, right: _compare_reversed_prefixes(tokens, left, right)
        ),
    )
    lcs = [0]
    lcs.extend(
        _common_suffix(tokens, pa[index - 1], pa[index]) for index in range(1, len(pa))
    )
    bwt = [tokens[endpoint] if endpoint < len(tokens) else _SENTINEL for endpoint in pa]
    return pa, lcs, bwt


class TestRLBWTBackend(unittest.TestCase):
    def assert_index(self, state: Any, tokens: list[int]) -> None:
        pa, lcs, bwt = _naive_index(tokens)
        self.assertEqual(state.pa[0][: len(pa)], pa)
        self.assertEqual(state.lcs[0][: len(lcs)], lcs)
        self.assertEqual(_reconstruct_rlbwt(state), bwt)
        self.assertEqual(reconstruct_rlbwt(state), bwt)

    def test_incremental_index_matches_naive_oracle(self) -> None:
        sequences = [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 1, 2, 0, 1],
            [5, -2, 5, -2, 4, 5],
            [-(2**63), 2**63 - 1, -(2**63), 0],
        ]
        for sequence in sequences:
            with self.subTest(sequence=sequence):
                state = _init_rlbwt_state(1, len(sequence))
                for position, token in enumerate(sequence):
                    _forward_step(state, torch.tensor([token]))
                    self.assert_index(state, sequence[: position + 1])

    def test_exhaustive_binary_matches_reference(self) -> None:
        tokens = torch.tensor(list(product(range(2), repeat=10)), dtype=torch.long)
        state = init_inference_state(1024, 10, backend="rlbwt")
        actual = prefill(state, tokens)
        expected, sources, lengths = reference_rosa(tokens)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(state.backend, "rlbwt")
        self.assertEqual(state.position, 10)
        self.assertTrue(torch.equal(torch.tensor(state._impl.sources), sources[:, -1]))
        self.assertTrue(
            torch.equal(torch.tensor(state._impl.lrs_lengths), lengths[:, -1])
        )

    def test_random_batch_prefill_and_continuation(self) -> None:
        generator = torch.Generator().manual_seed(20260811)
        tokens = torch.randint(17, (8, 96), generator=generator)
        state = init_inference_state(8, 96, backend="rlbwt")
        initial = prefill(state, tokens[:, :64])
        continuation = torch.stack(
            [forward_step(state, tokens[:, position]) for position in range(64, 96)],
            dim=1,
        )
        expected, _, _ = reference_rosa(tokens)
        self.assertTrue(
            torch.equal(torch.cat((initial, continuation), dim=1), expected)
        )
        self.assertEqual(state.positions.tolist(), [96] * 8)

        state.reset()
        self.assertEqual(state.position, 0)
        self.assertTrue(torch.equal(prefill(state, tokens), expected))

    def test_one_hundred_random_rows_match_all_reference_fields(self) -> None:
        generator = torch.Generator().manual_seed(20260811)
        for case_index in range(100):
            length = 1 + case_index % 32
            alphabet = 1 + case_index % 19
            tokens = torch.randint(alphabet, (1, length), generator=generator)
            expected, sources, match_lengths = reference_rosa(tokens)
            state = _init_rlbwt_state(1, length)
            actual_steps: list[torch.Tensor] = []
            for position in range(length):
                actual_steps.append(_forward_step(state, tokens[:, position]))
                self.assertEqual(state.sources[0], int(sources[0, position]))
                self.assertEqual(state.lrs_lengths[0], int(match_lengths[0, position]))
            actual = torch.stack(actual_steps, dim=1)
            with self.subTest(case=case_index):
                self.assertTrue(torch.equal(actual, expected))

    def test_scalar_empty_and_validation(self) -> None:
        state = init_inference_state(1, 3, backend="rlbwt")
        self.assertEqual(forward_step(state, torch.tensor(7)).item(), -1)
        self.assertEqual(forward_step(state, torch.tensor(7)).item(), 7)
        self.assertEqual(forward_step(state, torch.tensor(7)).item(), 7)
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            forward_step(state, torch.tensor(7))

        empty_state = init_inference_state(1, 1, backend="rlbwt")
        empty = prefill(empty_state, torch.empty(0, dtype=torch.long))
        self.assertEqual(tuple(empty.shape), (0,))
        with self.assertRaisesRegex(RuntimeError, "empty"):
            prefill(state, torch.tensor([], dtype=torch.long))

        with self.assertRaisesRegex(ValueError, "rich"):
            init_inference_state(1, 4, backend="rlbwt", mode="rich")
        with self.assertRaisesRegex(ValueError, "ragged"):
            init_inference_state(1, 4, backend="rlbwt", ragged=True)

    def test_native_capability_errors(self) -> None:
        with patch.dict("sys.modules", {"rosa_native_step": None}):
            with self.assertRaisesRegex(ImportError, "compatible"):
                _init_native_rlbwt_state(1, 4)
        incompatible = SimpleNamespace(rlbwt_abi_version=0)
        with patch.dict("sys.modules", {"rosa_native_step": incompatible}):
            with self.assertRaisesRegex(ImportError, "ABI 1"):
                _init_native_rlbwt_state(1, 4)
        missing_class = SimpleNamespace(rlbwt_abi_version=1)
        with patch.dict("sys.modules", {"rosa_native_step": missing_class}):
            with self.assertRaisesRegex(ImportError, "ABI 1"):
                _init_native_rlbwt_state(1, 4)

        real_import = __import__

        def unexpected_missing(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "rosa_native_step":
                raise ModuleNotFoundError("unexpected", name="unexpected")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=unexpected_missing):
            with self.assertRaises(ModuleNotFoundError):
                _init_native_rlbwt_state(1, 4)
            with self.assertRaises(ModuleNotFoundError):
                _init_native_rlbwt_mc_state(1, 4, 2)
            with self.assertRaises(ModuleNotFoundError):
                _init_native_rlbwt_compact_state(1, 4)

        with patch.dict("sys.modules", {"rosa_native_step": None}):
            with self.assertRaisesRegex(ImportError, "Monte-Carlo"):
                _init_native_rlbwt_mc_state(1, 4, 2)
        for capability in (
            SimpleNamespace(rlbwt_mc_abi_version=0),
            SimpleNamespace(rlbwt_mc_abi_version=1),
        ):
            with patch.dict("sys.modules", {"rosa_native_step": capability}):
                with self.assertRaisesRegex(ImportError, "MC ABI 1"):
                    _init_native_rlbwt_mc_state(1, 4, 2)

    def test_native_dispatch_with_capability_stub(self) -> None:
        class StubNativeRLBWTState:
            def __init__(self, batch_size: int, max_length: int) -> None:
                self.batch_size = batch_size
                self.max_length = max_length
                self.position = 0

            def step(self, tokens: np.ndarray) -> np.ndarray:
                self.position += 1
                return tokens.copy()

            def prefill(self, tokens: np.ndarray) -> np.ndarray:
                self.position = tokens.shape[1]
                return tokens.copy()

        capability = SimpleNamespace(
            rlbwt_abi_version=1, NativeRLBWTState=StubNativeRLBWTState
        )
        with patch.dict("sys.modules", {"rosa_native_step": capability}):
            state = init_inference_state(2, 3, backend="rlbwt_native")
            tokens = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
            self.assertTrue(torch.equal(prefill(state, tokens), tokens))
            self.assertEqual(state.position, 2)
            state.reset()
            self.assertTrue(
                torch.equal(
                    forward_step(state, torch.tensor([7, 9])), torch.tensor([7, 9])
                )
            )
            empty = init_inference_state(1, 1, backend="rlbwt_native")
            self.assertEqual(
                tuple(prefill(empty, torch.empty(0, dtype=torch.long)).shape), (0,)
            )

    def test_native_backend_matches_python_oracle_when_available(self) -> None:
        try:
            import rosa_native_step
        except ModuleNotFoundError:
            self.skipTest("native companion is unavailable")
        if getattr(rosa_native_step, "rlbwt_abi_version", None) != 1:
            self.skipTest("native companion lacks RLBWT ABI 1")

        tokens = torch.tensor(
            [[0, 1, 0, 1, 2, -1, 2**31, -1], [7] * 8], dtype=torch.long
        )
        expected, sources, lengths = reference_rosa(tokens)
        state = init_inference_state(2, 8, backend="rlbwt_native")
        self.assertTrue(torch.equal(prefill(state, tokens), expected))
        self.assertEqual(state.positions.tolist(), [8, 8])
        self.assertEqual(state._impl.sources.tolist(), sources[:, -1].tolist())
        self.assertEqual(state._impl.lrs_lengths.tolist(), lengths[:, -1].tolist())
        state.reset()
        self.assertEqual(forward_step(state, tokens[:, 0]).tolist(), [-1, -1])

        empty = init_inference_state(1, 1, backend="rlbwt_native")
        self.assertEqual(
            tuple(prefill(empty, torch.empty(0, dtype=torch.long)).shape), (0,)
        )
        with self.assertRaisesRegex(ValueError, "rich"):
            init_inference_state(1, 4, backend="rlbwt_native", mode="rich")
        with self.assertRaisesRegex(ValueError, "ragged"):
            init_inference_state(1, 4, backend="rlbwt_native", ragged=True)

    def test_native_compact_backend_when_available(self) -> None:
        try:
            import rosa_native_step
        except ModuleNotFoundError:
            self.skipTest("native companion is unavailable")
        if getattr(rosa_native_step, "rlbwt_compact_abi_version", None) != 1:
            self.skipTest("native companion lacks compact RLBWT ABI 1")

        tokens = torch.tensor(
            [[0, 15, 0, 15, 7, 255, 7, 0], [255, 1, 255, 2, 255, 3, 4, 5]],
            dtype=torch.long,
        )
        expected, sources, lengths = reference_rosa(tokens)
        state = init_inference_state(2, 8, backend="rlbwt_compact256")
        self.assertTrue(torch.equal(prefill(state, tokens), expected))
        self.assertEqual(state.positions.tolist(), [8, 8])
        self.assertEqual(state._impl.sources.tolist(), sources[:, -1].tolist())
        self.assertEqual(state._impl.lrs_lengths.tolist(), lengths[:, -1].tolist())
        self.assertEqual(state._impl.vocabulary_size, 256)
        state.reset()
        self.assertEqual(forward_step(state, tokens[:, 0]).tolist(), [-1, -1])

        for invalid in (-1, 256):
            with self.subTest(invalid=invalid):
                rejected = init_inference_state(1, 1, backend="rlbwt_compact256")
                with self.assertRaisesRegex(ValueError, r"\[0, 255\]"):
                    forward_step(rejected, torch.tensor([invalid]))
                with self.assertRaisesRegex(ValueError, r"\[0, 255\]"):
                    prefill(rejected, torch.tensor([[invalid]]))

        with self.assertRaisesRegex(ValueError, "rich"):
            init_inference_state(1, 4, backend="rlbwt_compact256", mode="rich")
        with self.assertRaisesRegex(ValueError, "ragged"):
            init_inference_state(1, 4, backend="rlbwt_compact256", ragged=True)

    def test_compact_native_import_validation(self) -> None:
        with patch.dict("sys.modules", {"rosa_native_step": None}):
            with self.assertRaisesRegex(ImportError, "compact RLBWT"):
                _init_native_rlbwt_compact_state(1, 4)
        for capability in (
            SimpleNamespace(rlbwt_compact_abi_version=0),
            SimpleNamespace(rlbwt_compact_abi_version=1),
        ):
            with patch.dict("sys.modules", {"rosa_native_step": capability}):
                with self.assertRaisesRegex(ImportError, "compact RLBWT ABI 1"):
                    _init_native_rlbwt_compact_state(1, 4)

    def test_native_mc_backends_when_available(self) -> None:
        try:
            import rosa_native_step
        except ModuleNotFoundError:
            self.skipTest("native companion is unavailable")
        if getattr(rosa_native_step, "rlbwt_mc_abi_version", None) != 1:
            self.skipTest("native companion lacks RLBWT MC ABI 1")

        generator = torch.Generator().manual_seed(20260811)
        workloads = (
            torch.tensor(list(product(range(2), repeat=8)), dtype=torch.long),
            torch.randint(11, (4, 192), generator=generator),
            torch.ones((2, 192), dtype=torch.long),
            torch.tensor([[index % 5 for index in range(192)]], dtype=torch.long),
        )
        for backend, lanes in (("rlbwt_mc128", 2), ("rlbwt_mc192", 3)):
            for tokens in workloads:
                with self.subTest(backend=backend, shape=tuple(tokens.shape)):
                    expected, _, _ = reference_rosa(tokens)
                    state = init_inference_state(
                        tokens.shape[0], tokens.shape[1], backend=backend
                    )
                    split = min(96, tokens.shape[1])
                    initial = prefill(state, tokens[:, :split])
                    continuation = (
                        torch.stack(
                            [
                                forward_step(state, tokens[:, position])
                                for position in range(split, tokens.shape[1])
                            ],
                            dim=1,
                        )
                        if split < tokens.shape[1]
                        else tokens[:, :0]
                    )
                    self.assertTrue(
                        torch.equal(torch.cat((initial, continuation), dim=1), expected)
                    )
                    self.assertEqual(state._impl.lanes, lanes)
                    self.assertEqual(state._impl.seed, 20260811)
                    with self.assertRaisesRegex(RuntimeError, "capacity"):
                        forward_step(state, tokens[:, 0])
                    state.reset()
                    self.assertEqual(state.position, 0)
                    self.assertEqual(state._impl.lanes, lanes)
                    self.assertEqual(
                        forward_step(state, tokens[:, 0]).shape[0], tokens.shape[0]
                    )

            with self.assertRaisesRegex(ValueError, "rich"):
                init_inference_state(1, 4, backend=backend, mode="rich")
            with self.assertRaisesRegex(ValueError, "ragged"):
                init_inference_state(1, 4, backend=backend, ragged=True)

    def test_private_validation_and_corrupt_sentinels(self) -> None:
        for batch_size, max_length, message in (
            (0, 1, "batch_size"),
            (1, 0, "max_length"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    _init_rlbwt_state(batch_size, max_length)

        state = _init_rlbwt_state(2, 2)
        self.assertEqual(repr(_SENTINEL), "$")
        self.assertEqual(state.history, [[0, 0], [0, 0]])
        with self.assertRaisesRegex(TypeError, "Tensor"):
            _forward_step(state, cast(Any, [1, 2]))
        with self.assertRaisesRegex(ValueError, "shape"):
            _forward_step(state, torch.tensor([1]))
        with self.assertRaisesRegex(TypeError, "Tensor"):
            _prefill(state, cast(Any, [[1], [2]]))
        with self.assertRaisesRegex(ValueError, "shape"):
            _prefill(state, torch.tensor([[1, 2]]))
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            _prefill(state, torch.zeros((2, 3), dtype=torch.long))
        _prefill(state, torch.zeros((2, 1), dtype=torch.long))
        with self.assertRaisesRegex(RuntimeError, "empty"):
            _prefill(state, torch.zeros((2, 1), dtype=torch.long))
        with self.assertRaisesRegex(IndexError, "batch_index"):
            _reconstruct_rlbwt(state, 2)
        with self.assertRaisesRegex(IndexError, "batch_index"):
            _reconstruct_rlbwt(state, -1)

        with self.assertRaisesRegex(RuntimeError, "missing"):
            _sentinel_location([_RLBWTRun(1)])
        with self.assertRaisesRegex(RuntimeError, "invalid"):
            _sentinel_location(
                [_RLBWTRun(is_sentinel=True), _RLBWTRun(is_sentinel=True)]
            )
        with self.assertRaisesRegex(RuntimeError, "invalid"):
            _sentinel_location([_RLBWTRun(length=2, is_sentinel=True)])
        with self.assertRaisesRegex(RuntimeError, "out of range"):
            _insert_sentinel([_RLBWTRun(1)], -1)
        with self.assertRaisesRegex(RuntimeError, "out of range"):
            _insert_sentinel([_RLBWTRun(1)], 2)
        with self.assertRaisesRegex(RuntimeError, "cannot split"):
            _insert_sentinel([_RLBWTRun(length=2, is_sentinel=True)], 1)
        with self.assertRaisesRegex(RuntimeError, "PA insertion"):
            _update_pa_lcs(state.rows[0], 0, -1)

        scalar_state = _init_rlbwt_state(1, 1)
        self.assertEqual(_forward_step(scalar_state, torch.tensor(3)).item(), -1)


if __name__ == "__main__":
    unittest.main()
