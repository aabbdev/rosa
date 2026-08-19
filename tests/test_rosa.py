from __future__ import annotations

import copy
import gc
import random
import threading
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

import rosa
from rosa import (
    NULL_KIND,
    ROSA,
    VIRTUAL_KIND,
    PreparedHardCandidates,
    _balance_kl,
    _build_forward_hard_candidates,
    _build_forward_hard_candidates_selected,
    _clear_soft_match_compile_cache,
    _gather_sequence,
    _soft_match,
    _soft_match_signature,
    _soft_match_torch,
    _st_categorical,
    _virtual_pool_single,
    build_hard_candidates,
    build_virtual_pool_indices,
    reference_rosa,
)


def factor_logits_from_tokens(
    tokens: torch.Tensor,
    codebook_sizes: tuple[int, int],
    hi: float = 20.0,
    lo: float = -20.0,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    c1, c2 = codebook_sizes
    id1 = tokens // c2
    id2 = tokens % c2
    l1 = torch.full((*tokens.shape, c1), lo, dtype=torch.float32)
    l2 = torch.full((*tokens.shape, c2), lo, dtype=torch.float32)
    l1.scatter_(-1, id1.unsqueeze(-1), hi)
    l2.scatter_(-1, id2.unsqueeze(-1), hi)
    return l1.requires_grad_(requires_grad), l2.requires_grad_(requires_grad)


def zero_learned_scorer(model: ROSA) -> None:
    with torch.no_grad():
        model.selector_query.weight.zero_()
        model.selector_key.weight.zero_()
        model.virtual_query.weight.zero_()
        model.virtual_key.weight.zero_()
        for layer in model.feature_mlp:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.zero_()
                layer.bias.zero_()
        model.kind_bias.zero_()
        model.null_head.weight.zero_()
        model.null_head.bias.zero_()


class _GatherFirstROSA(ROSA):
    """Local pre-fusion oracle retaining candidate-first projection order."""

    def _virtual_pool_keys(self, z_a: torch.Tensor, pool: torch.Tensor) -> torch.Tensor:
        return self.virtual_key(_gather_sequence(z_a, pool))

    def _candidate_selector_keys(
        self, z_a: torch.Tensor, source: torch.Tensor
    ) -> torch.Tensor:
        return self.selector_key(_gather_sequence(z_a, source))

    def _candidate_symbolic_values(
        self,
        st1: torch.Tensor,
        st2: torch.Tensor,
        next_position: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        g1 = _gather_sequence(st1, next_position)
        g2 = _gather_sequence(st2, next_position)
        e1 = g1 @ self.symbol_embedding_1.weight
        e2 = g2 @ self.symbol_embedding_2.weight
        return (e1 + e2) * mask.unsqueeze(-1).to(e1.dtype)

    def _candidate_neural_values(
        self, z_a: torch.Tensor, next_position: torch.Tensor
    ) -> torch.Tensor:
        return self.value_proj(_gather_sequence(z_a, next_position))


class TestReferenceROSA(unittest.TestCase):
    def test_reference_squeeze_batch_and_validation(self) -> None:
        one_d = torch.tensor([0, 1, 0, 2], dtype=torch.long)
        pred, src, length = reference_rosa(one_d)
        self.assertEqual(tuple(pred.shape), (4,))
        self.assertEqual(tuple(src.shape), (4,))
        self.assertEqual(tuple(length.shape), (4,))

        batched = one_d.unsqueeze(0).repeat(2, 1)
        pred_b, src_b, length_b = reference_rosa(batched)
        self.assertEqual(tuple(pred_b.shape), (2, 4))
        self.assertTrue(torch.equal(pred_b[0], pred))
        self.assertTrue(torch.equal(src_b[0], src))
        self.assertTrue(torch.equal(length_b[0], length))

        with self.assertRaisesRegex(ValueError, "shape"):
            reference_rosa(torch.zeros(1, 2, 3, dtype=torch.long))


class TestHardCandidateGenerator(unittest.TestCase):
    def test_exact_rosa_matches_bruteforce_random_and_exercises_clones(self) -> None:
        rng = random.Random(2026)
        for n in range(1, 28):
            for _ in range(12):
                seq = torch.tensor(
                    [[rng.randrange(5) for _ in range(n)]], dtype=torch.long
                )
                got = build_hard_candidates(seq, suffix_k=1, occurrences_r=1)
                ref_pred, ref_src, ref_len = reference_rosa(seq)
                self.assertTrue(torch.equal(got.rosa_predicted_tokens, ref_pred))
                self.assertTrue(torch.equal(got.rosa_source_index, ref_src))
                self.assertTrue(torch.equal(got.rosa_match_length, ref_len))
                valid = got.rosa_slot >= 0
                self.assertTrue(torch.equal(valid, ref_src >= 0))
                if valid.any():
                    self.assertTrue(torch.all(got.rosa_slot[valid] == 0))

    def test_multi_occurrence_cache_newest_first(self) -> None:
        seq = torch.tensor([[0, 1, 0, 2, 0, 3, 0]], dtype=torch.long)
        got = build_hard_candidates(seq, suffix_k=4, occurrences_r=3)
        self.assertEqual(got.source_index[0, -1, :3].tolist(), [4, 2, 0])
        self.assertEqual(got.match_length[0, -1, :3].tolist(), [1, 1, 1])
        self.assertEqual(got.frequency[0, -1, :3].tolist(), [3, 3, 3])
        self.assertTrue(got.mask[0, -1, :3].all())
        self.assertFalse(got.mask[0, -1, 3:].any())

    def test_squeeze_and_validation(self) -> None:
        one_d = torch.tensor([0, 1, 0], dtype=torch.long)
        got = build_hard_candidates(one_d, suffix_k=2, occurrences_r=2)
        self.assertEqual(got.source_index.ndim, 2)
        with self.assertRaisesRegex(ValueError, "suffix_k"):
            build_hard_candidates(one_d, suffix_k=0)
        with self.assertRaisesRegex(ValueError, "occurrences_r"):
            build_hard_candidates(one_d, occurrences_r=0)
        with self.assertRaisesRegex(ValueError, "shape"):
            build_hard_candidates(torch.zeros(1, 2, 3, dtype=torch.long))


class TestHelpers(unittest.TestCase):
    def test_virtual_pool_all_branches_and_causality(self) -> None:
        self.assertEqual(_virtual_pool_single(0, 4), [])
        self.assertEqual(_virtual_pool_single(2, 2), [0, 1])  # remaining == 1
        pool = _virtual_pool_single(10, 6)  # remaining > 1
        self.assertIn(0, pool)
        self.assertEqual(len(pool), len(set(pool)))
        self.assertTrue(all(0 <= j < 10 for j in pool))

        idx = build_virtual_pool_indices(2, 8, 6, torch.device("cpu"))
        self.assertEqual(tuple(idx.shape), (2, 8, 6))
        for i in range(8):
            valid = idx[:, i][idx[:, i] >= 0]
            if valid.numel():
                self.assertTrue(torch.all(valid < i))
        with self.assertRaisesRegex(ValueError, "batch_size"):
            build_virtual_pool_indices(0, 2, 2, torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "sequence_length"):
            build_virtual_pool_indices(1, 0, 2, torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "pool_size"):
            build_virtual_pool_indices(1, 2, 0, torch.device("cpu"))

    def test_gather_st_and_balance_helpers(self) -> None:
        x = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        index = torch.tensor([[[0, 3], [1, -1]], [[2, 1], [3, 0]]])
        got = _gather_sequence(x, index)
        self.assertEqual(tuple(got.shape), (2, 2, 2, 3))
        self.assertTrue(torch.equal(got[0, 0, 1], x[0, 3]))
        # -1 is safely clamped to zero; caller masks it afterwards.
        self.assertTrue(torch.equal(got[0, 1, 1], x[0, 0]))
        with self.assertRaisesRegex(ValueError, "x must"):
            _gather_sequence(torch.zeros(2, 3), torch.zeros(2, 1, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "batch"):
            _gather_sequence(x, torch.zeros(1, 2, dtype=torch.long))

        logits = torch.tensor([[[1.0, 3.0]]], requires_grad=True)
        soft, st, ids = _st_categorical(logits, 0.5)
        self.assertEqual(ids.item(), 1)
        self.assertTrue(
            torch.allclose(st.detach(), F.one_hot(ids, 2).float(), atol=1e-7, rtol=0.0)
        )
        st[..., 0].sum().backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)
        with self.assertRaisesRegex(ValueError, "temperature"):
            _st_categorical(logits.detach(), 0.0)

        uniform = torch.full((2, 3, 4), 0.25)
        peaked = F.one_hot(torch.zeros(2, 3, dtype=torch.long), 4).float()
        self.assertAlmostEqual(float(_balance_kl(uniform)), 0.0, places=6)
        self.assertGreater(float(_balance_kl(peaked)), 1.0)

    def test_compiled_soft_match_matches_eager_across_specializations(self) -> None:
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))

        for device in devices:
            with self.subTest(device=device.type):
                _clear_soft_match_compile_cache()
                cached_callable = None
                for shape in ((2, 7, 5), (1, 11, 3)):
                    batch, length, candidates = shape
                    torch.manual_seed(20260811 + length)
                    actual_st1 = torch.softmax(
                        torch.randn(batch, length, 4, device=device), dim=-1
                    ).requires_grad_()
                    actual_st2 = torch.softmax(
                        torch.randn(batch, length, 3, device=device), dim=-1
                    ).requires_grad_()
                    expected_st1 = actual_st1.detach().clone().requires_grad_()
                    expected_st2 = actual_st2.detach().clone().requires_grad_()
                    source = torch.randint(
                        -1, length, (batch, length, candidates), device=device
                    )
                    mask = torch.rand(batch, length, candidates, device=device) > 0.2

                    actual = _soft_match(actual_st1, actual_st2, source, mask, window=4)
                    expected = _soft_match_torch(
                        expected_st1, expected_st2, source, mask, window=4
                    )
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                    actual.square().sum().backward()
                    expected.square().sum().backward()
                    assert actual_st1.grad is not None
                    assert actual_st2.grad is not None
                    assert expected_st1.grad is not None
                    assert expected_st2.grad is not None
                    torch.testing.assert_close(
                        actual_st1.grad, expected_st1.grad, rtol=2e-5, atol=2e-6
                    )
                    torch.testing.assert_close(
                        actual_st2.grad, expected_st2.grad, rtol=2e-5, atol=2e-6
                    )

                    signature = _soft_match_signature(
                        actual_st1, actual_st2, source, mask, 4
                    )
                    if signature not in rosa._SOFT_MATCH_COMPILE_FAILURES:
                        self.assertEqual(len(rosa._SOFT_MATCH_COMPILED), 1)
                        self.assertIn(signature, rosa._SOFT_MATCH_COMPILE_READY)
                        if cached_callable is None:
                            cached_callable = rosa._SOFT_MATCH_COMPILED[4]
                        else:
                            self.assertIs(cached_callable, rosa._SOFT_MATCH_COMPILED[4])

    def test_soft_match_compile_failure_falls_back_before_caching(self) -> None:
        _clear_soft_match_compile_cache()
        st1 = torch.softmax(torch.randn(1, 5, 3), dim=-1)
        st2 = torch.softmax(torch.randn(1, 5, 2), dim=-1)
        source = torch.randint(-1, 5, (1, 5, 4))
        mask = source >= 0
        expected = _soft_match_torch(st1, st2, source, mask, window=3)
        with patch("rosa.torch.compile", side_effect=RuntimeError("unavailable")):
            actual = _soft_match(st1, st2, source, mask, window=3)
        signature = _soft_match_signature(st1, st2, source, mask, 3)
        self.assertTrue(torch.equal(actual, expected))
        self.assertNotIn(3, rosa._SOFT_MATCH_COMPILED)
        self.assertIn(signature, rosa._SOFT_MATCH_COMPILE_FAILURES)
        _clear_soft_match_compile_cache()

    def test_soft_match_disabled_and_ready_failure_fall_back(self) -> None:
        _clear_soft_match_compile_cache()
        st1 = torch.softmax(torch.randn(1, 5, 3), dim=-1)
        st2 = torch.softmax(torch.randn(1, 5, 2), dim=-1)
        source = torch.randint(-1, 5, (1, 5, 4))
        mask = source >= 0
        expected = _soft_match_torch(st1, st2, source, mask, window=3)
        with patch("rosa._SOFT_MATCH_COMPILE_ENABLED", False):
            disabled = _soft_match(st1, st2, source, mask, window=3)
        self.assertTrue(torch.equal(disabled, expected))

        signature = _soft_match_signature(st1, st2, source, mask, 3)

        def succeed(*args: torch.Tensor) -> torch.Tensor:
            return _soft_match_torch(*args, window=3)

        rosa._SOFT_MATCH_COMPILED[3] = succeed
        rosa._SOFT_MATCH_COMPILE_READY.add(signature)
        cached = _soft_match(st1, st2, source, mask, window=3)
        self.assertTrue(torch.equal(cached, expected))

        def fail(*args: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("cached specialization failed")

        rosa._SOFT_MATCH_COMPILED[3] = fail
        rosa._SOFT_MATCH_COMPILE_READY.add(signature)
        actual = _soft_match(st1, st2, source, mask, window=3)
        self.assertTrue(torch.equal(actual, expected))
        self.assertIn(signature, rosa._SOFT_MATCH_COMPILE_FAILURES)
        _clear_soft_match_compile_cache()

    def test_soft_match_compile_initialization_is_thread_safe(self) -> None:
        _clear_soft_match_compile_cache()
        st1 = torch.softmax(torch.randn(1, 5, 3), dim=-1)
        st2 = torch.softmax(torch.randn(1, 5, 2), dim=-1)
        source = torch.randint(-1, 5, (1, 5, 4))
        mask = source >= 0
        start = threading.Barrier(2)
        compile_calls = 0
        forward_calls = 0
        calls_lock = threading.Lock()

        def compiled(*args: torch.Tensor) -> torch.Tensor:
            nonlocal forward_calls
            with calls_lock:
                forward_calls += 1
            return _soft_match_torch(*args, window=3)

        def compile_once(*args: object, **kwargs: object) -> object:
            nonlocal compile_calls
            with calls_lock:
                compile_calls += 1
            return compiled

        def invoke() -> torch.Tensor:
            start.wait()
            return _soft_match(st1, st2, source, mask, window=3)

        with (
            patch("rosa.torch.compile", side_effect=compile_once),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            results = [
                future.result()
                for future in (executor.submit(invoke), executor.submit(invoke))
            ]

        self.assertEqual(compile_calls, 1)
        self.assertEqual(forward_calls, 2)
        self.assertTrue(torch.equal(results[0], results[1]))
        signature = _soft_match_signature(st1, st2, source, mask, 3)
        self.assertIn(signature, rosa._SOFT_MATCH_COMPILE_READY)

    def test_soft_match_waiter_reuses_newly_ready_specialization(self) -> None:
        _clear_soft_match_compile_cache()
        st1 = torch.softmax(torch.randn(1, 5, 3), dim=-1)
        st2 = torch.softmax(torch.randn(1, 5, 2), dim=-1)
        source = torch.randint(-1, 5, (1, 5, 4))
        mask = source >= 0
        signature = _soft_match_signature(st1, st2, source, mask, 3)
        compiled_started = threading.Event()
        second_waiting = threading.Event()
        release_compiled = threading.Event()

        class SignalingLock:
            def __init__(self) -> None:
                self._lock = threading.Lock()
                self._count_lock = threading.Lock()
                self._entries = 0

            def __enter__(self) -> None:
                with self._count_lock:
                    self._entries += 1
                    if self._entries == 2:
                        second_waiting.set()
                self._lock.acquire()

            def __exit__(self, *args: object) -> None:
                self._lock.release()

        rosa._SOFT_MATCH_SIGNATURE_LOCKS[signature] = SignalingLock()

        def compiled(*args: torch.Tensor) -> torch.Tensor:
            compiled_started.set()
            if not release_compiled.wait(timeout=5):
                raise RuntimeError("timed out waiting for concurrent caller")
            return _soft_match_torch(*args, window=3)

        with (
            patch("rosa.torch.compile", return_value=compiled) as compile_mock,
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            first = executor.submit(_soft_match, st1, st2, source, mask, 3)
            self.assertTrue(compiled_started.wait(timeout=5))
            second = executor.submit(_soft_match, st1, st2, source, mask, 3)
            try:
                self.assertTrue(second_waiting.wait(timeout=5))
            finally:
                release_compiled.set()
            first_result = first.result()
            second_result = second.result()

        self.assertEqual(compile_mock.call_count, 1)
        self.assertTrue(torch.equal(first_result, second_result))

    def test_soft_match_failure_does_not_poison_other_signature(self) -> None:
        _clear_soft_match_compile_cache()
        calls: list[tuple[int, ...]] = []

        def compiled(
            st1: torch.Tensor,
            st2: torch.Tensor,
            source: torch.Tensor,
            mask: torch.Tensor,
        ) -> torch.Tensor:
            calls.append(tuple(source.shape))
            if source.shape[1] == 5:
                raise RuntimeError("unsupported shape")
            return _soft_match_torch(st1, st2, source, mask, window=3)

        def inputs(length: int) -> tuple[torch.Tensor, ...]:
            st1 = torch.softmax(torch.randn(1, length, 3), dim=-1)
            st2 = torch.softmax(torch.randn(1, length, 2), dim=-1)
            source = torch.randint(-1, length, (1, length, 4))
            return st1, st2, source, source >= 0

        failing = inputs(5)
        working = inputs(7)
        with patch("rosa.torch.compile", return_value=compiled) as compile_mock:
            failed_result = _soft_match(*failing, window=3)
            working_result = _soft_match(*working, window=3)
            retried_result = _soft_match(*failing, window=3)

        self.assertEqual(compile_mock.call_count, 1)
        self.assertEqual(calls, [(1, 5, 4), (1, 7, 4)])
        self.assertTrue(
            torch.equal(failed_result, _soft_match_torch(*failing, window=3))
        )
        self.assertTrue(
            torch.equal(retried_result, _soft_match_torch(*failing, window=3))
        )
        self.assertTrue(
            torch.equal(working_result, _soft_match_torch(*working, window=3))
        )
        self.assertIn(3, rosa._SOFT_MATCH_COMPILED)
        self.assertIn(
            _soft_match_signature(*failing, window=3),
            rosa._SOFT_MATCH_COMPILE_FAILURES,
        )
        self.assertIn(
            _soft_match_signature(*working, window=3),
            rosa._SOFT_MATCH_COMPILE_READY,
        )

    def test_soft_match_signature_distinguishes_device(self) -> None:
        cpu = torch.empty(1, 2, 3)
        cpu_source = torch.empty(1, 2, 4, dtype=torch.long)
        meta = torch.empty(1, 2, 3, device="meta")
        meta_source = torch.empty(1, 2, 4, dtype=torch.long, device="meta")
        cpu_signature = _soft_match_signature(cpu, cpu, cpu_source, cpu_source >= 0, 3)
        meta_signature = _soft_match_signature(
            meta, meta, meta_source, meta_source >= 0, 3
        )
        self.assertNotEqual(cpu_signature, meta_signature)

    def test_soft_match_signature_distinguishes_layout_and_grad_mode(self) -> None:
        st1 = torch.randn(1, 5, 3, requires_grad=True)
        st2 = torch.randn(1, 5, 2, requires_grad=True)
        source = torch.zeros(1, 5, 4, dtype=torch.long)
        mask = torch.ones_like(source, dtype=torch.bool)
        baseline = _soft_match_signature(st1, st2, source, mask, 3)
        noncontiguous_st2 = torch.randn(1, 2, 5).transpose(1, 2).requires_grad_()
        self.assertNotEqual(
            baseline,
            _soft_match_signature(st1, noncontiguous_st2, source, mask, 3),
        )
        self.assertNotEqual(
            baseline,
            _soft_match_signature(st1.detach(), st2, source, mask, 3),
        )
        with torch.no_grad():
            no_grad = _soft_match_signature(st1, st2, source, mask, 3)
        self.assertNotEqual(baseline, no_grad)
        with torch.inference_mode():
            inference = _soft_match_signature(st1, st2, source, mask, 3)
        self.assertNotEqual(no_grad, inference)

    def test_soft_match_backward_compile_error_is_propagated(self) -> None:
        _clear_soft_match_compile_cache()
        st1 = torch.randn(1, 3, 2, requires_grad=True)
        st2 = torch.randn(1, 3, 2, requires_grad=True)
        source = torch.zeros(1, 3, 1, dtype=torch.long)
        mask = torch.ones_like(source, dtype=torch.bool)

        class BackwardFailure(torch.autograd.Function):
            @staticmethod
            def forward(ctx: object, value: torch.Tensor) -> torch.Tensor:
                return value.sum()

            @staticmethod
            def backward(ctx: object, grad: torch.Tensor) -> torch.Tensor:
                raise RuntimeError("deferred AOT backward failure")

        def compiled(*args: torch.Tensor) -> torch.Tensor:
            return BackwardFailure.apply(args[0])

        with patch("rosa.torch.compile", return_value=compiled):
            result = _soft_match(st1, st2, source, mask, window=2)

        with self.assertRaisesRegex(RuntimeError, "deferred AOT backward failure"):
            result.backward()

    def test_rosa_soft_match_is_eager_by_default(self) -> None:
        model = ROSA(d_model=4, soft_verify_window=2)
        st1 = torch.softmax(torch.randn(1, 4, 3), dim=-1)
        st2 = torch.softmax(torch.randn(1, 4, 2), dim=-1)
        source = torch.randint(-1, 4, (1, 4, 2))
        mask = source >= 0
        expected = _soft_match_torch(st1, st2, source, mask, window=2)

        with patch("rosa.torch.compile") as compile_mock:
            actual = model._soft_match(st1, st2, source, mask)

        compile_mock.assert_not_called()
        self.assertTrue(torch.equal(actual, expected))


class TestROSAConfiguration(unittest.TestCase):
    def test_constructor_validations(self) -> None:
        invalid_calls = [
            (dict(d_model=0), "d_model"),
            (dict(d_model=4, codebook_sizes=(2,)), "codebook_sizes"),
            (dict(d_model=4, codebook_sizes=(1, 2)), "codebook_sizes"),
            (dict(d_model=4, suffix_k=0), "suffix_k"),
            (dict(d_model=4, occurrences_r=0), "suffix_k"),
            (dict(d_model=4, soft_verify_window=0), "suffix_k"),
            (dict(d_model=4, virtual_candidates=-1), "virtual_pool_size"),
            (
                dict(d_model=4, virtual_candidates=4, virtual_pool_size=3),
                "virtual_pool_size",
            ),
            (dict(d_model=4, dense_recent_candidates=-1), "dense_recent"),
            (dict(d_model=4, sparse_old_candidates=-1), "sparse_old"),
            (
                dict(
                    d_model=4,
                    sparse_old_candidates=2,
                    sparse_old_pool_size=1,
                ),
                "sparse_old_pool_size",
            ),
            (dict(d_model=4, selector_dim=0), "selector_dim"),
            (dict(d_model=4, token_temperature=0), "temperatures"),
            (dict(d_model=4, retrieval_temperature=0), "temperatures"),
            (dict(d_model=4, tie_break_scale=1.0), "tie_break_scale"),
            (dict(d_model=4, virtual_prior_bias=0.2, null_prior_bias=0.1), "require"),
            (dict(d_model=4, null_prior_bias=1.0), "require"),
            (dict(d_model=4, learned_residual_scale=-0.1), "learned_residual_scale"),
            (dict(d_model=4, virtual_scale=1.1), "virtual_scale"),
            (dict(d_model=4, neural_value_scale=2.0), "neural_value_scale"),
            (dict(d_model=4, candidate_backend="invalid"), "candidate_backend"),
        ]
        for kwargs, pattern in invalid_calls:
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaisesRegex(ValueError, pattern),
            ):
                ROSA(**kwargs)
        with self.assertRaisesRegex(TypeError, "soft_candidates_forward"):
            ROSA(d_model=4, soft_candidates_forward=1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "compile_soft_match"):
            ROSA(d_model=4, compile_soft_match=1)  # type: ignore[arg-type]

    def test_setters_property_and_encode_validation(self) -> None:
        model = ROSA(
            d_model=6,
            codebook_sizes=(2, 3),
            suffix_k=2,
            occurrences_r=2,
            soft_verify_window=3,
            virtual_candidates=2,
            virtual_pool_size=4,
            selector_dim=5,
        )
        self.assertEqual(model.vocab_size, 6)
        self.assertFalse(model.compile_soft_match)
        self.assertTrue(ROSA(d_model=4, compile_soft_match=True).compile_soft_match)
        positional = ROSA(
            4,
            (2, 2),
            2,
            2,
            3,
            2,
            4,
            0,
            0,
            4,
            False,
            5,
        )
        self.assertEqual(positional.selector_dim, 5)
        self.assertFalse(positional.compile_soft_match)
        model.set_learned_residual_scale(0.4)
        model.set_virtual_scale(0.5)
        model.set_neural_value_scale(0.6)
        self.assertAlmostEqual(float(model.learned_residual_scale), 0.4, places=6)
        self.assertAlmostEqual(float(model.virtual_scale), 0.5, places=6)
        self.assertAlmostEqual(float(model.neural_value_scale), 0.6, places=6)
        for setter in (
            model.set_learned_residual_scale,
            model.set_virtual_scale,
            model.set_neural_value_scale,
        ):
            with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
                setter(-0.1)

        with self.assertRaisesRegex(ValueError, "z_a"):
            model.encode(torch.zeros(2, 6))
        z = torch.zeros(1, 3, 6)
        with self.assertRaisesRegex(ValueError, "pair"):
            model.encode(z, (torch.zeros(1, 3, 2),))  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "shapes"):
            model.encode(z, (torch.zeros(1, 3, 3), torch.zeros(1, 3, 3)))

        # Exercise the module-owned code heads path.
        soft, st, hard = model.encode(z)
        self.assertEqual(tuple(soft[0].shape), (1, 3, 2))
        self.assertEqual(tuple(st[1].shape), (1, 3, 3))
        self.assertEqual(tuple(hard.shape), (1, 3))


class TestROSASemantics(unittest.TestCase):
    def make_model(self, **overrides) -> ROSA:
        kwargs = dict(
            d_model=8,
            codebook_sizes=(2, 3),
            suffix_k=5,
            occurrences_r=3,
            soft_verify_window=6,
            virtual_candidates=2,
            virtual_pool_size=6,
            selector_dim=8,
            token_temperature=0.2,
            retrieval_temperature=0.7,
            learned_residual_scale=0.0,
            virtual_scale=1.0,
            neural_value_scale=0.0,
        )
        kwargs.update(overrides)
        return ROSA(**kwargs)

    def assert_nested_equal(self, actual, expected, name: str) -> None:
        if isinstance(actual, torch.Tensor):
            self.assertTrue(torch.equal(actual, expected), name)
        elif isinstance(actual, tuple):
            self.assertEqual(len(actual), len(expected), name)
            for index, (actual_item, expected_item) in enumerate(
                zip(actual, expected, strict=True)
            ):
                self.assert_nested_equal(actual_item, expected_item, f"{name}[{index}]")
        elif isinstance(actual, dict):
            self.assertEqual(actual.keys(), expected.keys(), name)
            for key in actual:
                self.assert_nested_equal(actual[key], expected[key], f"{name}.{key}")
        else:
            self.assertEqual(actual, expected, name)

    def test_zero_virtual_candidates_skip_virtual_work_and_have_exact_width(
        self,
    ) -> None:
        model = self.make_model(
            suffix_k=2,
            occurrences_r=3,
            virtual_candidates=0,
            dense_recent_candidates=2,
            sparse_old_candidates=1,
            sparse_old_pool_size=3,
        )
        z = torch.randn(2, 9, 8)
        positions = torch.tensor([[1, 5], [2, 8]])
        virtual_calls = 0

        def virtual_hook(_module, _args, _output) -> None:
            nonlocal virtual_calls
            virtual_calls += 1

        hooks = [
            model.virtual_query.register_forward_hook(virtual_hook),
            model.virtual_key.register_forward_hook(virtual_hook),
        ]
        try:
            with (
                patch.object(
                    model,
                    "_virtual_candidates",
                    side_effect=AssertionError("unexpected virtual candidates"),
                ),
                patch(
                    "rosa.build_virtual_pool_indices",
                    side_effect=AssertionError("unexpected virtual pool"),
                ),
                patch(
                    "rosa.torch.topk",
                    side_effect=AssertionError("unexpected virtual topk"),
                ),
            ):
                full = model(z)
                query_only = model(z, query_positions=positions)

                for output_name, query in (
                    ("retrieved", None),
                    ("updated", None),
                    ("retrieved", positions),
                    ("updated", positions),
                ):
                    model.zero_grad(set_to_none=True)
                    output = model(z, query_positions=query)
                    getattr(output, output_name).sum().backward()
                    for parameter in (
                        model.virtual_query.weight,
                        model.virtual_key.weight,
                    ):
                        self.assertIsNotNone(parameter.grad)
                        assert parameter.grad is not None
                        self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0)

                with torch.no_grad():
                    no_grad_full = model(z)
                    no_grad_query = model(z, query_positions=positions)
                with torch.inference_mode():
                    inference_full = model(z)
                    inference_query = model(z, query_positions=positions)
        finally:
            for hook in hooks:
                hook.remove()

        expected_width = 2 * 3 + 2 + 1 + 1
        self.assertEqual(full.candidate_source_index.shape[-1], expected_width)
        self.assertEqual(query_only.candidate_source_index.shape[-1], expected_width)
        self.assertEqual(full.value_gate.shape[-1], expected_width)
        self.assertEqual(query_only.value_gate.shape[-1], expected_width)
        self.assertEqual(virtual_calls, 0)
        for reference, no_grad, inference in (
            (full, no_grad_full, inference_full),
            (query_only, no_grad_query, inference_query),
        ):
            for name in ("retrieved", "updated"):
                self.assertTrue(
                    torch.equal(getattr(reference, name), getattr(no_grad, name))
                )
                self.assertTrue(
                    torch.equal(getattr(reference, name), getattr(inference, name))
                )

    def test_zero_virtual_checkpoint_compatibility_and_runtime_reactivation(
        self,
    ) -> None:
        source = self.make_model(
            suffix_k=4,
            occurrences_r=4,
            virtual_candidates=1,
        )
        target = self.make_model(
            suffix_k=1,
            occurrences_r=1,
            virtual_candidates=0,
        )
        incompatible = target.load_state_dict(source.state_dict(), strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertIn("virtual_query.weight", target.state_dict())
        self.assertIn("virtual_key.weight", target.state_dict())

        z = torch.randn(1, 8, 8)
        hard_tokens = target.encode(z)[2]
        prepared_k1r1 = target.prepare_hard_candidates(hard_tokens)
        prepared_k4r4 = source.prepare_hard_candidates(hard_tokens)
        with self.assertRaisesRegex(ValueError, "suffix_k"):
            target(z, hard_candidates=prepared_k4r4)

        target.suffix_k = 2
        with self.assertRaisesRegex(ValueError, "suffix_k"):
            target(z, hard_candidates=prepared_k1r1)
        target.suffix_k = 1
        target.occurrences_r = 2
        with self.assertRaisesRegex(ValueError, "occurrences_r"):
            target(z, hard_candidates=prepared_k1r1)
        target.occurrences_r = 1

        calls = 0

        def virtual_hook(_module, _args, _output) -> None:
            nonlocal calls
            calls += 1

        hooks = [
            target.virtual_query.register_forward_hook(virtual_hook),
            target.virtual_key.register_forward_hook(virtual_hook),
        ]
        try:
            target.virtual_candidates = 1
            output = target(z)
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(output.candidate_source_index.shape[-1], 1 * 1 + 1 + 1)
        self.assertEqual(calls, 2)

    def test_zero_neural_value_scale_skips_projection_and_reactivates(self) -> None:
        torch.manual_seed(20260811)
        model = self.make_model(neural_value_scale=0.0)
        z = torch.randn(2, 9, 8)
        positions = torch.tensor([[1, 6], [2, 8]])
        projection_calls = 0

        def projection_hook(_module, _args, _output) -> None:
            nonlocal projection_calls
            projection_calls += 1

        hook = model.value_proj.register_forward_hook(projection_hook)
        try:
            for query_positions in (None, positions):
                with self.subTest(scale=0.0, query_positions=query_positions):
                    model.zero_grad(set_to_none=True)
                    with patch.object(
                        model,
                        "_candidate_neural_values",
                        side_effect=AssertionError("unexpected neural values"),
                    ):
                        output = model(z, query_positions=query_positions)
                        output.updated.square().mean().backward()
                        with torch.no_grad():
                            model(z, query_positions=query_positions)
                    for parameter in model.value_proj.parameters():
                        self.assertIsNotNone(parameter.grad)
                        assert parameter.grad is not None
                        self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0)
            self.assertEqual(projection_calls, 0)

            model.set_neural_value_scale(1.0)
            for query_positions in (None, positions):
                with self.subTest(scale=1.0, query_positions=query_positions):
                    model.zero_grad(set_to_none=True)
                    output = model(z, query_positions=query_positions)
                    output.updated.square().mean().backward()
                    for parameter in model.value_proj.parameters():
                        self.assertIsNotNone(parameter.grad)
                        assert parameter.grad is not None
                        self.assertGreater(float(parameter.grad.abs().sum()), 0.0)
            self.assertEqual(projection_calls, 2)
        finally:
            hook.remove()

    def test_skipped_value_projection_zero_is_numerically_dormant(self) -> None:
        model = self.make_model(neural_value_scale=0.0).to(dtype=torch.bfloat16)
        symbolic = torch.randn(2, 3, 4, 8, dtype=torch.bfloat16)

        for corruption in ("maximum", "non_finite"):
            with self.subTest(corruption=corruption), torch.no_grad():
                model.value_proj.weight.fill_(torch.finfo(torch.bfloat16).max)
                if corruption == "non_finite":
                    flat_weight = model.value_proj.weight.reshape(-1)
                    flat_weight[0] = torch.nan
                    flat_weight[1] = torch.inf
                    flat_weight[2] = -torch.inf
            model.zero_grad(set_to_none=True)
            candidate_value = model._attach_skipped_value_projection_gradient(symbolic)
            self.assertTrue(torch.equal(candidate_value, symbolic))
            self.assertTrue(torch.isfinite(candidate_value).all())
            candidate_value.float().sum().backward()
            self.assertIsNotNone(model.value_proj.weight.grad)
            assert model.value_proj.weight.grad is not None
            self.assertEqual(
                torch.count_nonzero(model.value_proj.weight.grad).item(), 0
            )

    def test_skipped_virtual_zero_is_numerically_dormant(self) -> None:
        model = self.make_model(virtual_candidates=0).to(dtype=torch.bfloat16)
        retrieved = torch.randn(2, 3, 8, dtype=torch.bfloat16)
        updated = torch.randn(2, 3, 8, dtype=torch.bfloat16)

        for corruption in ("maximum", "non_finite"):
            with self.subTest(corruption=corruption), torch.no_grad():
                for parameter in (
                    model.virtual_query.weight,
                    model.virtual_key.weight,
                ):
                    parameter.fill_(torch.finfo(torch.bfloat16).max)
                    if corruption == "non_finite":
                        flat_parameter = parameter.reshape(-1)
                        flat_parameter[0] = torch.nan
                        flat_parameter[1] = torch.inf
                        flat_parameter[2] = -torch.inf
            model.zero_grad(set_to_none=True)
            actual_retrieved, actual_updated = model._attach_skipped_virtual_gradients(
                retrieved, updated
            )
            self.assertTrue(torch.equal(actual_retrieved, retrieved))
            self.assertTrue(torch.equal(actual_updated, updated))
            self.assertTrue(torch.isfinite(actual_retrieved).all())
            self.assertTrue(torch.isfinite(actual_updated).all())
            (actual_retrieved.float().sum() + actual_updated.float().sum()).backward()
            for parameter in (
                model.virtual_query.weight,
                model.virtual_key.weight,
            ):
                self.assertIsNotNone(parameter.grad)
                assert parameter.grad is not None
                self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0)

    def test_zero_neural_value_scale_matches_explicit_legacy_oracle(self) -> None:
        torch.manual_seed(20260811)
        base = self.make_model(
            learned_residual_scale=1.0,
            neural_value_scale=0.0,
            value_gate_bias=0.0,
        )
        positions = torch.tensor([[1, 6], [2, 8]])

        def legacy_oracle(
            model: ROSA,
            z: torch.Tensor,
            query_positions: torch.Tensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, object]:
            output = model(z, query_positions=query_positions)
            non_null = output.candidate_mask & (output.candidate_kind != NULL_KIND)
            next_position = torch.where(
                non_null,
                output.candidate_source_index + 1,
                torch.zeros_like(output.candidate_source_index),
            )
            if query_positions is None:
                symbol_sequence = (
                    output.code_st[0] @ model.symbol_embedding_1.weight
                    + output.code_st[1] @ model.symbol_embedding_2.weight
                )
                symbolic = _gather_sequence(symbol_sequence, next_position)
                neural = model.value_proj(_gather_sequence(z, next_position))
            else:
                next_st1 = _gather_sequence(output.code_st[0], next_position)
                next_st2 = _gather_sequence(output.code_st[1], next_position)
                symbolic = (
                    next_st1 @ model.symbol_embedding_1.weight
                    + next_st2 @ model.symbol_embedding_2.weight
                )
                neural = _gather_sequence(model.value_proj(z), next_position)
            symbolic = symbolic * non_null.unsqueeze(-1).to(z.dtype)
            neural = neural * output.value_gate.unsqueeze(-1)
            candidate = symbolic + model.neural_value_scale * neural
            st_weights = output.hard_weights + (
                output.soft_weights - output.soft_weights.detach()
            )
            retrieved = (st_weights.unsqueeze(-1) * candidate).sum(dim=-2)
            updated = z + output.read_gate * model.out_proj(retrieved)
            loss = (
                updated.square().mean()
                + retrieved.square().mean()
                + output.value_gate.square().mean()
            )
            return updated, retrieved, loss, output

        for query_positions in (None, positions):
            with self.subTest(query_positions=query_positions):
                optimized_model = copy.deepcopy(base)
                oracle_model = copy.deepcopy(base)
                z_optimized = torch.randn(2, 9, 8, requires_grad=True)
                z_oracle = z_optimized.detach().clone().requires_grad_()

                optimized = optimized_model(
                    z_optimized, query_positions=query_positions
                )
                optimized_loss = (
                    optimized.updated.square().mean()
                    + optimized.retrieved.square().mean()
                    + optimized.value_gate.square().mean()
                )
                oracle_updated, oracle_retrieved, oracle_loss, oracle = legacy_oracle(
                    oracle_model, z_oracle, query_positions
                )

                self.assertTrue(torch.equal(optimized.updated, oracle_updated))
                self.assertTrue(torch.equal(optimized.retrieved, oracle_retrieved))
                self.assertTrue(torch.equal(optimized.value_gate, oracle.value_gate))
                self.assertTrue(torch.equal(optimized_loss, oracle_loss))

                optimized_loss.backward()
                oracle_loss.backward()
                torch.testing.assert_close(
                    z_optimized.grad, z_oracle.grad, rtol=0, atol=0
                )
                for (optimized_name, optimized_parameter), (
                    oracle_name,
                    oracle_parameter,
                ) in zip(
                    optimized_model.named_parameters(),
                    oracle_model.named_parameters(),
                    strict=True,
                ):
                    self.assertEqual(optimized_name, oracle_name)
                    self.assertIsNotNone(optimized_parameter.grad, optimized_name)
                    self.assertIsNotNone(oracle_parameter.grad, oracle_name)
                    torch.testing.assert_close(
                        optimized_parameter.grad,
                        oracle_parameter.grad,
                        rtol=0,
                        atol=0,
                    )

    def test_prepared_candidates_are_bit_exact_and_shared_without_rebuild(self) -> None:
        torch.manual_seed(20260819)
        baseline_model = self.make_model(
            candidate_backend="python",
            learned_residual_scale=1.0,
            neural_value_scale=1.0,
        )
        prepared_model = copy.deepcopy(baseline_model)
        second_consumer = copy.deepcopy(baseline_model)
        tokens = torch.randint(6, (2, 13))
        baseline_z = torch.randn(2, 13, 8, requires_grad=True)
        prepared_z = baseline_z.detach().clone().requires_grad_()
        logits_base = factor_logits_from_tokens(
            tokens, (2, 3), hi=0.4, lo=-0.2, requires_grad=True
        )
        logits_prepared = (
            logits_base[0].detach().clone().requires_grad_(),
            logits_base[1].detach().clone().requires_grad_(),
        )

        baseline = baseline_model(baseline_z, code_logits=logits_base)
        baseline_loss = baseline.updated.square().sum() + sum(
            value.square().sum() for value in baseline.aux_losses.values()
        )
        baseline_loss.backward()

        calls = 0
        original = rosa._build_forward_hard_candidates

        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        with patch("rosa._build_forward_hard_candidates", side_effect=counted):
            hard_tokens = prepared_model.encode(prepared_z, logits_prepared)[2]
            prepared = prepared_model.prepare_hard_candidates(hard_tokens)
            self.assertIsInstance(prepared, PreparedHardCandidates)
            self.assertFalse(prepared.hard_tokens.requires_grad)
            self.assertEqual(calls, 1)
            actual = prepared_model(
                prepared_z, code_logits=logits_prepared, hard_candidates=prepared
            )
            # Different logits with the same argmax remain valid for another model.
            shifted_logits = tuple(value + 0.125 for value in logits_prepared)
            shared = second_consumer(
                prepared_z.detach(),
                code_logits=shifted_logits,
                hard_candidates=prepared,
            )
            self.assertEqual(calls, 1)

        for name in baseline.__dataclass_fields__:
            self.assert_nested_equal(
                getattr(actual, name), getattr(baseline, name), name
            )
        self.assertTrue(torch.equal(shared.hard_tokens, actual.hard_tokens))
        self.assertTrue(
            torch.equal(shared.candidate_source_index, actual.candidate_source_index)
        )
        actual_loss = actual.updated.square().sum() + sum(
            value.square().sum() for value in actual.aux_losses.values()
        )
        self.assertTrue(torch.equal(actual_loss, baseline_loss))
        actual_loss.backward()
        prepared_z_grad = prepared_z.grad
        baseline_z_grad = baseline_z.grad
        assert prepared_z_grad is not None
        assert baseline_z_grad is not None
        self.assertTrue(torch.equal(prepared_z_grad, baseline_z_grad))
        for actual_logit, baseline_logit in zip(
            logits_prepared, logits_base, strict=True
        ):
            actual_logit_grad = actual_logit.grad
            baseline_logit_grad = baseline_logit.grad
            assert actual_logit_grad is not None
            assert baseline_logit_grad is not None
            self.assertTrue(torch.equal(actual_logit_grad, baseline_logit_grad))
        for (_, actual_parameter), (_, baseline_parameter) in zip(
            prepared_model.named_parameters(),
            baseline_model.named_parameters(),
            strict=True,
        ):
            self.assertEqual(
                actual_parameter.grad is None, baseline_parameter.grad is None
            )
            actual_parameter_grad = actual_parameter.grad
            baseline_parameter_grad = baseline_parameter.grad
            if actual_parameter_grad is not None:
                assert baseline_parameter_grad is not None
                self.assertTrue(
                    torch.equal(actual_parameter_grad, baseline_parameter_grad)
                )

    def test_prepared_candidates_combine_with_query_positions(self) -> None:
        model = self.make_model(candidate_backend="python", learned_residual_scale=1.0)
        z = torch.randn(2, 11, 8)
        tokens = torch.randint(6, (2, 11))
        logits = factor_logits_from_tokens(tokens, (2, 3))
        prepared = model.prepare_hard_candidates(model.encode(z, logits)[2])
        positions = torch.tensor([[1, 5, 10], [0, 4, 8]])
        expected = model(z, code_logits=logits, query_positions=positions)
        with patch(
            "rosa._build_forward_hard_candidates",
            side_effect=AssertionError("unexpected rebuild"),
        ):
            actual = model(
                z,
                code_logits=logits,
                query_positions=positions,
                hard_candidates=prepared,
            )
        for name in expected.__dataclass_fields__:
            self.assert_nested_equal(
                getattr(actual, name), getattr(expected, name), name
            )

    def test_prepared_candidates_strict_rejections(self) -> None:
        model = self.make_model(candidate_backend="python")
        z = torch.randn(2, 9, 8)
        tokens = torch.randint(6, (2, 9))
        logits = factor_logits_from_tokens(tokens, (2, 3))

        def prepare():
            return model.prepare_hard_candidates(model.encode(z, logits)[2])

        def forged(field, replacement):
            base = prepare()
            candidates = copy.deepcopy(base.candidates)
            setattr(candidates, field, replacement)
            snapshot = base.hard_tokens.clone()
            tensors = (snapshot,) + tuple(
                getattr(candidates, name) for name in candidates.__dataclass_fields__
            )
            return PreparedHardCandidates(
                snapshot,
                candidates,
                base.suffix_k,
                base.occurrences_r,
                base.candidate_backend,
                base.shape,
                base.device,
                tuple(tensor._version for tensor in tensors),
                tuple(id(tensor) for tensor in tensors),
                tuple(tensor.clone() for tensor in tensors),
            )

        with self.assertRaisesRegex(TypeError, "PreparedHardCandidates"):
            model(z, code_logits=logits, hard_candidates=prepare().candidates)  # type: ignore[arg-type]
        with self.assertRaises(FrozenInstanceError):
            prepare().suffix_k = 99  # type: ignore[misc]

        stale_logits = factor_logits_from_tokens(tokens.clone(), (2, 3))
        # Force one encoded token to differ without changing shapes.
        stale_logits[0].data[0, 0].fill_(-20.0)
        stale_id = (int(tokens[0, 0] // 3) + 1) % 2
        stale_logits[0].data[0, 0, stale_id] = 20.0
        with self.assertRaisesRegex(ValueError, "stale"):
            model(z, code_logits=stale_logits, hard_candidates=prepare())

        for attribute, value, message in (
            ("suffix_k", model.suffix_k + 1, "suffix_k"),
            ("occurrences_r", model.occurrences_r + 1, "occurrences_r"),
            ("candidate_backend", "stateful", "candidate_backend"),
        ):
            consumer = copy.deepcopy(model)
            setattr(consumer, attribute, value)
            with self.assertRaisesRegex(ValueError, message):
                consumer(z, code_logits=logits, hard_candidates=prepare())

        short_z = z[:, :-1]
        short_logits = tuple(value[:, :-1] for value in logits)
        with self.assertRaisesRegex(ValueError, "B/N"):
            model(short_z, code_logits=short_logits, hard_candidates=prepare())

        with self.assertRaisesRegex(ValueError, "source_index.*shape"):
            model(
                z,
                code_logits=logits,
                hard_candidates=forged(
                    "source_index", torch.zeros(2, 9, 14, dtype=torch.long)
                ),
            )
        with self.assertRaisesRegex(TypeError, "match_length.*dtype"):
            model(
                z,
                code_logits=logits,
                hard_candidates=forged(
                    "match_length", torch.zeros(2, 9, 15, dtype=torch.float32)
                ),
            )
        with self.assertRaisesRegex(ValueError, "source_index.*device"):
            model(
                z,
                code_logits=logits,
                hard_candidates=forged(
                    "source_index",
                    torch.empty(2, 9, 15, dtype=torch.long, device="meta"),
                ),
            )

        for field, replacement in (
            ("source_index", torch.zeros(2, 9, 1, dtype=torch.long)),
            ("match_length", torch.zeros(2, 9, 15, dtype=torch.float32)),
        ):
            prepared = prepare()
            setattr(prepared.candidates, field, replacement)
            with self.assertRaises((TypeError, ValueError)):
                model(z, code_logits=logits, hard_candidates=prepared)

        prepared = prepare()
        prepared.candidates.mask.logical_not_()
        with self.assertRaisesRegex(ValueError, "mutated"):
            model(z, code_logits=logits, hard_candidates=prepared)
        prepared = prepare()
        prepared.candidates.mask.data.logical_not_()
        with self.assertRaisesRegex(ValueError, "mutated"):
            model(z, code_logits=logits, hard_candidates=prepared)
        prepared = prepare()
        prepared.candidates.source_index.numpy()[0, 0, 0] = 42
        with self.assertRaisesRegex(ValueError, "mutated"):
            model(z, code_logits=logits, hard_candidates=prepared)
        prepared = prepare()
        prepared.hard_tokens.add_(1)
        with self.assertRaisesRegex(ValueError, "mutated"):
            model(z, code_logits=logits, hard_candidates=prepared)

        if torch.cuda.is_available():
            cuda_model = copy.deepcopy(model).cuda()
            with self.assertRaisesRegex(ValueError, "device"):
                cuda_model(
                    z.cuda(),
                    code_logits=tuple(value.cuda() for value in logits),
                    hard_candidates=prepare(),
                )

    def test_prepared_candidates_cover_all_metadata_guards(self) -> None:
        model = self.make_model(candidate_backend="python")
        tokens = torch.zeros((1, 3), dtype=torch.long)

        for invalid, error_type, message in (
            (object(), TypeError, "must be a Tensor"),
            (tokens.float(), TypeError, "dtype torch.long"),
            (tokens[0], ValueError, r"\[B, N\]"),
            (torch.empty((1, 0), dtype=torch.long), ValueError, "dimensions"),
            (
                torch.empty((1, 3), dtype=torch.long, device="meta"),
                ValueError,
                "same device",
            ),
        ):
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(error_type, message),
            ):
                model.prepare_hard_candidates(invalid)  # type: ignore[arg-type]

        def prepare() -> PreparedHardCandidates:
            return model.prepare_hard_candidates(tokens)

        with self.assertRaisesRegex(ValueError, "device does not match"):
            model._validate_prepared_hard_candidates(
                replace(prepare(), device=torch.device("meta")), tokens
            )

        non_tensor_field = prepare()
        non_tensor_field.candidates.source_index = object()  # type: ignore[assignment]
        with self.assertRaisesRegex(TypeError, "fields must all be Tensors"):
            model._validate_prepared_hard_candidates(non_tensor_field, tokens)

        with self.assertRaisesRegex(ValueError, "metadata is invalid"):
            model._validate_prepared_hard_candidates(
                replace(prepare(), tensor_versions=()), tokens
            )
        with self.assertRaisesRegex(ValueError, "hard_tokens metadata"):
            model._validate_prepared_hard_candidates(
                replace(prepare(), hard_tokens=tokens.float()), tokens
            )

        candidate_type = prepare()
        original_tensors = model._prepared_tensors(candidate_type)
        candidate_type.candidates.source_index = object()  # type: ignore[assignment]
        with (
            patch.object(model, "_prepared_tensors", return_value=original_tensors),
            self.assertRaisesRegex(TypeError, "source_index.*Tensor"),
        ):
            model._validate_prepared_hard_candidates(candidate_type, tokens)

        snapshots = prepare()
        with self.assertRaisesRegex(TypeError, "snapshots must be Tensors"):
            model._validate_prepared_hard_candidates(
                replace(
                    snapshots,
                    _tensor_snapshots=(object(), *snapshots._tensor_snapshots[1:]),
                ),
                tokens,
            )
        with self.assertRaisesRegex(ValueError, "tensor metadata is invalid"):
            model._validate_prepared_hard_candidates(
                replace(
                    snapshots,
                    _tensor_snapshots=(
                        torch.empty(0, dtype=torch.long),
                        *snapshots._tensor_snapshots[1:],
                    ),
                ),
                tokens,
            )

    def test_selected_builder_optional_dependency_fallbacks(self) -> None:
        tokens = torch.tensor([[0, 1, 0]])
        positions = torch.tensor([[2, 0]])

        with (
            patch(
                "rosa._build_stateful_hard_candidates_selected",
                side_effect=ModuleNotFoundError("unexpected", name="unexpected"),
            ),
            self.assertRaises(ModuleNotFoundError),
        ):
            _build_forward_hard_candidates_selected(tokens, positions, 2, 1, "auto")

        missing_numba = ModuleNotFoundError("missing numba", name="numba")
        with patch(
            "rosa._build_stateful_hard_candidates_selected", side_effect=missing_numba
        ):
            with self.assertRaisesRegex(RuntimeError, "numba.*extra"):
                _build_forward_hard_candidates_selected(
                    tokens, positions, 2, 1, "stateful"
                )
            actual = _build_forward_hard_candidates_selected(
                tokens, positions, 2, 1, "auto"
            )
        expected = _build_forward_hard_candidates_selected(
            tokens, positions, 2, 1, "python"
        )
        for name in expected.__dataclass_fields__:
            self.assertTrue(torch.equal(getattr(actual, name), getattr(expected, name)))

    def test_prepared_candidates_have_explicit_external_ownership(self) -> None:
        model = self.make_model(candidate_backend="python")
        z = torch.randn(1, 8, 8)
        tokens = torch.randint(6, (1, 8))
        logits = factor_logits_from_tokens(tokens, (2, 3))
        prepared = model.prepare_hard_candidates(model.encode(z, logits)[2])
        state_keys = set(model.state_dict())
        self.assertFalse(
            any("prepared" in key or "candidate" in key for key in state_keys)
        )
        copied_model = copy.deepcopy(model)
        copied_prepared = copy.deepcopy(prepared)
        self.assertIsNot(copied_prepared.hard_tokens, prepared.hard_tokens)
        copied_model(z, code_logits=logits, hard_candidates=copied_prepared)
        model.to("cpu")
        self.assertEqual(prepared.device.type, "cpu")
        model(z, code_logits=logits, hard_candidates=prepared)

    def test_prepared_candidates_support_inference_tensors(self) -> None:
        model = self.make_model(candidate_backend="python")
        z = torch.randn(1, 8, 8)
        tokens = torch.randint(6, (1, 8))
        logits = factor_logits_from_tokens(tokens, (2, 3))
        with torch.inference_mode():
            hard_tokens = model.encode(z, logits)[2]
            prepared = model.prepare_hard_candidates(hard_tokens)
            copied = copy.deepcopy(prepared)
            expected = model(z, code_logits=logits)
            actual = model(z, code_logits=logits, hard_candidates=copied)
        for name in expected.__dataclass_fields__:
            self.assert_nested_equal(
                getattr(actual, name), getattr(expected, name), name
            )

    def test_query_positions_validation_and_keyword_only_signature(self) -> None:
        model = self.make_model(candidate_backend="python")
        z = torch.randn(2, 7, 8)
        with self.assertRaisesRegex(TypeError, "must be a Tensor"):
            model(z, query_positions=[[1], [2]])  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "dtype torch.long"):
            model(z, query_positions=torch.zeros(2, 1))
        for positions, message in (
            (torch.zeros(2, dtype=torch.long), r"\[B, Q\]"),
            (torch.zeros(1, 1, dtype=torch.long), r"\[B, Q\]"),
            (torch.empty(2, 0, dtype=torch.long), "at least one"),
            (torch.tensor([[0, 7], [1, 2]]), r"\[0, N\)"),
            (torch.tensor([[1, 1], [2, 3]]), "unique"),
        ):
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                model(z, query_positions=positions)
        with self.assertRaises(TypeError):
            model(z, None, None, torch.tensor([[1], [2]]))
        with self.assertRaisesRegex(ValueError, "same device"):
            model(
                z,
                query_positions=torch.empty((2, 1), dtype=torch.long, device="meta"),
            )

    def test_private_zero_virtual_and_query_only_branch_edges(self) -> None:
        no_virtual = self.make_model(virtual_candidates=0)
        z = torch.randn(1, 4, 8)
        source, mask, score = no_virtual._virtual_candidates(
            z,
            torch.empty((1, 4, 0), dtype=torch.long),
            torch.empty((1, 4, 0), dtype=torch.bool),
        )
        self.assertEqual(source.shape, (1, 4, 0))
        self.assertEqual(mask.shape, source.shape)
        self.assertEqual(score.shape, source.shape)

        one_anchor = self.make_model(
            virtual_candidates=0,
            dense_recent_candidates=1,
            sparse_old_candidates=1,
            sparse_old_pool_size=1,
            soft_candidates_forward=True,
        )
        positions = torch.tensor([[3, 1, 0, 2]])
        output = one_anchor(z, query_positions=positions)
        self.assertEqual(output.updated.shape, z.shape)

    def test_stateful_query_only_uses_selected_prefill_builder(self) -> None:
        from rosa._stateful_candidates_numba import prefill_candidates_selected

        model = self.make_model(
            candidate_backend="stateful", learned_residual_scale=1.0
        )
        z = torch.randn(2, 9, 8, requires_grad=True)
        positions = torch.tensor([[8, 0, 4], [1, 8, 0]])
        with (
            patch(
                "rosa._stateful_candidates_numba.prefill_candidates",
                side_effect=AssertionError("full stateful prefill"),
            ),
            patch(
                "rosa._stateful_candidates_numba.prefill_candidates_selected",
                wraps=prefill_candidates_selected,
            ) as selected,
        ):
            output = model(z, query_positions=positions)
            loss = output.updated.square().mean() + sum(output.aux_losses.values())
            loss.backward()
        selected.assert_called_once()
        self.assertEqual(output.candidate_source_index.shape[:2], (2, 9))
        self.assertIsNotNone(z.grad)

    def test_query_positions_matches_full_gradients_and_sentinels(self) -> None:
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        discrete_fields = (
            "hard_tokens",
            "candidate_source_index",
            "candidate_kind",
            "candidate_mask",
            "chosen_candidate",
            "chosen_source_index",
            "chosen_token",
            "chosen_match_length",
            "chosen_is_virtual",
            "hard_rosa_source_index",
            "hard_rosa_predicted_tokens",
            "hard_rosa_match_length",
        )
        float_fields = (
            "updated",
            "retrieved",
            "candidate_scores",
            "soft_weights",
            "hard_weights",
            "soft_match_score",
            "read_gate",
            "value_gate",
        )
        for device in devices:
            for backend in ("python", "stateful"):
                with self.subTest(device=device, backend=backend):
                    torch.manual_seed(20260819)
                    full_model = self.make_model(
                        candidate_backend=backend,
                        learned_residual_scale=1.0,
                        neural_value_scale=1.0,
                        dense_recent_candidates=2,
                        sparse_old_candidates=1,
                        sparse_old_pool_size=4,
                    ).to(device)
                    query_model = copy.deepcopy(full_model)
                    z_full = torch.randn(2, 17, 8, device=device, requires_grad=True)
                    z_query = z_full.detach().clone().requires_grad_()
                    positions = torch.tensor([[1, 6, 15], [2, 9, 16]], device=device)
                    full = full_model(z_full)
                    query = query_model(z_query, query_positions=positions)
                    batch = torch.arange(2, device=device).unsqueeze(1)
                    for name in discrete_fields:
                        expected = getattr(full, name)
                        actual = getattr(query, name)
                        if name != "hard_tokens":
                            expected = expected[batch, positions]
                            actual = actual[batch, positions]
                        self.assertTrue(torch.equal(actual, expected), name)
                    for name in float_fields:
                        expected = getattr(full, name)[batch, positions]
                        actual = getattr(query, name)[batch, positions]
                        torch.testing.assert_close(
                            actual, expected, rtol=2e-6, atol=2e-7
                        )

                    query_mask = torch.zeros((2, 17), dtype=torch.bool, device=device)
                    query_mask.scatter_(1, positions, True)
                    non_query = ~query_mask
                    self.assertTrue(
                        torch.equal(query.updated[non_query], z_query[non_query])
                    )
                    for name in ("retrieved", "soft_weights", "hard_weights"):
                        self.assertTrue(
                            torch.count_nonzero(getattr(query, name)[non_query]) == 0
                        )
                    self.assertTrue(
                        torch.isneginf(query.candidate_scores[non_query]).all()
                    )
                    self.assertTrue(
                        (query.candidate_source_index[non_query] == -1).all()
                    )
                    self.assertTrue((query.candidate_kind[non_query] == -1).all())
                    self.assertFalse(query.candidate_mask[non_query].any())
                    for name in (
                        "chosen_candidate",
                        "chosen_source_index",
                        "chosen_token",
                        "hard_rosa_source_index",
                        "hard_rosa_predicted_tokens",
                    ):
                        self.assertTrue(
                            (getattr(query, name)[non_query] == -1).all(), name
                        )
                    for name in (
                        "chosen_match_length",
                        "hard_rosa_match_length",
                        "soft_match_score",
                        "read_gate",
                        "value_gate",
                    ):
                        self.assertTrue(
                            torch.count_nonzero(getattr(query, name)[non_query]) == 0,
                            name,
                        )

                    def selected_loss(output):
                        return (
                            output.updated[batch, positions].square().mean()
                            + output.retrieved[batch, positions].square().mean()
                            + output.soft_weights[batch, positions].square().mean()
                            + sum(item.square().mean() for item in output.code_soft)
                        )

                    selected_loss(full).backward()
                    selected_loss(query).backward()
                    torch.testing.assert_close(
                        z_query.grad, z_full.grad, rtol=2e-6, atol=2e-7
                    )
                    for (_, full_parameter), (_, query_parameter) in zip(
                        full_model.named_parameters(),
                        query_model.named_parameters(),
                        strict=True,
                    ):
                        self.assertEqual(
                            full_parameter.grad is None, query_parameter.grad is None
                        )
                        if full_parameter.grad is not None:
                            torch.testing.assert_close(
                                query_parameter.grad,
                                full_parameter.grad,
                                rtol=2e-6,
                                atol=2e-7,
                            )

    def test_query_positions_full_arange_is_bit_exact_legacy_route(self) -> None:
        model = self.make_model(candidate_backend="python")
        z_a = torch.randn(2, 9, 8)
        z_b = torch.randn_like(z_a)
        legacy = model(z_a, z_b, None)
        positions = torch.arange(9).expand(2, -1)
        query = model(z_a, z_b, None, query_positions=positions)
        for name in legacy.__dataclass_fields__:
            self.assert_nested_equal(getattr(query, name), getattr(legacy, name), name)

    def test_python_and_stateful_backends_match_all_fields_outputs_and_gradients(
        self,
    ) -> None:
        torch.manual_seed(20260811)
        tokens = torch.randint(6, (2, 13))
        eager_hard = _build_forward_hard_candidates(tokens, 4, 3, "python")
        stateful_hard = _build_forward_hard_candidates(tokens, 4, 3, "stateful")
        for name in eager_hard.__dataclass_fields__:
            self.assertTrue(
                torch.equal(getattr(eager_hard, name), getattr(stateful_hard, name)),
                name,
            )

        python_model = self.make_model(
            candidate_backend="python",
            learned_residual_scale=1.0,
            neural_value_scale=1.0,
        )
        stateful_model = copy.deepcopy(python_model)
        stateful_model.candidate_backend = "stateful"
        z_python = torch.randn(2, 13, 8, requires_grad=True)
        z_stateful = z_python.detach().clone().requires_grad_()
        logits_python = factor_logits_from_tokens(
            tokens, (2, 3), hi=0.2, lo=-0.1, requires_grad=True
        )
        logits_stateful = tuple(
            item.detach().clone().requires_grad_() for item in logits_python
        )
        python_output = python_model(z_python, code_logits=logits_python)
        stateful_output = stateful_model(z_stateful, code_logits=logits_stateful)
        for name in python_output.__dataclass_fields__:
            self.assert_nested_equal(
                getattr(stateful_output, name), getattr(python_output, name), name
            )

        python_loss = python_output.updated.square().mean() + sum(
            python_output.aux_losses.values()
        )
        stateful_loss = stateful_output.updated.square().mean() + sum(
            stateful_output.aux_losses.values()
        )
        python_loss.backward()
        stateful_loss.backward()
        assert z_stateful.grad is not None
        assert z_python.grad is not None
        self.assertTrue(torch.equal(z_stateful.grad, z_python.grad))
        for actual, expected in zip(logits_stateful, logits_python, strict=True):
            assert actual.grad is not None
            assert expected.grad is not None
            self.assertTrue(torch.equal(actual.grad, expected.grad))
        for (actual_name, actual), (expected_name, expected) in zip(
            stateful_model.named_parameters(),
            python_model.named_parameters(),
            strict=True,
        ):
            self.assertEqual(actual_name, expected_name)
            self.assertEqual(actual.grad is None, expected.grad is None, actual_name)
            if actual.grad is not None:
                assert expected.grad is not None
                self.assertTrue(torch.equal(actual.grad, expected.grad), actual_name)

    def test_project_before_gather_matches_gather_first_oracle(self) -> None:
        # Moving a linear projection across a gather changes GEMM row batching,
        # so FP32 accumulation may differ slightly. These tolerances cover that
        # expected roundoff while remaining tight enough to catch path changes.
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))

        for device in devices:
            with self.subTest(device=device.type):
                torch.manual_seed(20260811)
                optimized = self.make_model(
                    dense_recent_candidates=3,
                    sparse_old_candidates=2,
                    sparse_old_pool_size=5,
                    learned_residual_scale=1.0,
                    virtual_scale=1.0,
                    neural_value_scale=1.0,
                    read_gate_bias=0.0,
                    value_gate_bias=0.0,
                    candidate_backend="python",
                    compile_soft_match=False,
                ).to(device)
                oracle = _GatherFirstROSA(
                    d_model=8,
                    codebook_sizes=(2, 3),
                    suffix_k=5,
                    occurrences_r=3,
                    soft_verify_window=6,
                    virtual_candidates=2,
                    virtual_pool_size=6,
                    dense_recent_candidates=3,
                    sparse_old_candidates=2,
                    sparse_old_pool_size=5,
                    selector_dim=8,
                    token_temperature=0.2,
                    retrieval_temperature=0.7,
                    learned_residual_scale=1.0,
                    virtual_scale=1.0,
                    neural_value_scale=1.0,
                    read_gate_bias=0.0,
                    value_gate_bias=0.0,
                    candidate_backend="python",
                    compile_soft_match=False,
                ).to(device)
                oracle.load_state_dict(optimized.state_dict())

                z_optimized = torch.randn(2, 13, 8, device=device, requires_grad=True)
                z_oracle = z_optimized.detach().clone().requires_grad_()
                target = torch.randn_like(z_optimized)
                logits_optimized = tuple(
                    torch.randn(2, 13, size, device=device, requires_grad=True)
                    for size in (2, 3)
                )
                logits_oracle = tuple(
                    item.detach().clone().requires_grad_() for item in logits_optimized
                )

                actual = optimized(z_optimized, code_logits=logits_optimized)
                expected = oracle(z_oracle, code_logits=logits_oracle)
                exact_fields = (
                    "hard_tokens",
                    "candidate_source_index",
                    "candidate_kind",
                    "candidate_mask",
                    "chosen_candidate",
                    "chosen_source_index",
                    "chosen_token",
                    "chosen_match_length",
                    "chosen_is_virtual",
                    "hard_rosa_source_index",
                    "hard_rosa_predicted_tokens",
                    "hard_rosa_match_length",
                )
                for name in exact_fields:
                    self.assertTrue(
                        torch.equal(getattr(actual, name), getattr(expected, name)),
                        name,
                    )

                rtol, atol = (1e-4, 2e-5) if device.type == "cuda" else (3e-5, 3e-6)

                def assert_close_nested(
                    actual_value, expected_value, name: str
                ) -> None:
                    if isinstance(actual_value, torch.Tensor):
                        if actual_value.is_floating_point():
                            torch.testing.assert_close(
                                actual_value,
                                expected_value,
                                rtol=rtol,
                                atol=atol,
                                msg=name,
                            )
                        else:
                            self.assertTrue(
                                torch.equal(actual_value, expected_value), name
                            )
                    elif isinstance(actual_value, tuple):
                        for index, (actual_item, expected_item) in enumerate(
                            zip(actual_value, expected_value, strict=True)
                        ):
                            assert_close_nested(
                                actual_item, expected_item, f"{name}[{index}]"
                            )
                    elif isinstance(actual_value, dict):
                        self.assertEqual(actual_value.keys(), expected_value.keys())
                        for key in actual_value:
                            assert_close_nested(
                                actual_value[key], expected_value[key], f"{name}.{key}"
                            )
                    else:
                        self.assertEqual(actual_value, expected_value, name)

                for name in actual.__dataclass_fields__:
                    assert_close_nested(
                        getattr(actual, name), getattr(expected, name), name
                    )

                actual_loss = F.mse_loss(actual.updated, target) + sum(
                    actual.aux_losses.values()
                )
                expected_loss = F.mse_loss(expected.updated, target) + sum(
                    expected.aux_losses.values()
                )
                torch.testing.assert_close(
                    actual_loss, expected_loss, rtol=rtol, atol=atol
                )
                actual_loss.backward()
                expected_loss.backward()

                assert z_optimized.grad is not None
                assert z_oracle.grad is not None
                torch.testing.assert_close(
                    z_optimized.grad, z_oracle.grad, rtol=rtol, atol=atol
                )
                for actual_logits, expected_logits in zip(
                    logits_optimized, logits_oracle, strict=True
                ):
                    assert actual_logits.grad is not None
                    assert expected_logits.grad is not None
                    torch.testing.assert_close(
                        actual_logits.grad,
                        expected_logits.grad,
                        rtol=rtol,
                        atol=atol,
                    )
                for (actual_name, actual_parameter), (
                    expected_name,
                    expected_parameter,
                ) in zip(
                    optimized.named_parameters(), oracle.named_parameters(), strict=True
                ):
                    self.assertEqual(actual_name, expected_name)
                    self.assertEqual(
                        actual_parameter.grad is None,
                        expected_parameter.grad is None,
                        actual_name,
                    )
                    if actual_parameter.grad is not None:
                        assert expected_parameter.grad is not None
                        torch.testing.assert_close(
                            actual_parameter.grad,
                            expected_parameter.grad,
                            rtol=rtol,
                            atol=atol,
                            msg=actual_name,
                        )

    def test_stateful_forward_does_not_call_eager_or_suffix_write(self) -> None:
        from rosa._stateful_candidates_numba import prefill_candidates

        tokens = torch.tensor([[0, 1, 0, 2, 0, 1]], dtype=torch.long)
        logits = factor_logits_from_tokens(tokens, (2, 3))
        expected = _build_forward_hard_candidates(tokens, 3, 2, "python")
        model = self.make_model(
            suffix_k=3,
            occurrences_r=2,
            candidate_backend="stateful",
        )
        with (
            patch("rosa.build_hard_candidates", side_effect=AssertionError("eager")),
            patch(
                "rosa._stateful_candidates_numba.forward_candidates_step",
                side_effect=AssertionError("step"),
            ),
            patch(
                "rosa._stateful_candidates_numba.prefill_candidates",
                wraps=prefill_candidates,
            ) as prefill_mock,
            patch.object(
                rosa._OnlineSuffixAutomaton,
                "write_current_end",
                side_effect=AssertionError("suffix write"),
            ),
        ):
            output = model(torch.randn(1, 6, 8), code_logits=logits)
        prefill_mock.assert_called_once()
        self.assertTrue(
            torch.equal(output.hard_rosa_source_index, expected.rosa_source_index)
        )

    def test_stateful_prefill_reuses_full_sequence_candidate_tensors(self) -> None:
        from rosa._stateful_candidates_numba import prefill_candidates

        tokens = torch.tensor([[0, 1, 0, 2, 0, 1]], dtype=torch.long)
        captured = None

        def capture_prefill(state, full_tokens):
            nonlocal captured
            captured = prefill_candidates(state, full_tokens)
            return captured

        with patch(
            "rosa._stateful_candidates_numba.prefill_candidates",
            side_effect=capture_prefill,
        ):
            hard = _build_forward_hard_candidates(tokens, 3, 2, "stateful")

        assert captured is not None
        for name in hard.__dataclass_fields__:
            self.assertIs(getattr(hard, name), getattr(captured, name), name)

    def test_stateful_one_shot_releases_native_state_and_storage(self) -> None:
        try:
            import rosa_native_step
        except ModuleNotFoundError:
            self.skipTest("native companion is unavailable")
        if getattr(rosa_native_step, "NativeCandidateState", None) is None:
            self.skipTest("native candidate backend is unavailable")

        from rosa._stateful_candidates_numba import (
            init_candidate_state,
            prefill_candidates,
        )

        tokens = torch.tensor([[0, 1, 0, 2, 0, 1]], dtype=torch.long)
        expected = build_hard_candidates(tokens, suffix_k=3, occurrences_r=2)
        state_ref = None
        array_refs = []
        native_used = False

        def tracked_initialize(*args, **kwargs):
            nonlocal state_ref, array_refs
            state = init_candidate_state(*args, **kwargs)
            state_ref = weakref.ref(state)
            array_refs = [
                weakref.ref(value)
                for value in vars(state).values()
                if isinstance(value, np.ndarray)
            ]
            return state

        def tracked_prefill(state, full_tokens):
            nonlocal native_used
            candidates = prefill_candidates(state, full_tokens)
            native_used = state.native_state not in (None, False)
            return candidates

        with (
            patch(
                "rosa._stateful_candidates_numba.init_candidate_state",
                side_effect=tracked_initialize,
            ),
            patch(
                "rosa._stateful_candidates_numba.prefill_candidates",
                side_effect=tracked_prefill,
            ),
        ):
            actual = rosa._build_stateful_hard_candidates(tokens, 3, 2)

        self.assertTrue(native_used)
        for name in expected.__dataclass_fields__:
            self.assertTrue(
                torch.equal(getattr(actual, name), getattr(expected, name)), name
            )
        gc.collect()
        assert state_ref is not None
        self.assertIsNone(state_ref())
        self.assertTrue(array_refs)
        self.assertTrue(all(array_ref() is None for array_ref in array_refs))

    def test_stateful_one_shot_detaches_native_state_on_exception(self) -> None:
        from rosa._stateful_candidates_numba import init_candidate_state

        state = init_candidate_state(1, 3, suffix_k=2, occurrences_r=2)

        class NativeOwner:
            def __init__(self, candidate_state):
                self.candidate_state = candidate_state

        native = NativeOwner(state)
        state.native_state = native
        with (
            patch(
                "rosa._stateful_candidates_numba.init_candidate_state",
                return_value=state,
            ),
            patch(
                "rosa._stateful_candidates_numba.prefill_candidates",
                side_effect=RuntimeError("prefill failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "prefill failed"),
        ):
            rosa._build_stateful_hard_candidates(torch.tensor([[0, 1, 0]]), 2, 2)
        self.assertIsNone(state.native_state)

    def test_stateful_full_sequence_preserves_scalar_batch_squeeze(self) -> None:
        tokens = torch.tensor([0, 1, 0, 2, 0, 1], dtype=torch.long)
        expected = _build_forward_hard_candidates(tokens, 3, 2, "python")
        actual = _build_forward_hard_candidates(tokens, 3, 2, "stateful")
        for name in expected.__dataclass_fields__:
            self.assertTrue(
                torch.equal(getattr(actual, name), getattr(expected, name)), name
            )
        with self.assertRaisesRegex(ValueError, r"\[N\].*\[B, N\]"):
            _build_forward_hard_candidates(
                torch.zeros((1, 1, 1), dtype=torch.long), 3, 2, "stateful"
            )

    def test_auto_fallback_is_limited_to_missing_optional_dependencies(self) -> None:
        tokens = torch.tensor([[0, 1, 0]], dtype=torch.long)
        expected = build_hard_candidates(tokens, 2, 2)
        for dependency in ("numba", "numpy"):
            missing = ModuleNotFoundError(
                f"No module named '{dependency}'", name=dependency
            )
            with patch("rosa._build_stateful_hard_candidates", side_effect=missing):
                actual = _build_forward_hard_candidates(tokens, 2, 2, "auto")
                self.assertTrue(
                    torch.equal(actual.source_index, expected.source_index), dependency
                )
                with self.assertRaisesRegex(RuntimeError, "numba.*extra"):
                    _build_forward_hard_candidates(tokens, 2, 2, "stateful")

        unrelated = ModuleNotFoundError("No module named 'other'", name="other")
        with (
            patch("rosa._build_stateful_hard_candidates", side_effect=unrelated),
            self.assertRaises(ModuleNotFoundError),
        ):
            _build_forward_hard_candidates(tokens, 2, 2, "auto")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_stateful_full_sequence_matches_eager_on_cuda(self) -> None:
        tokens = torch.randint(5, (3, 31), device="cuda")
        expected = _build_forward_hard_candidates(tokens, 5, 3, "python")
        actual = _build_forward_hard_candidates(tokens, 5, 3, "stateful")
        for name in expected.__dataclass_fields__:
            with self.subTest(name=name):
                self.assertTrue(
                    torch.equal(getattr(actual, name), getattr(expected, name))
                )

    def test_zero_residual_is_exact_rosa_even_with_virtuals(self) -> None:
        tokens = torch.tensor(
            [[0, 1, 0, 2, 0, 1, 0, 3, 0], [4, 4, 2, 4, 4, 2, 1, 4, 4]],
            dtype=torch.long,
        )
        logits = factor_logits_from_tokens(tokens, (2, 3))
        z_a = torch.randn(2, tokens.shape[1], 8)
        z_b = torch.randn_like(z_a)
        model = self.make_model()
        out = model(z_a, z_b=z_b, code_logits=logits)
        ref_pred, ref_src, ref_len = reference_rosa(tokens)

        self.assertTrue(torch.equal(out.hard_tokens, tokens))
        self.assertTrue(torch.equal(out.hard_rosa_source_index, ref_src))
        self.assertTrue(torch.equal(out.hard_rosa_predicted_tokens, ref_pred))
        self.assertTrue(torch.equal(out.hard_rosa_match_length, ref_len))
        self.assertTrue(torch.equal(out.chosen_source_index, ref_src))
        self.assertTrue(torch.equal(out.chosen_token, ref_pred))
        self.assertTrue(torch.equal(out.chosen_match_length, ref_len))
        self.assertFalse(out.chosen_is_virtual.any())
        self.assertTrue(
            torch.allclose(
                out.hard_weights.sum(-1), torch.ones_like(out.read_gate.squeeze(-1))
            )
        )
        self.assertEqual(tuple(out.updated.shape), tuple(z_a.shape))
        self.assertEqual(tuple(out.retrieved.shape), tuple(z_a.shape))
        self.assertTrue(torch.isfinite(out.candidate_scores).all())
        self.assertTrue(torch.isfinite(out.soft_weights).all())
        self.assertTrue(
            torch.allclose(
                out.soft_weights.sum(-1), torch.ones_like(out.soft_weights.sum(-1))
            )
        )
        for loss in out.aux_losses.values():
            self.assertTrue(torch.isfinite(loss))

    def test_soft_verification_forward_equals_truncated_common_suffix(self) -> None:
        tokens = torch.tensor([[0, 1, 0, 2, 0, 1, 0, 3]], dtype=torch.long)
        logits = factor_logits_from_tokens(tokens, (2, 3), hi=30.0, lo=-30.0)
        model = self.make_model(soft_verify_window=4)
        out = model(torch.randn(1, 8, 8), code_logits=logits)
        seq = tokens[0].tolist()
        for i in range(tokens.shape[1]):
            for c in range(out.candidate_source_index.shape[-1]):
                if not bool(out.candidate_mask[0, i, c]):
                    continue
                if int(out.candidate_kind[0, i, c]) == NULL_KIND:
                    self.assertEqual(float(out.soft_match_score[0, i, c]), 0.0)
                    continue
                j = int(out.candidate_source_index[0, i, c])
                k = 0
                while i - k >= 0 and j - k >= 0 and seq[i - k] == seq[j - k] and k < 4:
                    k += 1
                self.assertAlmostEqual(
                    float(out.soft_match_score[0, i, c]), float(k), places=5
                )

    def test_learned_residual_can_override_rosa_with_null(self) -> None:
        tokens = torch.tensor([[0, 1, 0, 2, 0]], dtype=torch.long)
        logits = factor_logits_from_tokens(tokens, (2, 3))
        z = torch.zeros(1, tokens.shape[1], 8)
        model = self.make_model()
        zero_learned_scorer(model)

        base = model(z, code_logits=logits)
        self.assertEqual(int(base.chosen_source_index[0, 2]), 0)

        with torch.no_grad():
            model.null_head.bias.fill_(50.0)
        model.set_learned_residual_scale(1.0)
        changed = model(z, code_logits=logits)
        self.assertEqual(int(changed.chosen_source_index[0, 2]), -1)
        self.assertEqual(int(changed.chosen_token[0, 2]), -1)
        self.assertEqual(
            int(changed.candidate_kind[0, 2, changed.chosen_candidate[0, 2]]), NULL_KIND
        )

    def test_virtual_residual_can_win_but_is_causal(self) -> None:
        # Unique hard symbols => standard ROSA is NULL everywhere.
        tokens = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
        logits = factor_logits_from_tokens(tokens, (2, 3))
        z = torch.randn(1, tokens.shape[1], 8)
        model = self.make_model(learned_residual_scale=1.0, virtual_scale=1.0)
        zero_learned_scorer(model)
        with torch.no_grad():
            model.kind_bias[VIRTUAL_KIND] = 30.0
            model.kind_bias[NULL_KIND] = -30.0

        out = model(z, code_logits=logits)
        self.assertEqual(int(out.chosen_source_index[0, 0]), -1)
        self.assertFalse(bool(out.chosen_is_virtual[0, 0]))
        self.assertTrue(out.chosen_is_virtual[0, 1:].all())
        positions = torch.arange(tokens.shape[1]).view(1, -1)
        self.assertTrue(torch.all(out.chosen_source_index[:, 1:] < positions[:, 1:]))
        self.assertTrue(torch.all(out.chosen_source_index[:, 1:] >= 0))

    def test_forward_validation_and_z_b_none_branch(self) -> None:
        model = self.make_model()
        with self.assertRaisesRegex(ValueError, "z_a"):
            model(torch.zeros(2, 8))
        with self.assertRaisesRegex(ValueError, "sequence length"):
            model(torch.zeros(1, 0, 8))
        with self.assertRaisesRegex(ValueError, "z_b"):
            model(torch.zeros(1, 2, 8), z_b=torch.zeros(1, 3, 8))
        out = model(torch.randn(1, 3, 8))
        self.assertEqual(tuple(out.updated.shape), (1, 3, 8))

    def test_full_gradient_path_and_auxiliary_loss_combination(self) -> None:
        torch.manual_seed(17)
        model = self.make_model(
            learned_residual_scale=1.0,
            virtual_scale=1.0,
            neural_value_scale=1.0,
            read_gate_bias=0.0,
            value_gate_bias=0.0,
        )
        with torch.no_grad():
            model.kind_bias[VIRTUAL_KIND] = 2.0
        z = torch.randn(2, 9, 8, requires_grad=True)
        target = torch.randn_like(z)
        out = model(z)
        lm_loss = F.mse_loss(out.updated, target)
        total = model.combine_losses(
            lm_loss,
            out.aux_losses,
            rosa_weight=0.1,
            consistency_weight=0.1,
            balance_weight=0.05,
            virtual_weight=0.02,
        )
        total.backward()

        self.assertIsNotNone(z.grad)
        self.assertGreater(float(z.grad.abs().sum()), 0.0)
        parameters_expected_to_learn = [
            model.code_head_1.weight,
            model.code_head_2.weight,
            model.symbol_embedding_1.weight,
            model.symbol_embedding_2.weight,
            model.selector_query.weight,
            model.selector_key.weight,
            model.virtual_query.weight,
            model.virtual_key.weight,
            model.feature_mlp[0].weight,
            model.feature_mlp[2].weight,
            model.kind_bias,
            model.null_head.weight,
            model.value_proj.weight,
            model.value_gate_head.weight,
            model.out_proj.weight,
            model.read_gate_head.weight,
        ]
        for p in parameters_expected_to_learn:
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        # Critical differentiable paths should receive non-zero credit.
        self.assertGreater(float(model.code_head_1.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.selector_query.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.symbol_embedding_1.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.value_proj.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.out_proj.weight.grad.abs().sum()), 0.0)

    def test_external_code_logits_receive_gradient(self) -> None:
        tokens = torch.tensor([[0, 1, 0, 2, 0, 1]], dtype=torch.long)
        logits = factor_logits_from_tokens(
            tokens, (2, 3), hi=3.0, lo=-3.0, requires_grad=True
        )
        model = self.make_model(learned_residual_scale=1.0, neural_value_scale=0.0)
        out = model(torch.randn(1, tokens.shape[1], 8), code_logits=logits)
        out.retrieved.square().mean().backward()
        self.assertIsNotNone(logits[0].grad)
        self.assertIsNotNone(logits[1].grad)
        self.assertGreater(
            float(logits[0].grad.abs().sum() + logits[1].grad.abs().sum()), 0.0
        )

    def test_hybrid_union_quotas_causality_dedup_and_constant_budget(self) -> None:
        model = ROSA(
            d_model=8,
            codebook_sizes=(4, 4),
            suffix_k=2,
            occurrences_r=2,
            soft_verify_window=3,
            virtual_candidates=1,
            virtual_pool_size=2,
            dense_recent_candidates=3,
            sparse_old_candidates=2,
            sparse_old_pool_size=4,
            selector_dim=8,
            virtual_scale=0.0,
        )
        for n in (7, 12):
            tokens = torch.arange(n, dtype=torch.long).unsqueeze(0)
            logits = factor_logits_from_tokens(tokens, (4, 4))
            out = model(torch.randn(1, n, 8), code_logits=logits)
            # K*R hard, one legacy virtual, D dense, S sparse, NULL.
            self.assertEqual(out.candidate_source_index.shape[-1], 4 + 1 + 3 + 2 + 1)
            dense = slice(5, 8)
            sparse = slice(8, 10)
            for i in range(n):
                dense_valid = out.candidate_source_index[0, i, dense][
                    out.candidate_mask[0, i, dense]
                ]
                sparse_valid = out.candidate_source_index[0, i, sparse][
                    out.candidate_mask[0, i, sparse]
                ]
                self.assertEqual(dense_valid.numel(), min(3, i))
                self.assertEqual(sparse_valid.numel(), min(2, max(0, i - 3)))
                if dense_valid.numel():
                    self.assertTrue(torch.all(dense_valid < i))
                    self.assertEqual(
                        dense_valid.tolist(), list(range(i - 1, max(-1, i - 4), -1))
                    )
                if sparse_valid.numel():
                    self.assertTrue(torch.all(sparse_valid < i - 3))
                valid_source = out.candidate_source_index[0, i][
                    out.candidate_mask[0, i] & (out.candidate_source_index[0, i] >= 0)
                ]
                self.assertEqual(len(valid_source), len(set(valid_source.tolist())))

        # All sparse scores tie for unique symbols: stable secondary ordering
        # chooses the newest anchors, [6, 4], from [0, 2, 4, 6].
        self.assertEqual(out.candidate_source_index[0, 10, 8:10].tolist(), [6, 4])

        one_anchor = self.make_model(
            dense_recent_candidates=1,
            sparse_old_candidates=1,
            sparse_old_pool_size=1,
        )
        one_anchor(
            torch.randn(1, 4, 8),
            code_logits=factor_logits_from_tokens(
                torch.arange(4, dtype=torch.long).unsqueeze(0), (2, 3)
            ),
        )

    def test_soft_only_union_preserves_hard_forward_and_opt_in_can_win(self) -> None:
        torch.manual_seed(2026)
        tokens = torch.tensor([[0, 1, 0, 2, 3, 1, 4, 5]], dtype=torch.long)
        logits = factor_logits_from_tokens(tokens, (2, 3), hi=4.0, lo=-4.0)
        z_a = torch.randn(1, tokens.shape[1], 8)
        z_b = torch.randn_like(z_a)
        baseline = self.make_model(virtual_scale=0.0)
        union = self.make_model(
            virtual_scale=0.0,
            dense_recent_candidates=2,
            sparse_old_candidates=2,
            sparse_old_pool_size=4,
            soft_candidates_forward=False,
        )
        union.load_state_dict(baseline.state_dict())
        expected = baseline(z_a, z_b=z_b, code_logits=logits)
        actual = union(z_a, z_b=z_b, code_logits=logits)
        for name in (
            "updated",
            "retrieved",
            "chosen_source_index",
            "chosen_token",
            "chosen_match_length",
            "hard_rosa_source_index",
            "hard_rosa_predicted_tokens",
        ):
            self.assertTrue(
                torch.equal(getattr(actual, name), getattr(expected, name)), name
            )

        opt_in = self.make_model(
            learned_residual_scale=1.0,
            virtual_scale=0.0,
            dense_recent_candidates=1,
            sparse_old_candidates=0,
            soft_candidates_forward=True,
        )
        zero_learned_scorer(opt_in)
        with torch.no_grad():
            opt_in.kind_bias[VIRTUAL_KIND] = 30.0
            opt_in.kind_bias[NULL_KIND] = -30.0
        unique = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
        unique_logits = factor_logits_from_tokens(unique, (2, 3))
        opted = opt_in(torch.zeros(1, 6, 8), code_logits=unique_logits)
        self.assertTrue(opted.chosen_is_virtual[0, 1:].all())
        self.assertEqual(opted.chosen_source_index[0, 1:].tolist(), [0, 1, 2, 3, 4])

    def test_recent_and_old_almost_matches_receive_targeted_gradient(self) -> None:
        torch.manual_seed(2026)
        tokens = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)
        # A small margin keeps hard argmaxes distinct while exposing useful
        # overlap to the soft backward path.
        logits = factor_logits_from_tokens(
            tokens, (2, 4), hi=0.1, lo=0.0, requires_grad=True
        )
        hard = build_hard_candidates(tokens, suffix_k=1, occurrences_r=1)
        self.assertFalse(hard.mask[0, 7, 0])
        model = ROSA(
            d_model=8,
            codebook_sizes=(2, 4),
            suffix_k=1,
            occurrences_r=1,
            soft_verify_window=3,
            virtual_candidates=1,
            virtual_pool_size=2,
            dense_recent_candidates=2,
            sparse_old_candidates=1,
            sparse_old_pool_size=4,
            selector_dim=8,
            learned_residual_scale=1.0,
            virtual_scale=0.0,
            soft_candidates_forward=False,
        )
        out = model(torch.randn(1, 8, 8), code_logits=logits)
        # Layout: hard[1], legacy[1], dense[2], sparse[1], NULL.
        recent_position, recent_slot = 7, 2
        old_position, old_slot = 7, 4
        self.assertEqual(
            int(out.candidate_source_index[0, recent_position, recent_slot]), 6
        )
        self.assertEqual(int(out.candidate_source_index[0, old_position, old_slot]), 4)
        self.assertLess(4, old_position - model.dense_recent_candidates)
        recent_loss = -torch.log(out.soft_weights[0, recent_position, recent_slot])
        old_loss = -torch.log(out.soft_weights[0, old_position, old_slot])
        recent_gradient = torch.autograd.grad(recent_loss, logits, retain_graph=True)
        old_gradient = torch.autograd.grad(old_loss, logits)
        self.assertGreater(
            float(sum(gradient[0, 4:8].abs().sum() for gradient in recent_gradient)),
            1e-8,
        )
        self.assertGreater(
            float(sum(gradient[0, 1:8].abs().sum() for gradient in old_gradient)),
            1e-8,
        )

    def test_combine_losses_validation(self) -> None:
        base = torch.tensor(2.0)
        aux = {
            "rosa_distillation": torch.tensor(1.0),
            "hard_soft_consistency": torch.tensor(2.0),
            "code_balance": torch.tensor(3.0),
            "virtual_usage": torch.tensor(4.0),
        }
        got = ROSA.combine_losses(base, aux, 1.0, 2.0, 3.0, 4.0)
        self.assertEqual(float(got), 2 + 1 + 4 + 9 + 16)
        bad = dict(aux)
        bad.pop("virtual_usage")
        with self.assertRaisesRegex(ValueError, "exactly the keys"):
            ROSA.combine_losses(base, bad)


if __name__ == "__main__":
    unittest.main()
