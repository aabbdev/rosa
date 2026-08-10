from __future__ import annotations

import random
import unittest

import torch
import torch.nn.functional as F

from rosa import (
    NULL_KIND,
    ROSA,
    VIRTUAL_KIND,
    _balance_kl,
    _gather_sequence,
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


class TestROSAConfiguration(unittest.TestCase):
    def test_constructor_validations(self) -> None:
        invalid_calls = [
            (dict(d_model=0), "d_model"),
            (dict(d_model=4, codebook_sizes=(2,)), "codebook_sizes"),
            (dict(d_model=4, codebook_sizes=(1, 2)), "codebook_sizes"),
            (dict(d_model=4, suffix_k=0), "suffix_k"),
            (dict(d_model=4, occurrences_r=0), "suffix_k"),
            (dict(d_model=4, soft_verify_window=0), "suffix_k"),
            (dict(d_model=4, virtual_candidates=0), "virtual_pool_size"),
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
        ]
        for kwargs, pattern in invalid_calls:
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaisesRegex(ValueError, pattern),
            ):
                ROSA(**kwargs)
        with self.assertRaisesRegex(TypeError, "soft_candidates_forward"):
            ROSA(d_model=4, soft_candidates_forward=1)  # type: ignore[arg-type]

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
