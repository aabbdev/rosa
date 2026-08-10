from __future__ import annotations

import unittest
from itertools import product

import torch

from rosa import build_hard_candidates
from rosa._stateful_candidates_numba import (
    CandidateState,
    CandidateStep,
    forward_candidates_step,
    init_candidate_state,
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


if __name__ == "__main__":
    unittest.main()
