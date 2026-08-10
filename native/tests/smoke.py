from __future__ import annotations

import numpy as np
import rosa_native_step
import torch

from rosa._stateful_numba import _forward_step, _init_inference_state, _prefill


def assert_same_initialized_state(oracle: object, candidate: object) -> None:
    assert oracle.position == candidate.position
    for batch in range(oracle.batch_size):
        size = int(oracle.size[batch])
        edges = int(oracle.edge_count[batch])
        assert int(candidate.size[batch]) == size
        assert int(candidate.edge_count[batch]) == edges
        assert int(candidate.last[batch]) == int(oracle.last[batch])
        for name in ("history",):
            assert np.array_equal(
                getattr(candidate, name)[batch, : oracle.position],
                getattr(oracle, name)[batch, : oracle.position],
            )
        for name in ("head", "hash_state"):
            assert np.array_equal(
                getattr(candidate, name)[batch], getattr(oracle, name)[batch]
            )
        for name in ("edge_token", "edge_target", "edge_next"):
            assert np.array_equal(
                getattr(candidate, name)[batch, :edges],
                getattr(oracle, name)[batch, :edges],
            )
        occupied = oracle.hash_state[batch] != -1
        for name in ("hash_token", "hash_edge"):
            assert np.array_equal(
                getattr(candidate, name)[batch, occupied],
                getattr(oracle, name)[batch, occupied],
            )
        for name in (
            "suffix_link",
            "length",
            "lct_left",
            "lct_right",
            "lct_parent",
            "lct_value",
            "lct_lazy_valid",
        ):
            assert np.array_equal(
                getattr(candidate, name)[batch, :size],
                getattr(oracle, name)[batch, :size],
            ), name


def main() -> None:
    tokens = torch.tensor(
        [[0, 1, 0, 1, 2, 0, 1, 0, 1, 3, -1, 2**31, -1, 7, -1, 7]] * 2,
        dtype=torch.long,
    )
    oracle = _init_inference_state(tokens.shape[0], tokens.shape[1])
    candidate = _init_inference_state(tokens.shape[0], tokens.shape[1])
    oracle.native_state = False

    split = 6
    candidate.native_state = rosa_native_step.NativeState(candidate)
    assert torch.equal(
        _prefill(oracle, tokens[:, :split]),
        torch.from_numpy(
            candidate.native_state.prefill(tokens[:, :split].contiguous().numpy())
        ),
    )
    for position in range(split, tokens.shape[1]):
        expected = _forward_step(oracle, tokens[:, position])
        actual = _forward_step(candidate, tokens[:, position])
        assert torch.equal(actual, expected), (position, actual, expected)

    assert isinstance(candidate.native_state, rosa_native_step.NativeState)
    assert candidate.native_state.position == tokens.shape[1]
    assert candidate.position == tokens.shape[1]

    generator = torch.Generator().manual_seed(20260811)
    cases = [
        torch.randint(-3, 9, (3, 257), generator=generator, dtype=torch.long),
        torch.tensor([[1, 2, 1, 2] * 64, [7] * 256], dtype=torch.long),
    ]
    for case in cases:
        oracle = _init_inference_state(case.shape[0], case.shape[1] + 8)
        candidate = _init_inference_state(case.shape[0], case.shape[1] + 8)
        oracle.native_state = False
        expected = _prefill(oracle, case)
        actual = _prefill(candidate, case)
        assert torch.equal(actual, expected)
        assert isinstance(candidate.native_state, rosa_native_step.NativeState)
        assert_same_initialized_state(oracle, candidate)
        # The offline-built LCT must be immediately usable by streaming.
        continuation = torch.randint(
            -3, 9, (case.shape[0], 8), generator=generator, dtype=torch.long
        )
        for position in range(continuation.shape[1]):
            expected_step = _forward_step(oracle, continuation[:, position])
            actual_step = _forward_step(candidate, continuation[:, position])
            assert torch.equal(actual_step, expected_step), position

    validation = _init_inference_state(2, 4)
    native_validation = rosa_native_step.NativeState(validation)
    for invalid in (
        np.zeros((2, 2), dtype=np.int32),
        np.zeros((2, 2), dtype=np.int64)[:, ::2],
        np.zeros((3, 2), dtype=np.int64),
    ):
        try:
            native_validation.prefill(invalid)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid prefill input was accepted")
    try:
        native_validation.prefill(np.zeros((2, 5), dtype=np.int64))
    except RuntimeError as error:
        assert "capacity" in str(error)
    else:
        raise AssertionError("over-capacity prefill was accepted")
    native_validation.prefill(np.zeros((2, 1), dtype=np.int64))
    try:
        native_validation.prefill(np.zeros((2, 1), dtype=np.int64))
    except RuntimeError as error:
        assert "empty" in str(error)
    else:
        raise AssertionError("prefill accepted a non-empty state")

    malformed = _init_inference_state(2, 4)
    malformed.edge_target = np.empty((2, 0), dtype=np.int32)
    try:
        rosa_native_step.NativeState(malformed)
    except ValueError as error:
        assert "layout" in str(error)
    else:
        raise AssertionError("malformed native state was accepted")
    print("rosa_native_step smoke: ok")


if __name__ == "__main__":
    main()
