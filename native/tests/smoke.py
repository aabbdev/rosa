from __future__ import annotations

import numpy as np
import rosa_native_step
import torch

from rosa._stateful_numba import _forward_step, _init_inference_state, _prefill


def main() -> None:
    tokens = torch.tensor(
        [[0, 1, 0, 1, 2, 0, 1, 0, 1, 3, -1, 2**31, -1, 7, -1, 7]] * 2,
        dtype=torch.long,
    )
    oracle = _init_inference_state(tokens.shape[0], tokens.shape[1])
    candidate = _init_inference_state(tokens.shape[0], tokens.shape[1])
    oracle.native_state = False

    split = 6
    assert torch.equal(
        _prefill(oracle, tokens[:, :split]),
        _prefill(candidate, tokens[:, :split]),
    )
    for position in range(split, tokens.shape[1]):
        expected = _forward_step(oracle, tokens[:, position])
        actual = _forward_step(candidate, tokens[:, position])
        assert torch.equal(actual, expected), (position, actual, expected)

    assert isinstance(candidate.native_state, rosa_native_step.NativeState)
    assert candidate.native_state.position == tokens.shape[1]
    assert candidate.position == tokens.shape[1]

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
