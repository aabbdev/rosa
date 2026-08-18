from __future__ import annotations

from itertools import product

import numpy as np
import rosa_native_step
import torch

from rosa import reference_rosa
from rosa._rlbwt_backend import (
    _init_rlbwt_state,
    _prefill,
    _reconstruct_rlbwt,
)


def main() -> None:
    assert rosa_native_step.rlbwt_abi_version == 1
    assert rosa_native_step.rlbwt_mc_abi_version == 1
    assert rosa_native_step.rlbwt_compact_abi_version == 1

    exhaustive = torch.tensor(list(product(range(3), repeat=8)), dtype=torch.long)
    expected, sources, lengths = reference_rosa(exhaustive)
    native = rosa_native_step.NativeRLBWTState(exhaustive.shape[0], exhaustive.shape[1])
    actual = torch.from_numpy(native.prefill(np.ascontiguousarray(exhaustive.numpy())))
    assert torch.equal(actual, expected)
    assert native.sources.tolist() == sources[:, -1].tolist()
    assert native.lrs_lengths.tolist() == lengths[:, -1].tolist()

    tokens = torch.tensor(
        [[-(2**63), 2**63 - 1, -(2**63), 0, 7, 0, 7]], dtype=torch.long
    )
    python_state = _init_rlbwt_state(1, tokens.shape[1])
    expected = _prefill(python_state, tokens)
    native = rosa_native_step.NativeRLBWTState(1, tokens.shape[1])
    actual = torch.from_numpy(native.prefill(np.ascontiguousarray(tokens.numpy())))
    assert torch.equal(actual, expected)
    pa, lcs, bwt, sentinel = native.row_snapshot(0)
    assert pa.tolist() == python_state.pa[0][: tokens.shape[1] + 1]
    assert lcs.tolist() == python_state.lcs[0][: tokens.shape[1] + 1]
    reconstructed = []
    for value, is_sentinel in zip(bwt.tolist(), sentinel.tolist(), strict=True):
        reconstructed.append("$" if is_sentinel else value)
    expected_bwt = [
        "$" if repr(value) == "$" else value
        for value in _reconstruct_rlbwt(python_state)
    ]
    assert reconstructed == expected_bwt

    generator = torch.Generator().manual_seed(20260811)
    random_tokens = torch.randint(16, (3, 300), generator=generator)
    expected, _, _ = reference_rosa(random_tokens)
    adaptive = rosa_native_step.NativeRLBWTState(3, 300)
    actual = torch.from_numpy(
        adaptive.prefill(np.ascontiguousarray(random_tokens.numpy()))
    )
    assert torch.equal(actual, expected)
    # Random alphabet-16 rows cross the adaptive RLE-to-literal threshold.
    assert all(count > 75 for count in adaptive.run_counts.tolist())

    compact_values = np.ascontiguousarray(random_tokens.numpy())
    compact = rosa_native_step.NativeRLBWTCompactState(3, 300, 256)
    compact_output = np.concatenate(
        (
            compact.prefill_append(np.ascontiguousarray(compact_values[:, :137])),
            compact.prefill_append(np.ascontiguousarray(compact_values[:, 137:])),
        ),
        axis=1,
    )
    assert np.array_equal(compact_output, actual.numpy())
    assert compact.vocabulary_size == 256
    assert sum(compact.storage_breakdown.values()) <= compact.storage_bytes
    try:
        compact.step(np.array([0, 1, 256], dtype=np.int64))
    except ValueError:
        pass
    else:
        raise AssertionError("compact RLBWT accepted an out-of-range token")

    wide_tokens = torch.randint(256, (1, 2048), generator=generator)
    wide_expected, _, _ = reference_rosa(wide_tokens)
    wide_compact = rosa_native_step.NativeRLBWTCompactState(1, 2048, 256)
    wide_actual = wide_compact.prefill(np.ascontiguousarray(wide_tokens.numpy()))
    assert np.array_equal(wide_actual, wide_expected.numpy())

    repeated = torch.tensor([[index & 1 for index in range(1200)]], dtype=torch.long)
    repeated_expected, _, _ = reference_rosa(repeated)
    repeated_compact = rosa_native_step.NativeRLBWTCompactState(1, 100_000_000, 256)
    repeated_actual = repeated_compact.prefill_append(
        np.ascontiguousarray(repeated.numpy())
    )
    assert np.array_equal(repeated_actual, repeated_expected.numpy())

    native.reset()
    assert native.position == 0
    assert native.run_counts.tolist() == [1]
    assert native.step(np.array([5], dtype=np.int64)).tolist() == [-1]

    monte_carlo_inputs = [
        torch.tensor(list(product(range(2), repeat=8)), dtype=torch.long),
        torch.randint(23, (4, 257), generator=generator),
        torch.zeros((2, 257), dtype=torch.long),
        torch.tensor([[index % 7 for index in range(257)]], dtype=torch.long),
    ]
    for lanes in (2, 3):
        for seed in (0, 20260811, 2**64 - 1):
            for values in monte_carlo_inputs:
                exact = rosa_native_step.NativeRLBWTState(
                    values.shape[0], values.shape[1]
                )
                mc = rosa_native_step.NativeRLBWTStateMC(
                    values.shape[0], values.shape[1], lanes, seed
                )
                array = np.ascontiguousarray(values.numpy())
                assert np.array_equal(mc.prefill(array), exact.prefill(array))
                assert np.array_equal(mc.sources, exact.sources)
                assert np.array_equal(mc.lrs_lengths, exact.lrs_lengths)
                for row in range(values.shape[0]):
                    for mc_field, exact_field in zip(
                        mc.row_snapshot(row), exact.row_snapshot(row), strict=True
                    ):
                        assert np.array_equal(mc_field, exact_field)
                assert mc.lanes == lanes
                assert mc.seed == seed
                assert mc.storage_bytes > exact.storage_bytes

        continuation = torch.randint(5, (2, 129), generator=generator)
        mc = rosa_native_step.NativeRLBWTStateMC(2, 129, lanes, 20260811)
        exact = rosa_native_step.NativeRLBWTState(2, 129)
        first = np.ascontiguousarray(continuation[:, :100].numpy())
        assert np.array_equal(mc.prefill(first), exact.prefill(first))
        for position in range(100, 129):
            column = np.ascontiguousarray(continuation[:, position].numpy())
            assert np.array_equal(mc.step(column), exact.step(column))
        try:
            mc.step(np.zeros(2, dtype=np.int64))
        except RuntimeError:
            pass
        else:
            raise AssertionError("MC state accepted a token beyond capacity")
        before_reset_bytes = mc.storage_bytes
        mc.reset()
        assert mc.position == 0
        assert mc.storage_bytes <= before_reset_bytes
        assert mc.step(np.array([9, 9], dtype=np.int64)).tolist() == [-1, -1]

    for invalid_lanes in (0, 1, 4):
        try:
            rosa_native_step.NativeRLBWTStateMC(1, 4, invalid_lanes, 0)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid MC lane count was accepted")

    try:
        rosa_native_step.NativeRLBWTState(1, 2**32 - 1)
    except ValueError:
        pass
    else:
        raise AssertionError("RLBWT accepted a max_length that overflows tree weights")

    for invalid_vocabulary in (0, 257):
        try:
            rosa_native_step.NativeRLBWTCompactState(1, 4, invalid_vocabulary)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid compact vocabulary size was accepted")

    for invalid in (
        np.zeros((1, 1), dtype=np.int32),
        np.zeros((2,), dtype=np.int64),
        np.zeros((1, 4), dtype=np.int64)[:, ::2],
    ):
        candidate = rosa_native_step.NativeRLBWTState(1, 2)
        try:
            if invalid.ndim == 1:
                candidate.step(invalid)
            else:
                candidate.prefill(invalid)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid native RLBWT input was accepted")


if __name__ == "__main__":
    main()
