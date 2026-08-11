# `rosa-torch-native`

Optional native companion for
[`rosa-torch`](https://github.com/aabbdev/rosa), accelerating the exact CPU
suffix-automaton and Link-Cut Tree inference step. The distribution installs
the importable `rosa_native_step` extension module; it does not replace the
main Python package.

The C++ core implements the validated exact production state machine. It binds
the NumPy arrays of a `_StatefulInferenceState` once, updates them in place,
and releases the GIL during computation. It neither includes nor calls
libtorch. The runtime dependency `rosa-torch[numba]>=0.2,<0.3` provides the
compatible state contract together with PyTorch, NumPy, and Numba.

The constructor validates every shape, dtype, counter, and ABI version before
retaining any pointer. The current native state ABI is `1`.

The module also exposes `NativeCandidateState` for the exact rich state in
`rosa._stateful_candidates_numba`. Its batched `step` maintains the same K
suffixes, R newest occurrences, unbounded frequencies, and
`newest-prefix + delta` LCT tags as the Python/Numba implementation. The R
capacity remains owned by the NumPy arrays of the Python `CandidateState`, whose
lifetime is retained by the native object. Capability can be detected through
the presence of `NativeCandidateState` and `candidate_abi_version == 1`.

The rich ABI retains `step`, global `reset`, and `position`, while detecting the
optional `prefill`, `step_masked`, `reset_masked`, `positions`, `step_into`, and
`prefill_into` capabilities. `prefill` emits all five native arrays at every
position in one C++ call and leaves the state ready for continuation. The
caller-owned `*_into` methods avoid repeated output allocation after validating
dtype, shape, contiguity, writability, and memory overlap.

Ragged mode stores one position per row. Uniform and ragged paths are
intentionally incompatible so that exactly one position authority exists at
any time. The Python wrapper falls back exactly to Numba when an older ABI-1
wheel does not provide a newer optional method.

## Installation and usage

`rosa-torch-native` is not currently published on PyPI. Build a wheel from a
Git checkout using the target Python interpreter, then install the matching
wheel for the current platform and ABI:

```bash
git clone https://github.com/aabbdev/rosa.git
cd rosa
uv sync --extra numba
uv build --python .venv/bin/python --wheel native --out-dir native/dist
uv pip install --python .venv/bin/python \
  native/dist/rosa_torch_native-0.2.0-*.whl
```

`rosa-torch` detects the extension automatically from its Numba inference
backend. The low-level API remains available for diagnostics:

```python
from rosa_native_step import NativeState
```

`NativeState(state).step(tokens_numpy)` expects a contiguous NumPy vector of
shape `[batch_size]` that is convertible to `int64`. The object retains a
reference to the Python state and exposes its read-only `position`.

`NativeCandidateState(candidate_state).step(tokens_numpy)` requires a
C-contiguous `int64` NumPy vector and returns the low-level tuple
`(source, match_length, state_id, frequency, count)`. `reset()` recycles the
whole batch in time proportional to the state nodes and hash slots that are
actually occupied.

Each native state lazily creates a small dependency-free persistent C++17
thread pool. It parallelizes independent batch rows for top-1 and rich prefill.
Prefill keeps a serial path below batch 4, where waking a worker costs more than
the measured computation. Uniform steps use the pool only from batch 64 onward
to preserve the inexpensive serial path for small batches.

The pool is bounded by the batch size, reported CPU count, and 16 total threads.
`ROSA_NATIVE_THREADS=1` forces the serial path; another positive integer sets a
lower cap. An unset variable selects the available limit automatically,
while an invalid value falls back conservatively to one thread. Only non-empty
digit strings representing a strictly positive integer are accepted.

Using one pool per state avoids fragile extension-singleton shutdown ordering,
and workers never call Python. No worker is created for a ragged state or a
batch below 4 that never uses prefill. A sufficiently large first call may
create at most 15 workers, which are reused until state destruction. Concurrent
mutating calls on the same instance are serialized by a mutex acquired while
the GIL is released. Public output allocation and array ownership remain
unchanged. Worker exceptions are captured and rethrown on the calling thread.

## Isolated local build

The PEP 517 backend is setuptools, with setuptools, wheel, and pybind11 as build
dependencies. `Pybind11Extension` selects C++17, while setuptools provides the
platform extension flags for macOS and Linux. Both platforms additionally use
`-O3` and `NDEBUG`.

From the repository root:

```bash
uv sync --extra numba
uv build --python .venv/bin/python --wheel native \
  --out-dir /tmp/rosa-native-dist
uv run --isolated \
  --with '.[numba]' \
  --with /tmp/rosa-native-dist/rosa_torch_native-0.2.0-*.whl \
  native/tests/smoke.py
```

The smoke test forces Numba as the oracle, lets the regular ROSA path load the
companion, and compares predictions step by step.

Run the rich smoke test and direct Numba comparison with
`native/tests/candidate_smoke.py` and `native/benchmark_candidates.py`,
respectively, in the same isolated environment.

## Multi-platform publication

The next publication step is a dedicated `cibuildwheel` workflow after defining
the supported Python and architecture matrix, manylinux targets, and release
policy. No such workflow is included yet because an unvalidated matrix must not
be published. Every wheel contains native code and must be built separately for
each Python ABI and platform.
