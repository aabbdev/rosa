# Changelog

All notable changes to `rosa-torch` are documented here. The project follows
semantic versioning while it remains in the 0.x development series.

## 0.4.0 — 2026-08-19

### Added

- Query-position execution through `ROSA.forward(..., query_positions=...)`,
  with full-shape public outputs and candidate computation restricted to Q.
- Reusable `PreparedHardCandidates` snapshots with strict token, geometry,
  backend, device, and mutation validation.
- Exact native and Numba selected-prefill APIs that ingest N tokens, emit only
  Q candidate rows, and preserve exact continuation.
- Deterministic `close()` and context-manager support for persistent inference
  states.

### Changed

- Removed native Python-owner reference cycles and made persistent state cleanup
  deterministic.
- Allowed `virtual_candidates=0` while preserving checkpoint compatibility and
  expected zero gradients for inactive parameters.
- Skipped inactive neural value projections without changing forward values or
  training gradient coverage.
- Preserved an exact one-hot straight-through forward while retaining the soft
  backward path.

### Performance

- Reduced selected native candidate prefill from 15.65 ms to 5.72 ms on the
  B16/N512/Q8 reference workload, while reducing candidate output storage from
  about 4.66 MiB to 70 KiB.
- Reduced the validated K4/R4/V1 query workload from 99.86 ms to 45.69 ms with
  a task-specific K1/R1/V0 configuration.
- Validated exact N32768/B1 query evaluation at 86.21 ms and 112.3 MiB peak CUDA
  allocation on the reference system.

### Compatibility

- Existing calls without `query_positions` or `hard_candidates` keep the
  historical execution path and output shapes.
- Default candidate budgets and `backend="auto"` behavior are unchanged.
- `rosa-torch-native 0.4.0` remains optional and requires
  `rosa-torch>=0.4,<0.5` plus NumPy.

## 0.3.0 — 2026-08-18

### Added

- Exact online RLBWT inference backends for top-1 dense workloads, including a
  Python semantic oracle and optional fused native implementations.
- Compact exact `rlbwt_compact256` storage for long contexts with adaptive
  leaves, packed position arrays, and bounded owned memory.
- Explicit opt-in Monte-Carlo RLBWT variants with independently seeded suffix
  fingerprints and clearly separated backend names.
- Native RLBWT smoke tests covering prefill, continuation, reset, compact token
  validation, and agreement with the exact Python oracle.

### Changed

- Reworked native RLBWT storage around unified cache-sized leaves and adaptive
  representations for BWT, position, and longest-common-suffix data.
- Allocated long-context history and tree arenas lazily from live length rather
  than configured capacity.
- Kept `backend="auto"` on the production suffix-automaton path; all RLBWT
  backends remain explicit opt-in choices.

### Compatibility

- The public `rosa` imports and existing stateful inference APIs are unchanged.
- The distribution remains `rosa-torch` and supports Python 3.10+.
- The optional `rosa-torch-native 0.3.0` companion remains separately versioned
  and requires `rosa-torch>=0.3,<0.4` plus NumPy. The Numba extra remains
  optional for suffix-automaton and rich-candidate integration.

## 0.2.0 — 2026-08-11

### Added

- Unified `ROSAInferenceState` facade for exact top-1 and rich candidate modes,
  with uniform and ragged batches, prefill, continuation, reset, and row
  recycling.
- Exact bounded rich candidate state with top-K suffix states, top-R newest
  occurrences, frequencies, masks, and compatibility wrappers.
- Optional native C++ companion capabilities for top-1/rich step and prefill,
  ragged operations, lazy persistent batch workers, and ABI-1-safe fallback.
- Caller-owned `step_into` and `prefill_into` candidate buffers with persistent
  NumPy/Torch views for latency-sensitive inference.
- Opt-in `compile_soft_match=True` Inductor island with per-signature cache
  isolation and eager forward fallback.

### Changed

- Replaced eager suffix-chain occurrence propagation with rooted Link-Cut Tree
  lazy path updates, reducing exact online updates to amortized `O(log N)`.
- `ROSA.forward` now uses fused stateful rich prefill when available while
  retaining `build_hard_candidates` as an independent exact oracle.
- Moved selector, value, virtual-key, and symbolic projections before candidate
  gather, avoiding repeated candidate-wise matrix multiplications.
- Gather chosen symbolic IDs directly instead of constructing temporary
  one-hot sequences.
- Parallelized native full-context batch prefill while retaining serial paths
  for small batches where dispatch overhead dominates.

### Compatibility

- The distribution remains `rosa-torch`; imports remain `from rosa import ...`.
- Python 3.10+ remains supported.
- Existing `forward_step`, `prefill`, `init_candidate_state`, and
  `forward_candidates_step` entry points remain available.
- Numba and the native companion remain optional; exact Python/Numba fallbacks
  are preserved when newer native capabilities are unavailable.
- `torch.compile` acceleration is opt-in because first-call and new-shape
  compilation can take several seconds.

## 0.1.1 — 2026-08-10

- Corrected attribution and clarified that this is an independent
  implementation of RWKV-8 ROSA.

## 0.1.0 — 2026-08-10

- Initial PyPI release of the differentiable exact suffix-automaton retrieval
  module.
