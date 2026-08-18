# Changelog

All notable changes to `rosa-torch` are documented here. The project follows
semantic versioning while it remains in the 0.x development series.

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
