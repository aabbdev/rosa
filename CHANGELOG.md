# Changelog

All notable changes to `rosa-torch` are documented here. The project follows
semantic versioning while it remains in the 0.x development series.

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
