# ROSA as Differentiable Sparse Retrieval with an Exact Suffix Automaton

This repository is an independent PyTorch implementation and differentiable
extension of **RWKV-8 ROSA (Rapid Online Suffix Automaton)**, described by
[Bo Peng (BlinkDL)](https://github.com/BlinkDL) in
[RWKV-8 ROSA: Beyond Attention](https://www.rwkv.com/images/RWKV-8-ROSA.png)
on [rwkv.com](https://www.rwkv.com/#rwkv-8-explained).

The implementation provides long-range associative retrieval over an internal
discrete code stream. It uses an exact online suffix automaton as a sparse
candidate generator, while keeping tokenization, candidate ranking, and value
retrieval differentiable.

The design avoids a trainable dense automaton transition tensor and avoids dense all-pairs attention over sequence positions. The discrete suffix-automaton structure remains exact; learning is concentrated on how symbols are produced and how a small causal candidate set is ranked and read.

## Highlights

- Exact online suffix-automaton backbone.
- Factorized straight-through discrete codebook.
- Top-K suffix-state candidate generation.
- Bounded multi-occurrence history per suffix state.
- Differentiable soft verification of candidate suffix matches.
- Causal sparse virtual candidates for learned non-suffix associations.
- Explicit NULL candidate when retrieval should be skipped.
- Hard top-1 forward selection with soft straight-through backward gradients.
- Symbolic retrieval with an optional gated neural value residual.
- Learned read gate before the retrieved value is added to the target stream.
- Exact ROSA prior plus a learned residual candidate score.
- Auxiliary losses for ROSA distillation, hard/soft consistency, codebook balance, and virtual-candidate usage.
- 100% statement and branch coverage for the `rosa` package.

## Core scoring rule

Candidate ranking is deliberately residual around standard ROSA behavior:

```text
candidate_score = rosa_prior + learned_residual_scale * learned_score
```

With `learned_residual_scale=0`, exact suffix candidates are ranked by match length with recency tie-breaking, reproducing standard ROSA selection. Increasing the scale allows the neural selector to override that prior when doing so improves the task loss.

The virtual-candidate branch and neural value residual have independent curriculum scales, so the module can start from strict ROSA behavior and gradually enable additional capacity.

## Requirements

- Python 3.10+
- PyTorch
- `coverage`, Ruff, and Pyright for development

Install the published package from PyPI:

```bash
uv add rosa-torch
```

Install the package and its locked development dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked --all-groups
```

The PyPI distribution is named `rosa-torch`; the Python import remains
`from rosa import ROSA`.

For a runtime-only installation from a built wheel, install the wheel with any
PEP 517-compatible Python package manager.

## Repository layout

```text
.
├── pyproject.toml
├── README.md
├── src
│   └── rosa
│       └── __init__.py
└── tests
    ├── __init__.py
    ├── run_coverage.py
    └── test_rosa.py
```

The implementation is distributed as an installable `rosa` package while
remaining in one source module to keep the exact suffix-automaton and neural
retrieval paths easy to inspect together.

## Quick start

```python
import torch

from rosa import ROSA

batch_size = 2
sequence_length = 128
d_model = 256

model = ROSA(
    d_model=d_model,
    codebook_sizes=(16, 16),
    suffix_k=16,
    occurrences_r=4,
    soft_verify_window=32,
    virtual_candidates=4,
    virtual_pool_size=64,
    selector_dim=128,
    learned_residual_scale=0.0,
    virtual_scale=0.0,
    neural_value_scale=0.0,
)

z_a = torch.randn(batch_size, sequence_length, d_model, requires_grad=True)
z_b = torch.randn_like(z_a)

out = model(z_a, z_b=z_b)
loss = out.updated.square().mean()
loss.backward()

print(out.updated.shape)  # [B, N, D]
print(out.chosen_source_index.shape)  # [B, N]
print(out.hard_rosa_match_length.shape)  # [B, N]
```

`z_a` is used to derive the internal symbolic stream and retrieval decisions. `z_b` is the stream receiving the gated retrieval residual. If `z_b` is omitted, `z_a` is used as the target stream as well.

## External code logits

If another module already produces the two factorized codebook logits, pass them directly:

```python
code_logits_1 = torch.randn(batch_size, sequence_length, 16, requires_grad=True)
code_logits_2 = torch.randn(batch_size, sequence_length, 16, requires_grad=True)

out = model(
    z_a,
    z_b=z_b,
    code_logits=(code_logits_1, code_logits_2),
)
```

The hard forward symbols are obtained with `argmax`; the backward path follows the corresponding softmax distributions through a straight-through estimator.

## Curriculum controls

The three runtime scales are registered buffers and are included in `state_dict`:

```python
# Start close to strict ROSA.
model.set_learned_residual_scale(0.0)
model.set_virtual_scale(0.0)
model.set_neural_value_scale(0.0)

# Gradually enable learned ranking and additional memory capacity.
model.set_learned_residual_scale(0.25)
model.set_virtual_scale(0.10)
model.set_neural_value_scale(0.10)

# Fully learned residual behavior if desired.
model.set_learned_residual_scale(1.0)
model.set_virtual_scale(1.0)
model.set_neural_value_scale(1.0)
```

A typical training schedule can anneal these values independently rather than changing architectures during training.

## Auxiliary losses

The forward result exposes:

```python
out.aux_losses
```

with the keys:

- `rosa_distillation`: encourages the soft selector to retain the exact ROSA choice.
- `hard_soft_consistency`: aligns the soft distribution with the hard top-1 forward choice.
- `code_balance`: discourages collapse of either factorized codebook.
- `virtual_usage`: provides an explicit regularizer for the virtual-candidate branch.

They can be combined with the task loss using:

```python
total_loss = model.combine_losses(
    lm_loss,
    out.aux_losses,
    rosa_weight=0.10,
    consistency_weight=0.10,
    balance_weight=0.01,
    virtual_weight=0.01,
)
```

## Output fields

`ROSA.forward` returns a `ROSAOutput` dataclass. The most commonly useful fields are:

- `updated`: target stream after the gated retrieval residual.
- `retrieved`: selected retrieval value before the output projection and read gate.
- `hard_tokens`: hard internal symbolic token IDs.
- `chosen_source_index`: selected historical source end-position, or `-1` for NULL.
- `chosen_token`: continuation token associated with the selected source, or `-1` for NULL.
- `chosen_match_length`: exact suffix length for exact suffix candidates.
- `chosen_is_virtual`: whether the selected candidate came from the virtual branch.
- `hard_rosa_source_index`: source selected by standard hard ROSA.
- `hard_rosa_predicted_tokens`: standard hard ROSA continuation token.
- `soft_match_score`: differentiable truncated common-suffix score for each candidate.
- `soft_weights` / `hard_weights`: soft selector distribution and hard top-1 decision.
- `read_gate` / `value_gate`: learned gates controlling residual injection and neural values.
- `aux_losses`: auxiliary training losses described above.

## Exact reference implementation

`reference_rosa` implements the ROSA definition directly in quadratic time and is intended for tests and diagnostics:

```python
from rosa import reference_rosa

predicted, source, match_length = reference_rosa(tokens)
```

`build_hard_candidates` uses the online suffix automaton and is tested against this brute-force definition over randomized sequences.

## Testing

Run linting, formatting checks, and static type checking:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

Run the unit tests against the installed development package:

```bash
uv run python -m unittest discover -s tests -v
```

Run the strict coverage gate:

```bash
uv run python tests/run_coverage.py
```

The coverage command exits non-zero unless both the test suite passes and the
`rosa` package reaches exactly 100% statement and branch coverage.

Build the wheel and source distribution:

```bash
uv build
```

## Complexity and implementation notes

The neural retrieval side operates on a bounded candidate set rather than all prior positions. For fixed `suffix_k`, `occurrences_r`, verification window, and virtual-pool size, its work per token is bounded independently of context length.

The current suffix-automaton control path is written in Python and stores bounded occurrence histories on suffix-link chains. It is designed for correctness, experimentation, and straightforward integration. For high-throughput training at very long context lengths, the discrete automaton and candidate-building path is the natural component to move to a C++/CUDA/Triton kernel while preserving the tensor-facing module interface and differentiable retrieval path.

## Design guarantees

- Reads happen before the current position is written into occurrence history, preventing self-retrieval.
- Virtual candidate pools contain only earlier positions.
- Disabling the learned residual restores exact ROSA ranking among suffix candidates.
- Disabling virtual candidates does not affect the exact suffix branch or NULL candidate.
- No dense trainable state-to-token-to-state transition tensor is used.

## Attribution

ROSA is an algorithm described by Bo Peng for RWKV-8. This package implements
and extends that algorithm; it does not claim authorship of ROSA itself. For
the original definition, pseudocode, and design notes, see
[RWKV-8 ROSA: Beyond Attention](https://www.rwkv.com/images/RWKV-8-ROSA.png)
on [rwkv.com](https://www.rwkv.com/#rwkv-8-explained).

The implementation in this repository is independently maintained and is not
an official RWKV distribution. The RWKV community can be found on the official
[RWKV Discord server](https://discord.gg/bDSBUMeFpc).
