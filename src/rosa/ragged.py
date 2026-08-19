"""Dynamic ragged batching for exact stateful ROSA inference.

This module is lazy: importing it does not import Numba or the optional native
companion. Constructing a state requires the ``rosa-torch[numba]`` extra; when
the native companion is installed, one masked native call handles every row.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor


class RaggedInferenceState:
    """Fixed-slot state with independent positions and per-step row masks.

    ``reset`` is applied only to active rows and happens before their token is
    consumed. Inactive rows always return ``-1`` and are not mutated.
    """

    def __init__(
        self,
        batch_size: int,
        max_length: int,
        *,
        use_native: bool = True,
    ) -> None:
        try:
            from ._stateful_numba import _init_inference_state
        except ModuleNotFoundError as error:  # pragma: no cover - optional extra
            if error.name == "numba":
                raise ModuleNotFoundError(
                    "ragged inference requires the 'rosa-torch[numba]' extra"
                ) from error
            raise

        self._state = _init_inference_state(batch_size, max_length)
        self._positions = np.zeros(batch_size, dtype=np.int64)
        # NativeState discovers this optional ABI extension without changing
        # the established scalar-position state layout.
        setattr(self._state, "positions", self._positions)
        self._use_native = use_native
        self._native: Any = None
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("state is closed")

    def close(self) -> None:
        """Release optional native ownership cycles idempotently."""

        if self._closed:
            return
        self._closed = True
        self._native = None
        self._state.native_state = None
        self._state.close()

    @property
    def batch_size(self) -> int:
        """Number of reusable row slots."""

        return int(self._state.batch_size)

    @property
    def max_length(self) -> int:
        """Independent token capacity of each row slot."""

        return int(self._state.max_length)

    @property
    def positions(self) -> Tensor:
        """Current consumed-token count for each row, returned as a copy."""

        self._ensure_open()
        return torch.from_numpy(self._positions.copy())

    @property
    def using_native(self) -> bool:
        """Whether the masked native companion has been selected."""

        return self._native not in (None, False)

    def _native_state(self) -> Any:
        self._ensure_open()
        if not self._use_native or self._native is False:
            return None
        if self._native is None:
            try:
                from rosa_native_step import (  # type: ignore[reportMissingImports]
                    NativeState,
                )
            except ModuleNotFoundError:  # pragma: no cover - optional companion
                self._native = False
                return None
            candidate = NativeState(  # pragma: no cover - optional native companion
                self._state
            )
            if getattr(candidate, "step_masked", None) is None:
                self._native = False
                return None
            self._native = candidate
        return self._native

    def step_masked(
        self,
        tokens: Tensor,
        active: Tensor,
        reset: Tensor | None = None,
    ) -> Tensor:
        """Consume tokens for active rows, optionally recycling selected slots."""

        self._ensure_open()
        if not isinstance(tokens, Tensor):
            raise TypeError("tokens must be a Tensor")
        if tokens.ndim == 0 and self.batch_size == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.ndim != 1 or tokens.shape[0] != self.batch_size:
            raise ValueError("tokens must have shape [batch_size]")
        if tokens.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError("tokens must use an integer dtype")
        active_cpu = self._mask(active, "active")
        if reset is None:
            reset_cpu = torch.zeros(self.batch_size, dtype=torch.uint8)
        else:
            reset_cpu = self._mask(reset, "reset")
        active_array = active_cpu.numpy()
        reset_array = reset_cpu.numpy()
        exhausted = (
            (active_array != 0)
            & (reset_array == 0)
            & (self._positions >= self.max_length)
        )
        if bool(np.any(exhausted)):
            raise RuntimeError("inference state capacity exceeded")

        device = tokens.device
        cpu_tokens = tokens.detach().to(device="cpu", dtype=torch.long).contiguous()
        native = self._native_state()
        if native is not None:  # pragma: no cover - optional native companion
            output = native.step_masked(cpu_tokens.numpy(), active_array, reset_array)
        else:
            from ._stateful_numba import _step_masked_kernel

            state = self._state
            output = _step_masked_kernel(
                cpu_tokens.numpy(),
                active_array,
                reset_array,
                self._positions,
                state.history,
                state.head,
                state.edge_token,
                state.edge_target,
                state.edge_next,
                state.hash_state,
                state.hash_token,
                state.hash_edge,
                state.suffix_link,
                state.length,
                state.lct_left,
                state.lct_right,
                state.lct_parent,
                state.lct_value,
                state.lct_lazy,
                state.lct_lazy_valid,
                state.lct_stack,
                state.last,
                state.size,
                state.edge_count,
            )
        return torch.from_numpy(output).to(device)

    def step(
        self,
        tokens: Tensor,
        active: Tensor | None = None,
        reset: Tensor | None = None,
    ) -> Tensor:
        """Convenience alias; all rows are active when ``active`` is omitted."""

        if active is None:
            active = torch.ones(self.batch_size, dtype=torch.uint8)
        return self.step_masked(tokens, active, reset)

    def _mask(self, mask: Tensor, name: str) -> Tensor:
        if mask.ndim == 0 and self.batch_size == 1:
            mask = mask.unsqueeze(0)
        if mask.ndim != 1 or mask.shape[0] != self.batch_size:
            raise ValueError(f"{name} must have shape [batch_size]")
        if mask.dtype not in (torch.bool, torch.uint8):
            raise TypeError(f"{name} must have dtype bool or uint8")
        return mask.detach().to(device="cpu", dtype=torch.uint8).contiguous()


def init_ragged_state(
    batch_size: int,
    max_length: int,
    *,
    use_native: bool = True,
) -> RaggedInferenceState:
    """Create a dynamic ragged inference state."""

    return RaggedInferenceState(batch_size, max_length, use_native=use_native)


__all__ = ["RaggedInferenceState", "init_ragged_state"]
