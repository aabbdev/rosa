"""Stateful exact top-R ROSA candidates without eager suffix propagation.

The suffix automaton is augmented with a Link-Cut Tree.  A write to the
current suffix chain is represented by a lazy tag containing the newest
bounded occurrence prefix and an unbounded frequency delta.  Tag composition
is exact: a newer prefix is prepended to the older pending prefix, while the
frequency deltas are added.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numba import njit
from torch import Tensor

from ._stateful_numba import (
    _add_transition,
    _find_transition,
    _replace_transition,
)


@njit(cache=True, nogil=True, inline="always")
def _is_aux_root(
    left: np.ndarray,
    right: np.ndarray,
    parent: np.ndarray,
    node: int,
) -> bool:  # pragma: no cover - executed as compiled Numba code
    ancestor = parent[node]
    return ancestor == -1 or (left[ancestor] != node and right[ancestor] != node)


@njit(cache=True, nogil=True)
def _apply_tag(
    left: np.ndarray,
    right: np.ndarray,
    occurrences: np.ndarray,
    occurrence_size: np.ndarray,
    frequency: np.ndarray,
    lazy_prefix: np.ndarray,
    lazy_size: np.ndarray,
    lazy_delta: np.ndarray,
    node: int,
    prefix: np.ndarray,
    prefix_size: int,
    delta: int,
) -> None:  # pragma: no cover - executed as compiled Numba code
    """Apply and compose ``(newest_prefix, frequency_delta)`` at ``node``."""

    if node == -1:
        return
    capacity = occurrences.shape[1]
    take = min(prefix_size, capacity)

    old_size = int(occurrence_size[node])
    updated_size = min(capacity, take + old_size)
    for index in range(updated_size - 1, take - 1, -1):
        occurrences[node, index] = occurrences[node, index - take]
    for index in range(take):
        occurrences[node, index] = prefix[index]
    occurrence_size[node] = updated_size
    frequency[node] += delta

    old_lazy_size = int(lazy_size[node])
    updated_lazy_size = min(capacity, take + old_lazy_size)
    for index in range(updated_lazy_size - 1, take - 1, -1):
        lazy_prefix[node, index] = lazy_prefix[node, index - take]
    for index in range(take):
        lazy_prefix[node, index] = prefix[index]
    lazy_size[node] = updated_lazy_size
    lazy_delta[node] += delta


@njit(cache=True, nogil=True)
def _push(
    left: np.ndarray,
    right: np.ndarray,
    occurrences: np.ndarray,
    occurrence_size: np.ndarray,
    frequency: np.ndarray,
    lazy_prefix: np.ndarray,
    lazy_size: np.ndarray,
    lazy_delta: np.ndarray,
    node: int,
) -> None:  # pragma: no cover - executed as compiled Numba code
    size = int(lazy_size[node])
    delta = int(lazy_delta[node])
    if size != 0 or delta != 0:
        _apply_tag(
            left,
            right,
            occurrences,
            occurrence_size,
            frequency,
            lazy_prefix,
            lazy_size,
            lazy_delta,
            int(left[node]),
            lazy_prefix[node],
            size,
            delta,
        )
        _apply_tag(
            left,
            right,
            occurrences,
            occurrence_size,
            frequency,
            lazy_prefix,
            lazy_size,
            lazy_delta,
            int(right[node]),
            lazy_prefix[node],
            size,
            delta,
        )
        lazy_size[node] = 0
        lazy_delta[node] = 0


@njit(cache=True, nogil=True, inline="always")
def _rotate(
    left: np.ndarray,
    right: np.ndarray,
    parent: np.ndarray,
    node: int,
) -> None:  # pragma: no cover - executed as compiled Numba code
    p = int(parent[node])
    g = int(parent[p])
    if left[p] == node:
        middle = int(right[node])
        right[node] = p
        left[p] = middle
    else:
        middle = int(left[node])
        left[node] = p
        right[p] = middle
    if middle != -1:
        parent[middle] = p
    parent[p] = node
    parent[node] = g
    if g != -1:
        if left[g] == p:
            left[g] = node
        elif right[g] == p:
            right[g] = node


@njit(cache=True, nogil=True)
def _splay(
    left: np.ndarray,
    right: np.ndarray,
    parent: np.ndarray,
    occurrences: np.ndarray,
    occurrence_size: np.ndarray,
    frequency: np.ndarray,
    lazy_prefix: np.ndarray,
    lazy_size: np.ndarray,
    lazy_delta: np.ndarray,
    node: int,
    stack: np.ndarray,
) -> None:  # pragma: no cover - executed as compiled Numba code
    depth = 0
    ancestor = node
    stack[depth] = ancestor
    depth += 1
    while not _is_aux_root(left, right, parent, ancestor):
        ancestor = int(parent[ancestor])
        stack[depth] = ancestor
        depth += 1
    while depth > 0:
        depth -= 1
        _push(
            left,
            right,
            occurrences,
            occurrence_size,
            frequency,
            lazy_prefix,
            lazy_size,
            lazy_delta,
            int(stack[depth]),
        )

    while not _is_aux_root(left, right, parent, node):
        p = int(parent[node])
        if not _is_aux_root(left, right, parent, p):
            g = int(parent[p])
            if (left[p] == node) == (left[g] == p):
                _rotate(left, right, parent, p)
            else:
                _rotate(left, right, parent, node)
        _rotate(left, right, parent, node)


@njit(cache=True, nogil=True)
def _access(
    left: np.ndarray,
    right: np.ndarray,
    parent: np.ndarray,
    occurrences: np.ndarray,
    occurrence_size: np.ndarray,
    frequency: np.ndarray,
    lazy_prefix: np.ndarray,
    lazy_size: np.ndarray,
    lazy_delta: np.ndarray,
    node: int,
    stack: np.ndarray,
) -> None:  # pragma: no cover - executed as compiled Numba code
    last = -1
    current = node
    while current != -1:
        _splay(
            left,
            right,
            parent,
            occurrences,
            occurrence_size,
            frequency,
            lazy_prefix,
            lazy_size,
            lazy_delta,
            current,
            stack,
        )
        right[current] = last
        if last != -1:
            parent[last] = current
        last = current
        current = int(parent[current])
    _splay(
        left,
        right,
        parent,
        occurrences,
        occurrence_size,
        frequency,
        lazy_prefix,
        lazy_size,
        lazy_delta,
        node,
        stack,
    )


@njit(cache=True, nogil=True)
def _materialize(
    left: np.ndarray,
    right: np.ndarray,
    parent: np.ndarray,
    occurrences: np.ndarray,
    occurrence_size: np.ndarray,
    frequency: np.ndarray,
    lazy_prefix: np.ndarray,
    lazy_size: np.ndarray,
    lazy_delta: np.ndarray,
    node: int,
    stack: np.ndarray,
) -> None:  # pragma: no cover - executed as compiled Numba code
    _access(
        left,
        right,
        parent,
        occurrences,
        occurrence_size,
        frequency,
        lazy_prefix,
        lazy_size,
        lazy_delta,
        node,
        stack,
    )


@njit(cache=True, nogil=True)
def _cut_parent(
    left: np.ndarray,
    right: np.ndarray,
    parent: np.ndarray,
    occurrences: np.ndarray,
    occurrence_size: np.ndarray,
    frequency: np.ndarray,
    lazy_prefix: np.ndarray,
    lazy_size: np.ndarray,
    lazy_delta: np.ndarray,
    node: int,
    stack: np.ndarray,
) -> None:  # pragma: no cover - executed as compiled Numba code
    _materialize(
        left,
        right,
        parent,
        occurrences,
        occurrence_size,
        frequency,
        lazy_prefix,
        lazy_size,
        lazy_delta,
        node,
        stack,
    )
    ancestors = int(left[node])
    left[node] = -1
    if ancestors != -1:
        parent[ancestors] = -1


@njit(cache=True, nogil=True)
def _link_parent(
    left: np.ndarray,
    right: np.ndarray,
    parent: np.ndarray,
    occurrences: np.ndarray,
    occurrence_size: np.ndarray,
    frequency: np.ndarray,
    lazy_prefix: np.ndarray,
    lazy_size: np.ndarray,
    lazy_delta: np.ndarray,
    node: int,
    represented_parent: int,
    stack: np.ndarray,
) -> None:  # pragma: no cover - executed as compiled Numba code
    _materialize(
        left,
        right,
        parent,
        occurrences,
        occurrence_size,
        frequency,
        lazy_prefix,
        lazy_size,
        lazy_delta,
        node,
        stack,
    )
    parent[node] = represented_parent


@njit(cache=True, nogil=True)
def _path_write(
    left: np.ndarray,
    right: np.ndarray,
    parent: np.ndarray,
    occurrences: np.ndarray,
    occurrence_size: np.ndarray,
    frequency: np.ndarray,
    lazy_prefix: np.ndarray,
    lazy_size: np.ndarray,
    lazy_delta: np.ndarray,
    node: int,
    position: int,
    stack: np.ndarray,
) -> None:  # pragma: no cover - executed as compiled Numba code
    _materialize(
        left,
        right,
        parent,
        occurrences,
        occurrence_size,
        frequency,
        lazy_prefix,
        lazy_size,
        lazy_delta,
        node,
        stack,
    )
    prefix = np.empty(1, dtype=np.int64)
    prefix[0] = position
    _apply_tag(
        left,
        right,
        occurrences,
        occurrence_size,
        frequency,
        lazy_prefix,
        lazy_size,
        lazy_delta,
        node,
        prefix,
        1,
        1,
    )


@njit(cache=True, nogil=True)
def _step_row(
    token: int,
    position: int,
    suffix_k: int,
    occurrences_r: int,
    history: np.ndarray,
    head: np.ndarray,
    edge_token: np.ndarray,
    edge_target: np.ndarray,
    edge_next: np.ndarray,
    hash_state: np.ndarray,
    hash_token: np.ndarray,
    hash_edge: np.ndarray,
    suffix_link: np.ndarray,
    length: np.ndarray,
    lct_left: np.ndarray,
    lct_right: np.ndarray,
    lct_parent: np.ndarray,
    occurrences: np.ndarray,
    occurrence_size: np.ndarray,
    frequency: np.ndarray,
    lazy_prefix: np.ndarray,
    lazy_size: np.ndarray,
    lazy_delta: np.ndarray,
    lct_stack: np.ndarray,
    last: int,
    size: int,
    edge_count: int,
    output_source: np.ndarray,
    output_length: np.ndarray,
    output_state: np.ndarray,
    output_frequency: np.ndarray,
) -> tuple[int, int, int, int]:  # pragma: no cover - compiled Numba code
    history[position] = token
    if size >= head.shape[0]:
        raise RuntimeError("suffix automaton state capacity exceeded")
    current = size
    size += 1
    length[current] = length[last] + 1
    state = last

    while (
        state != -1
        and _find_transition(hash_state, hash_token, hash_edge, state, token) == -1
    ):
        edge_count = _add_transition(
            head,
            edge_token,
            edge_target,
            edge_next,
            hash_state,
            hash_token,
            hash_edge,
            edge_count,
            state,
            token,
            current,
        )
        state = int(suffix_link[state])

    if state == -1:
        suffix_link[current] = 0
        _link_parent(
            lct_left,
            lct_right,
            lct_parent,
            occurrences,
            occurrence_size,
            frequency,
            lazy_prefix,
            lazy_size,
            lazy_delta,
            current,
            0,
            lct_stack,
        )
    else:
        transition = _find_transition(hash_state, hash_token, hash_edge, state, token)
        target = int(edge_target[transition])
        if length[state] + 1 == length[target]:
            suffix_link[current] = target
            _link_parent(
                lct_left,
                lct_right,
                lct_parent,
                occurrences,
                occurrence_size,
                frequency,
                lazy_prefix,
                lazy_size,
                lazy_delta,
                current,
                target,
                lct_stack,
            )
        else:
            if size >= head.shape[0]:
                raise RuntimeError("suffix automaton state capacity exceeded")
            clone = size
            size += 1
            length[clone] = length[state] + 1
            old_parent = int(suffix_link[target])
            suffix_link[clone] = old_parent

            # Materialize q before changing represented-tree edges.  The clone
            # receives q's exact bounded newest prefix and full count, but no
            # pending tag because it has no represented children yet.
            _materialize(
                lct_left,
                lct_right,
                lct_parent,
                occurrences,
                occurrence_size,
                frequency,
                lazy_prefix,
                lazy_size,
                lazy_delta,
                target,
                lct_stack,
            )
            clone_occurrence_size = int(occurrence_size[target])
            occurrence_size[clone] = clone_occurrence_size
            for index in range(clone_occurrence_size):
                occurrences[clone, index] = occurrences[target, index]
            frequency[clone] = frequency[target]
            lazy_size[clone] = 0
            lazy_delta[clone] = 0

            edge = int(head[target])
            while edge != -1:
                edge_count = _add_transition(
                    head,
                    edge_token,
                    edge_target,
                    edge_next,
                    hash_state,
                    hash_token,
                    hash_edge,
                    edge_count,
                    clone,
                    int(edge_token[edge]),
                    int(edge_target[edge]),
                )
                edge = int(edge_next[edge])

            transition = _find_transition(
                hash_state, hash_token, hash_edge, state, token
            )
            while (
                state != -1 and transition != -1 and edge_target[transition] == target
            ):
                _replace_transition(
                    edge_target,
                    hash_state,
                    hash_token,
                    hash_edge,
                    state,
                    token,
                    clone,
                )
                state = int(suffix_link[state])
                if state != -1:
                    transition = _find_transition(
                        hash_state, hash_token, hash_edge, state, token
                    )

            _link_parent(
                lct_left,
                lct_right,
                lct_parent,
                occurrences,
                occurrence_size,
                frequency,
                lazy_prefix,
                lazy_size,
                lazy_delta,
                clone,
                old_parent,
                lct_stack,
            )
            _cut_parent(
                lct_left,
                lct_right,
                lct_parent,
                occurrences,
                occurrence_size,
                frequency,
                lazy_prefix,
                lazy_size,
                lazy_delta,
                target,
                lct_stack,
            )
            suffix_link[target] = clone
            _link_parent(
                lct_left,
                lct_right,
                lct_parent,
                occurrences,
                occurrence_size,
                frequency,
                lazy_prefix,
                lazy_size,
                lazy_delta,
                target,
                clone,
                lct_stack,
            )
            suffix_link[current] = clone
            _link_parent(
                lct_left,
                lct_right,
                lct_parent,
                occurrences,
                occurrence_size,
                frequency,
                lazy_prefix,
                lazy_size,
                lazy_delta,
                current,
                clone,
                lct_stack,
            )

    last = current
    candidate_count = 0
    states_with_history = 0
    node = last
    while node != -1 and states_with_history < suffix_k:
        if length[node] > 0:
            _materialize(
                lct_left,
                lct_right,
                lct_parent,
                occurrences,
                occurrence_size,
                frequency,
                lazy_prefix,
                lazy_size,
                lazy_delta,
                node,
                lct_stack,
            )
            node_occurrences = int(occurrence_size[node])
            if node_occurrences > 0:
                states_with_history += 1
                for occurrence_index in range(min(occurrences_r, node_occurrences)):
                    source = int(occurrences[node, occurrence_index])
                    duplicate = False
                    for seen_index in range(candidate_count):
                        if output_source[seen_index] == source:
                            duplicate = True
                            break
                    if not duplicate:
                        output_source[candidate_count] = source
                        output_length[candidate_count] = length[node]
                        output_state[candidate_count] = node
                        output_frequency[candidate_count] = frequency[node]
                        candidate_count += 1
        node = int(suffix_link[node])

    _path_write(
        lct_left,
        lct_right,
        lct_parent,
        occurrences,
        occurrence_size,
        frequency,
        lazy_prefix,
        lazy_size,
        lazy_delta,
        current,
        position,
        lct_stack,
    )
    return candidate_count, last, size, edge_count


@njit(cache=True, nogil=True)
def _step_batch_kernel(
    tokens: np.ndarray,
    position: int,
    suffix_k: int,
    occurrences_r: int,
    history: np.ndarray,
    head: np.ndarray,
    edge_token: np.ndarray,
    edge_target: np.ndarray,
    edge_next: np.ndarray,
    hash_state: np.ndarray,
    hash_token: np.ndarray,
    hash_edge: np.ndarray,
    suffix_link: np.ndarray,
    length: np.ndarray,
    lct_left: np.ndarray,
    lct_right: np.ndarray,
    lct_parent: np.ndarray,
    occurrences: np.ndarray,
    occurrence_size: np.ndarray,
    frequency: np.ndarray,
    lazy_prefix: np.ndarray,
    lazy_size: np.ndarray,
    lazy_delta: np.ndarray,
    lct_stack: np.ndarray,
    last: np.ndarray,
    size: np.ndarray,
    edge_count: np.ndarray,
) -> tuple[  # pragma: no cover - executed as compiled Numba code
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    slots = suffix_k * occurrences_r
    batch_size = tokens.shape[0]
    source = np.full((batch_size, slots), -1, dtype=np.int64)
    match_length = np.zeros((batch_size, slots), dtype=np.int64)
    state_id = np.full((batch_size, slots), -1, dtype=np.int64)
    candidate_frequency = np.zeros((batch_size, slots), dtype=np.int64)
    count = np.zeros(batch_size, dtype=np.int32)
    for batch_index in range(batch_size):
        row_count, row_last, row_size, row_edge_count = _step_row(
            int(tokens[batch_index]),
            position,
            suffix_k,
            occurrences_r,
            history[batch_index],
            head[batch_index],
            edge_token[batch_index],
            edge_target[batch_index],
            edge_next[batch_index],
            hash_state[batch_index],
            hash_token[batch_index],
            hash_edge[batch_index],
            suffix_link[batch_index],
            length[batch_index],
            lct_left[batch_index],
            lct_right[batch_index],
            lct_parent[batch_index],
            occurrences[batch_index],
            occurrence_size[batch_index],
            frequency[batch_index],
            lazy_prefix[batch_index],
            lazy_size[batch_index],
            lazy_delta[batch_index],
            lct_stack[batch_index],
            int(last[batch_index]),
            int(size[batch_index]),
            int(edge_count[batch_index]),
            source[batch_index],
            match_length[batch_index],
            state_id[batch_index],
            candidate_frequency[batch_index],
        )
        count[batch_index] = row_count
        last[batch_index] = row_last
        size[batch_index] = row_size
        edge_count[batch_index] = row_edge_count
    return source, match_length, state_id, candidate_frequency, count


@njit(cache=True, nogil=True)
def _reset_candidate_rows_kernel(
    reset: np.ndarray,
    head: np.ndarray,
    hash_state: np.ndarray,
    suffix_link: np.ndarray,
    length: np.ndarray,
    lct_left: np.ndarray,
    lct_right: np.ndarray,
    lct_parent: np.ndarray,
    occurrence_size: np.ndarray,
    frequency: np.ndarray,
    lazy_size: np.ndarray,
    lazy_delta: np.ndarray,
    last: np.ndarray,
    size: np.ndarray,
    edge_count: np.ndarray,
    positions: np.ndarray,
) -> None:  # pragma: no cover - executed as compiled Numba code
    for batch_index in range(reset.shape[0]):
        if not reset[batch_index]:
            continue
        used_states = int(size[batch_index])
        for node in range(used_states):
            head[batch_index, node] = -1
            suffix_link[batch_index, node] = -1
            length[batch_index, node] = 0
            lct_left[batch_index, node] = -1
            lct_right[batch_index, node] = -1
            lct_parent[batch_index, node] = -1
            occurrence_size[batch_index, node] = 0
            frequency[batch_index, node] = 0
            lazy_size[batch_index, node] = 0
            lazy_delta[batch_index, node] = 0
        for slot in range(hash_state.shape[1]):
            hash_state[batch_index, slot] = -1
        last[batch_index] = 0
        size[batch_index] = 1
        edge_count[batch_index] = 0
        positions[batch_index] = 0


@njit(cache=True, nogil=True)
def _step_masked_batch_kernel(
    tokens: np.ndarray,
    active: np.ndarray,
    positions: np.ndarray,
    suffix_k: int,
    occurrences_r: int,
    history: np.ndarray,
    head: np.ndarray,
    edge_token: np.ndarray,
    edge_target: np.ndarray,
    edge_next: np.ndarray,
    hash_state: np.ndarray,
    hash_token: np.ndarray,
    hash_edge: np.ndarray,
    suffix_link: np.ndarray,
    length: np.ndarray,
    lct_left: np.ndarray,
    lct_right: np.ndarray,
    lct_parent: np.ndarray,
    occurrences: np.ndarray,
    occurrence_size: np.ndarray,
    frequency: np.ndarray,
    lazy_prefix: np.ndarray,
    lazy_size: np.ndarray,
    lazy_delta: np.ndarray,
    lct_stack: np.ndarray,
    last: np.ndarray,
    size: np.ndarray,
    edge_count: np.ndarray,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:  # pragma: no cover
    slots = suffix_k * occurrences_r
    batch_size = tokens.shape[0]
    source = np.full((batch_size, slots), -1, dtype=np.int64)
    match_length = np.zeros((batch_size, slots), dtype=np.int64)
    state_id = np.full((batch_size, slots), -1, dtype=np.int64)
    candidate_frequency = np.zeros((batch_size, slots), dtype=np.int64)
    count = np.zeros(batch_size, dtype=np.int32)
    for batch_index in range(batch_size):
        if not active[batch_index]:
            continue
        row_count, row_last, row_size, row_edge_count = _step_row(
            int(tokens[batch_index]),
            int(positions[batch_index]),
            suffix_k,
            occurrences_r,
            history[batch_index],
            head[batch_index],
            edge_token[batch_index],
            edge_target[batch_index],
            edge_next[batch_index],
            hash_state[batch_index],
            hash_token[batch_index],
            hash_edge[batch_index],
            suffix_link[batch_index],
            length[batch_index],
            lct_left[batch_index],
            lct_right[batch_index],
            lct_parent[batch_index],
            occurrences[batch_index],
            occurrence_size[batch_index],
            frequency[batch_index],
            lazy_prefix[batch_index],
            lazy_size[batch_index],
            lazy_delta[batch_index],
            lct_stack[batch_index],
            int(last[batch_index]),
            int(size[batch_index]),
            int(edge_count[batch_index]),
            source[batch_index],
            match_length[batch_index],
            state_id[batch_index],
            candidate_frequency[batch_index],
        )
        count[batch_index] = row_count
        last[batch_index] = row_last
        size[batch_index] = row_size
        edge_count[batch_index] = row_edge_count
        positions[batch_index] += 1
    return source, match_length, state_id, candidate_frequency, count


@njit(cache=True, nogil=True)
def _prefill_candidate_kernel(
    tokens: np.ndarray,
    suffix_k: int,
    occurrences_r: int,
    history: np.ndarray,
    head: np.ndarray,
    edge_token: np.ndarray,
    edge_target: np.ndarray,
    edge_next: np.ndarray,
    hash_state: np.ndarray,
    hash_token: np.ndarray,
    hash_edge: np.ndarray,
    suffix_link: np.ndarray,
    length: np.ndarray,
    lct_left: np.ndarray,
    lct_right: np.ndarray,
    lct_parent: np.ndarray,
    occurrences: np.ndarray,
    occurrence_size: np.ndarray,
    frequency: np.ndarray,
    lazy_prefix: np.ndarray,
    lazy_size: np.ndarray,
    lazy_delta: np.ndarray,
    lct_stack: np.ndarray,
    last: np.ndarray,
    size: np.ndarray,
    edge_count: np.ndarray,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:  # pragma: no cover
    batch_size, sequence_length = tokens.shape
    slots = suffix_k * occurrences_r
    source = np.full((batch_size, sequence_length, slots), -1, dtype=np.int64)
    match_length = np.zeros((batch_size, sequence_length, slots), dtype=np.int64)
    state_id = np.full((batch_size, sequence_length, slots), -1, dtype=np.int64)
    candidate_frequency = np.zeros((batch_size, sequence_length, slots), dtype=np.int64)
    count = np.zeros((batch_size, sequence_length), dtype=np.int32)
    for position in range(sequence_length):
        for batch_index in range(batch_size):
            row_count, row_last, row_size, row_edge_count = _step_row(
                int(tokens[batch_index, position]),
                position,
                suffix_k,
                occurrences_r,
                history[batch_index],
                head[batch_index],
                edge_token[batch_index],
                edge_target[batch_index],
                edge_next[batch_index],
                hash_state[batch_index],
                hash_token[batch_index],
                hash_edge[batch_index],
                suffix_link[batch_index],
                length[batch_index],
                lct_left[batch_index],
                lct_right[batch_index],
                lct_parent[batch_index],
                occurrences[batch_index],
                occurrence_size[batch_index],
                frequency[batch_index],
                lazy_prefix[batch_index],
                lazy_size[batch_index],
                lazy_delta[batch_index],
                lct_stack[batch_index],
                int(last[batch_index]),
                int(size[batch_index]),
                int(edge_count[batch_index]),
                source[batch_index, position],
                match_length[batch_index, position],
                state_id[batch_index, position],
                candidate_frequency[batch_index, position],
            )
            count[batch_index, position] = row_count
            last[batch_index] = row_last
            size[batch_index] = row_size
            edge_count[batch_index] = row_edge_count
    return source, match_length, state_id, candidate_frequency, count


@dataclass
class CandidateState:
    """Fixed-capacity tensor state for exact online hard candidates."""

    native_candidate_abi_version: int
    batch_size: int
    max_length: int
    suffix_k: int
    occurrences_r: int
    position: int
    ragged_mode: bool
    positions: np.ndarray
    history: np.ndarray
    head: np.ndarray
    edge_token: np.ndarray
    edge_target: np.ndarray
    edge_next: np.ndarray
    hash_state: np.ndarray
    hash_token: np.ndarray
    hash_edge: np.ndarray
    suffix_link: np.ndarray
    length: np.ndarray
    lct_left: np.ndarray
    lct_right: np.ndarray
    lct_parent: np.ndarray
    occurrences: np.ndarray
    occurrence_size: np.ndarray
    frequency: np.ndarray
    lazy_prefix: np.ndarray
    lazy_size: np.ndarray
    lazy_delta: np.ndarray
    lct_stack: np.ndarray
    last: np.ndarray
    size: np.ndarray
    edge_count: np.ndarray
    native_state: Any


@dataclass(frozen=True)
class CandidateStep:
    """Exact hard candidates emitted while consuming one token per row."""

    source_index: Tensor
    match_length: Tensor
    state_id: Tensor
    frequency: Tensor
    mask: Tensor
    rosa_slot: Tensor
    rosa_source_index: Tensor
    rosa_match_length: Tensor
    rosa_predicted_tokens: Tensor


def init_candidate_state(
    batch_size: int,
    max_length: int,
    *,
    suffix_k: int = 16,
    occurrences_r: int = 4,
    ragged: bool = False,
) -> CandidateState:
    """Allocate an exact bounded-candidate state backed by CPU tensors."""

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if max_length <= 0:
        raise ValueError("max_length must be > 0")
    if suffix_k <= 0:
        raise ValueError("suffix_k must be > 0")
    if occurrences_r <= 0:
        raise ValueError("occurrences_r must be > 0")
    max_states = 2 * max_length + 1
    max_edges = 4 * max_length + 1
    hash_capacity = 1 << (2 * max_edges - 1).bit_length()
    state_shape = (batch_size, max_states)
    edge_shape = (batch_size, max_edges)
    hash_shape = (batch_size, hash_capacity)
    occurrence_shape = (batch_size, max_states, occurrences_r)
    return CandidateState(
        native_candidate_abi_version=1,
        batch_size=batch_size,
        max_length=max_length,
        suffix_k=suffix_k,
        occurrences_r=occurrences_r,
        position=0,
        ragged_mode=ragged,
        positions=np.zeros(batch_size, dtype=np.int64),
        history=np.empty((batch_size, max_length), dtype=np.int64),
        head=np.full(state_shape, -1, dtype=np.int32),
        edge_token=np.empty(edge_shape, dtype=np.int64),
        edge_target=np.empty(edge_shape, dtype=np.int32),
        edge_next=np.empty(edge_shape, dtype=np.int32),
        hash_state=np.full(hash_shape, -1, dtype=np.int32),
        hash_token=np.empty(hash_shape, dtype=np.int64),
        hash_edge=np.empty(hash_shape, dtype=np.int32),
        suffix_link=np.full(state_shape, -1, dtype=np.int32),
        length=np.zeros(state_shape, dtype=np.int32),
        lct_left=np.full(state_shape, -1, dtype=np.int32),
        lct_right=np.full(state_shape, -1, dtype=np.int32),
        lct_parent=np.full(state_shape, -1, dtype=np.int32),
        occurrences=np.full(occurrence_shape, -1, dtype=np.int64),
        occurrence_size=np.zeros(state_shape, dtype=np.int32),
        frequency=np.zeros(state_shape, dtype=np.int64),
        lazy_prefix=np.full(occurrence_shape, -1, dtype=np.int64),
        lazy_size=np.zeros(state_shape, dtype=np.int32),
        lazy_delta=np.zeros(state_shape, dtype=np.int64),
        lct_stack=np.empty(state_shape, dtype=np.int32),
        last=np.zeros(batch_size, dtype=np.int32),
        size=np.ones(batch_size, dtype=np.int32),
        edge_count=np.zeros(batch_size, dtype=np.int32),
        native_state=None,
    )


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _native_candidate_step(  # pragma: no cover - optional native companion
    state: CandidateState,
    cpu_tokens: Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if state.native_state is False:
        return None
    if state.native_state is None:
        try:
            import rosa_native_step  # type: ignore[reportMissingImports]
        except ModuleNotFoundError:
            state.native_state = False
            return None
        native_type = getattr(rosa_native_step, "NativeCandidateState", None)
        if native_type is None:
            state.native_state = False
            return None
        state.native_state = native_type(state)
    return state.native_state.step(cpu_tokens.numpy())


def _native_candidate_call(  # pragma: no cover - optional native companion
    state: CandidateState,
    method: str,
    *args: np.ndarray,
) -> Any | None:
    """Call a post-ABI-1 capability, falling back for an older installed wheel."""

    if state.native_state is False:
        return None
    if state.native_state is None:
        try:
            import rosa_native_step  # type: ignore[reportMissingImports]
        except ModuleNotFoundError:
            state.native_state = False
            return None
        native_type = getattr(rosa_native_step, "NativeCandidateState", None)
        if native_type is None:
            state.native_state = False
            return None
        state.native_state = native_type(state)
    native_method = getattr(state.native_state, method, None)
    if native_method is None:
        state.native_state = False
        return None
    return native_method(*args)


def _validate_candidate_tokens(
    state: CandidateState, tokens: Tensor, *, sequence: bool = False
) -> tuple[Tensor, bool]:
    if not isinstance(state, CandidateState):
        raise TypeError("state must be a CandidateState")
    if not isinstance(tokens, Tensor):
        raise TypeError("tokens must be a Tensor")
    scalar = tokens.ndim == 0 and state.batch_size == 1 and not sequence
    if scalar:
        tokens = tokens.unsqueeze(0)
    expected_ndim = 2 if sequence else 1
    if tokens.ndim != expected_ndim or tokens.shape[0] != state.batch_size:
        shape = "[batch_size, sequence_length]" if sequence else "[batch_size]"
        raise ValueError(f"tokens must have shape {shape}")
    if tokens.dtype not in _INTEGER_DTYPES:
        raise TypeError("tokens must use an integer dtype")
    return tokens, scalar


def _candidate_step_from_arrays(
    state: CandidateState,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    device: torch.device,
) -> CandidateStep:
    source, match_length, state_id, frequency, count = arrays
    slots = state.suffix_k * state.occurrences_r
    slot_shape = (1,) * count.ndim + (slots,)
    slot_index = np.arange(slots, dtype=np.int32).reshape(slot_shape)
    mask = slot_index < count[..., None]
    rosa_source = source[..., 0].copy()
    rosa_length = match_length[..., 0].copy()
    rosa_slot = np.where(count > 0, 0, -1).astype(np.int64)
    rosa_predicted = np.full(count.shape, -1, dtype=np.int64)
    for index in np.ndindex(count.shape):
        if count[index] > 0:
            batch_index = index[0]
            source_position = int(rosa_source[index])
            rosa_predicted[index] = state.history[batch_index, source_position + 1]
    return CandidateStep(
        source_index=torch.from_numpy(source).to(device),
        match_length=torch.from_numpy(match_length).to(device),
        state_id=torch.from_numpy(state_id).to(device),
        frequency=torch.from_numpy(frequency).to(device),
        mask=torch.from_numpy(mask).to(device),
        rosa_slot=torch.from_numpy(rosa_slot).to(device),
        rosa_source_index=torch.from_numpy(rosa_source).to(device),
        rosa_match_length=torch.from_numpy(rosa_length).to(device),
        rosa_predicted_tokens=torch.from_numpy(rosa_predicted).to(device),
    )


def forward_candidates_step(state: CandidateState, tokens: Tensor) -> CandidateStep:
    """Consume one token per row and return exact top-R candidates for K suffixes."""

    tokens, _ = _validate_candidate_tokens(state, tokens)
    if state.ragged_mode:
        raise RuntimeError("uniform step is unavailable on a ragged candidate state")
    if state.position >= state.max_length:
        raise RuntimeError("candidate state capacity exceeded")

    device = tokens.device
    cpu_tokens = tokens.detach().to(device="cpu", dtype=torch.long).contiguous()
    native_output = _native_candidate_step(state, cpu_tokens)
    if native_output is None:
        source, match_length, state_id, frequency, count = _step_batch_kernel(
            cpu_tokens.numpy(),
            state.position,
            state.suffix_k,
            state.occurrences_r,
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
            state.occurrences,
            state.occurrence_size,
            state.frequency,
            state.lazy_prefix,
            state.lazy_size,
            state.lazy_delta,
            state.lct_stack,
            state.last,
            state.size,
            state.edge_count,
        )
        state.position += 1
        state.positions.fill(state.position)
    else:
        source, match_length, state_id, frequency, count = native_output
        state.positions.fill(state.position)
    return _candidate_step_from_arrays(
        state, (source, match_length, state_id, frequency, count), device
    )


def reset_candidates_masked(state: CandidateState, reset: Tensor) -> None:
    """Reset selected ragged rows without reallocating their fixed-capacity storage."""

    if not isinstance(state, CandidateState):
        raise TypeError("state must be a CandidateState")
    if not state.ragged_mode:
        raise RuntimeError("reset_masked requires a ragged candidate state")
    if not isinstance(reset, Tensor):
        raise TypeError("reset must be a Tensor")
    if reset.ndim == 0 and state.batch_size == 1:
        reset = reset.unsqueeze(0)
    if reset.ndim != 1 or reset.shape[0] != state.batch_size:
        raise ValueError("reset must have shape [batch_size]")
    if reset.dtype not in (torch.bool, torch.uint8):
        raise TypeError("reset must have dtype bool or uint8")
    cpu_reset = reset.detach().to(device="cpu", dtype=torch.bool).contiguous()
    native = _native_candidate_call(state, "reset_masked", cpu_reset.numpy())
    if native is None:  # pragma: no branch - native capability is optional
        _reset_candidate_rows_kernel(
            cpu_reset.numpy(),
            state.head,
            state.hash_state,
            state.suffix_link,
            state.length,
            state.lct_left,
            state.lct_right,
            state.lct_parent,
            state.occurrence_size,
            state.frequency,
            state.lazy_size,
            state.lazy_delta,
            state.last,
            state.size,
            state.edge_count,
            state.positions,
        )


def forward_candidates_step_masked(
    state: CandidateState,
    tokens: Tensor,
    active: Tensor,
    reset: Tensor | None = None,
) -> CandidateStep:
    """Consume tokens only on active rows, optionally recycling active rows first."""

    tokens, _ = _validate_candidate_tokens(state, tokens)
    if not state.ragged_mode:
        raise RuntimeError("step_masked requires a ragged candidate state")
    if not isinstance(active, Tensor):
        raise TypeError("active must be a Tensor")
    if active.ndim == 0 and state.batch_size == 1:
        active = active.unsqueeze(0)
    if active.ndim != 1 or active.shape[0] != state.batch_size:
        raise ValueError("active must have shape [batch_size]")
    if active.dtype not in (torch.bool, torch.uint8):
        raise TypeError("active must have dtype bool or uint8")
    if reset is None:
        reset = torch.zeros_like(active, dtype=torch.bool)
    if not isinstance(reset, Tensor):
        raise TypeError("reset must be a Tensor")
    if reset.ndim == 0 and state.batch_size == 1:
        reset = reset.unsqueeze(0)
    if reset.ndim != 1 or reset.shape[0] != state.batch_size:
        raise ValueError("reset must have shape [batch_size]")
    if reset.dtype not in (torch.bool, torch.uint8):
        raise TypeError("reset must have dtype bool or uint8")
    device = tokens.device
    cpu_tokens = tokens.detach().to(device="cpu", dtype=torch.long).contiguous()
    cpu_active = active.detach().to(device="cpu", dtype=torch.bool).contiguous()
    cpu_reset = reset.detach().to(device="cpu", dtype=torch.bool).contiguous()
    effective_reset = np.logical_and(cpu_active.numpy(), cpu_reset.numpy())
    future_positions = np.where(effective_reset, 0, state.positions)
    if np.any(np.logical_and(cpu_active.numpy(), future_positions < 0)):
        raise RuntimeError("candidate position must be non-negative")
    if np.any(np.logical_and(cpu_active.numpy(), future_positions >= state.max_length)):
        raise RuntimeError("candidate state capacity exceeded")
    native_output = _native_candidate_call(
        state,
        "step_masked",
        cpu_tokens.numpy(),
        cpu_active.numpy(),
        cpu_reset.numpy(),
    )
    if native_output is None:
        _reset_candidate_rows_kernel(
            effective_reset,
            state.head,
            state.hash_state,
            state.suffix_link,
            state.length,
            state.lct_left,
            state.lct_right,
            state.lct_parent,
            state.occurrence_size,
            state.frequency,
            state.lazy_size,
            state.lazy_delta,
            state.last,
            state.size,
            state.edge_count,
            state.positions,
        )
        native_output = _step_masked_batch_kernel(
            cpu_tokens.numpy(),
            cpu_active.numpy(),
            state.positions,
            state.suffix_k,
            state.occurrences_r,
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
            state.occurrences,
            state.occurrence_size,
            state.frequency,
            state.lazy_prefix,
            state.lazy_size,
            state.lazy_delta,
            state.lct_stack,
            state.last,
            state.size,
            state.edge_count,
        )
    return _candidate_step_from_arrays(state, native_output, device)


def prefill_candidates(state: CandidateState, tokens: Tensor) -> CandidateStep:
    """Consume a complete uniform sequence and emit candidates at every position."""

    tokens, _ = _validate_candidate_tokens(state, tokens, sequence=True)
    if state.ragged_mode:
        raise RuntimeError("prefill is unavailable on a ragged candidate state")
    if state.position != 0:
        raise RuntimeError("prefill requires an empty candidate state")
    sequence_length = tokens.shape[1]
    if sequence_length > state.max_length:
        raise RuntimeError("candidate state capacity exceeded")
    device = tokens.device
    cpu_tokens = tokens.detach().to(device="cpu", dtype=torch.long).contiguous()
    native_output = _native_candidate_call(state, "prefill", cpu_tokens.numpy())
    if native_output is None:
        native_output = _prefill_candidate_kernel(
            cpu_tokens.numpy(),
            state.suffix_k,
            state.occurrences_r,
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
            state.occurrences,
            state.occurrence_size,
            state.frequency,
            state.lazy_prefix,
            state.lazy_size,
            state.lazy_delta,
            state.lct_stack,
            state.last,
            state.size,
            state.edge_count,
        )
        state.position = sequence_length
    state.positions.fill(state.position)
    return _candidate_step_from_arrays(state, native_output, device)


# Explicit aliases keep naming discoverable while preserving the original API.
prefill_candidate_state = prefill_candidates
reset_candidate_rows = reset_candidates_masked
