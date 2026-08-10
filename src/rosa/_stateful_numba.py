"""Stateful exact ROSA inference using a suffix automaton and Link-Cut Tree."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numba import njit
from torch import Tensor


@njit(cache=True, nogil=True, inline="always")
def _transition_hash(  # pragma: no cover - executed as compiled Numba code
    state: int,
    token: int,
) -> np.uint64:
    """Mix the complete signed state/token key into a 64-bit hash."""

    value = np.uint64(token)
    value ^= np.uint64(state) + np.uint64(0x9E3779B97F4A7C15)
    value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return value ^ (value >> np.uint64(31))


@njit(cache=True, nogil=True, inline="always")
def _find_transition(  # pragma: no cover - executed as compiled Numba code
    hash_state: np.ndarray,
    hash_token: np.ndarray,
    hash_edge: np.ndarray,
    state: int,
    token: int,
) -> int:
    """Return the edge index for ``(state, token)``, or -1 when absent."""

    mask = hash_state.shape[0] - 1
    slot = np.int64(_transition_hash(state, token) & np.uint64(mask))
    while hash_state[slot] != -1:
        if hash_state[slot] == state and hash_token[slot] == token:
            return int(hash_edge[slot])
        slot = (slot + 1) & mask
    return -1


@njit(cache=True, nogil=True, inline="always")
def _add_transition(  # pragma: no cover - executed as compiled Numba code
    head: np.ndarray,
    edge_token: np.ndarray,
    edge_target: np.ndarray,
    edge_next: np.ndarray,
    hash_state: np.ndarray,
    hash_token: np.ndarray,
    hash_edge: np.ndarray,
    edge_count: int,
    state: int,
    token: int,
    target: int,
) -> int:
    if edge_count >= edge_token.shape[0]:
        raise RuntimeError("suffix automaton transition capacity exceeded")
    edge_token[edge_count] = token
    edge_target[edge_count] = target
    edge_next[edge_count] = head[state]
    head[state] = edge_count
    mask = hash_state.shape[0] - 1
    slot = np.int64(_transition_hash(state, token) & np.uint64(mask))
    while hash_state[slot] != -1:
        if hash_state[slot] == state and hash_token[slot] == token:
            raise RuntimeError("duplicate suffix automaton transition")
        slot = (slot + 1) & mask
    hash_state[slot] = state
    hash_token[slot] = token
    hash_edge[slot] = edge_count
    return edge_count + 1


@njit(cache=True, nogil=True, inline="always")
def _replace_transition(  # pragma: no cover - executed as compiled Numba code
    edge_target: np.ndarray,
    hash_state: np.ndarray,
    hash_token: np.ndarray,
    hash_edge: np.ndarray,
    state: int,
    token: int,
    target: int,
) -> None:
    edge = _find_transition(hash_state, hash_token, hash_edge, state, token)
    if edge == -1:
        raise RuntimeError("suffix automaton transition not found")
    edge_target[edge] = target


@njit(cache=True, nogil=True, inline="always")
def _lct_is_aux_root(  # pragma: no cover - executed as compiled Numba code
    left: np.ndarray,
    right: np.ndarray,
    parent: np.ndarray,
    node: int,
) -> bool:
    p = parent[node]
    return p == -1 or (left[p] != node and right[p] != node)


@njit(cache=True, nogil=True, inline="always")
def _lct_apply(  # pragma: no cover - executed as compiled Numba code
    value: np.ndarray,
    lazy: np.ndarray,
    lazy_valid: np.ndarray,
    node: int,
    assigned: int,
) -> None:
    if node != -1:
        value[node] = assigned
        lazy[node] = assigned
        lazy_valid[node] = 1


@njit(cache=True, nogil=True, inline="always")
def _lct_push(  # pragma: no cover - executed as compiled Numba code
    left: np.ndarray,
    right: np.ndarray,
    value: np.ndarray,
    lazy: np.ndarray,
    lazy_valid: np.ndarray,
    node: int,
) -> None:
    if lazy_valid[node] != 0:
        assigned = lazy[node]
        _lct_apply(value, lazy, lazy_valid, left[node], assigned)
        _lct_apply(value, lazy, lazy_valid, right[node], assigned)
        lazy_valid[node] = 0


@njit(cache=True, nogil=True, inline="always")
def _lct_rotate(  # pragma: no cover - executed as compiled Numba code
    left: np.ndarray,
    right: np.ndarray,
    parent: np.ndarray,
    node: int,
) -> None:
    p = parent[node]
    g = parent[p]
    if left[p] == node:
        middle = right[node]
        right[node] = p
        left[p] = middle
    else:
        middle = left[node]
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
def _lct_splay(  # pragma: no cover - executed as compiled Numba code
    left: np.ndarray,
    right: np.ndarray,
    parent: np.ndarray,
    value: np.ndarray,
    lazy: np.ndarray,
    lazy_valid: np.ndarray,
    node: int,
    stack: np.ndarray,
) -> None:
    depth = 0
    ancestor = node
    stack[depth] = ancestor
    depth += 1
    while not _lct_is_aux_root(left, right, parent, ancestor):
        ancestor = parent[ancestor]
        stack[depth] = ancestor
        depth += 1
    while depth > 0:
        depth -= 1
        _lct_push(left, right, value, lazy, lazy_valid, stack[depth])

    while not _lct_is_aux_root(left, right, parent, node):
        p = parent[node]
        if not _lct_is_aux_root(left, right, parent, p):
            g = parent[p]
            if (left[p] == node) == (left[g] == p):
                _lct_rotate(left, right, parent, p)
            else:
                _lct_rotate(left, right, parent, node)
        _lct_rotate(left, right, parent, node)


@njit(cache=True, nogil=True)
def _lct_access(  # pragma: no cover - executed as compiled Numba code
    left: np.ndarray,
    right: np.ndarray,
    parent: np.ndarray,
    value: np.ndarray,
    lazy: np.ndarray,
    lazy_valid: np.ndarray,
    node: int,
    stack: np.ndarray,
) -> None:
    last = -1
    current = node
    while current != -1:
        _lct_splay(left, right, parent, value, lazy, lazy_valid, current, stack)
        right[current] = last
        if last != -1:
            parent[last] = current
        last = current
        current = parent[current]
    _lct_splay(left, right, parent, value, lazy, lazy_valid, node, stack)


@njit(cache=True, nogil=True)
def _lct_point_query(  # pragma: no cover - executed as compiled Numba code
    left: np.ndarray,
    right: np.ndarray,
    parent: np.ndarray,
    value: np.ndarray,
    lazy: np.ndarray,
    lazy_valid: np.ndarray,
    node: int,
    stack: np.ndarray,
) -> int:
    _lct_access(left, right, parent, value, lazy, lazy_valid, node, stack)
    return int(value[node])


@njit(cache=True, nogil=True)
def _lct_path_assign(  # pragma: no cover - executed as compiled Numba code
    left: np.ndarray,
    right: np.ndarray,
    parent: np.ndarray,
    value: np.ndarray,
    lazy: np.ndarray,
    lazy_valid: np.ndarray,
    node: int,
    assigned: int,
    stack: np.ndarray,
) -> None:
    _lct_access(left, right, parent, value, lazy, lazy_valid, node, stack)
    _lct_apply(value, lazy, lazy_valid, node, assigned)


@njit(cache=True, nogil=True)
def _lct_cut_parent(  # pragma: no cover - executed as compiled Numba code
    left: np.ndarray,
    right: np.ndarray,
    parent: np.ndarray,
    value: np.ndarray,
    lazy: np.ndarray,
    lazy_valid: np.ndarray,
    node: int,
    stack: np.ndarray,
) -> None:
    _lct_access(left, right, parent, value, lazy, lazy_valid, node, stack)
    ancestors = left[node]
    left[node] = -1
    if ancestors != -1:
        parent[ancestors] = -1


@njit(cache=True, nogil=True)
def _lct_link_parent(  # pragma: no cover - executed as compiled Numba code
    left: np.ndarray,
    right: np.ndarray,
    parent: np.ndarray,
    value: np.ndarray,
    lazy: np.ndarray,
    lazy_valid: np.ndarray,
    node: int,
    represented_parent: int,
    stack: np.ndarray,
) -> None:
    _lct_access(left, right, parent, value, lazy, lazy_valid, node, stack)
    parent[node] = represented_parent


@njit(cache=True, nogil=True)
def _step_row(  # pragma: no cover - executed as compiled Numba code
    token: int,
    position: int,
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
    lct_value: np.ndarray,
    lct_lazy: np.ndarray,
    lct_lazy_valid: np.ndarray,
    lct_stack: np.ndarray,
    last: int,
    size: int,
    edge_count: int,
) -> tuple[int, int, int, int]:
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
        state = suffix_link[state]

    if state == -1:
        suffix_link[current] = 0
        _lct_link_parent(
            lct_left,
            lct_right,
            lct_parent,
            lct_value,
            lct_lazy,
            lct_lazy_valid,
            current,
            0,
            lct_stack,
        )
    else:
        transition = _find_transition(hash_state, hash_token, hash_edge, state, token)
        target = int(edge_target[transition])
        if length[state] + 1 == length[target]:
            suffix_link[current] = target
            _lct_link_parent(
                lct_left,
                lct_right,
                lct_parent,
                lct_value,
                lct_lazy,
                lct_lazy_valid,
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
            old_parent = suffix_link[target]
            suffix_link[clone] = old_parent
            clone_value = _lct_point_query(
                lct_left,
                lct_right,
                lct_parent,
                lct_value,
                lct_lazy,
                lct_lazy_valid,
                target,
                lct_stack,
            )
            lct_value[clone] = clone_value
            edge = head[target]
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
                edge = edge_next[edge]
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
                state = suffix_link[state]
                if state != -1:
                    transition = _find_transition(
                        hash_state, hash_token, hash_edge, state, token
                    )

            _lct_link_parent(
                lct_left,
                lct_right,
                lct_parent,
                lct_value,
                lct_lazy,
                lct_lazy_valid,
                clone,
                old_parent,
                lct_stack,
            )
            _lct_cut_parent(
                lct_left,
                lct_right,
                lct_parent,
                lct_value,
                lct_lazy,
                lct_lazy_valid,
                target,
                lct_stack,
            )
            suffix_link[target] = clone
            _lct_link_parent(
                lct_left,
                lct_right,
                lct_parent,
                lct_value,
                lct_lazy,
                lct_lazy_valid,
                target,
                clone,
                lct_stack,
            )
            suffix_link[current] = clone
            _lct_link_parent(
                lct_left,
                lct_right,
                lct_parent,
                lct_value,
                lct_lazy,
                lct_lazy_valid,
                current,
                clone,
                lct_stack,
            )

    last = current
    matched = suffix_link[current]
    source = -1
    if matched != 0:
        source = _lct_point_query(
            lct_left,
            lct_right,
            lct_parent,
            lct_value,
            lct_lazy,
            lct_lazy_valid,
            matched,
            lct_stack,
        )
    prediction = -1
    if source >= 0:
        prediction = int(history[source + 1])
    _lct_path_assign(
        lct_left,
        lct_right,
        lct_parent,
        lct_value,
        lct_lazy,
        lct_lazy_valid,
        current,
        position,
        lct_stack,
    )
    return prediction, last, size, edge_count


@njit(cache=True, nogil=True)
def _step_batch_kernel(  # pragma: no cover - executed as compiled Numba code
    tokens: np.ndarray,
    position: int,
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
    lct_value: np.ndarray,
    lct_lazy: np.ndarray,
    lct_lazy_valid: np.ndarray,
    lct_stack: np.ndarray,
    last: np.ndarray,
    size: np.ndarray,
    edge_count: np.ndarray,
) -> np.ndarray:
    output = np.empty(tokens.shape[0], dtype=np.int64)
    for batch_index in range(tokens.shape[0]):
        prediction, new_last, new_size, new_edge_count = _step_row(
            int(tokens[batch_index]),
            position,
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
            lct_value[batch_index],
            lct_lazy[batch_index],
            lct_lazy_valid[batch_index],
            lct_stack[batch_index],
            int(last[batch_index]),
            int(size[batch_index]),
            int(edge_count[batch_index]),
        )
        output[batch_index] = prediction
        last[batch_index] = new_last
        size[batch_index] = new_size
        edge_count[batch_index] = new_edge_count
    return output


@njit(cache=True, nogil=True)
def _replay_kernel(  # pragma: no cover - executed as compiled Numba code
    tokens: np.ndarray,
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
    lct_value: np.ndarray,
    lct_lazy: np.ndarray,
    lct_lazy_valid: np.ndarray,
    lct_stack: np.ndarray,
    last: np.ndarray,
    size: np.ndarray,
    edge_count: np.ndarray,
) -> np.ndarray:
    output = np.empty(tokens.shape, dtype=np.int64)
    for position in range(tokens.shape[1]):
        output[:, position] = _step_batch_kernel(
            tokens[:, position],
            position,
            history,
            head,
            edge_token,
            edge_target,
            edge_next,
            hash_state,
            hash_token,
            hash_edge,
            suffix_link,
            length,
            lct_left,
            lct_right,
            lct_parent,
            lct_value,
            lct_lazy,
            lct_lazy_valid,
            lct_stack,
            last,
            size,
            edge_count,
        )
    return output


@njit(cache=True, nogil=True, inline="always")
def _range_max(  # pragma: no cover - executed as compiled Numba code
    tree: np.ndarray,
    base: int,
    start: int,
    stop: int,
) -> int:
    result = -1
    left = base + start
    right = base + stop
    while left < right:
        if left & 1:
            if tree[left] > result:
                result = int(tree[left])
            left += 1
        if right & 1:
            right -= 1
            if tree[right] > result:
                result = int(tree[right])
        left >>= 1
        right >>= 1
    return result


@njit(cache=True, nogil=True, inline="always")
def _point_max(  # pragma: no cover - executed as compiled Numba code
    tree: np.ndarray,
    base: int,
    index: int,
    value: int,
) -> None:
    node = base + index
    if tree[node] >= value:
        return
    tree[node] = value
    node >>= 1
    while node != 0:
        updated = tree[node << 1]
        if tree[(node << 1) | 1] > updated:
            updated = tree[(node << 1) | 1]
        if tree[node] == updated:
            break
        tree[node] = updated
        node >>= 1


@njit(cache=True, nogil=True, inline="always")
def _fenwick_prefix(  # pragma: no cover - executed as compiled Numba code
    tree: np.ndarray,
    stop: int,
) -> int:
    result = 0
    node = stop
    while node > 0:
        result += int(tree[node])
        node -= node & -node
    return result


@njit(cache=True, nogil=True, inline="always")
def _fenwick_add(  # pragma: no cover - executed as compiled Numba code
    tree: np.ndarray,
    index: int,
) -> None:
    node = index + 1
    while node < tree.shape[0]:
        tree[node] += 1
        node += node & -node


@njit(cache=True, nogil=True, inline="always")
def _fenwick_select(  # pragma: no cover - executed as compiled Numba code
    tree: np.ndarray,
    rank: int,
) -> int:
    """Return the zero-based position of a one-based order statistic."""

    node = 0
    step = 1
    size = tree.shape[0] - 1
    while (step << 1) <= size:
        step <<= 1
    while step != 0:
        candidate = node + step
        if candidate <= size and tree[candidate] < rank:
            node = candidate
            rank -= int(tree[candidate])
        step >>= 1
    return node


@njit(cache=True, nogil=True, inline="always")
def _tree_lca(  # pragma: no cover - executed as compiled Numba code
    up: np.ndarray,
    tin: np.ndarray,
    tout: np.ndarray,
    first: int,
    second: int,
) -> int:
    if tin[first] <= tin[second] and tout[second] <= tout[first]:
        return first
    if tin[second] <= tin[first] and tout[first] <= tout[second]:
        return second
    current = first
    for level in range(up.shape[0] - 1, -1, -1):
        ancestor = up[level, current]
        if ancestor != -1 and not (
            tin[ancestor] <= tin[second] and tout[second] <= tout[ancestor]
        ):
            current = ancestor
    return int(up[0, current])


@njit(cache=True, nogil=True)
def _bulk_prefill_row(  # pragma: no cover - executed as compiled Numba code
    tokens: np.ndarray,
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
    lct_value: np.ndarray,
    lct_lazy_valid: np.ndarray,
) -> tuple[np.ndarray, int, int, int]:
    """Build the final SAM, answer prefixes offline, and seed the LCT."""

    token_count = tokens.shape[0]
    output = np.full(token_count, -1, dtype=np.int64)
    prefix_state = np.empty(token_count, dtype=np.int32)
    last = 0
    size = 1
    edge_count = 0

    for position in range(token_count):
        token = int(tokens[position])
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
            state = suffix_link[state]

        if state == -1:
            suffix_link[current] = 0
        else:
            transition = _find_transition(
                hash_state, hash_token, hash_edge, state, token
            )
            target = int(edge_target[transition])
            if length[state] + 1 == length[target]:
                suffix_link[current] = target
            else:
                if size >= head.shape[0]:
                    raise RuntimeError("suffix automaton state capacity exceeded")
                clone = size
                size += 1
                length[clone] = length[state] + 1
                suffix_link[clone] = suffix_link[target]
                edge = head[target]
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
                    edge = edge_next[edge]
                transition = _find_transition(
                    hash_state, hash_token, hash_edge, state, token
                )
                while (
                    state != -1
                    and transition != -1
                    and edge_target[transition] == target
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
                    state = suffix_link[state]
                    if state != -1:
                        transition = _find_transition(
                            hash_state, hash_token, hash_edge, state, token
                        )
                suffix_link[target] = clone
                suffix_link[current] = clone

        last = current
        prefix_state[position] = current

    first_child = np.full(size, -1, dtype=np.int32)
    next_sibling = np.full(size, -1, dtype=np.int32)
    for node in range(1, size):
        parent = suffix_link[node]
        next_sibling[node] = first_child[parent]
        first_child[parent] = node
    tin = np.empty(size, dtype=np.int32)
    tout = np.empty(size, dtype=np.int32)
    euler_node = np.empty(size, dtype=np.int32)
    dfs_nodes = np.empty(size, dtype=np.int32)
    dfs_next = np.empty(size, dtype=np.int32)
    depth = 0
    timer = 0
    dfs_nodes[0] = 0
    dfs_next[0] = first_child[0]
    tin[0] = timer
    euler_node[timer] = 0
    timer += 1
    while depth >= 0:
        child = dfs_next[depth]
        if child == -1:
            tout[dfs_nodes[depth]] = timer
            depth -= 1
        else:
            dfs_next[depth] = next_sibling[child]
            depth += 1
            dfs_nodes[depth] = child
            dfs_next[depth] = first_child[child]
            tin[child] = timer
            euler_node[timer] = child
            timer += 1

    levels = 1
    span = 1
    while span < size:
        levels += 1
        span <<= 1
    up = np.full((levels, size), -1, dtype=np.int32)
    for node in range(size):
        up[0, node] = suffix_link[node]
    for level in range(1, levels):
        for node in range(size):
            ancestor = up[level - 1, node]
            if ancestor != -1:
                up[level, node] = up[level - 1, ancestor]

    base = 1
    while base < size:
        base <<= 1
    active = np.full(base << 1, -1, dtype=np.int64)
    active_order = np.zeros(size + 1, dtype=np.int32)
    for position in range(token_count):
        node = int(suffix_link[prefix_state[position]])
        if node != -1:
            # Among a DFS-ordered set, one of the predecessor/successor has
            # the deepest LCA with the query node.  Thus two order-statistic
            # lookups replace a path of segment-tree range probes.
            node_tin = int(tin[node])
            preceding_count = _fenwick_prefix(active_order, node_tin + 1)
            best = 0
            if preceding_count > 0:
                preceding_tin = _fenwick_select(active_order, preceding_count)
                best = _tree_lca(up, tin, tout, node, int(euler_node[preceding_tin]))
            before_count = _fenwick_prefix(active_order, node_tin)
            if before_count < position:
                following_tin = _fenwick_select(active_order, before_count + 1)
                candidate = _tree_lca(
                    up, tin, tout, node, int(euler_node[following_tin])
                )
                if length[candidate] > length[best]:
                    best = candidate
            node = best
            source = _range_max(active, base, int(tin[node]), int(tout[node]))
            if node != 0 and source >= 0:
                output[position] = history[source + 1]
        _point_max(active, base, int(tin[prefix_state[position]]), position)
        _fenwick_add(active_order, int(tin[prefix_state[position]]))

    latest_end = np.full(size, -1, dtype=np.int64)
    for position in range(token_count):
        latest_end[prefix_state[position]] = position
    counts = np.zeros(token_count + 1, dtype=np.int32)
    for node in range(size):
        counts[length[node]] += 1
    for index in range(1, counts.shape[0]):
        counts[index] += counts[index - 1]
    order = np.empty(size, dtype=np.int32)
    for node in range(size - 1, -1, -1):
        node_length = length[node]
        counts[node_length] -= 1
        order[counts[node_length]] = node
    for index in range(size - 1, 0, -1):
        node = order[index]
        parent = suffix_link[node]
        if latest_end[node] > latest_end[parent]:
            latest_end[parent] = latest_end[node]
    for node in range(size):
        lct_left[node] = -1
        lct_right[node] = -1
        lct_parent[node] = suffix_link[node]
        lct_value[node] = latest_end[node]
        lct_lazy_valid[node] = 0

    return output, last, size, edge_count


@njit(cache=True, nogil=True)
def _bulk_prefill_kernel(  # pragma: no cover - executed as compiled Numba code
    tokens: np.ndarray,
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
    lct_value: np.ndarray,
    lct_lazy_valid: np.ndarray,
    last: np.ndarray,
    size: np.ndarray,
    edge_count: np.ndarray,
) -> np.ndarray:
    output = np.empty(tokens.shape, dtype=np.int64)
    for batch_index in range(tokens.shape[0]):
        row_output, row_last, row_size, row_edge_count = _bulk_prefill_row(
            tokens[batch_index],
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
            lct_value[batch_index],
            lct_lazy_valid[batch_index],
        )
        output[batch_index] = row_output
        last[batch_index] = row_last
        size[batch_index] = row_size
        edge_count[batch_index] = row_edge_count
    return output


@dataclass
class _StatefulInferenceState:
    """Fixed-capacity, independently batched exact ROSA inference state."""

    batch_size: int
    max_length: int
    native_abi_version: int
    position: int
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
    lct_value: np.ndarray
    lct_lazy: np.ndarray
    lct_lazy_valid: np.ndarray
    lct_stack: np.ndarray
    last: np.ndarray
    size: np.ndarray
    edge_count: np.ndarray
    native_state: Any


def _init_inference_state(
    batch_size: int,
    max_length: int,
) -> _StatefulInferenceState:
    """Allocate a fixed-capacity CPU state suitable for repeated steps."""

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if max_length <= 0:
        raise ValueError("max_length must be > 0")
    max_states = 2 * max_length + 1
    max_edges = 4 * max_length + 1
    # Open addressing needs a power-of-two capacity; at most half the slots
    # can be occupied even if the conservative edge bound is reached.
    hash_capacity = 1 << (2 * max_edges - 1).bit_length()
    state_shape = (batch_size, max_states)
    edge_shape = (batch_size, max_edges)
    hash_shape = (batch_size, hash_capacity)
    suffix_link = np.full(state_shape, -1, dtype=np.int32)
    return _StatefulInferenceState(
        batch_size=batch_size,
        max_length=max_length,
        native_abi_version=1,
        position=0,
        history=np.empty((batch_size, max_length), dtype=np.int64),
        head=np.full(state_shape, -1, dtype=np.int32),
        edge_token=np.empty(edge_shape, dtype=np.int64),
        edge_target=np.empty(edge_shape, dtype=np.int32),
        edge_next=np.empty(edge_shape, dtype=np.int32),
        hash_state=np.full(hash_shape, -1, dtype=np.int32),
        hash_token=np.empty(hash_shape, dtype=np.int64),
        hash_edge=np.empty(hash_shape, dtype=np.int32),
        suffix_link=suffix_link,
        length=np.zeros(state_shape, dtype=np.int32),
        lct_left=np.full(state_shape, -1, dtype=np.int32),
        lct_right=np.full(state_shape, -1, dtype=np.int32),
        lct_parent=np.full(state_shape, -1, dtype=np.int32),
        lct_value=np.full(state_shape, -1, dtype=np.int64),
        lct_lazy=np.empty(state_shape, dtype=np.int64),
        lct_lazy_valid=np.zeros(state_shape, dtype=np.uint8),
        lct_stack=np.empty(state_shape, dtype=np.int32),
        last=np.zeros(batch_size, dtype=np.int32),
        size=np.ones(batch_size, dtype=np.int32),
        edge_count=np.zeros(batch_size, dtype=np.int32),
        native_state=None,
    )


def _native_step(  # pragma: no cover - optional native companion
    state: _StatefulInferenceState,
    cpu_tokens: Tensor,
) -> np.ndarray | None:
    if state.native_state is False:
        return None
    if state.native_state is None:
        try:
            from rosa_native_step import (  # type: ignore[reportMissingImports]
                NativeState,
            )
        except ModuleNotFoundError:
            state.native_state = False
            return None
        state.native_state = NativeState(state)
    return state.native_state.step(cpu_tokens.numpy())


def _native_prefill(  # pragma: no cover - optional native companion
    state: _StatefulInferenceState,
    cpu_tokens: Tensor,
) -> np.ndarray | None:
    if state.native_state is False:
        return None
    if state.native_state is None:
        try:
            from rosa_native_step import (  # type: ignore[reportMissingImports]
                NativeState,
            )
        except ModuleNotFoundError:
            state.native_state = False
            return None
        state.native_state = NativeState(state)
    native_prefill = getattr(state.native_state, "prefill", None)
    if native_prefill is None:
        state.native_state = False
        return None
    return native_prefill(cpu_tokens.numpy())


def _forward_step(state: _StatefulInferenceState, tokens: Tensor) -> Tensor:
    """Consume one token per batch row and return exact top-1 predictions."""

    if tokens.ndim == 0 and state.batch_size == 1:
        tokens = tokens.unsqueeze(0)
    if tokens.ndim != 1 or tokens.shape[0] != state.batch_size:
        raise ValueError("tokens must have shape [batch_size]")
    if state.position >= state.max_length:
        raise RuntimeError("inference state capacity exceeded")
    device = tokens.device
    cpu_tokens = tokens.detach().to(device="cpu", dtype=torch.long).contiguous()
    native_output = _native_step(state, cpu_tokens)
    if native_output is not None:  # pragma: no cover - optional native companion
        return torch.from_numpy(native_output).to(device)
    output = _step_batch_kernel(
        cpu_tokens.numpy(),
        state.position,
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
    state.position += 1
    return torch.from_numpy(output).to(device)


def _prefill(state: _StatefulInferenceState, tokens: Tensor) -> Tensor:
    """Consume a full initial context through one fused compiled replay."""

    if state.position != 0:
        raise RuntimeError("prefill requires an empty inference state")
    if tokens.ndim != 2 or tokens.shape[0] != state.batch_size:
        raise ValueError("tokens must have shape [batch_size, sequence_length]")
    if tokens.shape[1] > state.max_length:
        raise RuntimeError("inference state capacity exceeded")
    device = tokens.device
    if tokens.shape[1] == 0:
        return torch.empty(tokens.shape, dtype=torch.long, device=device)
    cpu_tokens = tokens.detach().to(device="cpu", dtype=torch.long).contiguous()
    native_output = _native_prefill(state, cpu_tokens)
    if native_output is not None:  # pragma: no cover - optional native companion
        return torch.from_numpy(native_output).to(device)
    output_array = _bulk_prefill_kernel(
        cpu_tokens.numpy(),
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
        state.lct_lazy_valid,
        state.last,
        state.size,
        state.edge_count,
    )
    state.position = tokens.shape[1]
    return torch.from_numpy(output_array).to(device)


def predict_exact_stateful(tokens: Tensor) -> Tensor:
    """Return exact top-1 ROSA predictions through the stateful Numba backend."""

    squeeze = tokens.ndim == 1
    if squeeze:
        tokens = tokens.unsqueeze(0)
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape [N] or [B, N]")
    device = tokens.device
    batch_size, length = tokens.shape
    if length == 0:
        output = torch.empty(tokens.shape, dtype=torch.long, device=device)
        return output[0] if squeeze else output
    state = _init_inference_state(batch_size, length)
    cpu_tokens = tokens.detach().to(device="cpu", dtype=torch.long).contiguous()
    output_array = _replay_kernel(
        cpu_tokens.numpy(),
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
    output = torch.from_numpy(output_array).to(device)
    return output[0] if squeeze else output
