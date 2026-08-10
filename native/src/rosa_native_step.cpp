#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

class NativeState {
public:
  explicit NativeState(py::object state) : state_(std::move(state)) {
    const int64_t abi = py::cast<int64_t>(state_.attr("native_abi_version"));
    if (abi != 1)
      throw py::value_error("unsupported native state ABI");
    history_ = bind<int64_t>("history");
    head_ = bind<int32_t>("head");
    edge_token_ = bind<int64_t>("edge_token");
    edge_target_ = bind<int32_t>("edge_target");
    edge_next_ = bind<int32_t>("edge_next");
    hash_state_ = bind<int32_t>("hash_state");
    hash_token_ = bind<int64_t>("hash_token");
    hash_edge_ = bind<int32_t>("hash_edge");
    suffix_link_ = bind<int32_t>("suffix_link");
    length_ = bind<int32_t>("length");
    left_ = bind<int32_t>("lct_left");
    right_ = bind<int32_t>("lct_right");
    parent_ = bind<int32_t>("lct_parent");
    value_ = bind<int64_t>("lct_value");
    lazy_ = bind<int64_t>("lct_lazy");
    lazy_valid_ = bind<uint8_t>("lct_lazy_valid");
    stack_ = bind<int32_t>("lct_stack");
    last_ = bind<int32_t>("last");
    size_ = bind<int32_t>("size");
    edge_count_ = bind<int32_t>("edge_count");
    batch_ = py::cast<int64_t>(state_.attr("batch_size"));
    max_length_ = py::cast<int64_t>(state_.attr("max_length"));
    position_ = py::cast<int64_t>(state_.attr("position"));
    state_capacity_ = head_.shape(1);
    edge_capacity_ = edge_token_.shape(1);
    hash_capacity_ = hash_state_.shape(1);
    validate_shapes();
    positions_ = py::array_t<int64_t>(batch_);
    std::fill(positions_.mutable_data(), positions_.mutable_data() + batch_,
              position_);
    if (py::hasattr(state_, "positions")) {
      ragged_mode_ = true;
      py::object object = state_.attr("positions");
      if (!py::isinstance<py::array_t<int64_t>>(object))
        throw py::type_error("positions has an unexpected dtype");
      positions_ = py::cast<py::array_t<int64_t, py::array::c_style>>(object);
      if (!positions_.writeable() || !vector_shape(positions_, batch_))
        throw py::value_error("positions must be writable contiguous int64 [batch_size]");
    }
    occupied_slots_.resize(batch_);
    for (int64_t b = 0; b < batch_; ++b) {
      for (int64_t slot = 0; slot < hash_capacity_; ++slot) {
        if (hash_state_.data()[idx(b, hash_capacity_, slot)] != -1)
          occupied_slots_[b].push_back(static_cast<int32_t>(slot));
      }
    }
  }

  py::array_t<int64_t>
  step(py::array_t<int64_t, py::array::c_style | py::array::forcecast> tokens) {
    if (ragged_mode_)
      throw std::runtime_error("uniform step is unavailable on a ragged state");
    if (tokens.ndim() != 1 || tokens.shape(0) != batch_) {
      throw py::value_error("tokens must be contiguous int64 [batch_size]");
    }
    if (position_ >= max_length_) {
      throw std::runtime_error("inference state capacity exceeded");
    }
    py::array_t<int64_t> output(batch_);
    const int64_t *in = tokens.data();
    int64_t *out = output.mutable_data();
    {
      py::gil_scoped_release release;
      for (int64_t b = 0; b < batch_; ++b)
        out[b] = step_row(b, in[b], position_);
    }
    ++position_;
    std::fill(positions_.mutable_data(), positions_.mutable_data() + batch_,
              position_);
    state_.attr("position") = py::int_(position_);
    return output;
  }

  py::array_t<int64_t> step_masked(py::array tokens_object,
                                   py::array active_object,
                                   py::array reset_object) {
    if (!ragged_mode_)
      throw std::runtime_error("step_masked requires a ragged state");
    auto tokens = checked_vector<int64_t>(tokens_object, "tokens", "int64");
    const bool active_bool = py::isinstance<py::array_t<bool>>(active_object);
    const bool active_u8 = py::isinstance<py::array_t<uint8_t>>(active_object);
    const bool reset_bool = py::isinstance<py::array_t<bool>>(reset_object);
    const bool reset_u8 = py::isinstance<py::array_t<uint8_t>>(reset_object);
    if ((!active_bool && !active_u8) || (!reset_bool && !reset_u8))
      throw py::type_error("active and reset must have dtype bool or uint8");
    if ((active_object.flags() & py::array::c_style) == 0 ||
        (reset_object.flags() & py::array::c_style) == 0 ||
        active_object.ndim() != 1 || reset_object.ndim() != 1 ||
        active_object.shape(0) != batch_ || reset_object.shape(0) != batch_)
      throw py::value_error(
          "active and reset must be contiguous [batch_size]");
    std::vector<uint8_t> active(batch_), reset(batch_);
    for (int64_t b = 0; b < batch_; ++b) {
      active[b] = active_bool
                      ? static_cast<const bool *>(active_object.data())[b]
                      : static_cast<const uint8_t *>(active_object.data())[b] != 0;
      reset[b] = reset_bool
                     ? static_cast<const bool *>(reset_object.data())[b]
                     : static_cast<const uint8_t *>(reset_object.data())[b] != 0;
    }
    for (int64_t b = 0; b < batch_; ++b) {
      if (active[b] && !reset[b] && positions_.data()[b] >= max_length_)
        throw std::runtime_error("inference state capacity exceeded");
    }
    py::array_t<int64_t> output(batch_);
    std::fill(output.mutable_data(), output.mutable_data() + batch_, int64_t{-1});
    {
      py::gil_scoped_release release;
      for (int64_t b = 0; b < batch_; ++b) {
        if (!active[b])
          continue;
        if (reset[b])
          reset_row(b);
        const int64_t position = positions_.data()[b];
        output.mutable_data()[b] = step_row(b, tokens.data()[b], position);
        positions_.mutable_data()[b] = position + 1;
      }
    }
    return output;
  }

  py::array_t<int64_t> prefill(py::array tokens_object) {
    if (ragged_mode_)
      throw std::runtime_error("prefill is unavailable on a ragged state");
    if (!py::isinstance<py::array_t<int64_t>>(tokens_object)) {
      throw py::type_error("tokens must have dtype int64");
    }
    if ((tokens_object.flags() & py::array::c_style) == 0) {
      throw py::value_error("tokens must be C-contiguous");
    }
    auto tokens =
        py::cast<py::array_t<int64_t, py::array::c_style>>(tokens_object);
    if (tokens.ndim() != 2 || tokens.shape(0) != batch_) {
      throw py::value_error(
          "tokens must be contiguous int64 [batch_size, sequence_length]");
    }
    if (position_ != 0) {
      throw std::runtime_error("prefill requires an empty inference state");
    }
    const int64_t token_count = tokens.shape(1);
    if (token_count > max_length_) {
      throw std::runtime_error("inference state capacity exceeded");
    }
    py::array_t<int64_t> output({batch_, token_count});
    if (token_count == 0)
      return output;
    const int64_t *in = tokens.data();
    int64_t *out = output.mutable_data();
    {
      py::gil_scoped_release release;
      for (int64_t b = 0; b < batch_; ++b) {
        prefill_row(b, in + b * token_count, token_count,
                    out + b * token_count);
      }
    }
    position_ = token_count;
    std::fill(positions_.mutable_data(), positions_.mutable_data() + batch_,
              position_);
    state_.attr("position") = py::int_(position_);
    return output;
  }

  int64_t position() const { return position_; }
  py::array_t<int64_t> positions() const { return positions_; }

private:
  template <typename T>
  py::array_t<T, py::array::c_style>
  checked_vector(py::array object, const char *name, const char *dtype) const {
    if (!py::isinstance<py::array_t<T>>(object))
      throw py::type_error(std::string(name) + " must have dtype " + dtype);
    if ((object.flags() & py::array::c_style) == 0 || object.ndim() != 1 ||
        object.shape(0) != batch_)
      throw py::value_error(std::string(name) +
                            " must be contiguous [batch_size]");
    return py::cast<py::array_t<T, py::array::c_style>>(object);
  }

  template <typename T>
  py::array_t<T, py::array::c_style> bind(const char *name) {
    py::object object = state_.attr(name);
    if (!py::isinstance<py::array_t<T>>(object)) {
      throw py::type_error(std::string(name) + " has an unexpected dtype");
    }
    auto array = py::cast<py::array_t<T, py::array::c_style>>(object);
    if (!array.writeable())
      throw py::value_error(std::string(name) + " is readonly");
    return array;
  }

  void validate_shapes() {
    if (batch_ <= 0 || max_length_ <= 0 || position_ < 0 ||
        position_ > max_length_ || state_capacity_ <= 0 ||
        edge_capacity_ <= 0 || hash_capacity_ <= 0) {
      throw py::value_error("incompatible _StatefulInferenceState layout");
    }
    if (!matrix_shape(history_, batch_, max_length_) ||
        !matrix_shape(head_, batch_, state_capacity_) ||
        !matrix_shape(edge_token_, batch_, edge_capacity_) ||
        !matrix_shape(edge_target_, batch_, edge_capacity_) ||
        !matrix_shape(edge_next_, batch_, edge_capacity_) ||
        !matrix_shape(hash_state_, batch_, hash_capacity_) ||
        !matrix_shape(hash_token_, batch_, hash_capacity_) ||
        !matrix_shape(hash_edge_, batch_, hash_capacity_) ||
        !matrix_shape(suffix_link_, batch_, state_capacity_) ||
        !matrix_shape(length_, batch_, state_capacity_) ||
        !matrix_shape(left_, batch_, state_capacity_) ||
        !matrix_shape(right_, batch_, state_capacity_) ||
        !matrix_shape(parent_, batch_, state_capacity_) ||
        !matrix_shape(value_, batch_, state_capacity_) ||
        !matrix_shape(lazy_, batch_, state_capacity_) ||
        !matrix_shape(lazy_valid_, batch_, state_capacity_) ||
        !matrix_shape(stack_, batch_, state_capacity_) ||
        !vector_shape(last_, batch_) || !vector_shape(size_, batch_) ||
        !vector_shape(edge_count_, batch_)) {
      throw py::value_error("incompatible _StatefulInferenceState layout");
    }
    if ((hash_capacity_ & (hash_capacity_ - 1)) != 0) {
      throw py::value_error("hash capacity must be a power of two");
    }
    for (int64_t b = 0; b < batch_; ++b) {
      if (last_.data()[b] < 0 || last_.data()[b] >= state_capacity_ ||
          size_.data()[b] < 1 || size_.data()[b] > state_capacity_ ||
          edge_count_.data()[b] < 0 || edge_count_.data()[b] > edge_capacity_) {
        throw py::value_error("incompatible _StatefulInferenceState counters");
      }
    }
  }

  template <typename T>
  bool matrix_shape(const py::array_t<T, py::array::c_style> &array,
                    int64_t rows, int64_t columns) const {
    return array.ndim() == 2 && array.shape(0) == rows &&
           array.shape(1) == columns;
  }

  template <typename T>
  bool vector_shape(const py::array_t<T, py::array::c_style> &array,
                    int64_t length) const {
    return array.ndim() == 1 && array.shape(0) == length;
  }

  inline uint64_t transition_hash(int32_t state, int64_t token) const {
    uint64_t v = static_cast<uint64_t>(token);
    v ^= static_cast<uint64_t>(state) + UINT64_C(0x9E3779B97F4A7C15);
    v = (v ^ (v >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    v = (v ^ (v >> 27)) * UINT64_C(0x94D049BB133111EB);
    return v ^ (v >> 31);
  }

  inline int64_t idx(int64_t b, int64_t width, int64_t i) const {
    return b * width + i;
  }

  int32_t find_transition(int64_t b, int32_t state, int64_t token) const {
    const int32_t *hs = hash_state_.data();
    const int64_t *ht = hash_token_.data();
    const int32_t *he = hash_edge_.data();
    int64_t slot =
        static_cast<int64_t>(transition_hash(state, token) &
                             static_cast<uint64_t>(hash_capacity_ - 1));
    while (hs[idx(b, hash_capacity_, slot)] != -1) {
      const int64_t at = idx(b, hash_capacity_, slot);
      if (hs[at] == state && ht[at] == token)
        return he[at];
      slot = (slot + 1) & (hash_capacity_ - 1);
    }
    return -1;
  }

  int32_t add_transition(int64_t b, int32_t count, int32_t state, int64_t token,
                         int32_t target) {
    if (count >= edge_capacity_)
      throw std::runtime_error("transition capacity exceeded");
    int32_t *head = head_.mutable_data();
    int64_t *et = edge_token_.mutable_data();
    int32_t *eg = edge_target_.mutable_data();
    int32_t *en = edge_next_.mutable_data();
    const int64_t edge_at = idx(b, edge_capacity_, count);
    const int64_t state_at = idx(b, state_capacity_, state);
    et[edge_at] = token;
    eg[edge_at] = target;
    en[edge_at] = head[state_at];
    head[state_at] = count;
    int64_t slot =
        static_cast<int64_t>(transition_hash(state, token) &
                             static_cast<uint64_t>(hash_capacity_ - 1));
    int32_t *hs = hash_state_.mutable_data();
    int64_t *ht = hash_token_.mutable_data();
    int32_t *he = hash_edge_.mutable_data();
    while (hs[idx(b, hash_capacity_, slot)] != -1) {
      const int64_t at = idx(b, hash_capacity_, slot);
      if (hs[at] == state && ht[at] == token)
        throw std::runtime_error("duplicate suffix automaton transition");
      slot = (slot + 1) & (hash_capacity_ - 1);
    }
    const int64_t at = idx(b, hash_capacity_, slot);
    hs[at] = state;
    ht[at] = token;
    he[at] = count;
    occupied_slots_[b].push_back(static_cast<int32_t>(slot));
    return count + 1;
  }

  void reset_row(int64_t b) {
    const int32_t used_states = size_.data()[b];
    for (int32_t state = 0; state < used_states; ++state) {
      const int64_t at = idx(b, state_capacity_, state);
      head_.mutable_data()[at] = -1;
      suffix_link_.mutable_data()[at] = -1;
      length_.mutable_data()[at] = 0;
      left_.mutable_data()[at] = -1;
      right_.mutable_data()[at] = -1;
      parent_.mutable_data()[at] = -1;
      value_.mutable_data()[at] = -1;
      lazy_valid_.mutable_data()[at] = 0;
    }
    for (const int32_t slot : occupied_slots_[b]) {
      hash_state_.mutable_data()[idx(b, hash_capacity_, slot)] = -1;
    }
    occupied_slots_[b].clear();
    last_.mutable_data()[b] = 0;
    size_.mutable_data()[b] = 1;
    edge_count_.mutable_data()[b] = 0;
    positions_.mutable_data()[b] = 0;
  }

  void replace_transition(int64_t b, int32_t state, int64_t token,
                          int32_t target) {
    const int32_t edge = find_transition(b, state, token);
    if (edge == -1)
      throw std::runtime_error("transition not found");
    edge_target_.mutable_data()[idx(b, edge_capacity_, edge)] = target;
  }

  inline bool is_aux_root(int64_t b, int32_t node) const {
    const int32_t *left = left_.data();
    const int32_t *right = right_.data();
    const int32_t *parent = parent_.data();
    const int64_t at = idx(b, state_capacity_, node);
    const int32_t p = parent[at];
    return p == -1 || (left[idx(b, state_capacity_, p)] != node &&
                       right[idx(b, state_capacity_, p)] != node);
  }

  inline void apply(int64_t b, int32_t node, int64_t assigned) {
    if (node == -1)
      return;
    const int64_t at = idx(b, state_capacity_, node);
    value_.mutable_data()[at] = assigned;
    lazy_.mutable_data()[at] = assigned;
    lazy_valid_.mutable_data()[at] = 1;
  }

  inline void push(int64_t b, int32_t node) {
    const int64_t at = idx(b, state_capacity_, node);
    if (lazy_valid_.data()[at]) {
      const int64_t assigned = lazy_.data()[at];
      apply(b, left_.data()[at], assigned);
      apply(b, right_.data()[at], assigned);
      lazy_valid_.mutable_data()[at] = 0;
    }
  }

  inline void rotate(int64_t b, int32_t node) {
    int32_t *left = left_.mutable_data();
    int32_t *right = right_.mutable_data();
    int32_t *parent = parent_.mutable_data();
    const auto at = [&](int32_t n) { return idx(b, state_capacity_, n); };
    const int32_t p = parent[at(node)], g = parent[at(p)];
    int32_t middle;
    if (left[at(p)] == node) {
      middle = right[at(node)];
      right[at(node)] = p;
      left[at(p)] = middle;
    } else {
      middle = left[at(node)];
      left[at(node)] = p;
      right[at(p)] = middle;
    }
    if (middle != -1)
      parent[at(middle)] = p;
    parent[at(p)] = node;
    parent[at(node)] = g;
    if (g != -1) {
      if (left[at(g)] == p)
        left[at(g)] = node;
      else if (right[at(g)] == p)
        right[at(g)] = node;
    }
  }

  void splay(int64_t b, int32_t node) {
    int32_t *stack = stack_.mutable_data() + b * state_capacity_;
    int32_t depth = 0, ancestor = node;
    stack[depth++] = ancestor;
    while (!is_aux_root(b, ancestor)) {
      ancestor = parent_.data()[idx(b, state_capacity_, ancestor)];
      stack[depth++] = ancestor;
    }
    while (depth > 0)
      push(b, stack[--depth]);
    while (!is_aux_root(b, node)) {
      const int32_t p = parent_.data()[idx(b, state_capacity_, node)];
      if (!is_aux_root(b, p)) {
        const int32_t g = parent_.data()[idx(b, state_capacity_, p)];
        if ((left_.data()[idx(b, state_capacity_, p)] == node) ==
            (left_.data()[idx(b, state_capacity_, g)] == p))
          rotate(b, p);
        else
          rotate(b, node);
      }
      rotate(b, node);
    }
  }

  void access(int64_t b, int32_t node) {
    int32_t last = -1, current = node;
    while (current != -1) {
      splay(b, current);
      right_.mutable_data()[idx(b, state_capacity_, current)] = last;
      if (last != -1)
        parent_.mutable_data()[idx(b, state_capacity_, last)] = current;
      last = current;
      current = parent_.data()[idx(b, state_capacity_, current)];
    }
    splay(b, node);
  }

  int64_t point_query(int64_t b, int32_t node) {
    access(b, node);
    return value_.data()[idx(b, state_capacity_, node)];
  }
  void path_assign(int64_t b, int32_t node, int64_t assigned) {
    access(b, node);
    apply(b, node, assigned);
  }
  void cut_parent(int64_t b, int32_t node) {
    access(b, node);
    const int64_t at = idx(b, state_capacity_, node);
    const int32_t ancestors = left_.data()[at];
    left_.mutable_data()[at] = -1;
    if (ancestors != -1)
      parent_.mutable_data()[idx(b, state_capacity_, ancestors)] = -1;
  }
  void link_parent(int64_t b, int32_t node, int32_t represented_parent) {
    access(b, node);
    parent_.mutable_data()[idx(b, state_capacity_, node)] = represented_parent;
  }

  int64_t step_row(int64_t b, int64_t token, int64_t position) {
    int32_t *last_a = last_.mutable_data();
    int32_t *size_a = size_.mutable_data();
    int32_t *edge_count_a = edge_count_.mutable_data();
    int32_t last = last_a[b], size = size_a[b], edge_count = edge_count_a[b];
    history_.mutable_data()[idx(b, max_length_, position)] = token;
    if (size >= state_capacity_)
      throw std::runtime_error("state capacity exceeded");
    const int32_t current = size++;
    length_.mutable_data()[idx(b, state_capacity_, current)] =
        length_.data()[idx(b, state_capacity_, last)] + 1;
    int32_t state = last;
    while (state != -1 && find_transition(b, state, token) == -1) {
      edge_count = add_transition(b, edge_count, state, token, current);
      state = suffix_link_.data()[idx(b, state_capacity_, state)];
    }
    if (state == -1) {
      suffix_link_.mutable_data()[idx(b, state_capacity_, current)] = 0;
      link_parent(b, current, 0);
    } else {
      int32_t transition = find_transition(b, state, token);
      const int32_t target =
          edge_target_.data()[idx(b, edge_capacity_, transition)];
      if (length_.data()[idx(b, state_capacity_, state)] + 1 ==
          length_.data()[idx(b, state_capacity_, target)]) {
        suffix_link_.mutable_data()[idx(b, state_capacity_, current)] = target;
        link_parent(b, current, target);
      } else {
        if (size >= state_capacity_)
          throw std::runtime_error("state capacity exceeded");
        const int32_t clone = size++;
        length_.mutable_data()[idx(b, state_capacity_, clone)] =
            length_.data()[idx(b, state_capacity_, state)] + 1;
        const int32_t old_parent =
            suffix_link_.data()[idx(b, state_capacity_, target)];
        suffix_link_.mutable_data()[idx(b, state_capacity_, clone)] =
            old_parent;
        value_.mutable_data()[idx(b, state_capacity_, clone)] =
            point_query(b, target);
        int32_t edge = head_.data()[idx(b, state_capacity_, target)];
        while (edge != -1) {
          edge_count =
              add_transition(b, edge_count, clone,
                             edge_token_.data()[idx(b, edge_capacity_, edge)],
                             edge_target_.data()[idx(b, edge_capacity_, edge)]);
          edge = edge_next_.data()[idx(b, edge_capacity_, edge)];
        }
        transition = find_transition(b, state, token);
        while (state != -1 && transition != -1 &&
               edge_target_.data()[idx(b, edge_capacity_, transition)] ==
                   target) {
          replace_transition(b, state, token, clone);
          state = suffix_link_.data()[idx(b, state_capacity_, state)];
          if (state != -1)
            transition = find_transition(b, state, token);
        }
        link_parent(b, clone, old_parent);
        cut_parent(b, target);
        suffix_link_.mutable_data()[idx(b, state_capacity_, target)] = clone;
        link_parent(b, target, clone);
        suffix_link_.mutable_data()[idx(b, state_capacity_, current)] = clone;
        link_parent(b, current, clone);
      }
    }
    last = current;
    const int32_t matched =
        suffix_link_.data()[idx(b, state_capacity_, current)];
    int64_t source = -1;
    if (matched != 0)
      source = point_query(b, matched);
    int64_t prediction = -1;
    if (source >= 0)
      prediction = history_.data()[idx(b, max_length_, source + 1)];
    path_assign(b, current, position);
    last_a[b] = last;
    size_a[b] = size;
    edge_count_a[b] = edge_count;
    return prediction;
  }

  void prefill_row(int64_t b, const int64_t *tokens, int64_t token_count,
                   int64_t *output) {
    std::fill(output, output + token_count, int64_t{-1});
    std::vector<int32_t> prefix_state(token_count);
    int32_t last = 0, size = 1, edge_count = 0;

    // Build exactly the same final suffix automaton as the bulk Numba path,
    // deliberately postponing all link-cut-tree work until the final tree is
    // known.
    for (int64_t position = 0; position < token_count; ++position) {
      const int64_t token = tokens[position];
      history_.mutable_data()[idx(b, max_length_, position)] = token;
      if (size >= state_capacity_)
        throw std::runtime_error("suffix automaton state capacity exceeded");
      const int32_t current = size++;
      length_.mutable_data()[idx(b, state_capacity_, current)] =
          length_.data()[idx(b, state_capacity_, last)] + 1;
      int32_t state = last;
      while (state != -1 && find_transition(b, state, token) == -1) {
        edge_count = add_transition(b, edge_count, state, token, current);
        state = suffix_link_.data()[idx(b, state_capacity_, state)];
      }
      if (state == -1) {
        suffix_link_.mutable_data()[idx(b, state_capacity_, current)] = 0;
      } else {
        int32_t transition = find_transition(b, state, token);
        const int32_t target =
            edge_target_.data()[idx(b, edge_capacity_, transition)];
        if (length_.data()[idx(b, state_capacity_, state)] + 1 ==
            length_.data()[idx(b, state_capacity_, target)]) {
          suffix_link_.mutable_data()[idx(b, state_capacity_, current)] =
              target;
        } else {
          if (size >= state_capacity_)
            throw std::runtime_error(
                "suffix automaton state capacity exceeded");
          const int32_t clone = size++;
          length_.mutable_data()[idx(b, state_capacity_, clone)] =
              length_.data()[idx(b, state_capacity_, state)] + 1;
          suffix_link_.mutable_data()[idx(b, state_capacity_, clone)] =
              suffix_link_.data()[idx(b, state_capacity_, target)];
          int32_t edge = head_.data()[idx(b, state_capacity_, target)];
          while (edge != -1) {
            edge_count = add_transition(
                b, edge_count, clone,
                edge_token_.data()[idx(b, edge_capacity_, edge)],
                edge_target_.data()[idx(b, edge_capacity_, edge)]);
            edge = edge_next_.data()[idx(b, edge_capacity_, edge)];
          }
          transition = find_transition(b, state, token);
          while (state != -1 && transition != -1 &&
                 edge_target_.data()[idx(b, edge_capacity_, transition)] ==
                     target) {
            replace_transition(b, state, token, clone);
            state = suffix_link_.data()[idx(b, state_capacity_, state)];
            if (state != -1)
              transition = find_transition(b, state, token);
          }
          suffix_link_.mutable_data()[idx(b, state_capacity_, target)] = clone;
          suffix_link_.mutable_data()[idx(b, state_capacity_, current)] = clone;
        }
      }
      last = current;
      prefix_state[position] = current;
    }

    std::vector<int32_t> first_child(size, -1), next_sibling(size, -1);
    for (int32_t node = 1; node < size; ++node) {
      const int32_t p = suffix_link_.data()[idx(b, state_capacity_, node)];
      next_sibling[node] = first_child[p];
      first_child[p] = node;
    }
    std::vector<int32_t> tin(size), tout(size), euler_node(size),
        dfs_nodes(size), dfs_next(size);
    int32_t depth = 0, timer = 0;
    dfs_nodes[0] = 0;
    dfs_next[0] = first_child[0];
    tin[0] = timer;
    euler_node[timer++] = 0;
    while (depth >= 0) {
      const int32_t child = dfs_next[depth];
      if (child == -1) {
        tout[dfs_nodes[depth]] = timer;
        --depth;
      } else {
        dfs_next[depth] = next_sibling[child];
        ++depth;
        dfs_nodes[depth] = child;
        dfs_next[depth] = first_child[child];
        tin[child] = timer;
        euler_node[timer++] = child;
      }
    }

    int32_t levels = 1;
    for (int32_t span = 1; span < size; span <<= 1)
      ++levels;
    std::vector<int32_t> up(static_cast<size_t>(levels) * size, -1);
    const auto up_at = [size](int32_t level, int32_t node) {
      return static_cast<size_t>(level) * size + node;
    };
    for (int32_t node = 0; node < size; ++node)
      up[up_at(0, node)] = suffix_link_.data()[idx(b, state_capacity_, node)];
    for (int32_t level = 1; level < levels; ++level) {
      for (int32_t node = 0; node < size; ++node) {
        const int32_t ancestor = up[up_at(level - 1, node)];
        if (ancestor != -1)
          up[up_at(level, node)] = up[up_at(level - 1, ancestor)];
      }
    }
    const auto lca = [&](int32_t first, int32_t second) {
      if (tin[first] <= tin[second] && tout[second] <= tout[first])
        return first;
      if (tin[second] <= tin[first] && tout[first] <= tout[second])
        return second;
      int32_t current = first;
      for (int32_t level = levels - 1; level >= 0; --level) {
        const int32_t ancestor = up[up_at(level, current)];
        if (ancestor != -1 &&
            !(tin[ancestor] <= tin[second] && tout[second] <= tout[ancestor]))
          current = ancestor;
      }
      return up[up_at(0, current)];
    };

    int32_t base = 1;
    while (base < size)
      base <<= 1;
    std::vector<int64_t> active(static_cast<size_t>(base) * 2, -1);
    std::vector<int32_t> fenwick(size + 1, 0);
    const auto fenwick_prefix = [&](int32_t index) {
      int32_t total = 0;
      while (index > 0) {
        total += fenwick[index];
        index -= index & -index;
      }
      return total;
    };
    const auto fenwick_select = [&](int32_t rank) {
      int32_t node = 0, step = 1;
      while ((step << 1) <= size)
        step <<= 1;
      while (step != 0) {
        const int32_t candidate = node + step;
        if (candidate <= size && fenwick[candidate] < rank) {
          node = candidate;
          rank -= fenwick[candidate];
        }
        step >>= 1;
      }
      return node;
    };
    const auto range_max = [&](int32_t left, int32_t right) {
      int64_t result = -1;
      for (left += base, right += base; left < right; left >>= 1, right >>= 1) {
        if (left & 1)
          result = std::max(result, active[left++]);
        if (right & 1)
          result = std::max(result, active[--right]);
      }
      return result;
    };

    for (int64_t position = 0; position < token_count; ++position) {
      int32_t node =
          suffix_link_.data()[idx(b, state_capacity_, prefix_state[position])];
      if (node != -1) {
        const int32_t node_tin = tin[node];
        const int32_t preceding_count = fenwick_prefix(node_tin + 1);
        int32_t best = 0;
        if (preceding_count > 0)
          best = lca(node, euler_node[fenwick_select(preceding_count)]);
        const int32_t before_count = fenwick_prefix(node_tin);
        if (before_count < position) {
          const int32_t candidate =
              lca(node, euler_node[fenwick_select(before_count + 1)]);
          if (length_.data()[idx(b, state_capacity_, candidate)] >
              length_.data()[idx(b, state_capacity_, best)])
            best = candidate;
        }
        node = best;
        const int64_t source = range_max(tin[node], tout[node]);
        if (node != 0 && source >= 0)
          output[position] = history_.data()[idx(b, max_length_, source + 1)];
      }
      int32_t tree_node = base + tin[prefix_state[position]];
      active[tree_node] = std::max(active[tree_node], position);
      while ((tree_node >>= 1) != 0)
        active[tree_node] =
            std::max(active[tree_node << 1], active[(tree_node << 1) | 1]);
      for (int32_t index = tin[prefix_state[position]] + 1; index <= size;
           index += index & -index)
        ++fenwick[index];
    }

    std::vector<int64_t> latest_end(size, -1);
    for (int64_t position = 0; position < token_count; ++position)
      latest_end[prefix_state[position]] = position;
    std::vector<int32_t> counts(token_count + 1, 0), order(size);
    for (int32_t node = 0; node < size; ++node)
      ++counts[length_.data()[idx(b, state_capacity_, node)]];
    for (size_t i = 1; i < counts.size(); ++i)
      counts[i] += counts[i - 1];
    for (int32_t node = size - 1; node >= 0; --node) {
      const int32_t node_length = length_.data()[idx(b, state_capacity_, node)];
      order[--counts[node_length]] = node;
    }
    for (int32_t index = size - 1; index > 0; --index) {
      const int32_t node = order[index];
      const int32_t p = suffix_link_.data()[idx(b, state_capacity_, node)];
      latest_end[p] = std::max(latest_end[p], latest_end[node]);
    }
    for (int32_t node = 0; node < size; ++node) {
      const int64_t at = idx(b, state_capacity_, node);
      left_.mutable_data()[at] = -1;
      right_.mutable_data()[at] = -1;
      parent_.mutable_data()[at] = suffix_link_.data()[at];
      value_.mutable_data()[at] = latest_end[node];
      lazy_valid_.mutable_data()[at] = 0;
    }
    last_.mutable_data()[b] = last;
    size_.mutable_data()[b] = size;
    edge_count_.mutable_data()[b] = edge_count;
  }

  py::object state_;
  py::array_t<int64_t, py::array::c_style> history_, edge_token_, hash_token_,
      value_, lazy_;
  py::array_t<int32_t, py::array::c_style> head_, edge_target_, edge_next_,
      hash_state_, hash_edge_, suffix_link_, length_, left_, right_, parent_,
      stack_, last_, size_, edge_count_;
  py::array_t<uint8_t, py::array::c_style> lazy_valid_;
  py::array_t<int64_t, py::array::c_style> positions_;
  std::vector<std::vector<int32_t>> occupied_slots_;
  bool ragged_mode_ = false;
  int64_t batch_, max_length_, position_, state_capacity_, edge_capacity_,
      hash_capacity_;
};

PYBIND11_MODULE(rosa_native_step, m) {
  m.doc() = "Exact CPU SAM+LCT step prototype (no libtorch calls in core)";
  py::class_<NativeState>(m, "NativeState")
      .def(py::init<py::object>(), py::keep_alive<1, 2>())
      .def("step", &NativeState::step)
      .def("step_masked", &NativeState::step_masked)
      .def("prefill", &NativeState::prefill)
      .def_property_readonly("position", &NativeState::position)
      .def_property_readonly("positions", &NativeState::positions);
}
