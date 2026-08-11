#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace py = pybind11;

namespace {

constexpr size_t kMaximumNativeThreads = 16;

size_t native_thread_count(int64_t rows) {
  if (rows <= 1)
    return 1;
  size_t available = std::thread::hardware_concurrency();
  if (available == 0)
    available = 1;
  size_t requested = available;
  if (const char *value = std::getenv("ROSA_NATIVE_THREADS")) {
    try {
      const std::string text(value);
      if (text.empty() ||
          !std::all_of(text.begin(), text.end(), [](unsigned char character) {
            return character >= '0' && character <= '9';
          }))
        throw std::invalid_argument("thread count must contain only digits");
      size_t consumed = 0;
      const unsigned long parsed = std::stoul(text, &consumed);
      if (consumed == text.size() && parsed > 0)
        requested = static_cast<size_t>(parsed);
      else
        requested = 1;
    } catch (const std::exception &) {
      requested = 1;
    }
  }
  return std::max<size_t>(
      1, std::min({requested, available, kMaximumNativeThreads,
                   static_cast<size_t>(rows)}));
}

// A pool belongs to one native state. Its workers never touch Python and stay
// alive across step/prefill calls; this avoids singleton shutdown ordering and
// lets ROSA_NATIVE_THREADS be selected independently when each state is made.
class RowThreadPool {
public:
  explicit RowThreadPool(int64_t rows) : thread_count_(native_thread_count(rows)) {
    try {
      for (size_t worker = 1; worker < thread_count_; ++worker)
        workers_.emplace_back([this] { worker_loop(); });
    } catch (...) {
      // Without this cleanup, unwinding a partially constructed vector of
      // joinable std::threads calls std::terminate. Retain a serial pool.
      {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
        ++generation_;
      }
      work_ready_.notify_all();
      for (std::thread &worker : workers_)
        worker.join();
      workers_.clear();
      thread_count_ = 1;
      return;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    workers_ready_.wait(lock,
                        [this] { return ready_workers_ == workers_.size(); });
  }

  RowThreadPool(const RowThreadPool &) = delete;
  RowThreadPool &operator=(const RowThreadPool &) = delete;

  size_t worker_count() const { return workers_.size(); }

  ~RowThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
      ++generation_;
    }
    work_ready_.notify_all();
    for (std::thread &worker : workers_)
      worker.join();
  }

  template <typename Function>
  void parallel_for_rows(int64_t rows, int64_t minimum_parallel_rows,
                         Function &&function) {
    if (rows < minimum_parallel_rows || workers_.empty()) {
      for (int64_t row = 0; row < rows; ++row)
        function(row);
      return;
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      next_row_.store(0, std::memory_order_relaxed);
      end_row_ = rows;
      cancelled_.store(false, std::memory_order_relaxed);
      exception_ = nullptr;
      function_ = std::forward<Function>(function);
      active_workers_ = workers_.size();
      ++generation_;
    }
    work_ready_.notify_all();
    run_rows();
    {
      std::unique_lock<std::mutex> lock(mutex_);
      work_done_.wait(lock, [this] { return active_workers_ == 0; });
      function_ = nullptr;
      if (exception_)
        std::rethrow_exception(exception_);
    }
  }

private:
  void capture_exception() {
    cancelled_.store(true, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(mutex_);
    if (!exception_)
      exception_ = std::current_exception();
  }

  void run_rows() {
    try {
      while (!cancelled_.load(std::memory_order_relaxed)) {
        const int64_t row = next_row_.fetch_add(1, std::memory_order_relaxed);
        if (row >= end_row_)
          break;
        function_(row);
      }
    } catch (...) {
      capture_exception();
    }
  }

  void worker_loop() {
    size_t observed_generation = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ++ready_workers_;
      workers_ready_.notify_one();
    }
    for (;;) {
      {
        std::unique_lock<std::mutex> lock(mutex_);
        work_ready_.wait(lock, [this, observed_generation] {
          return stopping_ || generation_ != observed_generation;
        });
        if (stopping_)
          return;
        observed_generation = generation_;
      }
      run_rows();
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (--active_workers_ == 0)
          work_done_.notify_one();
      }
    }
  }

  size_t thread_count_;
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable work_ready_, work_done_, workers_ready_;
  std::function<void(int64_t)> function_;
  std::atomic<int64_t> next_row_{0};
  std::atomic<bool> cancelled_{false};
  int64_t end_row_ = 0;
  size_t generation_ = 0, active_workers_ = 0, ready_workers_ = 0;
  bool stopping_ = false;
  std::exception_ptr exception_;
};

} // namespace

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
    py::array_t<int64_t> output(batch_);
    const int64_t *in = tokens.data();
    int64_t *out = output.mutable_data();
    ensure_pool(64);
    std::unique_lock<std::mutex> call_lock;
    {
      py::gil_scoped_release release;
      call_lock = std::unique_lock<std::mutex>(call_mutex_);
      if (position_ >= max_length_)
        throw std::runtime_error("inference state capacity exceeded");
      parallel_for_rows(64, [&](int64_t b) {
        out[b] = step_row(b, in[b], position_);
      });
    }
    ++position_;
    std::fill(positions_.mutable_data(), positions_.mutable_data() + batch_,
              position_);
    state_.attr("position") = py::int_(position_);
    call_lock.unlock();
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
    py::array_t<int64_t> output(batch_);
    std::fill(output.mutable_data(), output.mutable_data() + batch_, int64_t{-1});
    std::vector<int64_t> next_positions(batch_);
    std::unique_lock<std::mutex> call_lock;
    {
      py::gil_scoped_release release;
      call_lock = std::unique_lock<std::mutex>(call_mutex_);
      for (int64_t b = 0; b < batch_; ++b) {
        if (active[b] && !reset[b] && positions_.data()[b] >= max_length_)
          throw std::runtime_error("inference state capacity exceeded");
      }
      for (int64_t b = 0; b < batch_; ++b) {
        const int64_t current_position =
            active[b] && reset[b] ? 0 : positions_.data()[b];
        next_positions[b] = current_position;
        if (!active[b])
          continue;
        if (reset[b])
          reset_row(b);
        output.mutable_data()[b] =
            step_row(b, tokens.data()[b], current_position);
        next_positions[b] = current_position + 1;
      }
    }
    std::copy(next_positions.begin(), next_positions.end(),
              positions_.mutable_data());
    call_lock.unlock();
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
    const int64_t token_count = tokens.shape(1);
    py::array_t<int64_t> output({batch_, token_count});
    const int64_t *in = tokens.data();
    int64_t *out = output.mutable_data();
    ensure_pool(4);
    std::unique_lock<std::mutex> call_lock;
    {
      py::gil_scoped_release release;
      call_lock = std::unique_lock<std::mutex>(call_mutex_);
      if (position_ != 0)
        throw std::runtime_error("prefill requires an empty inference state");
      if (token_count > max_length_)
        throw std::runtime_error("inference state capacity exceeded");
      if (token_count > 0)
        parallel_for_rows(4, [&](int64_t b) {
          prefill_row(b, in + b * token_count, token_count,
                      out + b * token_count);
        });
    }
    position_ = token_count;
    std::fill(positions_.mutable_data(), positions_.mutable_data() + batch_,
              position_);
    state_.attr("position") = py::int_(position_);
    call_lock.unlock();
    return output;
  }

  int64_t position() const { return position_; }
  py::array_t<int64_t> positions() const { return positions_; }
  size_t worker_count() const {
    return row_pool_ ? row_pool_->worker_count() : 0;
  }

private:
  void ensure_pool(int64_t minimum_parallel_rows) {
    if (batch_ >= minimum_parallel_rows && !row_pool_)
      row_pool_ = std::make_unique<RowThreadPool>(batch_);
  }

  template <typename Function>
  void parallel_for_rows(int64_t minimum_parallel_rows, Function &&function) {
    if (batch_ < minimum_parallel_rows) {
      for (int64_t b = 0; b < batch_; ++b)
        function(b);
      return;
    }
    row_pool_->parallel_for_rows(batch_, minimum_parallel_rows,
                                 std::forward<Function>(function));
  }

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
  std::unique_ptr<RowThreadPool> row_pool_;
  std::mutex call_mutex_;
  bool ragged_mode_ = false;
  int64_t batch_, max_length_, position_, state_capacity_, edge_capacity_,
      hash_capacity_;
};

class NativeCandidateState {
public:
  explicit NativeCandidateState(py::object state) : state_(std::move(state)) {
    if (!py::hasattr(state_, "native_candidate_abi_version") ||
        py::cast<int64_t>(state_.attr("native_candidate_abi_version")) != 1)
      throw py::value_error("unsupported native candidate state ABI");
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
    occurrences_ = bind<int64_t>("occurrences");
    occurrence_size_ = bind<int32_t>("occurrence_size");
    frequency_ = bind<int64_t>("frequency");
    lazy_prefix_ = bind<int64_t>("lazy_prefix");
    lazy_size_ = bind<int32_t>("lazy_size");
    lazy_delta_ = bind<int64_t>("lazy_delta");
    stack_ = bind<int32_t>("lct_stack");
    last_ = bind<int32_t>("last");
    size_ = bind<int32_t>("size");
    edge_count_ = bind<int32_t>("edge_count");
    batch_ = py::cast<int64_t>(state_.attr("batch_size"));
    max_length_ = py::cast<int64_t>(state_.attr("max_length"));
    suffix_k_ = py::cast<int64_t>(state_.attr("suffix_k"));
    occurrences_r_ = py::cast<int64_t>(state_.attr("occurrences_r"));
    position_ = py::cast<int64_t>(state_.attr("position"));
    state_capacity_ = head_.shape(1);
    edge_capacity_ = edge_token_.shape(1);
    hash_capacity_ = hash_state_.shape(1);
    validate_shapes();
    ragged_mode_ = py::hasattr(state_, "ragged_mode") &&
                   py::cast<bool>(state_.attr("ragged_mode"));
    positions_ = py::array_t<int64_t>(batch_);
    std::fill(positions_.mutable_data(), positions_.mutable_data() + batch_,
              position_);
    if (py::hasattr(state_, "positions")) {
      positions_ = bind<int64_t>("positions");
      if (!vector_shape(positions_, batch_))
        throw py::value_error(
            "positions must be contiguous int64 [batch_size]");
    } else if (ragged_mode_) {
      throw py::value_error("ragged candidate state requires positions");
    }
    for (int64_t b = 0; b < batch_; ++b)
      if (positions_.data()[b] < 0 || positions_.data()[b] > max_length_)
        throw py::value_error("candidate positions are outside capacity");
    occupied_slots_.resize(batch_);
    for (int64_t b = 0; b < batch_; ++b)
      for (int64_t slot = 0; slot < hash_capacity_; ++slot)
        if (hash_state_.data()[idx(b, hash_capacity_, slot)] != -1)
          occupied_slots_[b].push_back(static_cast<int32_t>(slot));
  }

  py::tuple step(py::array tokens_object) {
    if (ragged_mode_)
      throw std::runtime_error("uniform step is unavailable on a ragged candidate state");
    if (!py::isinstance<py::array_t<int64_t>>(tokens_object))
      throw py::type_error("tokens must have dtype int64");
    if ((tokens_object.flags() & py::array::c_style) == 0 ||
        tokens_object.ndim() != 1 || tokens_object.shape(0) != batch_)
      throw py::value_error("tokens must be contiguous int64 [batch_size]");
    auto tokens = py::cast<py::array_t<int64_t, py::array::c_style>>(tokens_object);
    const int64_t slots = suffix_k_ * occurrences_r_;
    py::array_t<int64_t> source({batch_, slots}), match_length({batch_, slots}),
        state_id({batch_, slots}), candidate_frequency({batch_, slots});
    py::array_t<int32_t> count(batch_);
    std::fill(source.mutable_data(), source.mutable_data() + batch_ * slots,
              int64_t{-1});
    std::fill(match_length.mutable_data(),
              match_length.mutable_data() + batch_ * slots, int64_t{0});
    std::fill(state_id.mutable_data(), state_id.mutable_data() + batch_ * slots,
              int64_t{-1});
    std::fill(candidate_frequency.mutable_data(),
              candidate_frequency.mutable_data() + batch_ * slots, int64_t{0});
    ensure_pool(64);
    std::unique_lock<std::mutex> call_lock;
    {
      py::gil_scoped_release release;
      call_lock = std::unique_lock<std::mutex>(call_mutex_);
      if (position_ >= max_length_)
        throw std::runtime_error("candidate state capacity exceeded");
      parallel_for_rows(64, [&](int64_t b) {
        count.mutable_data()[b] =
            step_row(b, tokens.data()[b], position_,
                     source.mutable_data() + b * slots,
                     match_length.mutable_data() + b * slots,
                     state_id.mutable_data() + b * slots,
                     candidate_frequency.mutable_data() + b * slots);
      });
    }
    ++position_;
    std::fill(positions_.mutable_data(), positions_.mutable_data() + batch_,
              position_);
    state_.attr("position") = py::int_(position_);
    call_lock.unlock();
    return py::make_tuple(source, match_length, state_id, candidate_frequency,
                          count);
  }

  void reset() {
    if (ragged_mode_)
      throw std::runtime_error("uniform reset is unavailable on a ragged candidate state");
    std::unique_lock<std::mutex> call_lock;
    {
      py::gil_scoped_release release;
      call_lock = std::unique_lock<std::mutex>(call_mutex_);
      for (int64_t b = 0; b < batch_; ++b)
        reset_row(b);
    }
    position_ = 0;
    std::fill(positions_.mutable_data(), positions_.mutable_data() + batch_,
              int64_t{0});
    state_.attr("position") = py::int_(0);
    call_lock.unlock();
  }

  py::tuple step_masked(py::array tokens_object, py::array active_object,
                        py::array reset_object) {
    if (!ragged_mode_)
      throw std::runtime_error("step_masked requires a ragged candidate state");
    if (!py::isinstance<py::array_t<int64_t>>(tokens_object) ||
        (tokens_object.flags() & py::array::c_style) == 0 ||
        tokens_object.ndim() != 1 || tokens_object.shape(0) != batch_)
      throw py::value_error("tokens must be contiguous int64 [batch_size]");
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
      throw py::value_error("active and reset must be contiguous [batch_size]");
    auto tokens = py::cast<py::array_t<int64_t, py::array::c_style>>(tokens_object);
    std::vector<uint8_t> active(batch_), reset(batch_);
    for (int64_t b = 0; b < batch_; ++b) {
      active[b] = active_bool ? static_cast<const bool *>(active_object.data())[b]
                              : static_cast<const uint8_t *>(active_object.data())[b] != 0;
      reset[b] = reset_bool ? static_cast<const bool *>(reset_object.data())[b]
                            : static_cast<const uint8_t *>(reset_object.data())[b] != 0;
    }
    const int64_t slots = suffix_k_ * occurrences_r_;
    py::array_t<int64_t> source({batch_, slots}), match_length({batch_, slots}),
        state_id({batch_, slots}), candidate_frequency({batch_, slots});
    py::array_t<int32_t> count(batch_);
    std::fill(source.mutable_data(), source.mutable_data() + batch_ * slots, int64_t{-1});
    std::fill(match_length.mutable_data(), match_length.mutable_data() + batch_ * slots, int64_t{0});
    std::fill(state_id.mutable_data(), state_id.mutable_data() + batch_ * slots, int64_t{-1});
    std::fill(candidate_frequency.mutable_data(), candidate_frequency.mutable_data() + batch_ * slots, int64_t{0});
    std::fill(count.mutable_data(), count.mutable_data() + batch_, int32_t{0});
    std::vector<int64_t> next_positions(batch_);
    std::unique_lock<std::mutex> call_lock;
    {
      py::gil_scoped_release release;
      call_lock = std::unique_lock<std::mutex>(call_mutex_);
      for (int64_t b = 0; b < batch_; ++b) {
        const int64_t next_position = reset[b] ? 0 : positions_.data()[b];
        if (active[b] && next_position < 0)
          throw std::runtime_error("candidate position must be non-negative");
        if (active[b] && next_position >= max_length_)
          throw std::runtime_error("candidate state capacity exceeded");
      }
      for (int64_t b = 0; b < batch_; ++b) {
        const int64_t current_position =
            active[b] && reset[b] ? 0 : positions_.data()[b];
        next_positions[b] = current_position;
        if (!active[b])
          continue;
        if (reset[b])
          reset_row(b);
        count.mutable_data()[b] = step_row(
            b, tokens.data()[b], current_position,
            source.mutable_data() + b * slots,
            match_length.mutable_data() + b * slots,
            state_id.mutable_data() + b * slots,
            candidate_frequency.mutable_data() + b * slots);
        next_positions[b] = current_position + 1;
      }
    }
    std::copy(next_positions.begin(), next_positions.end(),
              positions_.mutable_data());
    call_lock.unlock();
    return py::make_tuple(source, match_length, state_id, candidate_frequency,
                          count);
  }

  void reset_masked(py::array reset_object) {
    if (!ragged_mode_)
      throw std::runtime_error("reset_masked requires a ragged candidate state");
    const bool reset_bool = py::isinstance<py::array_t<bool>>(reset_object);
    const bool reset_u8 = py::isinstance<py::array_t<uint8_t>>(reset_object);
    if ((!reset_bool && !reset_u8))
      throw py::type_error("reset must have dtype bool or uint8");
    if ((reset_object.flags() & py::array::c_style) == 0 ||
        reset_object.ndim() != 1 || reset_object.shape(0) != batch_)
      throw py::value_error("reset must be contiguous [batch_size]");
    std::vector<uint8_t> reset(batch_);
    std::vector<int64_t> next_positions(batch_);
    for (int64_t b = 0; b < batch_; ++b)
      reset[b] = reset_bool ? static_cast<const bool *>(reset_object.data())[b]
                            : static_cast<const uint8_t *>(reset_object.data())[b] != 0;
    std::unique_lock<std::mutex> call_lock;
    {
      py::gil_scoped_release release;
      call_lock = std::unique_lock<std::mutex>(call_mutex_);
      for (int64_t b = 0; b < batch_; ++b) {
        next_positions[b] = positions_.data()[b];
        if (reset[b]) {
          reset_row(b);
          next_positions[b] = 0;
        }
      }
    }
    std::copy(next_positions.begin(), next_positions.end(),
              positions_.mutable_data());
    call_lock.unlock();
  }

  py::tuple prefill(py::array tokens_object) {
    if (ragged_mode_)
      throw std::runtime_error("prefill is unavailable on a ragged candidate state");
    if (!py::isinstance<py::array_t<int64_t>>(tokens_object))
      throw py::type_error("tokens must have dtype int64");
    if ((tokens_object.flags() & py::array::c_style) == 0 ||
        tokens_object.ndim() != 2 || tokens_object.shape(0) != batch_)
      throw py::value_error(
          "tokens must be contiguous int64 [batch_size, sequence_length]");
    auto tokens = py::cast<py::array_t<int64_t, py::array::c_style>>(tokens_object);
    const int64_t sequence_length = tokens.shape(1);
    const int64_t slots = suffix_k_ * occurrences_r_;
    py::array_t<int64_t> source({batch_, sequence_length, slots}),
        match_length({batch_, sequence_length, slots}),
        state_id({batch_, sequence_length, slots}),
        candidate_frequency({batch_, sequence_length, slots});
    py::array_t<int32_t> count({batch_, sequence_length});
    const int64_t output_size = batch_ * sequence_length * slots;
    std::fill(source.mutable_data(), source.mutable_data() + output_size, int64_t{-1});
    std::fill(match_length.mutable_data(), match_length.mutable_data() + output_size, int64_t{0});
    std::fill(state_id.mutable_data(), state_id.mutable_data() + output_size, int64_t{-1});
    std::fill(candidate_frequency.mutable_data(), candidate_frequency.mutable_data() + output_size, int64_t{0});
    std::fill(count.mutable_data(), count.mutable_data() + batch_ * sequence_length, int32_t{0});
    ensure_pool(4);
    std::unique_lock<std::mutex> call_lock;
    {
      py::gil_scoped_release release;
      call_lock = std::unique_lock<std::mutex>(call_mutex_);
      if (position_ != 0)
        throw std::runtime_error("prefill requires an empty candidate state");
      if (sequence_length > max_length_)
        throw std::runtime_error("candidate state capacity exceeded");
      parallel_for_rows(4, [&](int64_t b) {
        for (int64_t position = 0; position < sequence_length; ++position) {
          const int64_t output_at = (b * sequence_length + position) * slots;
          count.mutable_data()[b * sequence_length + position] = step_row(
              b, tokens.data()[b * sequence_length + position], position,
              source.mutable_data() + output_at,
              match_length.mutable_data() + output_at,
              state_id.mutable_data() + output_at,
              candidate_frequency.mutable_data() + output_at);
        }
      });
    }
    position_ = sequence_length;
    std::fill(positions_.mutable_data(), positions_.mutable_data() + batch_,
              position_);
    state_.attr("position") = py::int_(position_);
    call_lock.unlock();
    return py::make_tuple(source, match_length, state_id, candidate_frequency,
                          count);
  }

  int64_t position() const { return position_; }
  py::array_t<int64_t> positions() const { return positions_; }
  size_t worker_count() const {
    return row_pool_ ? row_pool_->worker_count() : 0;
  }

private:
  void ensure_pool(int64_t minimum_parallel_rows) {
    if (batch_ >= minimum_parallel_rows && !row_pool_)
      row_pool_ = std::make_unique<RowThreadPool>(batch_);
  }

  template <typename Function>
  void parallel_for_rows(int64_t minimum_parallel_rows, Function &&function) {
    if (batch_ < minimum_parallel_rows) {
      for (int64_t b = 0; b < batch_; ++b)
        function(b);
      return;
    }
    row_pool_->parallel_for_rows(batch_, minimum_parallel_rows,
                                 std::forward<Function>(function));
  }

  template <typename T> py::array_t<T, py::array::c_style> bind(const char *name) {
    py::object object = state_.attr(name);
    if (!py::isinstance<py::array_t<T>>(object))
      throw py::type_error(std::string(name) + " has an unexpected dtype");
    auto array = py::cast<py::array_t<T, py::array::c_style>>(object);
    if (!array.writeable())
      throw py::value_error(std::string(name) + " is readonly");
    return array;
  }
  template <typename T>
  bool matrix_shape(const py::array_t<T, py::array::c_style> &a, int64_t rows,
                    int64_t columns) const {
    return a.ndim() == 2 && a.shape(0) == rows && a.shape(1) == columns;
  }
  template <typename T>
  bool vector_shape(const py::array_t<T, py::array::c_style> &a,
                    int64_t length) const {
    return a.ndim() == 1 && a.shape(0) == length;
  }
  void validate_shapes() {
    if (batch_ <= 0 || max_length_ <= 0 || suffix_k_ <= 0 ||
        occurrences_r_ <= 0 || position_ < 0 || position_ > max_length_ ||
        state_capacity_ <= 0 || edge_capacity_ <= 0 || hash_capacity_ <= 0 ||
        (hash_capacity_ & (hash_capacity_ - 1)) != 0)
      throw py::value_error("incompatible CandidateState layout");
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
        occurrences_.ndim() != 3 || occurrences_.shape(0) != batch_ ||
        occurrences_.shape(1) != state_capacity_ ||
        occurrences_.shape(2) != occurrences_r_ ||
        lazy_prefix_.ndim() != 3 || lazy_prefix_.shape(0) != batch_ ||
        lazy_prefix_.shape(1) != state_capacity_ ||
        lazy_prefix_.shape(2) != occurrences_r_ ||
        !matrix_shape(occurrence_size_, batch_, state_capacity_) ||
        !matrix_shape(frequency_, batch_, state_capacity_) ||
        !matrix_shape(lazy_size_, batch_, state_capacity_) ||
        !matrix_shape(lazy_delta_, batch_, state_capacity_) ||
        !matrix_shape(stack_, batch_, state_capacity_) ||
        !vector_shape(last_, batch_) || !vector_shape(size_, batch_) ||
        !vector_shape(edge_count_, batch_))
      throw py::value_error("incompatible CandidateState layout");
    for (int64_t b = 0; b < batch_; ++b)
      if (last_.data()[b] < 0 || last_.data()[b] >= state_capacity_ ||
          size_.data()[b] < 1 || size_.data()[b] > state_capacity_ ||
          edge_count_.data()[b] < 0 || edge_count_.data()[b] > edge_capacity_)
        throw py::value_error("incompatible CandidateState counters");
  }
  inline int64_t idx(int64_t b, int64_t width, int64_t i) const {
    return b * width + i;
  }
  inline int64_t occ_idx(int64_t b, int32_t node, int64_t i) const {
    return (b * state_capacity_ + node) * occurrences_r_ + i;
  }
  inline uint64_t transition_hash(int32_t state, int64_t token) const {
    uint64_t v = static_cast<uint64_t>(token);
    v ^= static_cast<uint64_t>(state) + UINT64_C(0x9E3779B97F4A7C15);
    v = (v ^ (v >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    v = (v ^ (v >> 27)) * UINT64_C(0x94D049BB133111EB);
    return v ^ (v >> 31);
  }
  int32_t find_transition(int64_t b, int32_t state, int64_t token) const {
    int64_t slot = static_cast<int64_t>(transition_hash(state, token) &
                                        static_cast<uint64_t>(hash_capacity_ - 1));
    while (hash_state_.data()[idx(b, hash_capacity_, slot)] != -1) {
      const int64_t at = idx(b, hash_capacity_, slot);
      if (hash_state_.data()[at] == state && hash_token_.data()[at] == token)
        return hash_edge_.data()[at];
      slot = (slot + 1) & (hash_capacity_ - 1);
    }
    return -1;
  }
  int32_t add_transition(int64_t b, int32_t count, int32_t state,
                         int64_t token, int32_t target) {
    if (count >= edge_capacity_)
      throw std::runtime_error("transition capacity exceeded");
    const int64_t edge_at = idx(b, edge_capacity_, count);
    edge_token_.mutable_data()[edge_at] = token;
    edge_target_.mutable_data()[edge_at] = target;
    edge_next_.mutable_data()[edge_at] = head_.data()[idx(b, state_capacity_, state)];
    head_.mutable_data()[idx(b, state_capacity_, state)] = count;
    int64_t slot = static_cast<int64_t>(transition_hash(state, token) &
                                        static_cast<uint64_t>(hash_capacity_ - 1));
    while (hash_state_.data()[idx(b, hash_capacity_, slot)] != -1) {
      const int64_t at = idx(b, hash_capacity_, slot);
      if (hash_state_.data()[at] == state && hash_token_.data()[at] == token)
        throw std::runtime_error("duplicate suffix automaton transition");
      slot = (slot + 1) & (hash_capacity_ - 1);
    }
    const int64_t at = idx(b, hash_capacity_, slot);
    hash_state_.mutable_data()[at] = state;
    hash_token_.mutable_data()[at] = token;
    hash_edge_.mutable_data()[at] = count;
    occupied_slots_[b].push_back(static_cast<int32_t>(slot));
    return count + 1;
  }
  void replace_transition(int64_t b, int32_t state, int64_t token,
                          int32_t target) {
    const int32_t edge = find_transition(b, state, token);
    if (edge == -1)
      throw std::runtime_error("transition not found");
    edge_target_.mutable_data()[idx(b, edge_capacity_, edge)] = target;
  }
  inline bool is_aux_root(int64_t b, int32_t node) const {
    const int64_t at = idx(b, state_capacity_, node);
    const int32_t p = parent_.data()[at];
    return p == -1 || (left_.data()[idx(b, state_capacity_, p)] != node &&
                       right_.data()[idx(b, state_capacity_, p)] != node);
  }
  void apply_tag(int64_t b, int32_t node, const int64_t *prefix,
                 int32_t prefix_size, int64_t delta) {
    if (node == -1)
      return;
    const int32_t take = std::min<int64_t>(prefix_size, occurrences_r_);
    const int64_t at = idx(b, state_capacity_, node);
    const int32_t old_size = occurrence_size_.data()[at];
    const int32_t updated = std::min<int64_t>(occurrences_r_, take + old_size);
    for (int32_t i = updated - 1; i >= take; --i)
      occurrences_.mutable_data()[occ_idx(b, node, i)] =
          occurrences_.data()[occ_idx(b, node, i - take)];
    for (int32_t i = 0; i < take; ++i)
      occurrences_.mutable_data()[occ_idx(b, node, i)] = prefix[i];
    occurrence_size_.mutable_data()[at] = updated;
    frequency_.mutable_data()[at] += delta;
    const int32_t old_lazy = lazy_size_.data()[at];
    const int32_t updated_lazy =
        std::min<int64_t>(occurrences_r_, take + old_lazy);
    for (int32_t i = updated_lazy - 1; i >= take; --i)
      lazy_prefix_.mutable_data()[occ_idx(b, node, i)] =
          lazy_prefix_.data()[occ_idx(b, node, i - take)];
    for (int32_t i = 0; i < take; ++i)
      lazy_prefix_.mutable_data()[occ_idx(b, node, i)] = prefix[i];
    lazy_size_.mutable_data()[at] = updated_lazy;
    lazy_delta_.mutable_data()[at] += delta;
  }
  void push(int64_t b, int32_t node) {
    const int64_t at = idx(b, state_capacity_, node);
    const int32_t size = lazy_size_.data()[at];
    const int64_t delta = lazy_delta_.data()[at];
    if (size != 0 || delta != 0) {
      const int64_t *prefix = lazy_prefix_.data() + occ_idx(b, node, 0);
      apply_tag(b, left_.data()[at], prefix, size, delta);
      apply_tag(b, right_.data()[at], prefix, size, delta);
      lazy_size_.mutable_data()[at] = 0;
      lazy_delta_.mutable_data()[at] = 0;
    }
  }
  void rotate(int64_t b, int32_t node) {
    const auto at = [&](int32_t n) { return idx(b, state_capacity_, n); };
    int32_t *left = left_.mutable_data(), *right = right_.mutable_data(),
            *parent = parent_.mutable_data();
    const int32_t p = parent[at(node)], g = parent[at(p)];
    int32_t middle;
    if (left[at(p)] == node) {
      middle = right[at(node)]; right[at(node)] = p; left[at(p)] = middle;
    } else {
      middle = left[at(node)]; left[at(node)] = p; right[at(p)] = middle;
    }
    if (middle != -1) parent[at(middle)] = p;
    parent[at(p)] = node; parent[at(node)] = g;
    if (g != -1) {
      if (left[at(g)] == p) left[at(g)] = node;
      else if (right[at(g)] == p) right[at(g)] = node;
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
    while (depth > 0) push(b, stack[--depth]);
    while (!is_aux_root(b, node)) {
      const int32_t p = parent_.data()[idx(b, state_capacity_, node)];
      if (!is_aux_root(b, p)) {
        const int32_t g = parent_.data()[idx(b, state_capacity_, p)];
        if ((left_.data()[idx(b, state_capacity_, p)] == node) ==
            (left_.data()[idx(b, state_capacity_, g)] == p)) rotate(b, p);
        else rotate(b, node);
      }
      rotate(b, node);
    }
  }
  void access(int64_t b, int32_t node) {
    int32_t last = -1, current = node;
    while (current != -1) {
      splay(b, current);
      right_.mutable_data()[idx(b, state_capacity_, current)] = last;
      if (last != -1) parent_.mutable_data()[idx(b, state_capacity_, last)] = current;
      last = current;
      current = parent_.data()[idx(b, state_capacity_, current)];
    }
    splay(b, node);
  }
  void materialize(int64_t b, int32_t node) { access(b, node); }
  void cut_parent(int64_t b, int32_t node) {
    materialize(b, node);
    const int64_t at = idx(b, state_capacity_, node);
    const int32_t ancestors = left_.data()[at];
    left_.mutable_data()[at] = -1;
    if (ancestors != -1) parent_.mutable_data()[idx(b, state_capacity_, ancestors)] = -1;
  }
  void link_parent(int64_t b, int32_t node, int32_t represented_parent) {
    materialize(b, node);
    parent_.mutable_data()[idx(b, state_capacity_, node)] = represented_parent;
  }
  void path_write(int64_t b, int32_t node, int64_t position) {
    materialize(b, node);
    apply_tag(b, node, &position, 1, 1);
  }
  int32_t step_row(int64_t b, int64_t token, int64_t position,
                   int64_t *source_out, int64_t *length_out,
                   int64_t *state_out, int64_t *frequency_out) {
    int32_t last = last_.data()[b], size = size_.data()[b],
            edge_count = edge_count_.data()[b];
    history_.mutable_data()[idx(b, max_length_, position)] = token;
    if (size >= state_capacity_) throw std::runtime_error("state capacity exceeded");
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
      const int32_t target = edge_target_.data()[idx(b, edge_capacity_, transition)];
      if (length_.data()[idx(b, state_capacity_, state)] + 1 ==
          length_.data()[idx(b, state_capacity_, target)]) {
        suffix_link_.mutable_data()[idx(b, state_capacity_, current)] = target;
        link_parent(b, current, target);
      } else {
        if (size >= state_capacity_) throw std::runtime_error("state capacity exceeded");
        const int32_t clone = size++;
        length_.mutable_data()[idx(b, state_capacity_, clone)] =
            length_.data()[idx(b, state_capacity_, state)] + 1;
        const int32_t old_parent = suffix_link_.data()[idx(b, state_capacity_, target)];
        suffix_link_.mutable_data()[idx(b, state_capacity_, clone)] = old_parent;
        materialize(b, target);
        const int32_t copied = occurrence_size_.data()[idx(b, state_capacity_, target)];
        occurrence_size_.mutable_data()[idx(b, state_capacity_, clone)] = copied;
        for (int32_t i = 0; i < copied; ++i)
          occurrences_.mutable_data()[occ_idx(b, clone, i)] = occurrences_.data()[occ_idx(b, target, i)];
        frequency_.mutable_data()[idx(b, state_capacity_, clone)] = frequency_.data()[idx(b, state_capacity_, target)];
        lazy_size_.mutable_data()[idx(b, state_capacity_, clone)] = 0;
        lazy_delta_.mutable_data()[idx(b, state_capacity_, clone)] = 0;
        int32_t edge = head_.data()[idx(b, state_capacity_, target)];
        while (edge != -1) {
          edge_count = add_transition(b, edge_count, clone,
              edge_token_.data()[idx(b, edge_capacity_, edge)],
              edge_target_.data()[idx(b, edge_capacity_, edge)]);
          edge = edge_next_.data()[idx(b, edge_capacity_, edge)];
        }
        transition = find_transition(b, state, token);
        while (state != -1 && transition != -1 &&
               edge_target_.data()[idx(b, edge_capacity_, transition)] == target) {
          replace_transition(b, state, token, clone);
          state = suffix_link_.data()[idx(b, state_capacity_, state)];
          if (state != -1) transition = find_transition(b, state, token);
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
    int32_t candidate_count = 0, states_with_history = 0, node = last;
    while (node != -1 && states_with_history < suffix_k_) {
      const int64_t at = idx(b, state_capacity_, node);
      if (length_.data()[at] > 0) {
        materialize(b, node);
        const int32_t node_occurrences = occurrence_size_.data()[at];
        if (node_occurrences > 0) {
          ++states_with_history;
          const int32_t take = std::min<int64_t>(occurrences_r_, node_occurrences);
          for (int32_t occurrence_index = 0; occurrence_index < take; ++occurrence_index) {
            const int64_t source = occurrences_.data()[occ_idx(b, node, occurrence_index)];
            bool duplicate = false;
            for (int32_t seen = 0; seen < candidate_count; ++seen)
              if (source_out[seen] == source) { duplicate = true; break; }
            if (!duplicate) {
              source_out[candidate_count] = source;
              length_out[candidate_count] = length_.data()[at];
              state_out[candidate_count] = node;
              frequency_out[candidate_count] = frequency_.data()[at];
              ++candidate_count;
            }
          }
        }
      }
      node = suffix_link_.data()[idx(b, state_capacity_, node)];
    }
    path_write(b, current, position);
    last_.mutable_data()[b] = last;
    size_.mutable_data()[b] = size;
    edge_count_.mutable_data()[b] = edge_count;
    return candidate_count;
  }
  void reset_row(int64_t b) {
    const int32_t used_states = size_.data()[b];
    for (int32_t node = 0; node < used_states; ++node) {
      const int64_t at = idx(b, state_capacity_, node);
      head_.mutable_data()[at] = -1;
      suffix_link_.mutable_data()[at] = -1;
      length_.mutable_data()[at] = 0;
      left_.mutable_data()[at] = -1;
      right_.mutable_data()[at] = -1;
      parent_.mutable_data()[at] = -1;
      occurrence_size_.mutable_data()[at] = 0;
      frequency_.mutable_data()[at] = 0;
      lazy_size_.mutable_data()[at] = 0;
      lazy_delta_.mutable_data()[at] = 0;
    }
    for (int32_t slot : occupied_slots_[b])
      hash_state_.mutable_data()[idx(b, hash_capacity_, slot)] = -1;
    occupied_slots_[b].clear();
    last_.mutable_data()[b] = 0;
    size_.mutable_data()[b] = 1;
    edge_count_.mutable_data()[b] = 0;
  }

  py::object state_;
  py::array_t<int64_t, py::array::c_style> history_, edge_token_, hash_token_,
      occurrences_, frequency_, lazy_prefix_, lazy_delta_, positions_;
  py::array_t<int32_t, py::array::c_style> head_, edge_target_, edge_next_,
      hash_state_, hash_edge_, suffix_link_, length_, left_, right_, parent_,
      occurrence_size_, lazy_size_, stack_, last_, size_, edge_count_;
  std::vector<std::vector<int32_t>> occupied_slots_;
  std::unique_ptr<RowThreadPool> row_pool_;
  std::mutex call_mutex_;
  int64_t batch_, max_length_, suffix_k_, occurrences_r_, position_,
      state_capacity_, edge_capacity_, hash_capacity_;
  bool ragged_mode_ = false;
};

PYBIND11_MODULE(rosa_native_step, m) {
  m.doc() = "Exact CPU SAM+LCT step prototype (no libtorch calls in core)";
  py::class_<NativeState>(m, "NativeState")
      .def(py::init<py::object>(), py::keep_alive<1, 2>())
      .def("step", &NativeState::step)
      .def("step_masked", &NativeState::step_masked)
      .def("prefill", &NativeState::prefill)
      .def_property_readonly("position", &NativeState::position)
      .def_property_readonly("positions", &NativeState::positions)
      .def_property_readonly("worker_count", &NativeState::worker_count);
  py::class_<NativeCandidateState>(m, "NativeCandidateState")
      .def(py::init<py::object>(), py::keep_alive<1, 2>())
      .def("step", &NativeCandidateState::step)
      .def("reset", &NativeCandidateState::reset)
      .def("step_masked", &NativeCandidateState::step_masked)
      .def("reset_masked", &NativeCandidateState::reset_masked)
      .def("prefill", &NativeCandidateState::prefill)
      .def_property_readonly("position", &NativeCandidateState::position)
      .def_property_readonly("positions", &NativeCandidateState::positions)
      .def_property_readonly("worker_count",
                             &NativeCandidateState::worker_count);
  m.attr("candidate_abi_version") = py::int_(1);
}
