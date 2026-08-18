#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
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

uint32_t leading_zero_count64(uint64_t value) noexcept {
  if (value == 0)
    return 64;
#if defined(_MSC_VER)
  uint32_t count = 0;
  if ((value & UINT64_C(0xffffffff00000000)) == 0) {
    count += 32;
    value <<= 32;
  }
  if ((value & UINT64_C(0xffff000000000000)) == 0) {
    count += 16;
    value <<= 16;
  }
  if ((value & UINT64_C(0xff00000000000000)) == 0) {
    count += 8;
    value <<= 8;
  }
  if ((value & UINT64_C(0xf000000000000000)) == 0) {
    count += 4;
    value <<= 4;
  }
  if ((value & UINT64_C(0xc000000000000000)) == 0) {
    count += 2;
    value <<= 2;
  }
  return count + ((value & UINT64_C(0x8000000000000000)) == 0 ? 1 : 0);
#else
  return static_cast<uint32_t>(__builtin_clzll(value));
#endif
}

uint32_t trailing_zero_count64(uint64_t value) noexcept {
  if (value == 0)
    return 64;
#if defined(_MSC_VER)
  uint32_t count = 0;
  if ((value & UINT64_C(0x00000000ffffffff)) == 0) {
    count += 32;
    value >>= 32;
  }
  if ((value & UINT64_C(0x000000000000ffff)) == 0) {
    count += 16;
    value >>= 16;
  }
  if ((value & UINT64_C(0x00000000000000ff)) == 0) {
    count += 8;
    value >>= 8;
  }
  if ((value & UINT64_C(0x000000000000000f)) == 0) {
    count += 4;
    value >>= 4;
  }
  if ((value & UINT64_C(0x0000000000000003)) == 0) {
    count += 2;
    value >>= 2;
  }
  return count + ((value & UINT64_C(0x1)) == 0 ? 1 : 0);
#else
  return static_cast<uint32_t>(__builtin_ctzll(value));
#endif
}

uint32_t population_count64(uint64_t value) noexcept {
#if defined(_MSC_VER)
  value -= (value >> 1) & UINT64_C(0x5555555555555555);
  value = (value & UINT64_C(0x3333333333333333)) +
          ((value >> 2) & UINT64_C(0x3333333333333333));
  value = (value + (value >> 4)) & UINT64_C(0x0f0f0f0f0f0f0f0f);
  return static_cast<uint32_t>((value * UINT64_C(0x0101010101010101)) >> 56);
#else
  return static_cast<uint32_t>(__builtin_popcountll(value));
#endif
}

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
  size_t storage_bytes() const noexcept {
    return sizeof(*this) + workers_.capacity() * sizeof(std::thread);
  }

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
    const int64_t slots = suffix_k_ * occurrences_r_;
    py::array_t<int64_t> source({batch_, slots}), match_length({batch_, slots}),
        state_id({batch_, slots}), candidate_frequency({batch_, slots});
    py::array_t<int32_t> count(batch_);
    step_into(tokens_object, source, match_length, state_id,
              candidate_frequency, count);
    return py::make_tuple(source, match_length, state_id, candidate_frequency,
                          count);
  }

  void step_into(py::array tokens_object, py::array source_object,
                 py::array match_length_object, py::array state_id_object,
                 py::array candidate_frequency_object,
                 py::array count_object) {
    if (ragged_mode_)
      throw std::runtime_error("uniform step is unavailable on a ragged candidate state");
    const int64_t slots = suffix_k_ * occurrences_r_;
    auto tokens = checked_input<int64_t>(tokens_object, "tokens", {batch_});
    auto source = checked_output<int64_t>(source_object, "source", {batch_, slots});
    auto match_length = checked_output<int64_t>(
        match_length_object, "length", {batch_, slots});
    auto state_id = checked_output<int64_t>(state_id_object, "state", {batch_, slots});
    auto candidate_frequency = checked_output<int64_t>(
        candidate_frequency_object, "frequency", {batch_, slots});
    auto count = checked_output<int32_t>(count_object, "count", {batch_});
    validate_disjoint({tokens_object, source_object, match_length_object,
                       state_id_object, candidate_frequency_object,
                       count_object});
    validate_runtime_positions();
    ensure_pool(64);
    std::unique_lock<std::mutex> call_lock;
    {
      py::gil_scoped_release release;
      call_lock = std::unique_lock<std::mutex>(call_mutex_);
      if (position_ >= max_length_)
        throw std::runtime_error("candidate state capacity exceeded");
      std::fill(source.mutable_data(), source.mutable_data() + batch_ * slots,
              int64_t{-1});
      std::fill(match_length.mutable_data(),
              match_length.mutable_data() + batch_ * slots, int64_t{0});
      std::fill(state_id.mutable_data(), state_id.mutable_data() + batch_ * slots,
              int64_t{-1});
      std::fill(candidate_frequency.mutable_data(),
              candidate_frequency.mutable_data() + batch_ * slots, int64_t{0});
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
    if (!py::isinstance<py::array_t<int64_t>>(tokens_object) ||
        tokens_object.ndim() != 2)
      throw py::type_error("tokens must be a two-dimensional int64 array");
    const int64_t sequence_length = tokens_object.shape(1);
    const int64_t slots = suffix_k_ * occurrences_r_;
    py::array_t<int64_t> source({batch_, sequence_length, slots}),
        match_length({batch_, sequence_length, slots}),
        state_id({batch_, sequence_length, slots}),
        candidate_frequency({batch_, sequence_length, slots});
    py::array_t<int32_t> count({batch_, sequence_length});
    prefill_into(tokens_object, source, match_length, state_id,
                 candidate_frequency, count);
    return py::make_tuple(source, match_length, state_id, candidate_frequency,
                          count);
  }

  void prefill_into(py::array tokens_object, py::array source_object,
                    py::array match_length_object, py::array state_id_object,
                    py::array candidate_frequency_object,
                    py::array count_object) {
    if (ragged_mode_)
      throw std::runtime_error("prefill is unavailable on a ragged candidate state");
    if (!py::isinstance<py::array_t<int64_t>>(tokens_object) ||
        tokens_object.ndim() != 2 || tokens_object.shape(0) != batch_)
      throw py::value_error("tokens must be contiguous int64 [batch_size, sequence_length]");
    const int64_t sequence_length = tokens_object.shape(1);
    const int64_t slots = suffix_k_ * occurrences_r_;
    auto tokens = checked_input<int64_t>(tokens_object, "tokens",
                                         {batch_, sequence_length});
    auto source = checked_output<int64_t>(
        source_object, "source", {batch_, sequence_length, slots});
    auto match_length = checked_output<int64_t>(
        match_length_object, "length", {batch_, sequence_length, slots});
    auto state_id = checked_output<int64_t>(
        state_id_object, "state", {batch_, sequence_length, slots});
    auto candidate_frequency = checked_output<int64_t>(
        candidate_frequency_object, "frequency",
        {batch_, sequence_length, slots});
    auto count = checked_output<int32_t>(count_object, "count",
                                         {batch_, sequence_length});
    validate_disjoint({tokens_object, source_object, match_length_object,
                       state_id_object, candidate_frequency_object,
                       count_object});
    validate_runtime_positions();
    const int64_t output_size = batch_ * sequence_length * slots;
    ensure_pool(4);
    std::unique_lock<std::mutex> call_lock;
    {
      py::gil_scoped_release release;
      call_lock = std::unique_lock<std::mutex>(call_mutex_);
      if (position_ != 0)
        throw std::runtime_error("prefill requires an empty candidate state");
      if (sequence_length > max_length_)
        throw std::runtime_error("candidate state capacity exceeded");
      std::fill(source.mutable_data(), source.mutable_data() + output_size, int64_t{-1});
      std::fill(match_length.mutable_data(), match_length.mutable_data() + output_size, int64_t{0});
      std::fill(state_id.mutable_data(), state_id.mutable_data() + output_size, int64_t{-1});
      std::fill(candidate_frequency.mutable_data(), candidate_frequency.mutable_data() + output_size, int64_t{0});
      std::fill(count.mutable_data(), count.mutable_data() + batch_ * sequence_length, int32_t{0});
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
  }

  int64_t position() const { return position_; }
  py::array_t<int64_t> positions() const { return positions_; }
  size_t worker_count() const {
    return row_pool_ ? row_pool_->worker_count() : 0;
  }

private:
  template <typename T>
  py::array_t<T, py::array::c_style>
  checked_input(py::array object, const char *name,
                std::initializer_list<int64_t> shape) const {
    if (!py::isinstance<py::array_t<T>>(object))
      throw py::type_error(std::string(name) + " has an unexpected dtype");
    if ((object.flags() & py::array::c_style) == 0 ||
        object.ndim() != static_cast<int64_t>(shape.size()))
      throw py::value_error(std::string(name) + " must be C-contiguous with the expected shape");
    int64_t dimension = 0;
    for (const int64_t extent : shape)
      if (object.shape(dimension++) != extent)
        throw py::value_error(std::string(name) + " must be C-contiguous with the expected shape");
    return py::cast<py::array_t<T, py::array::c_style>>(object);
  }

  template <typename T>
  py::array_t<T, py::array::c_style>
  checked_output(py::array object, const char *name,
                 std::initializer_list<int64_t> shape) const {
    auto output = checked_input<T>(object, name, shape);
    if (!output.writeable())
      throw py::value_error(std::string(name) + " must be writable");
    return output;
  }

  static bool overlaps(const py::array &first, const py::array &second) {
    if (first.nbytes() == 0 || second.nbytes() == 0)
      return false;
    const auto first_begin = reinterpret_cast<uintptr_t>(first.data());
    const auto second_begin = reinterpret_cast<uintptr_t>(second.data());
    const auto first_end = first_begin + static_cast<uintptr_t>(first.nbytes());
    const auto second_end = second_begin + static_cast<uintptr_t>(second.nbytes());
    return first_begin < second_end && second_begin < first_end;
  }

  void validate_disjoint(std::initializer_list<py::array> call_arrays) const {
    const std::vector<py::array> arrays(call_arrays);
    for (size_t first = 0; first < arrays.size(); ++first)
      for (size_t second = first + 1; second < arrays.size(); ++second)
        if (overlaps(arrays[first], arrays[second]))
          throw py::value_error("tokens and output buffers must not overlap");
    const py::array state_arrays[] = {
        history_, head_, edge_token_, edge_target_, edge_next_, hash_state_,
        hash_token_, hash_edge_, suffix_link_, length_, left_, right_, parent_,
        occurrences_, occurrence_size_, frequency_, lazy_prefix_, lazy_size_,
        lazy_delta_, stack_, last_, size_, edge_count_, positions_};
    for (const py::array &array : arrays)
      for (const py::array &state_array : state_arrays)
        if (overlaps(array, state_array))
          throw py::value_error("tokens and output buffers must not overlap candidate state storage");
  }

  void validate_runtime_positions() const {
    for (int64_t b = 0; b < batch_; ++b) {
      const int64_t position = positions_.data()[b];
      if (position < 0 || position > max_length_)
        throw py::value_error("candidate positions are outside capacity");
      if (!ragged_mode_ && position != position_)
        throw py::value_error("uniform candidate positions are inconsistent");
    }
  }

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

// An owning implementation of the exact reversed-prefix RLBWT prototype.
// Unlike NativeState and NativeCandidateState, this class does not borrow any
// Python-owned storage.  Each row is independent, so the existing persistent
// row pool can safely update rows in parallel.
class NativeRLBWTState {
public:
  NativeRLBWTState(int64_t batch_size, int64_t max_length)
      : NativeRLBWTState(batch_size, max_length, 0, 0, 0) {}

  NativeRLBWTState(int64_t batch_size, int64_t max_length, uint32_t lanes,
                   uint64_t seed)
      : NativeRLBWTState(batch_size, max_length, lanes, seed, 0) {}

protected:
  NativeRLBWTState(int64_t batch_size, int64_t max_length, uint32_t lanes,
                   uint64_t seed, uint32_t vocabulary_size)
      : batch_(batch_size), max_length_(max_length), lanes_(lanes), seed_(seed),
        vocabulary_size_(vocabulary_size), identity_codes_(vocabulary_size != 0) {
    if (batch_ <= 0)
      throw py::value_error("batch_size must be > 0");
    if (max_length_ <= 0)
      throw py::value_error("max_length must be > 0");
    if (static_cast<uint64_t>(max_length_) >=
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))
      throw py::value_error("max_length must be < UINT32_MAX");
    if (lanes_ != 0 && lanes_ != 2 && lanes_ != 3)
      throw py::value_error("lanes must be 2 or 3");
    if (identity_codes_ && (vocabulary_size_ < 1 || vocabulary_size_ > 256))
      throw py::value_error("vocabulary_size must be in [1, 256]");
    uint64_t splitmix_state = seed_;
    for (uint32_t lane = 0; lane < lanes_; ++lane) {
      bases_[lane] = splitmix64(splitmix_state) | uint64_t{1};
      if (bases_[lane] < 257)
        bases_[lane] += 256;
      while (std::find(bases_.begin(), bases_.begin() + lane, bases_[lane]) !=
             bases_.begin() + lane) {
        bases_[lane] += 2;
        if (bases_[lane] < 257)
          bases_[lane] += 256;
      }
      powers_[lane].resize(static_cast<size_t>(max_length_) + 1);
      powers_[lane][0] = 1;
      for (size_t index = 1; index < powers_[lane].size(); ++index)
        powers_[lane][index] = powers_[lane][index - 1] * bases_[lane];
    }
    rows_.reserve(static_cast<size_t>(batch_));
    for (int64_t b = 0; b < batch_; ++b)
      rows_.emplace_back(max_length_, identity_codes_);
    if (lanes_ != 0) {
      row_hashes_.resize(static_cast<size_t>(batch_));
      for (auto &row_hashes : row_hashes_)
        for (uint32_t lane = 0; lane < lanes_; ++lane)
          row_hashes[lane].resize(static_cast<size_t>(max_length_) + 1);
    }
  }

public:

  py::array_t<int64_t> step(py::array tokens_object) {
    auto tokens = checked_tokens(tokens_object, 1);
    py::array_t<int64_t> output(batch_);
    const int64_t *input = tokens.data();
    int64_t *result = output.mutable_data();
    {
      py::gil_scoped_release release;
      std::lock_guard<std::mutex> lock(call_mutex_);
      if (position_ >= max_length_)
        throw std::runtime_error("inference state capacity exceeded");
      ensure_pool(128);
      parallel_for_rows(128, [&](int64_t b) {
        result[b] = step_row(rows_[static_cast<size_t>(b)], position_, input[b]);
      });
      ++position_;
    }
    return output;
  }

  py::array_t<int64_t> prefill(py::array tokens_object) {
    auto tokens = checked_tokens(tokens_object, 2);
    if (tokens.shape(0) != batch_)
      throw py::value_error(
          "tokens must be contiguous int64 [batch_size, sequence_length]");
    const int64_t count = tokens.shape(1);
    py::array_t<int64_t> output({batch_, count});
    const int64_t *input = tokens.data();
    int64_t *result = output.mutable_data();
    {
      py::gil_scoped_release release;
      std::lock_guard<std::mutex> lock(call_mutex_);
      if (position_ != 0)
        throw std::runtime_error("prefill requires an empty inference state");
      if (count > max_length_)
        throw std::runtime_error("inference state capacity exceeded");
      if (count != 0) {
        ensure_pool(4);
        parallel_for_rows(4, [&](int64_t b) {
          Row &row = rows_[static_cast<size_t>(b)];
          const int64_t *row_input = input + b * count;
          int64_t *row_result = result + b * count;
          // Keep the token loop inside the row job: no transposes, temporary
          // columns, or repeated calls through the Python/C++ boundary.
          for (int64_t position = 0; position < count; ++position)
            row_result[position] =
                step_row(row, position, row_input[position]);
        });
      }
      position_ = count;
    }
    return output;
  }

  // Chunked long-context ingestion.  Unlike the compatibility `prefill`,
  // this method deliberately resumes at the current position so callers can
  // bound their temporary token/output buffers while the owned index grows.
  py::array_t<int64_t> prefill_append(py::array tokens_object) {
    auto tokens = checked_tokens(tokens_object, 2);
    const int64_t count = tokens.shape(1);
    py::array_t<int64_t> output({batch_, count});
    const int64_t *input = tokens.data();
    int64_t *result = output.mutable_data();
    {
      py::gil_scoped_release release;
      std::lock_guard<std::mutex> lock(call_mutex_);
      if (count > max_length_ - position_)
        throw std::runtime_error("inference state capacity exceeded");
      const int64_t base = position_;
      if (count != 0) {
        ensure_pool(4);
        parallel_for_rows(4, [&](int64_t b) {
          Row &row = rows_[static_cast<size_t>(b)];
          const int64_t *row_input = input + b * count;
          int64_t *row_result = result + b * count;
          for (int64_t offset = 0; offset < count; ++offset)
            row_result[offset] =
                step_row(row, base + offset, row_input[offset]);
        });
      }
      position_ += count;
    }
    return output;
  }

  void reset() {
    py::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(call_mutex_);
    // Prepare every allocation first so reset cannot leave a partially reset
    // batch. Histories release all pages and return to PACKED4 during commit.
    std::vector<UnifiedSequence> reset_sequences;
    reset_sequences.reserve(rows_.size());
    for (size_t row = 0; row < rows_.size(); ++row) {
      reset_sequences.emplace_back(static_cast<size_t>(max_length_) + 1);
    }
    for (size_t row = 0; row < rows_.size(); ++row)
      rows_[row].reset(std::move(reset_sequences[row]));
    for (auto &row_hashes : row_hashes_)
      for (uint32_t lane = 0; lane < lanes_; ++lane)
        std::fill(row_hashes[lane].begin(), row_hashes[lane].end(), uint64_t{0});
    position_ = 0;
  }

  int64_t position() const {
    std::lock_guard<std::mutex> lock(call_mutex_);
    return position_;
  }
  int64_t batch_size() const { return batch_; }
  int64_t max_length() const { return max_length_; }
  uint32_t lanes() const { return lanes_; }
  uint64_t seed() const { return seed_; }

  py::array_t<int64_t> sources() const {
    std::lock_guard<std::mutex> lock(call_mutex_);
    py::array_t<int64_t> result(batch_);
    for (int64_t b = 0; b < batch_; ++b)
      result.mutable_data()[b] = rows_[static_cast<size_t>(b)].source;
    return result;
  }

  py::array_t<int64_t> lrs_lengths() const {
    std::lock_guard<std::mutex> lock(call_mutex_);
    py::array_t<int64_t> result(batch_);
    for (int64_t b = 0; b < batch_; ++b)
      result.mutable_data()[b] = rows_[static_cast<size_t>(b)].lrs;
    return result;
  }

  py::array_t<int64_t> run_counts() const {
    std::lock_guard<std::mutex> lock(call_mutex_);
    py::array_t<int64_t> result(batch_);
    for (int64_t b = 0; b < batch_; ++b) {
      const Row &row = rows_[static_cast<size_t>(b)];
      result.mutable_data()[b] = row.sequence.run_count();
    }
    return result;
  }

  int64_t storage_bytes() const {
    std::lock_guard<std::mutex> lock(call_mutex_);
    size_t bytes = sizeof(NativeRLBWTState);
    checked_add(bytes, checked_product(rows_.capacity(), sizeof(Row)));
    checked_add(bytes,
                checked_product(row_hashes_.capacity(),
                                sizeof(decltype(row_hashes_)::value_type)));
    for (size_t row_index = 0; row_index < rows_.size(); ++row_index) {
      const Row &row = rows_[row_index];
      checked_add(bytes, row.history.storage_bytes());
      checked_add(bytes, row.sequence.storage_bytes());
      if (identity_codes_) {
        checked_add(bytes, sizeof(Row::IdentityCounts));
      } else {
        checked_add(bytes,
                    checked_product(row.counts.capacity(), sizeof(SymbolCount)));
        checked_add(bytes, checked_product(row.code_values.capacity(),
                                           sizeof(int64_t)));
      }
      for (uint32_t lane = 0; lane < lanes_; ++lane)
        checked_add(bytes, checked_product(row_hashes_[row_index][lane].capacity(),
                                           sizeof(uint64_t)));
    }
    for (uint32_t lane = 0; lane < lanes_; ++lane)
      checked_add(bytes, checked_product(powers_[lane].capacity(),
                                         sizeof(uint64_t)));
    if (row_pool_)
      checked_add(bytes, row_pool_->storage_bytes());
    if (bytes > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
      throw std::overflow_error("native RLBWT storage size exceeds int64");
    return static_cast<int64_t>(bytes);
  }

  py::dict storage_breakdown() const {
    std::lock_guard<std::mutex> lock(call_mutex_);
    size_t history = 0;
    size_t dictionaries = 0;
    std::array<size_t, 5> sequence{};
    for (const Row &row : rows_) {
      checked_add(history, row.history.storage_bytes());
      if (identity_codes_) {
        checked_add(dictionaries, sizeof(Row::IdentityCounts));
      } else {
        checked_add(dictionaries,
                    checked_product(row.counts.capacity(), sizeof(SymbolCount)));
        checked_add(dictionaries, checked_product(row.code_values.capacity(),
                                                  sizeof(int64_t)));
      }
      const auto components = row.sequence.storage_components();
      for (size_t index = 0; index < sequence.size(); ++index)
        checked_add(sequence[index], components[index]);
    }
    py::dict result;
    result["state"] = py::int_(sizeof(NativeRLBWTState));
    result["rows"] = py::int_(checked_product(rows_.capacity(), sizeof(Row)));
    result["history"] = py::int_(history);
    result["dictionary"] = py::int_(dictionaries);
    result["arenas"] = py::int_(sequence[0]);
    result["bwt"] = py::int_(sequence[1]);
    result["pa"] = py::int_(sequence[2]);
    result["lcs"] = py::int_(sequence[3]);
    result["histograms"] = py::int_(sequence[4]);
    return result;
  }

  py::tuple row_snapshot(int64_t batch_index) const {
    std::lock_guard<std::mutex> lock(call_mutex_);
    if (batch_index < 0 || batch_index >= batch_)
      throw py::index_error("batch_index is out of range");
    const Row &row = rows_[static_cast<size_t>(batch_index)];
    const int64_t live = position_ + 1;
    py::array_t<int64_t> pa(live), lcs(live), bwt(live);
    py::array_t<bool> sentinel_mask(live);
    row.sequence.snapshot(row.code_values, identity_codes_, pa.mutable_data(),
                          lcs.mutable_data(), bwt.mutable_data(),
                          sentinel_mask.mutable_data());
    return py::make_tuple(std::move(pa), std::move(lcs), std::move(bwt),
                          std::move(sentinel_mask));
  }

private:
  static uint64_t splitmix64(uint64_t &state) noexcept {
    uint64_t value = (state += 0x9e3779b97f4a7c15ULL);
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
  }

  static size_t checked_product(size_t left, size_t right) {
    if (right != 0 && left > std::numeric_limits<size_t>::max() / right)
      throw std::overflow_error("native RLBWT storage size overflow");
    return left * right;
  }

  static void checked_add(size_t &total, size_t increment) {
    if (increment > std::numeric_limits<size_t>::max() - total)
      throw std::overflow_error("native RLBWT storage size overflow");
    total += increment;
  }

  static constexpr uint32_t kSentinelCode =
      std::numeric_limits<uint32_t>::max();

  struct Run {
    uint32_t code = kSentinelCode;
    uint32_t length = 1;
  };

  struct SymbolCount {
    int64_t symbol = 0;
    uint32_t code = 0;
    uint32_t count = 0;
  };

  // Append-only paged token history. No storage or page directory is sized
  // from max_length: both grow only as live endpoints cross page boundaries.
  // Exactly one payload width owns pages at a time, and promotion stages a
  // complete replacement before swapping it into the live history.
  class CompactArray {
  public:
    static constexpr size_t kPageCodes = 4096;
    static constexpr size_t kPackedPageBytes = kPageCodes / 2;
    template <typename Value>
    using Pages = std::vector<std::unique_ptr<Value[]>>;

    CompactArray() = default;
    explicit CompactArray(size_t) {}

    size_t size() const noexcept { return live_size_; }
    bool is_byte() const noexcept { return width_ == 1; }
    uint32_t width() const noexcept { return width_; }

    void prepare_append(size_t index) {
      if (index != live_size_)
        throw std::runtime_error("history append position is out of range");
      const size_t page = index / kPageCodes;
      if (width_ == 0)
        ensure_packed_page(page);
      else if (width_ == 4)
        ensure_page(wide_pages_, page);
      else if (width_ == 2)
        ensure_page(narrow_pages_, page);
      else
        ensure_page(byte_pages_, page);
    }

    void finish_append() noexcept { ++live_size_; }

    uint32_t get(size_t index) const noexcept {
      const size_t page = index / kPageCodes, slot = index % kPageCodes;
      if (width_ == 0) return get_packed(packed_pages_[page].get(), slot);
      if (width_ == 4) return wide_pages_[page][slot];
      if (width_ == 2) return narrow_pages_[page][slot];
      return byte_pages_[page][slot];
    }
    void set(size_t index, uint32_t value) noexcept {
      const size_t page = index / kPageCodes, slot = index % kPageCodes;
      if (width_ == 0) set_packed(packed_pages_[page].get(), slot, value);
      else if (width_ == 4) wide_pages_[page][slot] = value;
      else if (width_ == 2)
        narrow_pages_[page][slot] = static_cast<uint16_t>(value);
      else byte_pages_[page][slot] = static_cast<uint8_t>(value);
    }

    size_t common_suffix(size_t left, size_t right) const noexcept {
      const size_t available = std::min(left, right);
      if (width_ == 0)
        return common_suffix_packed(left, right, available);
      if (width_ == 4)
        return common_suffix_pages(wide_pages_, left, right, available);
      if (width_ == 2)
        return common_suffix_pages(narrow_pages_, left, right, available);
      return common_suffix_pages(byte_pages_, left, right, available);
    }

    template <typename Value>
    Pages<Value> prepare() const {
      const size_t active_pages = width_ == 0   ? packed_pages_.size()
                                  : width_ == 4 ? wide_pages_.size()
                                  : width_ == 2 ? narrow_pages_.size()
                                                : byte_pages_.size();
      Pages<Value> result;
      result.reserve(active_pages);
      for (size_t page = 0; page < active_pages; ++page)
        result.push_back(std::make_unique<Value[]>(kPageCodes));
      for (size_t index = 0; index < live_size_; ++index)
        result[index / kPageCodes][index % kPageCodes] =
            static_cast<Value>(get(index));
      return result;
    }

    template <typename Value>
    Pages<Value> prepare_promotion_append() const {
      const size_t required_pages =
          (live_size_ + 1 + kPageCodes - 1) / kPageCodes;
      Pages<Value> result;
      result.reserve(required_pages);
      for (size_t page = 0; page < required_pages; ++page)
        result.push_back(std::make_unique<Value[]>(kPageCodes));
      for (size_t index = 0; index < live_size_; ++index)
        result[index / kPageCodes][index % kPageCodes] =
            static_cast<Value>(get(index));
      return result;
    }

    void commit(Pages<uint8_t> values) noexcept {
      byte_pages_.swap(values);
      Pages<uint8_t>().swap(packed_pages_);
      Pages<uint16_t>().swap(narrow_pages_);
      Pages<uint32_t>().swap(wide_pages_);
      width_ = 1;
    }

    void commit(Pages<uint32_t> values) noexcept {
      wide_pages_.swap(values);
      Pages<uint8_t>().swap(packed_pages_);
      Pages<uint16_t>().swap(narrow_pages_);
      Pages<uint8_t>().swap(byte_pages_);
      width_ = 4;
    }

    void commit(Pages<uint16_t> values) noexcept {
      narrow_pages_.swap(values);
      Pages<uint8_t>().swap(packed_pages_);
      Pages<uint32_t>().swap(wide_pages_);
      Pages<uint8_t>().swap(byte_pages_);
      width_ = 2;
    }

    void reset() noexcept {
      Pages<uint8_t>().swap(packed_pages_);
      Pages<uint8_t>().swap(byte_pages_);
      Pages<uint16_t>().swap(narrow_pages_);
      Pages<uint32_t>().swap(wide_pages_);
      width_ = 0;
      live_size_ = 0;
    }

    size_t storage_bytes() const {
      size_t bytes = checked_product(packed_pages_.capacity(),
                                     sizeof(std::unique_ptr<uint8_t[]>));
      checked_add(bytes, checked_product(byte_pages_.capacity(),
                                         sizeof(std::unique_ptr<uint8_t[]>)));
      checked_add(bytes, checked_product(narrow_pages_.capacity(),
                                         sizeof(std::unique_ptr<uint16_t[]>)));
      checked_add(bytes, checked_product(wide_pages_.capacity(),
                                         sizeof(std::unique_ptr<uint32_t[]>)));
      checked_add(bytes,
                  checked_product(packed_pages_.size(), kPackedPageBytes));
      checked_add(bytes, checked_product(byte_pages_.size(), kPageCodes));
      checked_add(bytes, checked_product(narrow_pages_.size(),
                                         kPageCodes * sizeof(uint16_t)));
      checked_add(bytes, checked_product(wide_pages_.size(),
                                         kPageCodes * sizeof(uint32_t)));
      return bytes;
    }

  private:
    static uint32_t get_packed(const uint8_t *values, size_t index) noexcept {
      const uint8_t byte = values[index >> 1];
      return (index & 1u) == 0 ? byte & 0x0fu : byte >> 4;
    }

    static void set_packed(uint8_t *values, size_t index,
                           uint32_t value) noexcept {
      uint8_t &byte = values[index >> 1];
      if ((index & 1u) == 0)
        byte = static_cast<uint8_t>((byte & 0xf0u) | value);
      else
        byte = static_cast<uint8_t>((byte & 0x0fu) | (value << 4));
    }

    void ensure_packed_page(size_t page) {
      if (page < packed_pages_.size()) return;
      if (page != packed_pages_.size())
        throw std::runtime_error("history page position is out of range");
      auto payload = std::make_unique<uint8_t[]>(kPackedPageBytes);
      packed_pages_.push_back(std::move(payload));
    }

    static uint64_t load_little_u64(const uint8_t *data) noexcept {
      uint64_t value;
      std::memcpy(&value, data, sizeof(value));
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
      value = __builtin_bswap64(value);
#endif
      return value;
    }

    static uint64_t packed_word(const uint8_t *page,
                                size_t start) noexcept {
      const size_t byte = start >> 1;
      const uint64_t low = load_little_u64(page + byte);
      if ((start & 1u) == 0) return low;
      return (low >> 4) |
             (static_cast<uint64_t>(page[byte + 8] & 0x0fu) << 60);
    }

    size_t common_suffix_packed(size_t left, size_t right,
                                size_t available) const noexcept {
      static constexpr size_t kChunkCodes = 16;
      if (left == right) return available;
      size_t matched = 0;
      // Random streams almost always differ on their newest code.  Avoid two
      // unaligned word loads and normalization in that overwhelmingly common
      // case, while retaining the word path for repetitive suffixes.
      const size_t scalar_prefix = std::min(available, size_t{4});
      while (matched < scalar_prefix) {
        if (get(left - matched - 1) != get(right - matched - 1))
          return matched;
        ++matched;
      }
      while (matched < available) {
        const size_t left_end = left - matched;
        const size_t right_end = right - matched;
        const size_t contiguous = std::min(
            {available - matched, (left_end - 1) % kPageCodes + 1,
             (right_end - 1) % kPageCodes + 1});
        if (contiguous >= kChunkCodes) {
          const size_t left_start = left_end - kChunkCodes;
          const size_t right_start = right_end - kChunkCodes;
          const uint64_t difference =
              packed_word(packed_pages_[left_start / kPageCodes].get(),
                          left_start % kPageCodes) ^
              packed_word(packed_pages_[right_start / kPageCodes].get(),
                          right_start % kPageCodes);
          if (difference == 0) {
            matched += kChunkCodes;
            continue;
          }
          return matched +
                 static_cast<size_t>(leading_zero_count64(difference) / 4);
        }
        size_t edge = contiguous;
        while (edge != 0) {
          if (get(left - matched - 1) != get(right - matched - 1))
            return matched;
          ++matched;
          --edge;
        }
      }
      return matched;
    }

    template <typename Value>
    static void ensure_page(Pages<Value> &pages, size_t page) {
      if (page < pages.size())
        return;
      if (page != pages.size())
        throw std::runtime_error("history page position is out of range");
      auto payload = std::make_unique<Value[]>(kPageCodes);
      pages.push_back(std::move(payload));
    }

    template <typename Value>
    static size_t common_suffix_pages(const Pages<Value> &pages, size_t left,
                                      size_t right,
                                      size_t available) noexcept {
      size_t matched = 0;
      while (matched < available) {
        const size_t left_end = left - matched;
        const size_t right_end = right - matched;
        const size_t contiguous = std::min(
            {available - matched, (left_end - 1) % kPageCodes + 1,
             (right_end - 1) % kPageCodes + 1});
        const Value *left_data =
            pages[(left_end - 1) / kPageCodes].get() +
            (left_end % kPageCodes == 0 ? kPageCodes : left_end % kPageCodes) -
            contiguous;
        const Value *right_data =
            pages[(right_end - 1) / kPageCodes].get() +
            (right_end % kPageCodes == 0 ? kPageCodes : right_end % kPageCodes) -
            contiguous;
        if (std::memcmp(left_data, right_data,
                        contiguous * sizeof(Value)) == 0) {
          matched += contiguous;
          continue;
        }
        while (matched < available &&
               pages[(left - matched - 1) / kPageCodes]
                        [(left - matched - 1) % kPageCodes] ==
                   pages[(right - matched - 1) / kPageCodes]
                        [(right - matched - 1) % kPageCodes])
          ++matched;
        return matched;
      }
      return matched;
    }

    Pages<uint8_t> packed_pages_;
    Pages<uint8_t> byte_pages_;
    Pages<uint16_t> narrow_pages_;
    Pages<uint32_t> wide_pages_;
    size_t live_size_ = 0;
    uint32_t width_ = 0;
  };

  // BWT, PA and LCS share one blocked rank space.  Leaves are stable arena
  // entries and a weighted B+ directory (fanout 16) owns their order.  Every
  // directory node carries the exact aggregate needed by rank/LCS/PA queries;
  // the linked leaves are used only by snapshot and run_count.
  class UnifiedSequence {
  public:
    // Scratch remains sized for the largest route. Logical leaf geometry is
    // selected once from PA width by the constructor below.
    static constexpr uint32_t kMaxLeafCapacity = 2048;
    static constexpr uint32_t kFanout = 16;
    static constexpr uint32_t kNil = std::numeric_limits<uint32_t>::max();

    UnifiedSequence() = default;
    explicit UnifiedSequence(size_t capacity)
        : capacity_(capacity), maximum_pa_bits_(pa_bit_width(capacity - 1)),
          pa_bits_(1),
          leaf_limit_(maximum_pa_bits_ <= 20 ? 256u : 2048u),
          split_left_size_(leaf_limit_ / 2),
          split_right_size_(split_left_size_ + 1),
          split_leaf_capacity_(maximum_pa_bits_ <= 20 ? 136u : 1088u),
          leaf_growth_(maximum_pa_bits_ <= 20 ? 32u : 256u),
          lcs_width_(capacity - 1 > 65535 ? 4u : 2u) {
      // Arena capacity is deliberately unrelated to the logical sequence
      // limit. prepare_insert grows both index-addressed arenas before commit.
      leaves_.reserve(2);
      nodes_.reserve(2);
      pending_nodes_.reserve(4);
      auto initial_leaf = make_leaf(0, leaf_limit_ / 2);
      leaves_.push_back(std::move(*initial_leaf));
      Leaf &leaf = leaves_[0];
      leaf.size = 1;
      set_pa_bits(leaf, 0, 0);
      const uint32_t initial_lcs = 0;
      encode_lcs_into(&initial_lcs, 1, leaf.lcs, leaf.payload_capacity);
      sentinel_leaf_ = 0;
      sentinel_slot_ = 0;
      refresh(leaf);
      first_leaf_ = last_leaf_ = root_ = 0;
      root_is_leaf_ = true;
      size_ = 1;
    }

    UnifiedSequence(UnifiedSequence &&) noexcept = default;
    UnifiedSequence &operator=(UnifiedSequence &&) noexcept = default;
    UnifiedSequence(const UnifiedSequence &) = delete;
    UnifiedSequence &operator=(const UnifiedSequence &) = delete;

    size_t size() const noexcept { return size_; }
    uint32_t code_width() const noexcept { return code_width_; }

    uint32_t pa(size_t rank) const noexcept {
      const Location location = locate(rank);
      return get_pa(leaves_[location.leaf], location.slot);
    }
    uint32_t lcs(size_t rank) const noexcept {
      const Location location = locate(rank);
      return get_lcs(leaves_[location.leaf], location.slot);
    }

    uint32_t rank(uint32_t code) const noexcept {
      return rank_prefix(prefix(sentinel_leaf_) + sentinel_slot_, code);
    }

    // PA width follows the largest live endpoint rather than the configured
    // capacity. Preparing every replacement first makes the representation
    // change transactional; the following commit is allocation-free. The
    // sentinel remains physically implicit, while get_pa() supplies its
    // logical endpoint during repack.
    void promote_pa_for_endpoint(uint32_t endpoint) {
      const uint32_t width = pa_bit_width(endpoint);
      if (width <= pa_bits_)
        return;
      if (width > maximum_pa_bits_)
        throw std::runtime_error("PA endpoint exceeds configured capacity");
      if (pending_left_payload_ || pending_leaf_ || !pending_nodes_.empty() ||
          pending_lcs_active_ || pending_successor_lcs_active_ ||
          pending_raw8_in_place_ || pending_raw8_successor_in_place_)
        throw std::runtime_error("PA promotion requires no pending insertion");

      std::vector<std::unique_ptr<uint8_t[]>> payloads;
      payloads.reserve(leaves_.size());
      for (uint32_t leaf_index = 0; leaf_index < leaves_.size(); ++leaf_index) {
        const Leaf &leaf = leaves_[leaf_index];
        const size_t bytes = pa_storage_bytes(leaf.payload_capacity, width);
        auto payload = std::make_unique<uint8_t[]>(bytes);
        std::memset(payload.get(), 0, bytes);
        for (uint32_t slot = 0; slot < leaf.size; ++slot) {
          const uint32_t value = get_pa(leaf, slot);
          if (leaf_index != sentinel_leaf_ || slot != sentinel_slot_)
            set_pa_bits(payload.get(), slot, value, width);
        }
        payloads.push_back(std::move(payload));
      }
      for (size_t index = 0; index < leaves_.size(); ++index)
        leaves_[index].pa.swap(payloads[index]);
      pa_bits_ = width;
    }

    void prepare_insert(size_t rank, uint32_t replacement_code, uint32_t x,
                        bool has_successor, uint32_t y) {
      pending_left_payload_.reset();
      pending_leaf_.reset();
      pending_lcs_active_ = false;
      pending_successor_lcs_active_ = false;
      pending_raw8_in_place_ = false;
      pending_raw8_successor_in_place_ = false;
      pending_successor_leaf_ = kNil;
      pending_nodes_.clear();
      if (rank > size_ || size_ >= capacity_)
        throw std::runtime_error("unified insertion position is out of range");
      Location location = rank == size_ ? locate(size_ - 1) : locate(rank);
      if (rank == size_)
        location.slot = leaves_[location.leaf].size;
      const bool splits_leaf = leaves_[location.leaf].size == leaf_limit_;
      size_t node_count = 0;
      if (splits_leaf) {
        if (root_is_leaf_) {
          node_count = 1;
        } else {
          uint32_t parent = leaves_[location.leaf].parent;
          while (parent != kNil && nodes_[parent].count == kFanout) {
            ++node_count;
            parent = nodes_[parent].parent;
          }
          if (parent == kNil)
            ++node_count;
        }
      }
      // Materialize only the exact chunks needed by the pending commit. The
      // objects already stored in prior chunks never move; indices remain the
      // arena authority throughout preparation and promotion.
      // No Leaf/Node reference is retained across these reserves. Indices are
      // the arena authority and remain valid when vector storage relocates.
      ensure_arena_capacity(leaves_, leaves_.size() + (splits_leaf ? 1 : 0));
      ensure_arena_capacity(nodes_, nodes_.size() + node_count);
      if (pending_nodes_.capacity() < node_count)
        pending_nodes_.reserve(node_count);
      const size_t histogram_capacity = root_histogram_size() + 1;
      reserve_leaf_histogram(
          sentinel_leaf_,
          leaf_histogram_size(leaves_[sentinel_leaf_], code_width_) + 1);
      reserve_leaf_histogram(
          location.leaf,
          leaf_histogram_size(leaves_[location.leaf], code_width_) + 1);
      reserve_path(sentinel_leaf_, histogram_capacity);
      reserve_path(location.leaf, histogram_capacity);
      if (rank < size_) {
        const Location successor = locate(rank);
        reserve_leaf_histogram(
            successor.leaf,
            leaf_histogram_size(leaves_[successor.leaf], code_width_) + 1);
        reserve_path(successor.leaf, histogram_capacity);
      }
      const bool adds_code =
          leaf_histogram_count(leaves_[sentinel_leaf_], replacement_code,
                               code_width_) == 0;
      (void)adds_code; // capacity above covers the one possible new symbol.
      Leaf &target = leaves_[location.leaf];
      const uint32_t required_width =
          replacement_code > std::numeric_limits<uint16_t>::max()
              ? 4u
              : (replacement_code > std::numeric_limits<uint8_t>::max()
                     ? 2u
                     : (replacement_code > 15u ? 1u : 0u));
      const uint32_t target_width = std::max(code_width_, required_width);
      if (target.size == leaf_limit_) {
        // Both halves replace physical payloads at the split commit. Keeping
        // a small insertion runway avoids an immediate reallocation while
        // still releasing the full leaf's old payload.
        pending_left_payload_ = make_leaf(target_width, split_leaf_capacity_);
        pending_leaf_ = make_leaf(target_width, split_leaf_capacity_);
        reserve_leaf_histogram(
            *pending_leaf_, leaf_histogram_size(target, code_width_) + 1,
            target_width);
        for (size_t index = 0; index < node_count; ++index) {
          auto node = std::make_unique<Node>();
          histogram_reserve(node->histogram, histogram_capacity, code_width_);
          pending_nodes_.push_back(std::move(node));
        }
      } else if (target.size == target.payload_capacity) {
        const uint32_t grown_capacity =
            std::min(leaf_limit_, target.payload_capacity + leaf_growth_);
        pending_left_payload_ = make_leaf(target_width, grown_capacity);
        copy_payload(target, *pending_left_payload_);
      }

      Location successor{kNil, 0};
      if (has_successor)
        successor = locate(rank);
      const bool changes_payload =
          target.size == leaf_limit_ || target.size == target.payload_capacity;
      const bool target_is_packed4 = target.lcs.kind == LcsKind::Packed4;
      const bool target_is_raw8 = target.lcs.kind == LcsKind::Raw8;
      bool raw8_can_shrink = target_is_raw8 && x <= 15u;
      if (raw8_can_shrink && has_successor && successor.leaf == location.leaf)
        raw8_can_shrink = y <= 15u;
      if (raw8_can_shrink) {
        for (uint32_t slot = 0; slot < target.size; ++slot) {
          const uint32_t value =
              has_successor && successor.leaf == location.leaf &&
                      slot == successor.slot
                  ? y
                  : target.lcs.payload[slot];
          if (value > 15u) {
            raw8_can_shrink = false;
            break;
          }
        }
      }
      bool successor_raw8_can_shrink =
          has_successor && successor.leaf != location.leaf && y <= 15u &&
          leaves_[successor.leaf].lcs.kind == LcsKind::Raw8;
      if (successor_raw8_can_shrink) {
        const Leaf &successor_leaf = leaves_[successor.leaf];
        for (uint32_t slot = 0; slot < successor_leaf.size; ++slot) {
          const uint32_t value = slot == successor.slot
                                     ? y
                                     : successor_leaf.lcs.payload[slot];
          if (value > 15u) {
            successor_raw8_can_shrink = false;
            break;
          }
        }
      }
      const bool target_stays_mutable =
          !changes_payload &&
          ((target_is_packed4 && x <= 15u &&
           (!has_successor || successor.leaf != location.leaf || y <= 15u)) ||
           (target_is_raw8 && x <= 255u &&
            (!has_successor || successor.leaf != location.leaf || y <= 255u) &&
            !raw8_can_shrink));
      const bool successor_stays_mutable =
          !has_successor || successor.leaf == location.leaf ||
          (leaves_[successor.leaf].lcs.kind == LcsKind::Packed4
               ? y <= 15u
               : leaves_[successor.leaf].lcs.kind == LcsKind::Raw8 &&
                     y <= 255u && !successor_raw8_can_shrink);
      if (target_stays_mutable && successor_stays_mutable) {
        // PACKED4 and RAW8 payloads are physically sized to leaf capacity, so
        // insertion and the optional point update need no allocation. Growth,
        // split and any value outside the active codec are staged below.
        pending_raw8_in_place_ = true;
        pending_raw8_successor_in_place_ =
            has_successor && successor.leaf != location.leaf;
        pending_successor_leaf_ =
            pending_raw8_successor_in_place_ ? successor.leaf : kNil;
        pending_successor_slot_ = successor.slot;
        return;
      }

      // Decode only the leaves changed by this insertion.  The replacement
      // codecs are fully allocated before any sequence state is committed.
      std::array<uint32_t, kMaxLeafCapacity> old_values{};
      decode_lcs(target, old_values.data());
      std::array<uint32_t, kMaxLeafCapacity + 1> values{};
      for (uint32_t out = 0; out <= target.size; ++out) {
        if (out == location.slot)
          values[out] = x;
        else
          values[out] = old_values[out - (out > location.slot ? 1u : 0u)];
      }
      if (has_successor) {
        if (successor.leaf == location.leaf)
          values[successor.slot + (successor.slot >= location.slot ? 1u : 0u)] = y;
      }
      if (target.size == leaf_limit_) {
        encode_lcs_into(values.data(), split_left_size_, staging_lcs_,
                        pending_left_payload_->payload_capacity);
        pending_lcs_active_ = true;
        // A split's new right leaf has no old codec to recycle. Encode its
        // one-off payload directly into the already-prepared Leaf.
        encode_lcs_into(values.data() + split_left_size_, split_right_size_,
                        pending_leaf_->lcs,
                        pending_leaf_->payload_capacity);
      } else {
        const uint32_t physical_capacity = pending_left_payload_
                                               ? pending_left_payload_->payload_capacity
                                               : target.payload_capacity;
        encode_lcs_into(values.data(), target.size + 1, staging_lcs_,
                        physical_capacity);
        pending_lcs_active_ = true;
      }
      if (has_successor && successor.leaf != location.leaf) {
        const Leaf &successor_leaf = leaves_[successor.leaf];
        std::array<uint32_t, kMaxLeafCapacity> successor_values{};
        decode_lcs(successor_leaf, successor_values.data());
        successor_values[successor.slot] = y;
        encode_lcs_into(successor_values.data(), successor_leaf.size,
                        staging_successor_lcs_,
                        successor_leaf.payload_capacity);
        pending_successor_lcs_active_ = true;
        pending_successor_leaf_ = successor.leaf;
      }
    }

    std::vector<std::unique_ptr<uint16_t[]>> prepare_codes16() const {
      std::vector<std::unique_ptr<uint16_t[]>> result;
      result.reserve(leaves_.size());
      for (const Leaf &leaf : leaves_) {
        auto values = std::make_unique<uint16_t[]>(leaf.payload_capacity);
        for (uint32_t slot = 0; slot < leaf.size; ++slot)
          values[slot] = static_cast<uint16_t>(get_code(leaf, slot));
        result.push_back(std::move(values));
      }
      return result;
    }

    std::vector<std::unique_ptr<uint8_t[]>> prepare_codes8() const {
      std::vector<std::unique_ptr<uint8_t[]>> result;
      result.reserve(leaves_.size());
      for (const Leaf &leaf : leaves_) {
        auto values = std::make_unique<uint8_t[]>(leaf.payload_capacity);
        if (code_width_ == 0) {
          const uint32_t pairs = leaf.size >> 1;
          for (uint32_t pair = 0; pair < pairs; ++pair) {
            const uint8_t packed = leaf.codes4[pair];
            values[pair * 2] = packed & 0x0fu;
            values[pair * 2 + 1] = packed >> 4;
          }
          if ((leaf.size & 1u) != 0)
            values[leaf.size - 1] = leaf.codes4[leaf.size >> 1] & 0x0fu;
        } else {
          std::memcpy(values.get(), leaf.codes8.get(), leaf.size);
        }
        result.push_back(std::move(values));
      }
      return result;
    }

    std::vector<std::unique_ptr<uint32_t[]>> prepare_codes32() const {
      std::vector<std::unique_ptr<uint32_t[]>> result;
      result.reserve(leaves_.size());
      for (const Leaf &leaf : leaves_) {
        auto values = std::make_unique<uint32_t[]>(leaf.payload_capacity);
        for (uint32_t slot = 0; slot < leaf.size; ++slot)
          values[slot] = get_code(leaf, slot);
        result.push_back(std::move(values));
      }
      return result;
    }

    struct HistogramRepack {
      std::vector<std::vector<uint8_t>> leaves;
      std::vector<std::vector<uint8_t>> nodes;
      std::vector<std::vector<uint8_t>> pending_nodes;
      std::vector<uint8_t> pending_leaf;
      std::vector<uint8_t> pending_left;
      bool has_pending_leaf = false;
      bool has_pending_left = false;
      bool repack_leaves = false;
    };

    HistogramRepack prepare_histograms(uint32_t width) const {
      HistogramRepack result;
      result.repack_leaves = width > 1;
      if (result.repack_leaves) {
        result.leaves.reserve(leaves_.size());
        for (const Leaf &leaf : leaves_)
          result.leaves.push_back(
              repack_leaf_histogram(leaf, code_width_, width));
      }
      result.nodes.reserve(nodes_.size());
      for (const Node &node : nodes_)
        result.nodes.push_back(repack_histogram(node.histogram, code_width_,
                                                width));
      result.pending_nodes.reserve(pending_nodes_.size());
      for (const auto &node : pending_nodes_)
        result.pending_nodes.push_back(repack_histogram(
            node->histogram, code_width_, width));
      if (pending_leaf_ && result.repack_leaves) {
        result.pending_leaf = repack_leaf_histogram(
            *pending_leaf_, code_width_, width);
        result.has_pending_leaf = true;
      }
      if (pending_left_payload_ && result.repack_leaves) {
        result.pending_left = repack_leaf_histogram(
            *pending_left_payload_, code_width_, width);
        result.has_pending_left = true;
      }
      return result;
    }

    void commit_histograms(HistogramRepack values) noexcept {
      if (values.repack_leaves)
        for (size_t index = 0; index < leaves_.size(); ++index)
          commit_leaf_histogram(leaves_[index],
                                std::move(values.leaves[index]));
      for (size_t index = 0; index < nodes_.size(); ++index)
        nodes_[index].histogram.swap(values.nodes[index]);
      for (size_t index = 0; index < pending_nodes_.size(); ++index)
        pending_nodes_[index]->histogram.swap(values.pending_nodes[index]);
      if (values.has_pending_leaf)
        commit_leaf_histogram(*pending_leaf_, std::move(values.pending_leaf));
      if (values.has_pending_left)
        commit_leaf_histogram(*pending_left_payload_,
                              std::move(values.pending_left));
    }

    void commit_codes16(std::vector<std::unique_ptr<uint16_t[]>> values,
                        HistogramRepack histograms) noexcept {
      for (size_t index = 0; index < leaves_.size(); ++index) {
        leaves_[index].codes16 = std::move(values[index]);
        leaves_[index].codes4.reset();
        leaves_[index].codes8.reset();
        leaves_[index].codes32.reset();
      }
      commit_histograms(std::move(histograms));
      code_width_ = 2;
    }

    void commit_codes8(std::vector<std::unique_ptr<uint8_t[]>> values,
                       HistogramRepack histograms) noexcept {
      for (size_t index = 0; index < leaves_.size(); ++index) {
        leaves_[index].codes8 = std::move(values[index]);
        leaves_[index].codes4.reset();
        leaves_[index].codes16.reset();
        leaves_[index].codes32.reset();
      }
      commit_histograms(std::move(histograms));
      code_width_ = 1;
    }

    void commit_codes32(std::vector<std::unique_ptr<uint32_t[]>> values,
                        HistogramRepack histograms) noexcept {
      for (size_t index = 0; index < leaves_.size(); ++index) {
        leaves_[index].codes32 = std::move(values[index]);
        leaves_[index].codes4.reset();
        leaves_[index].codes8.reset();
        leaves_[index].codes16.reset();
      }
      commit_histograms(std::move(histograms));
      code_width_ = 4;
    }

    // Allocation-free after prepare_insert().  The old sentinel is first
    // replaced by code, then the new aligned sentinel/PA/LCS row is inserted.
    void replace_and_insert(uint32_t code, size_t rank, uint32_t endpoint,
                            uint32_t x, bool has_successor,
                            uint32_t y) noexcept {
      const uint32_t replaced_index = sentinel_leaf_;
      Location location = rank == size_ ? locate(size_ - 1) : locate(rank);
      if (rank == size_)
        location.slot = leaves_[location.leaf].size;

      const uint32_t left_index = location.leaf;

      // The overwhelmingly common path does not split.  Shifting rows does
      // not change any aggregate: the former sentinel contributes exactly
      // one `code`, while the inserted sentinel contributes no histogram
      // entry.  Update those deltas directly and rebuild only the tiny
      // (fanout 16) extrema summaries on each distinct ancestor path.
      if (leaves_[location.leaf].size < leaf_limit_) {
        Leaf &left = leaves_[left_index];
        const bool mutable_lcs_in_place = pending_raw8_in_place_;
        if (pending_left_payload_) {
          commit_payload(left, std::move(pending_left_payload_));
        }
        uint32_t old_successor_lcs = 0;
        if (mutable_lcs_in_place && has_successor) {
          old_successor_lcs = pending_raw8_successor_in_place_
                                  ? decode_lcs_value(
                                        leaves_[pending_successor_leaf_],
                                        pending_successor_slot_)
                                  : decode_lcs_value(left, location.slot);
        }
        if (mutable_lcs_in_place) {
          move_mutable_lcs_right(left, location.slot,
                                 left.size - location.slot);
          set_mutable_lcs(left, location.slot, x);
          if (has_successor && !pending_raw8_successor_in_place_)
            set_mutable_lcs(left, location.slot + 1, y);
          if (pending_raw8_successor_in_place_)
            set_mutable_lcs(leaves_[pending_successor_leaf_],
                            pending_successor_slot_, y);
        } else {
          swap_lcs(left.lcs, staging_lcs_);
          pending_lcs_active_ = false;
        }
        if (pending_successor_lcs_active_) {
          swap_lcs(leaves_[pending_successor_leaf_].lcs,
                   staging_successor_lcs_);
          pending_successor_lcs_active_ = false;
        }
        // The current sentinel endpoint is implicit. Materialize it before
        // this physical slot is shifted and becomes an ordinary PA row.
        set_pa_bits(leaves_[replaced_index], sentinel_slot_,
                    static_cast<uint32_t>(size_ - 1));
        set_code(leaves_[replaced_index], sentinel_slot_, code);
        increment_leaf_histogram(leaves_[replaced_index], code, code_width_);
        increment_histogram_up(leaves_[replaced_index].parent, code);

        insert_local(left, location.slot, 0, 0, x);
        add_max(endpoint, left.max1_pa, left.max2_pa);
        increment_weight_up(left_index);
        sentinel_leaf_ = left_index;
        sentinel_slot_ = location.slot;
        ++size_;

        uint32_t changed[2] = {left_index, kNil};
        uint32_t changed_count = 1;
        if (mutable_lcs_in_place) {
          // The local multiset replaces old_successor with {x, y}; by the
          // adjacent-LCP identity min(x, y) == old_successor, its minimum
          // cannot increase. Update the exact count without scanning.
          if (has_successor && !pending_raw8_successor_in_place_) {
            if (old_successor_lcs == left.min_lcs)
              --left.min_lcs_count;
            add_leaf_lcs(left, y);
          }
          add_leaf_lcs(left, x);
          if (pending_raw8_successor_in_place_) {
            Leaf &successor_leaf = leaves_[pending_successor_leaf_];
            if (old_successor_lcs == successor_leaf.min_lcs) {
              if (successor_leaf.min_lcs_count > 1) {
                --successor_leaf.min_lcs_count;
                add_leaf_lcs(successor_leaf, y);
              } else if (y <= old_successor_lcs) {
                successor_leaf.min_lcs = y;
                successor_leaf.min_lcs_count = 1;
              } else {
                // This geometry is not produced by locate(rank) today, but
                // keep the aggregate exact if leaf-boundary policy changes.
                recompute_leaf_lcs(successor_leaf);
              }
            }
            if (pending_successor_leaf_ != left_index)
              changed[changed_count++] = pending_successor_leaf_;
          }
          pending_raw8_in_place_ = false;
          pending_raw8_successor_in_place_ = false;
        } else if (has_successor) {
          const Location successor = locate(rank + 1);
          Leaf &successor_leaf = leaves_[successor.leaf];
          // The replacement codec is already committed; derive the exact
          // aggregate from it rather than retaining stale raw-slot metadata.
          recompute_leaf_lcs(successor_leaf);
          if (successor.leaf != left_index)
            changed[changed_count++] = successor.leaf;
        }
        if (!mutable_lcs_in_place) {
          // Insertion changed a staged codec wholesale; derive its exact
          // aggregate from decoded values rather than stale raw slots.
          recompute_leaf_lcs(left);
        }
        refresh_extrema_paths(changed, changed_count);
        return;
      }

      set_pa_bits(leaves_[replaced_index], sentinel_slot_,
                  static_cast<uint32_t>(size_ - 1));
      set_code(leaves_[replaced_index], sentinel_slot_, code);
      refresh(leaves_[replaced_index]);
      refresh_up(leaves_[replaced_index].parent);

      {
        std::array<uint32_t, kMaxLeafCapacity + 1> codes, pas;
        for (uint32_t out = 0; out <= leaf_limit_; ++out) {
          if (out == location.slot) {
            codes[out] = 0;
            pas[out] = 0;
          } else {
            const uint32_t old = out - (out > location.slot ? 1u : 0u);
            codes[out] = get_code(leaves_[left_index], old);
            pas[out] = get_pa(leaves_[left_index], old);
          }
        }
        std::unique_ptr<Leaf> prepared_left =
            std::move(pending_left_payload_);
        std::unique_ptr<Leaf> prepared = std::move(pending_leaf_);
        const uint32_t right_index = static_cast<uint32_t>(leaves_.size());
        leaves_.push_back(std::move(*prepared));
        // push_back may relocate the Leaf arena. Reacquire by authoritative
        // index rather than retaining a reference across the operation.
        Leaf &left = leaves_[left_index];
        Leaf &right = leaves_[right_index];
        right.parent = left.parent;
        right.previous = left_index;
        right.next = left.next;
        if (left.next != kNil)
          leaves_[left.next].previous = right_index;
        else
          last_leaf_ = right_index;
        left.next = right_index;
        commit_payload(left, std::move(prepared_left));
        swap_lcs(left.lcs, staging_lcs_);
        pending_lcs_active_ = false;
        write_leaf(left, codes.data(), pas.data(), nullptr, split_left_size_);
        write_leaf(right, codes.data() + split_left_size_,
                   pas.data() + split_left_size_, nullptr, split_right_size_);
        if (location.slot < split_left_size_) {
          sentinel_leaf_ = left_index;
          sentinel_slot_ = location.slot;
        } else {
          sentinel_leaf_ = right_index;
          sentinel_slot_ = location.slot - split_left_size_;
        }
        // refresh() reads the current sentinel endpoint implicitly from
        // size_, so publish the new logical size before rebuilding maxima.
        ++size_;
        refresh(left);
        refresh(right);
        insert_leaf_after(left_index, right_index);
      }
      // Once the sentinel moves, its former slot becomes an ordinary code.
      // If insertion happened in another leaf, refresh that old leaf again;
      // the pre-insertion refresh intentionally excluded the old sentinel.
      if (replaced_index != left_index) {
        refresh(leaves_[replaced_index]);
        refresh_up(leaves_[replaced_index].parent);
      }
      if (has_successor) {
        const Location successor = locate(rank + 1);
        if (pending_successor_lcs_active_) {
          swap_lcs(leaves_[pending_successor_leaf_].lcs,
                   staging_successor_lcs_);
          pending_successor_lcs_active_ = false;
        }
        refresh(leaves_[successor.leaf]);
        refresh_up(leaves_[successor.leaf].parent);
      }
    }

    size_t nearest_previous_lcs_less(size_t rank,
                                     uint32_t threshold) const noexcept {
      Location location = locate(rank);
      for (;;) {
        std::array<uint32_t, kMaxLeafCapacity> values{};
        decode_lcs(leaves_[location.leaf], values.data());
        for (uint32_t slot = location.slot + 1; slot-- > 0;)
          if (values[slot] < threshold)
            return prefix(location.leaf) + slot;
        const uint32_t candidate = previous_leaf_with_min(location.leaf, threshold);
        if (candidate == kNil)
          return 0;
        location.leaf = candidate;
        location.slot = leaves_[candidate].size - 1;
      }
    }

    size_t nearest_next_lcs_less(size_t rank,
                                 uint32_t threshold) const noexcept {
      if (rank >= size_)
        return size_;
      Location location = locate(rank);
      for (;;) {
        std::array<uint32_t, kMaxLeafCapacity> values{};
        decode_lcs(leaves_[location.leaf], values.data());
        for (uint32_t slot = location.slot; slot < leaves_[location.leaf].size; ++slot)
          if (values[slot] < threshold)
            return prefix(location.leaf) + slot;
        const uint32_t candidate = next_leaf_with_min(location.leaf, threshold);
        if (candidate == kNil)
          return size_;
        location.leaf = candidate;
        location.slot = 0;
      }
    }

    uint32_t range_max_excluding(size_t first, size_t last,
                                 uint32_t excluded) const noexcept {
      if (first >= last)
        return 0;
      return root_is_leaf_
                 ? query_leaf_max(root_, first, last, 0, excluded)
                 : query_node_max(root_, first, last, 0, excluded);
    }

    int64_t run_count() const noexcept {
      int64_t runs = 0;
      bool previous_sentinel = false;
      uint32_t previous_code = 0;
      size_t rank = 0;
      for (uint32_t leaf_index = first_leaf_; leaf_index != kNil;
           leaf_index = leaves_[leaf_index].next) {
        const Leaf &leaf = leaves_[leaf_index];
        for (uint32_t slot = 0; slot < leaf.size; ++slot, ++rank) {
          const bool sentinel = leaf_index == sentinel_leaf_ &&
                                slot == sentinel_slot_;
          const uint32_t code = get_code(leaf, slot);
          if (rank == 0 || sentinel != previous_sentinel ||
              (!sentinel && code != previous_code)) ++runs;
          previous_sentinel = sentinel;
          previous_code = code;
        }
      }
      return runs;
    }

    void snapshot(const std::vector<int64_t> &values, bool identity_codes,
                  int64_t *pa_output, int64_t *lcs_output,
                  int64_t *bwt_output, bool *sentinel_output) const noexcept {
      size_t output = 0;
      for (uint32_t leaf_index = first_leaf_; leaf_index != kNil;
           leaf_index = leaves_[leaf_index].next) {
        const Leaf &leaf = leaves_[leaf_index];
        std::array<uint32_t, kMaxLeafCapacity> lcs_values{};
        decode_lcs(leaf, lcs_values.data());
        for (uint32_t slot = 0; slot < leaf.size; ++slot, ++output) {
          const bool sentinel = leaf_index == sentinel_leaf_ &&
                                slot == sentinel_slot_;
          pa_output[output] = get_pa(leaf, slot);
          lcs_output[output] = lcs_values[slot];
          sentinel_output[output] = sentinel;
          const uint32_t code = get_code(leaf, slot);
          bwt_output[output] =
              sentinel ? 0 : identity_codes ? static_cast<int64_t>(code)
                                            : values[code];
        }
      }
    }

    size_t storage_bytes() const {
      size_t bytes = checked_product(leaves_.capacity(), sizeof(Leaf));
      checked_add(bytes, checked_product(nodes_.capacity(), sizeof(Node)));
      checked_add(bytes, checked_product(pending_nodes_.capacity(),
                                         sizeof(std::unique_ptr<Node>)));
      for (const Leaf &leaf : leaves_) {
        checked_add(bytes, code_storage_bytes(leaf.payload_capacity,
                                              code_width_));
        checked_add(bytes, pa_storage_bytes(leaf.payload_capacity));
        checked_add(bytes, leaf.lcs.payload.capacity());
        checked_add(bytes, leaf.lcs.checkpoints.capacity());
        checked_add(bytes, leaf.histogram_counts.capacity());
        checked_add(bytes, leaf.histogram.capacity());
      }
      for (const Node &node : nodes_) {
        checked_add(bytes, node.histogram.capacity());
      }
      if (pending_leaf_) {
        checked_add(bytes, sizeof(Leaf));
        checked_add(bytes, code_storage_bytes(
                               pending_leaf_->payload_capacity,
                               payload_code_width(*pending_leaf_)));
        checked_add(bytes, pa_storage_bytes(pending_leaf_->payload_capacity));
        checked_add(bytes, pending_leaf_->lcs.payload.capacity());
        checked_add(bytes, pending_leaf_->lcs.checkpoints.capacity());
        checked_add(bytes, pending_leaf_->histogram_counts.capacity());
        checked_add(bytes, pending_leaf_->histogram.capacity());
      }
      if (pending_left_payload_) {
        checked_add(bytes, sizeof(Leaf));
        checked_add(bytes, code_storage_bytes(
                               pending_left_payload_->payload_capacity,
                               payload_code_width(*pending_left_payload_)));
        checked_add(bytes,
                    pa_storage_bytes(pending_left_payload_->payload_capacity));
        checked_add(bytes, pending_left_payload_->lcs.payload.capacity());
        checked_add(bytes, pending_left_payload_->lcs.checkpoints.capacity());
        checked_add(bytes,
                    pending_left_payload_->histogram_counts.capacity());
        checked_add(bytes, pending_left_payload_->histogram.capacity());
      }
      for (const auto &node : pending_nodes_) {
        checked_add(bytes, sizeof(Node));
        checked_add(bytes, node->histogram.capacity());
      }
      checked_add(bytes, staging_lcs_.payload.capacity());
      checked_add(bytes, staging_lcs_.checkpoints.capacity());
      checked_add(bytes, staging_successor_lcs_.payload.capacity());
      checked_add(bytes, staging_successor_lcs_.checkpoints.capacity());
      return bytes;
    }

    std::array<size_t, 5> storage_components() const {
      std::array<size_t, 5> result{};  // arenas, codes, PA, LCS, histograms
      result[0] = checked_product(leaves_.capacity(), sizeof(Leaf));
      checked_add(result[0], checked_product(nodes_.capacity(), sizeof(Node)));
      checked_add(result[0], checked_product(pending_nodes_.capacity(),
                                             sizeof(std::unique_ptr<Node>)));
      for (const Leaf &leaf : leaves_) {
        checked_add(result[1], code_storage_bytes(leaf.payload_capacity,
                                                  code_width_));
        checked_add(result[2], pa_storage_bytes(leaf.payload_capacity));
        checked_add(result[3], leaf.lcs.payload.capacity());
        checked_add(result[3], leaf.lcs.checkpoints.capacity());
        checked_add(result[4], leaf.histogram_counts.capacity());
        checked_add(result[4], leaf.histogram.capacity());
      }
      for (const Node &node : nodes_)
        checked_add(result[4], node.histogram.capacity());
      checked_add(result[3], staging_lcs_.payload.capacity());
      checked_add(result[3], staging_lcs_.checkpoints.capacity());
      checked_add(result[3], staging_successor_lcs_.payload.capacity());
      checked_add(result[3], staging_successor_lcs_.checkpoints.capacity());
      return result;
    }

  private:
    enum class LcsKind : uint8_t {
      Packed4,
      Raw8,
      Raw16,
      Raw32,
      For8,
      Delta4,
      Delta8
    };
    struct LcsCodec {
      std::vector<uint8_t> payload;
      std::vector<uint8_t> checkpoints;
      uint32_t anchor = 0;
      LcsKind kind = LcsKind::Raw8;
    };
    struct Leaf {
      std::unique_ptr<uint8_t[]> codes4;
      std::unique_ptr<uint8_t[]> codes8;
      std::unique_ptr<uint16_t[]> codes16;
      std::unique_ptr<uint32_t[]> codes32;
      std::unique_ptr<uint8_t[]> pa;
      LcsCodec lcs;
      // Codes in the byte alphabet use one count byte per present code. Zero
      // encodes 256 and a small inline table stores every exact excess.
      std::array<uint64_t, 4> histogram_bitmap{};
      std::vector<uint8_t> histogram_counts;
      std::array<uint32_t, 8> histogram_overflows{};
      uint8_t histogram_overflow_size = 0;
      std::vector<uint8_t> histogram;
      bool histogram_wide = false;
      uint32_t size = 0;
      uint32_t payload_capacity = 0;
      uint32_t min_lcs = std::numeric_limits<uint32_t>::max();
      uint32_t min_lcs_count = 0;
      uint32_t max1_pa = 0, max2_pa = 0;
      uint32_t parent = kNil, parent_slot = 0;
      uint32_t previous = kNil, next = kNil;
    };
    struct Node {
      std::array<uint32_t, kFanout> children{};
      std::array<uint32_t, kFanout> weights{};
      std::vector<uint8_t> histogram;
      uint32_t parent = kNil, parent_slot = 0;
      uint32_t count = 0, weight = 0;
      uint32_t min_lcs = std::numeric_limits<uint32_t>::max();
      uint32_t max1_pa = 0, max2_pa = 0;
      bool children_are_leaves = false;
    };
    struct Location { uint32_t leaf; uint32_t slot; };

    template <typename Value>
    static void ensure_arena_capacity(std::vector<Value> &arena,
                                      size_t required) {
      if (required <= arena.capacity())
        return;
      size_t capacity = std::max<size_t>(arena.capacity(), 2);
      while (capacity < required) {
        // Bound dead arena slots more tightly for long contexts.  Leaf and
        // node objects retain stable logical indices across relocation, so a
        // 1.125x geometric factor trades a handful of rare moves for a much
        // smaller worst-case owned-memory sawtooth.
        const size_t grown = capacity + std::max<size_t>(capacity / 8, 1);
        if (grown <= capacity) {
          capacity = required;
          break;
        }
        capacity = grown;
      }
      arena.reserve(capacity);
    }

    static uint32_t histogram_code_width(uint32_t width) noexcept {
      return width <= 1 ? 1u : width;
    }
    static size_t histogram_record_width(uint32_t width) noexcept {
      return static_cast<size_t>(histogram_code_width(width)) + 4;
    }
    static size_t histogram_size(const std::vector<uint8_t> &histogram,
                                 uint32_t width) noexcept {
      return histogram.size() / histogram_record_width(width);
    }
    static uint32_t load_little_endian_exact(const uint8_t *source,
                                             uint32_t width) noexcept {
      uint32_t value = 0;
      std::memcpy(&value, source, width);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
      value = __builtin_bswap32(value);
#endif
      return value;
    }
    static void store_little_endian_exact(uint8_t *destination,
                                          uint32_t value,
                                          uint32_t width) noexcept {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
      value = __builtin_bswap32(value);
#endif
      std::memcpy(destination, &value, width);
    }
    static uint32_t histogram_code(const std::vector<uint8_t> &histogram,
                                   size_t index,
                                   uint32_t width) noexcept {
      const uint32_t code_width = histogram_code_width(width);
      return load_little_endian_exact(
          histogram.data() + index * histogram_record_width(width),
          code_width);
    }
    static uint32_t histogram_entry_count(
        const std::vector<uint8_t> &histogram, size_t index,
        uint32_t width) noexcept {
      const uint32_t code_width = histogram_code_width(width);
      return load_little_endian_exact(
          histogram.data() + index * histogram_record_width(width) +
              code_width,
          4);
    }
    static void histogram_set_count(std::vector<uint8_t> &histogram,
                                    size_t index, uint32_t count,
                                    uint32_t width) noexcept {
      const uint32_t code_width = histogram_code_width(width);
      store_little_endian_exact(
          histogram.data() + index * histogram_record_width(width) +
              code_width,
          count, 4);
    }
    static void histogram_set(std::vector<uint8_t> &histogram, size_t index,
                              uint32_t code, uint32_t count,
                              uint32_t width) noexcept {
      const uint32_t code_width = histogram_code_width(width);
      uint8_t *const record =
          histogram.data() + index * histogram_record_width(width);
      store_little_endian_exact(record, code, code_width);
      store_little_endian_exact(record + code_width, count, 4);
    }
    static size_t histogram_lower_bound(
        const std::vector<uint8_t> &histogram, uint32_t code,
        uint32_t width) noexcept {
      const size_t entries = histogram_size(histogram, width);
      // Identity-coded compact streams quickly form a dense low-ID prefix.
      // Once present, the sorted entry index is the code itself and avoids
      // four branchy unaligned loads on every B+ rank/update.
      if (width <= 1 && code < entries &&
          histogram[code * histogram_record_width(width)] == code)
        return code;
      size_t first = 0;
      size_t count = entries;
      while (count != 0) {
        const size_t step = count / 2;
        const size_t middle = first + step;
        if (histogram_code(histogram, middle, width) < code) {
          first = middle + 1;
          count -= step + 1;
        } else {
          count = step;
        }
      }
      return first;
    }
    static void histogram_reserve(std::vector<uint8_t> &histogram,
                                  size_t entries, uint32_t width) {
      const size_t bytes = checked_product(entries,
                                           histogram_record_width(width));
      if (histogram.capacity() < bytes)
        histogram.reserve(bytes);
    }
    static void histogram_insert(std::vector<uint8_t> &histogram,
                                 size_t index, uint32_t code,
                                 uint32_t count, uint32_t width) noexcept {
      const size_t record_width = histogram_record_width(width);
      const size_t offset = index * record_width;
      const size_t old_size = histogram.size();
      histogram.resize(old_size + record_width);
      std::memmove(histogram.data() + offset + record_width,
                   histogram.data() + offset, old_size - offset);
      histogram_set(histogram, index, code, count, width);
    }
    static std::vector<uint8_t> repack_histogram(
        const std::vector<uint8_t> &histogram, uint32_t source_width,
        uint32_t destination_width) {
      const size_t source_record_width = histogram_record_width(source_width);
      const size_t destination_record_width =
          histogram_record_width(destination_width);
      const size_t entries = histogram_size(histogram, source_width);
      const size_t capacity_entries =
          histogram.capacity() / source_record_width;
      std::vector<uint8_t> result;
      result.reserve(checked_product(capacity_entries,
                                     destination_record_width));
      result.resize(checked_product(entries, destination_record_width));
      for (size_t index = 0; index < entries; ++index)
        histogram_set(result, index,
                      histogram_code(histogram, index, source_width),
                      histogram_entry_count(histogram, index, source_width),
                      destination_width);
      return result;
    }

    static uint32_t popcount64(uint64_t value) noexcept {
      return population_count64(value);
    }
    static size_t compact_histogram_rank(const Leaf &leaf,
                                         uint32_t code) noexcept {
      const uint32_t word = code >> 6;
      size_t rank = 0;
      for (uint32_t index = 0; index < word; ++index)
        rank += popcount64(leaf.histogram_bitmap[index]);
      const uint32_t bit = code & 63u;
      const uint64_t before = bit == 0 ? 0 : ((uint64_t{1} << bit) - 1);
      return rank + popcount64(leaf.histogram_bitmap[word] & before);
    }
    static bool compact_histogram_contains(const Leaf &leaf,
                                           uint32_t code) noexcept {
      return code <= 255u &&
             (leaf.histogram_bitmap[code >> 6] &
              (uint64_t{1} << (code & 63u))) != 0;
    }
    static size_t leaf_histogram_size(const Leaf &leaf,
                                      uint32_t width) noexcept {
      if (leaf.histogram_wide)
        return histogram_size(leaf.histogram, width);
      size_t result = 0;
      for (uint64_t word : leaf.histogram_bitmap)
        result += popcount64(word);
      return result;
    }
    static uint32_t leaf_histogram_count(const Leaf &leaf, uint32_t code,
                                         uint32_t width) noexcept {
      if (leaf.histogram_wide)
      {
        const size_t found = histogram_lower_bound(leaf.histogram, code, width);
        return found != histogram_size(leaf.histogram, width) &&
                       histogram_code(leaf.histogram, found, width) == code
                   ? histogram_entry_count(leaf.histogram, found, width)
                   : 0;
      }
      if (!compact_histogram_contains(leaf, code))
        return 0;
      const uint8_t encoded =
          leaf.histogram_counts[compact_histogram_rank(leaf, code)];
      if (encoded != 0)
        return encoded;
      for (uint32_t index = 0; index < leaf.histogram_overflow_size; ++index) {
        const uint32_t overflow = leaf.histogram_overflows[index];
        if ((overflow & 255u) == code)
          return 256u + (overflow >> 8);
      }
      return 256u;
    }
    static void reserve_leaf_histogram(Leaf &leaf, size_t entries,
                                       uint32_t width) {
      if (leaf.histogram_wide) {
        histogram_reserve(leaf.histogram, entries, width);
      } else if (leaf.histogram_counts.capacity() < entries) {
        leaf.histogram_counts.reserve(std::min<size_t>(256, (entries + 15) & ~size_t{15}));
      }
    }
    static void clear_leaf_histogram(Leaf &leaf) noexcept {
      if (leaf.histogram_wide) {
        leaf.histogram.clear();
        return;
      }
      leaf.histogram_bitmap.fill(0);
      leaf.histogram_counts.clear();
      leaf.histogram_overflow_size = 0;
    }
    static void increment_leaf_histogram(Leaf &leaf, uint32_t code,
                                         uint32_t width) noexcept {
      if (leaf.histogram_wide) {
        const size_t found = histogram_lower_bound(leaf.histogram, code, width);
        if (found != histogram_size(leaf.histogram, width) &&
            histogram_code(leaf.histogram, found, width) == code) {
          histogram_set_count(
              leaf.histogram, found,
              histogram_entry_count(leaf.histogram, found, width) + 1u,
              width);
        } else {
          histogram_insert(leaf.histogram, found, code, 1, width);
        }
        return;
      }
      // The width-promotion transaction converts all leaves before a code
      // above 255 can become live.
      if (code > 255u)
        std::terminate();
      const size_t rank = compact_histogram_rank(leaf, code);
      const uint64_t bit = uint64_t{1} << (code & 63u);
      uint64_t &word = leaf.histogram_bitmap[code >> 6];
      if ((word & bit) == 0) {
        const size_t old_size = leaf.histogram_counts.size();
        leaf.histogram_counts.resize(old_size + 1);
        std::memmove(leaf.histogram_counts.data() + rank + 1,
                     leaf.histogram_counts.data() + rank, old_size - rank);
        leaf.histogram_counts[rank] = 1;
        word |= bit;
        return;
      }
      uint8_t &encoded = leaf.histogram_counts[rank];
      if (encoded == 255) {
        encoded = 0;
        return;
      }
      if (encoded != 0) {
        ++encoded;
        return;
      }
      for (uint32_t index = 0; index < leaf.histogram_overflow_size; ++index) {
        uint32_t &overflow = leaf.histogram_overflows[index];
        if ((overflow & 255u) == code) {
          overflow += 1u << 8;
          return;
        }
      }
      if (leaf.histogram_overflow_size >= leaf.histogram_overflows.size())
        std::terminate();
      leaf.histogram_overflows[leaf.histogram_overflow_size++] =
          code | (1u << 8);
    }
    static std::vector<uint8_t> repack_leaf_histogram(
        const Leaf &leaf, uint32_t source_width, uint32_t destination_width) {
      if (leaf.histogram_wide)
        return repack_histogram(leaf.histogram, source_width,
                                destination_width);
      std::vector<uint8_t> result;
      const size_t entries = leaf_histogram_size(leaf, source_width);
      result.reserve(checked_product(
          std::max(entries, leaf.histogram_counts.capacity()),
          histogram_record_width(destination_width)));
      result.resize(checked_product(entries,
                                    histogram_record_width(destination_width)));
      size_t index = 0;
      for (uint32_t word_index = 0; word_index < 4; ++word_index) {
        uint64_t word = leaf.histogram_bitmap[word_index];
        while (word != 0) {
          const uint32_t bit = trailing_zero_count64(word);
          const uint32_t code = word_index * 64u + bit;
          histogram_set(result, index++, code,
                        leaf_histogram_count(leaf, code, source_width),
                        destination_width);
          word &= word - 1;
        }
      }
      return result;
    }
    static void commit_leaf_histogram(Leaf &leaf,
                                      std::vector<uint8_t> wide) noexcept {
      leaf.histogram.swap(wide);
      std::vector<uint8_t>().swap(leaf.histogram_counts);
      leaf.histogram_bitmap.fill(0);
      leaf.histogram_overflow_size = 0;
      leaf.histogram_wide = true;
    }

    static void store_bytes(uint8_t *destination, uint32_t value,
                            uint32_t width) noexcept {
      for (uint32_t byte = 0; byte < width; ++byte)
        destination[byte] = static_cast<uint8_t>(value >> (byte * 8));
    }
    static uint32_t load_bytes(const uint8_t *source,
                               uint32_t width) noexcept {
      uint32_t value = 0;
      for (uint32_t byte = 0; byte < width; ++byte)
        value |= static_cast<uint32_t>(source[byte]) << (byte * 8);
      return value;
    }
    static bool zigzag_delta(uint32_t previous, uint32_t value,
                             uint32_t &encoded) noexcept {
      const int64_t delta = static_cast<int64_t>(value) - previous;
      const uint64_t zigzag = delta >= 0
                                  ? static_cast<uint64_t>(delta) * 2
                                  : static_cast<uint64_t>(-delta) * 2 - 1;
      if (zigzag > std::numeric_limits<uint32_t>::max())
        return false;
      encoded = static_cast<uint32_t>(zigzag);
      return true;
    }
    static uint32_t undo_zigzag(uint32_t previous,
                                uint32_t encoded) noexcept {
      const int64_t delta = (encoded & 1u)
                                ? -static_cast<int64_t>((encoded >> 1) + 1u)
                                : static_cast<int64_t>(encoded >> 1);
      return static_cast<uint32_t>(static_cast<int64_t>(previous) + delta);
    }
    void encode_lcs_into(const uint32_t *values, uint32_t count,
                         LcsCodec &codec,
                         uint32_t raw8_physical_capacity) const {
      uint32_t raw_max = 0;
      uint32_t minimum = count == 0 ? 0 : values[0];
      for (uint32_t slot = 0; slot < count; ++slot) {
        raw_max = std::max(raw_max, values[slot]);
        minimum = std::min(minimum, values[slot]);
      }
      const uint32_t raw_width = raw_max <= 255u
                                     ? 1u
                                     : (raw_max <= 65535u ? 2u : 4u);
      const LcsKind raw_kind = raw_width == 1
                                   ? LcsKind::Raw8
                                   : (raw_width == 2 ? LcsKind::Raw16
                                                     : LcsKind::Raw32);
      const size_t raw_cost = static_cast<size_t>(count) * raw_width;
      const bool packed4_valid = raw_max <= 15u;
      const size_t packed4_cost = (static_cast<size_t>(count) + 1) / 2;

      const bool for_valid = raw_max - minimum <= 255u;
      const size_t for_cost = count;

      bool delta4_valid = true;
      bool delta8_valid = true;
      size_t delta_count = 0;
      for (uint32_t slot = 0; slot < count; ++slot) {
        if ((slot & 15u) == 0)
          continue;
        uint32_t encoded = 0;
        if (!zigzag_delta(values[slot - 1], values[slot], encoded) ||
            encoded > 255u) {
          delta4_valid = false;
          delta8_valid = false;
        } else if (encoded > 15u) {
          delta4_valid = false;
        }
        ++delta_count;
      }
      const size_t checkpoint_count = (count + 15u) / 16u;
      const size_t checkpoint_cost = checkpoint_count * lcs_width_;
      const size_t delta4_payload = (delta_count + 1) / 2;
      const size_t delta4_cost = checkpoint_cost + delta4_payload;
      const size_t delta8_cost = checkpoint_cost + delta_count;

      // Select by exact allocated payload size. RAW wins ties; FOR wins a tie
      // with DELTA after it has already beaten RAW.
      LcsKind kind = raw_kind;
      size_t payload_size = raw_cost;
      size_t best_cost = raw_cost;
      if (packed4_valid && packed4_cost < best_cost) {
        kind = LcsKind::Packed4;
        payload_size = packed4_cost;
        best_cost = packed4_cost;
      }
      // RAW8 is deliberately sticky: it enables allocation-free in-place
      // mutation, and compressed alternatives save too little to repay a
      // full leaf rebuild on the low-LCS random workload.
      if (raw_width != 1) {
        if (for_valid && for_cost < best_cost) {
          kind = LcsKind::For8;
          payload_size = for_cost;
          best_cost = for_cost;
        }
        if (delta4_valid && delta4_cost < best_cost) {
          kind = LcsKind::Delta4;
          payload_size = delta4_payload;
          best_cost = delta4_cost;
        } else if (delta8_valid && delta8_cost < best_cost) {
          kind = LcsKind::Delta8;
          payload_size = delta_count;
          best_cost = delta8_cost;
        }
      }

      // Mutable codecs keep vector size (not just allocation capacity) equal
      // to their physical leaf payload, so insertion may touch the next byte
      // without allocation. Selection above remains based on live bytes.
      if (kind == LcsKind::Packed4)
        payload_size = (static_cast<size_t>(raw8_physical_capacity) + 1) / 2;
      if (kind == LcsKind::Raw8)
        payload_size = raw8_physical_capacity;

      // This is the only payload preparation: resize, clear, then pack the
      // selected representation directly into the reusable destination.
      // reserve(exact) avoids resize's geometric growth, keeping the three
      // retained scratch capacities below the long-context memory budget.
      if (kind == LcsKind::Packed4 &&
          codec.payload.capacity() != payload_size) {
        std::vector<uint8_t> exact_payload(payload_size);
        codec.payload.swap(exact_payload);
      } else if (codec.payload.capacity() < payload_size) {
        codec.payload.reserve(payload_size);
      }
      codec.payload.resize(payload_size);
      std::fill(codec.payload.begin(), codec.payload.end(), uint8_t{0});
      if (kind == LcsKind::Packed4) {
        // Packed absolute nibbles have neither an anchor nor checkpoints; do
        // not retain stale DELTA scratch in a live Packed4 leaf.
        std::vector<uint8_t>().swap(codec.checkpoints);
      } else {
        codec.checkpoints.clear();
      }
      codec.anchor = kind == LcsKind::For8 ? minimum : 0;
      codec.kind = kind;
      if (kind == LcsKind::Packed4) {
        for (uint32_t slot = 0; slot < count; ++slot) {
          const uint32_t shift = (slot & 1u) * 4;
          codec.payload[slot >> 1] |=
              static_cast<uint8_t>(values[slot] << shift);
        }
      } else if (kind == LcsKind::Raw8) {
        for (uint32_t slot = 0; slot < count; ++slot)
          codec.payload[slot] = static_cast<uint8_t>(values[slot]);
      } else if (kind == LcsKind::Raw16 || kind == LcsKind::Raw32) {
        for (uint32_t slot = 0; slot < count; ++slot)
          store_bytes(codec.payload.data() + static_cast<size_t>(slot) * raw_width,
                      values[slot], raw_width);
      } else if (kind == LcsKind::For8) {
        for (uint32_t slot = 0; slot < count; ++slot)
          codec.payload[slot] = static_cast<uint8_t>(values[slot] - minimum);
      } else {
        const size_t checkpoint_bytes = checkpoint_count * lcs_width_;
        if (codec.checkpoints.capacity() < checkpoint_bytes)
          codec.checkpoints.reserve(checkpoint_bytes);
        codec.checkpoints.resize(checkpoint_bytes);
        size_t delta_index = 0;
        for (uint32_t slot = 0; slot < count; ++slot) {
          if ((slot & 15u) == 0) {
            store_bytes(codec.checkpoints.data() +
                            static_cast<size_t>(slot >> 4) * lcs_width_,
                        values[slot], lcs_width_);
          } else {
            uint32_t encoded = 0;
            (void)zigzag_delta(values[slot - 1], values[slot], encoded);
            if (kind == LcsKind::Delta4) {
              uint8_t &byte = codec.payload[delta_index >> 1];
              if ((delta_index & 1u) == 0)
                byte = static_cast<uint8_t>(encoded);
              else
                byte |= static_cast<uint8_t>(encoded << 4);
              ++delta_index;
            } else {
              codec.payload[delta_index++] = static_cast<uint8_t>(encoded);
            }
          }
        }
      }
    }
    static void swap_lcs(LcsCodec &left, LcsCodec &right) noexcept {
      left.payload.swap(right.payload);
      left.checkpoints.swap(right.checkpoints);
      std::swap(left.anchor, right.anchor);
      std::swap(left.kind, right.kind);
    }
    uint32_t decode_lcs_value(const Leaf &leaf,
                              uint32_t slot) const noexcept {
      const LcsCodec &codec = leaf.lcs;
      if (codec.kind == LcsKind::Packed4)
        return (codec.payload[slot >> 1] >> ((slot & 1u) * 4)) & 15u;
      if (codec.kind == LcsKind::Raw8)
        return codec.payload[slot];
      if (codec.kind == LcsKind::Raw16)
        return load_bytes(codec.payload.data() + static_cast<size_t>(slot) * 2,
                          2);
      if (codec.kind == LcsKind::Raw32)
        return load_bytes(codec.payload.data() + static_cast<size_t>(slot) * 4,
                          4);
      if (codec.kind == LcsKind::For8)
        return codec.anchor + codec.payload[slot];
      const uint32_t block = slot >> 4;
      const uint32_t block_start = block << 4;
      uint32_t value = load_bytes(
          codec.checkpoints.data() + static_cast<size_t>(block) * lcs_width_,
          lcs_width_);
      size_t delta_index = static_cast<size_t>(block) * 15;
      for (uint32_t current = block_start + 1; current <= slot; ++current) {
        const uint32_t encoded = codec.kind == LcsKind::Delta4
                                     ? (codec.payload[delta_index >> 1] >>
                                        ((delta_index & 1u) * 4)) & 15u
                                     : codec.payload[delta_index];
        value = undo_zigzag(value, encoded);
        ++delta_index;
      }
      return value;
    }
    void decode_lcs(const Leaf &leaf, uint32_t *values) const noexcept {
      const LcsCodec &codec = leaf.lcs;
      if (codec.kind == LcsKind::Packed4) {
        const uint32_t pairs = leaf.size >> 1;
        for (uint32_t pair = 0; pair < pairs; ++pair) {
          const uint8_t packed = codec.payload[pair];
          values[pair * 2] = packed & 15u;
          values[pair * 2 + 1] = packed >> 4;
        }
        if ((leaf.size & 1u) != 0)
          values[leaf.size - 1] = codec.payload[leaf.size >> 1] & 15u;
        return;
      }
      if (codec.kind == LcsKind::Raw8) {
        for (uint32_t slot = 0; slot < leaf.size; ++slot)
          values[slot] = codec.payload[slot];
        return;
      }
      if (codec.kind == LcsKind::Raw16 || codec.kind == LcsKind::Raw32) {
        const uint32_t width = codec.kind == LcsKind::Raw16 ? 2u : 4u;
        for (uint32_t slot = 0; slot < leaf.size; ++slot)
          values[slot] = load_bytes(
              codec.payload.data() + static_cast<size_t>(slot) * width, width);
        return;
      }
      if (codec.kind == LcsKind::For8) {
        for (uint32_t slot = 0; slot < leaf.size; ++slot)
          values[slot] = codec.anchor + codec.payload[slot];
        return;
      }
      size_t delta_index = 0;
      for (uint32_t slot = 0; slot < leaf.size; ++slot) {
        if ((slot & 15u) == 0) {
          values[slot] = load_bytes(
              codec.checkpoints.data() +
                  static_cast<size_t>(slot >> 4) * lcs_width_,
              lcs_width_);
          continue;
        }
        const uint32_t encoded = codec.kind == LcsKind::Delta4
                                     ? (codec.payload[delta_index >> 1] >>
                                        ((delta_index & 1u) * 4)) & 15u
                                     : codec.payload[delta_index];
        values[slot] = undo_zigzag(values[slot - 1], encoded);
        ++delta_index;
      }
    }

    static uint32_t payload_code_width(const Leaf &leaf) noexcept {
      return leaf.codes32 ? 4u : (leaf.codes16 ? 2u : (leaf.codes8 ? 1u : 0u));
    }
    static size_t code_storage_bytes(uint32_t capacity,
                                     uint32_t width) {
      if (width == 0)
        return (static_cast<size_t>(capacity) + 1) / 2;
      return checked_product(capacity, width);
    }
    static uint32_t pa_bit_width(size_t maximum_endpoint) noexcept {
      uint32_t bits = 1;
      while (maximum_endpoint > 1) {
        ++bits;
        maximum_endpoint >>= 1;
      }
      return bits;
    }
    static size_t pa_payload_bytes(uint32_t payload_capacity,
                                   uint32_t width) {
      const size_t bits = checked_product(payload_capacity, width);
      return bits / 8 + (bits % 8 != 0);
    }
    size_t pa_payload_bytes(uint32_t payload_capacity) const {
      return pa_payload_bytes(payload_capacity, pa_bits_);
    }
    static size_t pa_storage_bytes(uint32_t payload_capacity,
                                   uint32_t width) {
      size_t bytes = pa_payload_bytes(payload_capacity, width);
      checked_add(bytes, 8);
      return bytes;
    }
    size_t pa_storage_bytes(uint32_t payload_capacity) const {
      return pa_storage_bytes(payload_capacity, pa_bits_);
    }
    std::unique_ptr<Leaf> make_leaf(uint32_t width,
                                    uint32_t payload_capacity) const {
      auto result = std::make_unique<Leaf>();
      result->payload_capacity = payload_capacity;
      result->histogram_wide = width > 1;
      if (width == 4)
        result->codes32 = std::make_unique<uint32_t[]>(payload_capacity);
      else if (width == 2)
        result->codes16 = std::make_unique<uint16_t[]>(payload_capacity);
      else if (width == 1)
        result->codes8 = std::make_unique<uint8_t[]>(payload_capacity);
      else
        result->codes4 = std::make_unique<uint8_t[]>(
            code_storage_bytes(payload_capacity, 0));
      result->pa =
          std::make_unique<uint8_t[]>(pa_storage_bytes(payload_capacity));
      std::memset(result->pa.get(), 0, pa_storage_bytes(payload_capacity));
      return result;
    }
    void copy_payload(const Leaf &source, Leaf &destination) const noexcept {
      const uint32_t destination_width = payload_code_width(destination);
      if (destination_width == code_width_) {
        if (code_width_ == 0)
          std::memcpy(destination.codes4.get(), source.codes4.get(),
                      (static_cast<size_t>(source.size) + 1) / 2);
        else if (code_width_ == 1)
          std::memcpy(destination.codes8.get(), source.codes8.get(), source.size);
        else if (code_width_ == 2)
          std::memcpy(destination.codes16.get(), source.codes16.get(),
                      static_cast<size_t>(source.size) * 2);
        else
          std::memcpy(destination.codes32.get(), source.codes32.get(),
                      static_cast<size_t>(source.size) * 4);
      } else if (code_width_ == 0 && destination_width == 1) {
        const uint32_t pairs = source.size >> 1;
        for (uint32_t pair = 0; pair < pairs; ++pair) {
          const uint8_t packed = source.codes4[pair];
          destination.codes8[pair * 2] = packed & 0x0fu;
          destination.codes8[pair * 2 + 1] = packed >> 4;
        }
        if ((source.size & 1u) != 0)
          destination.codes8[source.size - 1] =
              source.codes4[source.size >> 1] & 0x0fu;
      } else {
        // Width-changing growth is a rare transactional staging operation.
        for (uint32_t slot = 0; slot < source.size; ++slot) {
          const uint32_t code = get_code(source, slot);
          if (destination.codes32)
            destination.codes32[slot] = code;
          else if (destination.codes16)
            destination.codes16[slot] = static_cast<uint16_t>(code);
          else
            destination.codes8[slot] = static_cast<uint8_t>(code);
        }
      }
      std::memcpy(destination.pa.get(), source.pa.get(),
                  pa_payload_bytes(source.payload_capacity));
    }
    static void commit_payload(Leaf &leaf,
                               std::unique_ptr<Leaf> prepared) noexcept {
      leaf.codes4.swap(prepared->codes4);
      leaf.codes16.swap(prepared->codes16);
      leaf.codes32.swap(prepared->codes32);
      leaf.codes8.swap(prepared->codes8);
      leaf.pa.swap(prepared->pa);
      leaf.payload_capacity = prepared->payload_capacity;
    }
    uint32_t get_code(const Leaf &leaf, uint32_t slot) const noexcept {
      if (code_width_ == 4) return leaf.codes32[slot];
      if (code_width_ == 2) return leaf.codes16[slot];
      if (code_width_ == 1) return leaf.codes8[slot];
      const uint8_t packed = leaf.codes4[slot >> 1];
      return (packed >> ((slot & 1u) * 4)) & 0x0fu;
    }
    void set_code(Leaf &leaf, uint32_t slot, uint32_t value) noexcept {
      if (code_width_ == 4) leaf.codes32[slot] = value;
      else if (code_width_ == 2) leaf.codes16[slot] = static_cast<uint16_t>(value);
      else if (code_width_ == 1) leaf.codes8[slot] = static_cast<uint8_t>(value);
      else {
        uint8_t &packed = leaf.codes4[slot >> 1];
        const uint32_t shift = (slot & 1u) * 4;
        packed = static_cast<uint8_t>((packed & ~(0x0fu << shift)) |
                                      ((value & 0x0fu) << shift));
      }
    }
    static uint64_t little_endian_window(const uint8_t *source) noexcept {
      uint64_t value;
      std::memcpy(&value, source, sizeof(value));
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
      value = __builtin_bswap64(value);
#endif
      return value;
    }
    static void store_little_endian_window(uint8_t *destination,
                                           uint64_t value) noexcept {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
      value = __builtin_bswap64(value);
#endif
      std::memcpy(destination, &value, sizeof(value));
    }
    uint32_t get_pa_bits(const Leaf &leaf, uint32_t slot) const noexcept {
      return get_pa_bits(leaf.pa.get(), slot, pa_bits_);
    }
    static uint32_t get_pa_bits(const uint8_t *payload, uint32_t slot,
                                uint32_t width) noexcept {
      const size_t first_bit = static_cast<size_t>(slot) * width;
      const uint32_t bit_offset = static_cast<uint32_t>(first_bit & 7u);
      const uint64_t mask = (uint64_t{1} << width) - 1;
      const uint64_t window =
          little_endian_window(payload + (first_bit >> 3));
      return static_cast<uint32_t>((window >> bit_offset) & mask);
    }
    uint32_t get_pa(const Leaf &leaf, uint32_t slot) const noexcept {
      if (sentinel_leaf_ < leaves_.size() &&
          &leaf == &leaves_[sentinel_leaf_] && slot == sentinel_slot_)
        return static_cast<uint32_t>(size_ - 1);
      return get_pa_bits(leaf, slot);
    }
    uint32_t get_lcs(const Leaf &leaf, uint32_t slot) const noexcept {
      return decode_lcs_value(leaf, slot);
    }
    static void set_packed4(uint8_t *payload, uint32_t slot,
                            uint32_t value) noexcept {
      uint8_t &packed = payload[slot >> 1];
      const uint32_t shift = (slot & 1u) * 4;
      packed = static_cast<uint8_t>((packed & ~(15u << shift)) |
                                    ((value & 15u) << shift));
    }
    static void move_packed4_right(uint8_t *values, uint32_t slot,
                                   uint32_t count) noexcept {
      if (count == 0)
        return;
      // Shift one nibble with a reverse byte pass. Each destination byte is
      // assembled from adjacent source bytes; no per-value or per-bit loop.
      const uint32_t first = slot + 1;
      const uint32_t last = slot + count;
      const uint32_t first_byte = first >> 1;
      const uint32_t last_byte = last >> 1;
      for (uint32_t byte = last_byte + 1; byte-- > first_byte;) {
        const uint8_t original = values[byte];
        const uint8_t previous = byte == 0 ? 0 : values[byte - 1];
        uint8_t shifted =
            static_cast<uint8_t>((previous >> 4) | (original << 4));
        if (byte == first_byte && (first & 1u) != 0)
          shifted = static_cast<uint8_t>((shifted & 0xf0u) |
                                         (original & 0x0fu));
        if (byte == last_byte && (last & 1u) == 0)
          shifted = static_cast<uint8_t>((shifted & 0x0fu) |
                                         (original & 0xf0u));
        values[byte] = shifted;
      }
    }
    static void move_mutable_lcs_right(Leaf &leaf, uint32_t slot,
                                       uint32_t count) noexcept {
      if (leaf.lcs.kind == LcsKind::Packed4) {
        move_packed4_right(leaf.lcs.payload.data(), slot, count);
      } else if (count != 0) {
        std::memmove(leaf.lcs.payload.data() + slot + 1,
                     leaf.lcs.payload.data() + slot, count);
      }
    }
    static void set_mutable_lcs(Leaf &leaf, uint32_t slot,
                                uint32_t value) noexcept {
      if (leaf.lcs.kind == LcsKind::Packed4)
        set_packed4(leaf.lcs.payload.data(), slot, value);
      else
        leaf.lcs.payload[slot] = static_cast<uint8_t>(value);
    }
    void set_pa_bits(Leaf &leaf, uint32_t slot, uint32_t value) const noexcept {
      set_pa_bits(leaf.pa.get(), slot, value, pa_bits_);
    }
    static void set_pa_bits(uint8_t *payload, uint32_t slot, uint32_t value,
                            uint32_t width) noexcept {
      const size_t first_bit = static_cast<size_t>(slot) * width;
      const uint32_t bit_offset = static_cast<uint32_t>(first_bit & 7u);
      const uint64_t value_mask = (uint64_t{1} << width) - 1;
      const uint64_t field_mask = value_mask << bit_offset;
      uint8_t *const destination = payload + (first_bit >> 3);
      uint64_t window = little_endian_window(destination);
      window = (window & ~field_mask) |
               ((static_cast<uint64_t>(value) & value_mask) << bit_offset);
      store_little_endian_window(destination, window);
    }
    static void move_pa_bits_right(uint8_t *payload, uint32_t slot,
                                   uint32_t count, uint32_t width) noexcept {
      if (count == 0)
        return;
      const size_t source_first = static_cast<size_t>(slot) * width;
      const size_t bit_count = static_cast<size_t>(count) * width;
      const size_t destination_first = source_first + width;
      const size_t destination_last = destination_first + bit_count;
      if ((width & 7u) == 0) {
        std::memmove(payload + (destination_first >> 3),
                     payload + (source_first >> 3), bit_count >> 3);
        return;
      }

      // Shift the packed stream right by one field.  The reverse byte pass is
      // overlap-safe: every source byte is at or below the destination byte,
      // and the current destination is read before it is replaced.  Boundary
      // masks preserve every bit outside the exact destination range.
      const size_t byte_shift = width >> 3;
      const uint32_t bit_shift = width & 7u;
      const size_t first_byte = destination_first >> 3;
      const size_t last_byte = (destination_last - 1) >> 3;
      for (size_t byte = last_byte + 1; byte-- > first_byte;) {
        const size_t upper_source = byte - byte_shift;
        uint8_t shifted =
            static_cast<uint8_t>(payload[upper_source] << bit_shift);
        if (upper_source != 0)
          shifted = static_cast<uint8_t>(
              shifted | (payload[upper_source - 1] >> (8u - bit_shift)));

        const size_t byte_first = byte << 3;
        const uint32_t first = static_cast<uint32_t>(
            destination_first > byte_first ? destination_first - byte_first
                                           : 0);
        const uint32_t last = static_cast<uint32_t>(
            destination_last < byte_first + 8
                ? destination_last - byte_first
                : 8);
        const uint8_t mask = static_cast<uint8_t>(
            ((uint16_t{1} << last) - 1u) &
            ~((uint16_t{1} << first) - 1u));
        payload[byte] = static_cast<uint8_t>(
            (payload[byte] & static_cast<uint8_t>(~mask)) | (shifted & mask));
      }
    }
    Location locate(size_t rank) const noexcept {
      if (root_is_leaf_)
        return {root_, static_cast<uint32_t>(rank)};
      uint32_t node_index = root_;
      for (;;) {
        const Node &node = nodes_[node_index];
        uint32_t slot = 0;
        while (slot + 1 < node.count && rank >= node.weights[slot]) {
          rank -= node.weights[slot++];
        }
        if (node.children_are_leaves)
          return {node.children[slot], static_cast<uint32_t>(rank)};
        node_index = node.children[slot];
      }
    }
    size_t prefix(uint32_t leaf_index) const noexcept {
      size_t result = 0;
      uint32_t parent = leaves_[leaf_index].parent;
      uint32_t slot = leaves_[leaf_index].parent_slot;
      while (parent != kNil) {
        const Node &node = nodes_[parent];
        for (uint32_t index = 0; index < slot; ++index)
          result += node.weights[index];
        slot = node.parent_slot;
        parent = node.parent;
      }
      return result;
    }
    void move_right(Leaf &leaf, uint32_t slot, uint32_t count) noexcept {
      if (!count) return;
      if (code_width_ == 4) std::memmove(leaf.codes32.get() + slot + 1, leaf.codes32.get() + slot, count * 4);
      else if (code_width_ == 2) std::memmove(leaf.codes16.get() + slot + 1, leaf.codes16.get() + slot, count * 2);
      else if (code_width_ == 1) std::memmove(leaf.codes8.get() + slot + 1, leaf.codes8.get() + slot, count);
      else {
        move_packed4_right(leaf.codes4.get(), slot, count);
      }
      if ((pa_bits_ & 7u) == 0) {
        move_pa_bits_right(leaf.pa.get(), slot, count, pa_bits_);
      } else {
        // For non-byte widths, a field loop touches fewer cache lines than a
        // byte-stream shift on large leaves. Each field remains one pair of
        // unaligned uint64 windows, never a per-bit loop.
        for (uint32_t offset = count; offset != 0; --offset)
          set_pa_bits(leaf, slot + offset,
                      get_pa_bits(leaf, slot + offset - 1));
      }
    }
    void insert_local(Leaf &leaf, uint32_t slot, uint32_t code,
                      uint32_t pa_value, uint32_t lcs_value) noexcept {
      move_right(leaf, slot, leaf.size - slot);
      set_code(leaf, slot, code); set_pa_bits(leaf, slot, pa_value);
      (void)lcs_value; // The prepared codec already contains the inserted row.
      ++leaf.size;
    }
    void write_leaf(Leaf &leaf, const uint32_t *codes, const uint32_t *pas,
                    const uint32_t *lcss, uint32_t count) noexcept {
      leaf.size = count;
      if (code_width_ == 0) {
        const uint32_t pairs = count >> 1;
        for (uint32_t pair = 0; pair < pairs; ++pair)
          leaf.codes4[pair] = static_cast<uint8_t>(
              codes[pair * 2] | (codes[pair * 2 + 1] << 4));
        if ((count & 1u) != 0)
          leaf.codes4[count >> 1] = static_cast<uint8_t>(codes[count - 1]);
      } else {
        // Split rebuilding is rare transactional staging; live insertion
        // shifts contiguous byte/word payloads in move_right().
        for (uint32_t slot = 0; slot < count; ++slot)
          set_code(leaf, slot, codes[slot]);
      }
      for (uint32_t slot = 0; slot < count; ++slot)
        set_pa_bits(leaf, slot, pas[slot]);
      (void)lcss; // LCS was encoded transactionally during prepare_insert().
    }
    void refresh(Leaf &leaf) noexcept {
      clear_leaf_histogram(leaf);
      leaf.min_lcs = std::numeric_limits<uint32_t>::max();
      leaf.min_lcs_count = 0;
      leaf.max1_pa = leaf.max2_pa = 0;
      std::array<uint32_t, kMaxLeafCapacity> lcs_values{};
      decode_lcs(leaf, lcs_values.data());
      for (uint32_t slot = 0; slot < leaf.size; ++slot) {
        if (!(&leaf == &leaves_[sentinel_leaf_] && slot == sentinel_slot_)) {
          const uint32_t code = get_code(leaf, slot);
          increment_leaf_histogram(leaf, code, code_width_);
        }
        const uint32_t lcs = lcs_values[slot];
        if (lcs < leaf.min_lcs) {
          leaf.min_lcs = lcs;
          leaf.min_lcs_count = 1;
        } else if (lcs == leaf.min_lcs) {
          ++leaf.min_lcs_count;
        }
        const uint32_t value = get_pa(leaf, slot);
        if (value >= leaf.max1_pa) { leaf.max2_pa = leaf.max1_pa; leaf.max1_pa = value; }
        else if (value > leaf.max2_pa) leaf.max2_pa = value;
      }
    }

    static void increment_histogram(std::vector<uint8_t> &histogram,
                                    uint32_t code,
                                    uint32_t width) noexcept {
      const size_t found = histogram_lower_bound(histogram, code, width);
      if (found != histogram_size(histogram, width) &&
          histogram_code(histogram, found, width) == code) {
        histogram_set_count(
            histogram, found,
            histogram_entry_count(histogram, found, width) + 1u, width);
      } else {
        histogram_insert(histogram, found, code, 1, width);
      }
    }

    void increment_histogram_up(uint32_t node, uint32_t code) noexcept {
      while (node != kNil) {
        increment_histogram(nodes_[node].histogram, code, code_width_);
        node = nodes_[node].parent;
      }
    }

    void increment_weight_up(uint32_t leaf) noexcept {
      uint32_t node = leaves_[leaf].parent;
      uint32_t slot = leaves_[leaf].parent_slot;
      while (node != kNil) {
        Node &entry = nodes_[node];
        ++entry.weights[slot];
        ++entry.weight;
        slot = entry.parent_slot;
        node = entry.parent;
      }
    }

    void add_leaf_lcs(Leaf &leaf, uint32_t value) noexcept {
      if (value < leaf.min_lcs) {
        leaf.min_lcs = value;
        leaf.min_lcs_count = 1;
      } else if (value == leaf.min_lcs) {
        ++leaf.min_lcs_count;
      }
    }

    void recompute_leaf_lcs(Leaf &leaf) noexcept {
      leaf.min_lcs = std::numeric_limits<uint32_t>::max();
      leaf.min_lcs_count = 0;
      std::array<uint32_t, kMaxLeafCapacity> values{};
      decode_lcs(leaf, values.data());
      for (uint32_t slot = 0; slot < leaf.size; ++slot) {
        const uint32_t value = values[slot];
        if (value < leaf.min_lcs) {
          leaf.min_lcs = value;
          leaf.min_lcs_count = 1;
        } else if (value == leaf.min_lcs) {
          ++leaf.min_lcs_count;
        }
      }
    }

    static uint32_t histogram_count(const std::vector<uint8_t> &histogram,
                                    uint32_t code,
                                    uint32_t width) noexcept {
      const size_t found = histogram_lower_bound(histogram, code, width);
      return found != histogram_size(histogram, width) &&
                     histogram_code(histogram, found, width) == code
                 ? histogram_entry_count(histogram, found, width)
                 : 0;
    }
    static void add_histogram_entry(std::vector<uint8_t> &histogram,
                                    uint32_t code, uint32_t count,
                                    uint32_t width) noexcept {
      const size_t found = histogram_lower_bound(histogram, code, width);
      if (found != histogram_size(histogram, width) &&
          histogram_code(histogram, found, width) == code) {
        histogram_set_count(
            histogram, found,
            histogram_entry_count(histogram, found, width) + count, width);
      } else {
        histogram_insert(histogram, found, code, count, width);
      }
    }
    static void merge_leaf_histogram(std::vector<uint8_t> &destination,
                                     const Leaf &leaf,
                                     uint32_t width) noexcept {
      if (leaf.histogram_wide) {
        const size_t entries = histogram_size(leaf.histogram, width);
        for (size_t index = 0; index < entries; ++index)
          add_histogram_entry(
              destination, histogram_code(leaf.histogram, index, width),
              histogram_entry_count(leaf.histogram, index, width), width);
        return;
      }
      for (uint32_t word_index = 0; word_index < 4; ++word_index) {
        uint64_t word = leaf.histogram_bitmap[word_index];
        while (word != 0) {
          const uint32_t bit = trailing_zero_count64(word);
          const uint32_t code = word_index * 64u + bit;
          add_histogram_entry(destination, code,
                              leaf_histogram_count(leaf, code, width), width);
          word &= word - 1;
        }
      }
    }
    size_t root_histogram_size() const noexcept {
      return root_is_leaf_
                 ? leaf_histogram_size(leaves_[root_], code_width_)
                 : histogram_size(nodes_[root_].histogram, code_width_);
    }
    void reserve_leaf_histogram(uint32_t leaf, size_t capacity) {
      reserve_leaf_histogram(leaves_[leaf], capacity, code_width_);
    }
    void reserve_path(uint32_t leaf, size_t capacity) {
      for (uint32_t node = leaves_[leaf].parent; node != kNil;
           node = nodes_[node].parent)
        histogram_reserve(nodes_[node].histogram, capacity, code_width_);
    }
    uint32_t child_weight(const Node &node, uint32_t slot) const noexcept {
      return node.children_are_leaves ? leaves_[node.children[slot]].size
                                      : nodes_[node.children[slot]].weight;
    }
    uint32_t child_min(const Node &node, uint32_t slot) const noexcept {
      return node.children_are_leaves ? leaves_[node.children[slot]].min_lcs
                                      : nodes_[node.children[slot]].min_lcs;
    }
    void child_max(const Node &node, uint32_t slot, uint32_t &max1,
                   uint32_t &max2) const noexcept {
      if (node.children_are_leaves) {
        max1 = leaves_[node.children[slot]].max1_pa;
        max2 = leaves_[node.children[slot]].max2_pa;
      } else {
        max1 = nodes_[node.children[slot]].max1_pa;
        max2 = nodes_[node.children[slot]].max2_pa;
      }
    }
    static void add_max(uint32_t value, uint32_t &max1,
                        uint32_t &max2) noexcept {
      if (value >= max1) { max2 = max1; max1 = value; }
      else if (value > max2) max2 = value;
    }
    void refresh_node_extrema(uint32_t node_index) noexcept {
      Node &node = nodes_[node_index];
      node.min_lcs = std::numeric_limits<uint32_t>::max();
      node.max1_pa = node.max2_pa = 0;
      for (uint32_t slot = 0; slot < node.count; ++slot) {
        node.min_lcs = std::min(node.min_lcs, child_min(node, slot));
        uint32_t first, second;
        child_max(node, slot, first, second);
        add_max(first, node.max1_pa, node.max2_pa);
        add_max(second, node.max1_pa, node.max2_pa);
      }
    }
    void refresh_extrema_paths(const uint32_t *leaves,
                               uint32_t leaf_count) noexcept {
      uint32_t current[2];
      uint32_t current_count = 0;
      for (uint32_t index = 0; index < leaf_count; ++index) {
        const uint32_t parent = leaves_[leaves[index]].parent;
        bool seen = parent == kNil;
        for (uint32_t prior = 0; prior < current_count; ++prior)
          seen = seen || current[prior] == parent;
        if (!seen)
          current[current_count++] = parent;
      }
      while (current_count != 0) {
        uint32_t next[2];
        uint32_t next_count = 0;
        for (uint32_t index = 0; index < current_count; ++index) {
          const uint32_t node = current[index];
          refresh_node_extrema(node);
          const uint32_t parent = nodes_[node].parent;
          bool seen = parent == kNil;
          for (uint32_t prior = 0; prior < next_count; ++prior)
            seen = seen || next[prior] == parent;
          if (!seen)
            next[next_count++] = parent;
        }
        current_count = next_count;
        for (uint32_t index = 0; index < next_count; ++index)
          current[index] = next[index];
      }
    }
    void set_parent(const Node &node, uint32_t slot,
                    uint32_t parent_index) noexcept {
      if (node.children_are_leaves) {
        leaves_[node.children[slot]].parent = parent_index;
        leaves_[node.children[slot]].parent_slot = slot;
      } else {
        nodes_[node.children[slot]].parent = parent_index;
        nodes_[node.children[slot]].parent_slot = slot;
      }
    }
    void refresh_node(uint32_t node_index) noexcept {
      Node &node = nodes_[node_index];
      node.weight = 0;
      node.min_lcs = std::numeric_limits<uint32_t>::max();
      node.max1_pa = node.max2_pa = 0;
      node.histogram.clear();
      for (uint32_t slot = 0; slot < node.count; ++slot) {
        set_parent(node, slot, node_index);
        node.weights[slot] = child_weight(node, slot);
        node.weight += node.weights[slot];
        node.min_lcs = std::min(node.min_lcs, child_min(node, slot));
        uint32_t first, second;
        child_max(node, slot, first, second);
        add_max(first, node.max1_pa, node.max2_pa);
        add_max(second, node.max1_pa, node.max2_pa);
        if (node.children_are_leaves) {
          merge_leaf_histogram(node.histogram, leaves_[node.children[slot]],
                               code_width_);
        } else {
          const std::vector<uint8_t> &child =
              nodes_[node.children[slot]].histogram;
          const size_t child_size = histogram_size(child, code_width_);
          for (size_t index = 0; index < child_size; ++index)
            add_histogram_entry(
                node.histogram, histogram_code(child, index, code_width_),
                histogram_entry_count(child, index, code_width_),
                code_width_);
        }
      }
    }
    void refresh_up(uint32_t node) noexcept {
      while (node != kNil) {
        refresh_node(node);
        node = nodes_[node].parent;
      }
    }
    uint32_t take_pending_node() noexcept {
      std::unique_ptr<Node> prepared = std::move(pending_nodes_.back());
      pending_nodes_.pop_back();
      const uint32_t index = static_cast<uint32_t>(nodes_.size());
      nodes_.push_back(std::move(*prepared));
      return index;
    }
    void make_root(uint32_t left, uint32_t right,
                   bool children_are_leaves) noexcept {
      const uint32_t root = take_pending_node();
      Node &node = nodes_[root];
      node.children_are_leaves = children_are_leaves;
      node.count = 2;
      node.children[0] = left;
      node.children[1] = right;
      node.parent = kNil;
      refresh_node(root);
      root_ = root;
      root_is_leaf_ = false;
    }
    void insert_child_after(uint32_t parent, uint32_t left_slot,
                            uint32_t child) noexcept {
      Node &node = nodes_[parent];
      if (node.count < kFanout) {
        for (uint32_t slot = node.count; slot > left_slot + 1; --slot)
          node.children[slot] = node.children[slot - 1];
        node.children[left_slot + 1] = child;
        ++node.count;
        refresh_node(parent);
        refresh_up(node.parent);
        return;
      }
      std::array<uint32_t, kFanout + 1> children;
      for (uint32_t out = 0; out <= kFanout; ++out) {
        if (out == left_slot + 1) children[out] = child;
        else children[out] = node.children[out - (out > left_slot + 1 ? 1u : 0u)];
      }
      const uint32_t old_parent = node.parent;
      const uint32_t old_parent_slot = node.parent_slot;
      const uint32_t sibling_index = take_pending_node();
      Node &sibling = nodes_[sibling_index];
      sibling.children_are_leaves = node.children_are_leaves;
      sibling.parent = old_parent;
      node.count = 8;
      sibling.count = 9;
      for (uint32_t slot = 0; slot < 8; ++slot) node.children[slot] = children[slot];
      for (uint32_t slot = 0; slot < 9; ++slot) sibling.children[slot] = children[slot + 8];
      refresh_node(parent);
      refresh_node(sibling_index);
      if (old_parent == kNil)
        make_root(parent, sibling_index, false);
      else
        insert_child_after(old_parent, old_parent_slot, sibling_index);
    }
    void insert_leaf_after(uint32_t left, uint32_t right) noexcept {
      if (root_is_leaf_) {
        make_root(left, right, true);
        return;
      }
      insert_child_after(leaves_[left].parent, leaves_[left].parent_slot, right);
    }
    uint32_t rank_prefix(size_t rank, uint32_t code) const noexcept {
      if (root_is_leaf_) {
        const Leaf &leaf = leaves_[root_];
        uint32_t result = 0;
        for (uint32_t slot = 0; slot < rank; ++slot)
          result += get_code(leaf, slot) == code;
        return result;
      }
      uint32_t result = 0, node_index = root_;
      for (;;) {
        const Node &node = nodes_[node_index];
        uint32_t slot = 0;
        while (slot < node.count && rank >= node.weights[slot]) {
          result += node.children_are_leaves
                        ? leaf_histogram_count(leaves_[node.children[slot]],
                                               code, code_width_)
                        : histogram_count(nodes_[node.children[slot]].histogram,
                                          code, code_width_);
          rank -= node.weights[slot++];
        }
        if (node.children_are_leaves) {
          if (slot == node.count) return result;
          const Leaf &leaf = leaves_[node.children[slot]];
          for (uint32_t leaf_slot = 0; leaf_slot < rank; ++leaf_slot)
            result += get_code(leaf, leaf_slot) == code;
          return result;
        }
        node_index = node.children[slot];
      }
    }
    uint32_t previous_leaf_with_min(uint32_t leaf, uint32_t threshold) const noexcept {
      uint32_t parent = leaves_[leaf].parent, slot = leaves_[leaf].parent_slot;
      while (parent != kNil) {
        const Node &node = nodes_[parent];
        while (slot > 0) {
          --slot;
          if (child_min(node, slot) < threshold)
            return rightmost_leaf_below(node.children[slot], node.children_are_leaves, threshold);
        }
        slot = node.parent_slot;
        parent = node.parent;
      }
      return kNil;
    }
    uint32_t next_leaf_with_min(uint32_t leaf, uint32_t threshold) const noexcept {
      uint32_t parent = leaves_[leaf].parent, slot = leaves_[leaf].parent_slot;
      while (parent != kNil) {
        const Node &node = nodes_[parent];
        while (++slot < node.count)
          if (child_min(node, slot) < threshold)
            return leftmost_leaf_below(node.children[slot], node.children_are_leaves, threshold);
        slot = node.parent_slot;
        parent = node.parent;
      }
      return kNil;
    }
    uint32_t rightmost_leaf_below(uint32_t child, bool leaf,
                                  uint32_t threshold) const noexcept {
      while (!leaf) {
        const Node &node = nodes_[child];
        uint32_t slot = node.count;
        do { --slot; } while (child_min(node, slot) >= threshold);
        child = node.children[slot];
        leaf = node.children_are_leaves;
      }
      return child;
    }
    uint32_t leftmost_leaf_below(uint32_t child, bool leaf,
                                 uint32_t threshold) const noexcept {
      while (!leaf) {
        const Node &node = nodes_[child];
        uint32_t slot = 0;
        while (child_min(node, slot) >= threshold) ++slot;
        child = node.children[slot];
        leaf = node.children_are_leaves;
      }
      return child;
    }
    static uint32_t aggregate_max(uint32_t max1, uint32_t max2,
                                  uint32_t excluded) noexcept {
      return max1 == excluded ? max2 : max1;
    }
    uint32_t query_leaf_max(uint32_t leaf_index, size_t first, size_t last,
                            size_t base, uint32_t excluded) const noexcept {
      const Leaf &leaf = leaves_[leaf_index];
      const size_t end = base + leaf.size;
      if (last <= base || end <= first) return 0;
      if (first <= base && end <= last)
        return aggregate_max(leaf.max1_pa, leaf.max2_pa, excluded);
      uint32_t result = 0;
      const uint32_t begin = first > base ? static_cast<uint32_t>(first - base) : 0;
      const uint32_t finish = last < end ? static_cast<uint32_t>(last - base) : leaf.size;
      for (uint32_t slot = begin; slot < finish; ++slot) {
        const uint32_t value = get_pa(leaf, slot);
        if (value != excluded) result = std::max(result, value);
      }
      return result;
    }
    uint32_t query_node_max(uint32_t node_index, size_t first, size_t last,
                            size_t base, uint32_t excluded) const noexcept {
      const Node &node = nodes_[node_index];
      if (first <= base && base + node.weight <= last)
        return aggregate_max(node.max1_pa, node.max2_pa, excluded);
      uint32_t result = 0;
      size_t child_base = base;
      for (uint32_t slot = 0; slot < node.count; ++slot) {
        const size_t child_end = child_base + node.weights[slot];
        if (child_end > first && child_base < last) {
          uint32_t value;
          if (first <= child_base && child_end <= last) {
            uint32_t first_max, second_max;
            child_max(node, slot, first_max, second_max);
            value = aggregate_max(first_max, second_max, excluded);
          } else if (node.children_are_leaves) {
            value = query_leaf_max(node.children[slot], first, last,
                                   child_base, excluded);
          } else {
            value = query_node_max(node.children[slot], first, last,
                                   child_base, excluded);
          }
          result = std::max(result, value);
        }
        child_base = child_end;
        if (child_base >= last) break;
      }
      return result;
    }

    std::vector<Leaf> leaves_;
    std::vector<Node> nodes_;
    std::unique_ptr<Leaf> pending_left_payload_;
    std::unique_ptr<Leaf> pending_leaf_;
    LcsCodec staging_lcs_;
    LcsCodec staging_successor_lcs_;
    bool pending_lcs_active_ = false;
    bool pending_successor_lcs_active_ = false;
    bool pending_raw8_in_place_ = false;
    bool pending_raw8_successor_in_place_ = false;
    uint32_t pending_successor_leaf_ = kNil;
    uint32_t pending_successor_slot_ = 0;
    std::vector<std::unique_ptr<Node>> pending_nodes_;
    uint32_t root_ = kNil, first_leaf_ = kNil, last_leaf_ = kNil;
    uint32_t sentinel_leaf_ = kNil;
    uint32_t sentinel_slot_ = 0;
    size_t capacity_ = 0, size_ = 0;
    uint32_t maximum_pa_bits_ = 1;
    uint32_t pa_bits_ = 0;
    uint32_t leaf_limit_ = 2048;
    uint32_t split_left_size_ = 1024;
    uint32_t split_right_size_ = 1025;
    uint32_t split_leaf_capacity_ = 1088;
    uint32_t leaf_growth_ = 256;
    uint32_t lcs_width_ = 2;
    uint32_t code_width_ = 0;
    bool root_is_leaf_ = true;
  };

  struct Row {
    struct IdentityCounts {
      std::array<uint32_t, 256> values{};
      std::array<uint32_t, 16> blocks{};
    };

    explicit Row(int64_t capacity, bool identity_codes)
        : sequence(static_cast<size_t>(capacity) + 1),
          identity_counts(identity_codes ? std::make_unique<IdentityCounts>()
                                         : nullptr) {}

    void reset(UnifiedSequence reset_sequence) noexcept {
      sequence = std::move(reset_sequence);
      history.reset();
      std::vector<SymbolCount>().swap(counts);
      std::vector<int64_t>().swap(code_values);
      if (identity_counts) *identity_counts = IdentityCounts{};
      source = -1;
      lrs = 0;
    }

    CompactArray history;
    UnifiedSequence sequence;
    std::vector<SymbolCount> counts;
    std::vector<int64_t> code_values;
    std::unique_ptr<IdentityCounts> identity_counts;
    int64_t source = -1;
    int64_t lrs = 0;
  };

  py::array_t<int64_t, py::array::c_style>
  checked_tokens(py::array object, int dimensions) const {
    if (!py::isinstance<py::array_t<int64_t>>(object))
      throw py::type_error("tokens must have dtype int64");
    if ((object.flags() & py::array::c_style) == 0)
      throw py::value_error("tokens must be C-contiguous");
    auto tokens =
        py::cast<py::array_t<int64_t, py::array::c_style>>(object);
    if (tokens.ndim() != dimensions || tokens.shape(0) != batch_) {
      if (dimensions == 1)
        throw py::value_error(
            "tokens must be contiguous int64 [batch_size]");
      throw py::value_error(
          "tokens must be contiguous int64 [batch_size, sequence_length]");
    }
    if (identity_codes_) {
      const int64_t count = static_cast<int64_t>(tokens.size());
      for (int64_t index = 0; index < count; ++index) {
        const int64_t token = tokens.data()[index];
        if (token < 0 || token >= static_cast<int64_t>(vocabulary_size_))
          throw py::value_error("compact RLBWT tokens must be in [0, vocabulary_size)");
      }
    }
    return tokens;
  }

  static uint64_t identity_prefix(const Row &row, uint32_t code) noexcept {
    uint64_t total = 0;
    const uint32_t full_blocks = code >> 4;
    for (uint32_t block = 0; block < full_blocks; ++block)
      total += row.identity_counts->blocks[block];
    const uint32_t first = full_blocks << 4;
    for (uint32_t index = first; index < code; ++index)
      total += row.identity_counts->values[index];
    return total;
  }

  static void identity_increment(Row &row, uint32_t code) noexcept {
    ++row.identity_counts->values[code];
    ++row.identity_counts->blocks[code >> 4];
  }

  static std::vector<SymbolCount>::iterator count_location(Row &row,
                                                            int64_t symbol) {
    return std::lower_bound(
        row.counts.begin(), row.counts.end(), symbol,
        [](const SymbolCount &entry, int64_t value) {
          return entry.symbol < value;
        });
  }

  static void prepare_dictionary_capacity(
      const Row &row, bool is_new, std::vector<SymbolCount> &grown_counts,
      std::vector<int64_t> &grown_codes, bool &replace_counts,
      bool &replace_codes) {
    const size_t count_needed = row.counts.size() + (is_new ? 1 : 0);
    const size_t code_needed = row.code_values.size() + (is_new ? 1 : 0);
    replace_counts = count_needed > row.counts.capacity();
    replace_codes = code_needed > row.code_values.capacity();
    if (replace_counts) {
      grown_counts = row.counts;
      grown_counts.reserve(std::max(count_needed,
                                    row.counts.capacity() * 2));
    }
    if (replace_codes) {
      grown_codes = row.code_values;
      grown_codes.reserve(std::max(code_needed,
                                   row.code_values.capacity() * 2));
    }
  }

  bool equal_suffix(const Row &row, size_t left, size_t right,
                    size_t length) const noexcept {
    const auto &hashes = row_hashes_[static_cast<size_t>(&row - rows_.data())];
    for (uint32_t lane = 0; lane < lanes_; ++lane) {
      const auto &prefix = hashes[lane];
      const uint64_t left_hash =
          prefix[left] - prefix[left - length] * powers_[lane][length];
      const uint64_t right_hash =
          prefix[right] - prefix[right - length] * powers_[lane][length];
      if (left_hash != right_hash)
        return false;
    }
    return true;
  }

  int64_t common_suffix(const Row &row, int64_t left, int64_t right) const {
    const size_t left_endpoint = static_cast<size_t>(left);
    const size_t right_endpoint = static_cast<size_t>(right);
    const size_t available = std::min(left_endpoint, right_endpoint);
    if (lanes_ == 0 || available <= 64)
      return static_cast<int64_t>(
          row.history.common_suffix(left_endpoint, right_endpoint));
    size_t low = 0, high = available + 1;
    while (low + 1 < high) {
      const size_t middle = low + (high - low) / 2;
      if (equal_suffix(row, left_endpoint, right_endpoint, middle))
        low = middle;
      else
        high = middle;
    }
    // Short LCEs remain exact. Long results are intentionally not verified:
    // this is the Monte-Carlo contract and the source of the speedup.
    if (low <= 64)
      return static_cast<int64_t>(
          row.history.common_suffix(left_endpoint, right_endpoint));
    return static_cast<int64_t>(low);
  }

  void compute_pa_lcs(const Row &row, int64_t old_length,
                      int64_t insertion_index, int64_t &x,
                      int64_t &y) const {
    const int64_t old_size = old_length + 1;
    const int64_t new_endpoint = old_length + 1;
    if (insertion_index < 0 || insertion_index > old_size)
      throw std::runtime_error("PA insertion position is out of range");
    const bool has_predecessor = insertion_index > 0;
    const bool has_successor = insertion_index < old_size;
    x = has_predecessor
            ? common_suffix(row, new_endpoint,
                            row.sequence.pa(
                                static_cast<size_t>(insertion_index - 1)))
            : 0;
    y = has_successor
            ? common_suffix(row, new_endpoint,
                            row.sequence.pa(static_cast<size_t>(insertion_index)))
            : 0;
  }

  static int64_t select_source(const Row &row, int64_t new_rank,
                               int64_t new_size, int64_t lrs) {
    if (lrs == 0)
      return -1;
    const size_t left = row.sequence.nearest_previous_lcs_less(
        static_cast<size_t>(new_rank), static_cast<uint32_t>(lrs));
    const size_t right = row.sequence.nearest_next_lcs_less(
        static_cast<size_t>(new_rank + 1), static_cast<uint32_t>(lrs));
    const int64_t new_endpoint = new_size - 1;
    // Exact recent-occurrence tie break: choose the largest old endpoint in
    // the maximal PA interval, skipping complete leaves via cached maxima.
    const uint32_t previous_endpoint = row.sequence.range_max_excluding(
        left, right, static_cast<uint32_t>(new_endpoint));
    if (previous_endpoint == static_cast<uint32_t>(new_endpoint))
      throw std::runtime_error("RLBWT source interval has no old endpoint");
    return static_cast<int64_t>(previous_endpoint) - 1;
  }

  int64_t step_row(Row &row, int64_t old_length, int64_t token) {
    const auto token_location = identity_codes_ ? row.counts.end()
                                                : count_location(row, token);
    const bool new_symbol = !identity_codes_ &&
                            (token_location == row.counts.end() ||
                             token_location->symbol != token);
    if (new_symbol && row.code_values.size() >= kSentinelCode)
      throw std::runtime_error("RLBWT symbol dictionary is exhausted");
    const uint32_t code = identity_codes_
                              ? static_cast<uint32_t>(token)
                              : new_symbol
                              ? static_cast<uint32_t>(row.code_values.size())
                              : token_location->code;
    const bool promote8 = row.history.width() == 0 &&
                          (identity_codes_ ? code >= 16
                                           : new_symbol && row.code_values.size() == 16);
    const bool promote16 = !identity_codes_ && new_symbol && row.history.width() == 1 &&
                           row.code_values.size() ==
                               static_cast<size_t>(std::numeric_limits<uint8_t>::max()) + 1;
    const bool promote32 = new_symbol && row.history.width() == 2 &&
                           row.code_values.size() ==
                               static_cast<size_t>(std::numeric_limits<uint16_t>::max()) + 1;
    CompactArray::Pages<uint8_t> byte_history;
    CompactArray::Pages<uint16_t> narrow_history;
    CompactArray::Pages<uint32_t> wide_history;
    std::vector<std::unique_ptr<uint8_t[]>> byte_sequence;
    std::vector<std::unique_ptr<uint16_t[]>> narrow_sequence;
    std::vector<std::unique_ptr<uint32_t[]>> wide_sequence;
    UnifiedSequence::HistogramRepack sequence_histograms;
    // Prepare the destination before history mutation. PACKED4 promotion
    // stages both conversion and a possible new page in the replacement;
    // later preparation failures therefore leave the live history untouched.
    if (promote8) {
      byte_history = row.history.prepare_promotion_append<uint8_t>();
    } else
      row.history.prepare_append(static_cast<size_t>(old_length));
    if (promote16) {
      narrow_history = row.history.prepare<uint16_t>();
    } else if (promote32) {
      wide_history = row.history.prepare<uint32_t>();
    }
    std::vector<SymbolCount> grown_counts;
    std::vector<int64_t> grown_codes;
    bool replace_counts = false, replace_codes = false;
    if (!identity_codes_)
      prepare_dictionary_capacity(row, new_symbol, grown_counts, grown_codes,
                                  replace_counts, replace_codes);
    uint64_t less = identity_codes_ ? 1 + identity_prefix(row, code) : 1;
    if (!identity_codes_)
      for (const SymbolCount &entry : row.counts) {
        if (entry.symbol >= token) break;
        less += entry.count;
      }
    const uint32_t rank = static_cast<uint32_t>(less + row.sequence.rank(code));
    // The history slot is not live until the outer position advances, so it
    // may be populated before preparation and overwritten after a failure.
    // At a width promotion the token is necessarily new and therefore both
    // exact suffix lengths are zero without storing its not-yet-representable
    // code in the old narrow array.
    int64_t x = 0, y = 0;
    row.sequence.promote_pa_for_endpoint(
        static_cast<uint32_t>(old_length + 1));
    if (!promote8 && !promote16 && !promote32) {
      row.history.set(static_cast<size_t>(old_length), code);
      append_hash(row, static_cast<size_t>(old_length), code);
      compute_pa_lcs(row, old_length, rank, x, y);
    }
    row.sequence.prepare_insert(
        rank, code, static_cast<uint32_t>(rank > 0 ? x : 0),
        rank < static_cast<uint32_t>(old_length + 1),
        static_cast<uint32_t>(y));
    // Width promotion is one transaction across BWT payloads and every
    // live or pending histogram. Prepare all replacement allocations only
    // after prepare_insert() has materialized the pending split objects.
    if (promote8) {
      byte_sequence = row.sequence.prepare_codes8();
      sequence_histograms = row.sequence.prepare_histograms(1);
    } else if (promote16) {
      narrow_sequence = row.sequence.prepare_codes16();
      sequence_histograms = row.sequence.prepare_histograms(2);
    } else if (promote32) {
      wide_sequence = row.sequence.prepare_codes32();
      sequence_histograms = row.sequence.prepare_histograms(4);
    }

    // Every codec and growth payload has succeeded.  From here all commits
    // are vector/unique_ptr swaps and fixed-capacity moves.
    if (promote8) {
      row.history.commit(std::move(byte_history));
      row.sequence.commit_codes8(std::move(byte_sequence),
                                 std::move(sequence_histograms));
    } else if (promote16) {
      row.history.commit(std::move(narrow_history));
      row.sequence.commit_codes16(std::move(narrow_sequence),
                                  std::move(sequence_histograms));
    } else if (promote32) {
      row.history.commit(std::move(wide_history));
      row.sequence.commit_codes32(std::move(wide_sequence),
                                  std::move(sequence_histograms));
    }
    if (replace_counts)
      row.counts.swap(grown_counts);
    if (replace_codes)
      row.code_values.swap(grown_codes);
    if (promote8 || promote16 || promote32) {
      row.history.set(static_cast<size_t>(old_length), code);
      append_hash(row, static_cast<size_t>(old_length), code);
    }
    auto location = row.counts.end();
    if (!identity_codes_) {
      location = count_location(row, token);
      if (new_symbol) {
        row.code_values.push_back(token);
        location = row.counts.insert(location, SymbolCount{token, code, 0});
      }
    }
    row.sequence.replace_and_insert(
        code, rank, static_cast<uint32_t>(old_length + 1),
        static_cast<uint32_t>(rank > 0 ? x : 0),
        rank < static_cast<uint32_t>(old_length + 1),
        static_cast<uint32_t>(y));
    row.history.finish_append();
    if (identity_codes_)
      identity_increment(row, code);
    else
      ++location->count;
    row.lrs = std::max(x, y);
    row.source = select_source(row, rank, old_length + 2, row.lrs);
    if (row.source < 0) return -1;
    const uint32_t output_code =
        row.history.get(static_cast<size_t>(row.source + 1));
    return identity_codes_ ? static_cast<int64_t>(output_code)
                           : row.code_values[output_code];
  }

  void append_hash(Row &row, size_t index, uint32_t code) noexcept {
    const uint64_t value = static_cast<uint64_t>(code) + 1;
    auto &hashes = row_hashes_[static_cast<size_t>(&row - rows_.data())];
    for (uint32_t lane = 0; lane < lanes_; ++lane)
      hashes[lane][index + 1] = hashes[lane][index] * bases_[lane] + value;
  }

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

  std::vector<Row> rows_;
  std::vector<std::array<std::vector<uint64_t>, 3>> row_hashes_;
  std::array<std::vector<uint64_t>, 3> powers_;
  std::array<uint64_t, 3> bases_{};
  std::unique_ptr<RowThreadPool> row_pool_;
  mutable std::mutex call_mutex_;
  int64_t batch_, max_length_, position_ = 0;
  uint32_t lanes_ = 0;
  uint64_t seed_ = 0;
  uint32_t vocabulary_size_ = 0;
  bool identity_codes_ = false;
};

class NativeRLBWTCompactState : public NativeRLBWTState {
public:
  NativeRLBWTCompactState(int64_t batch_size, int64_t max_length,
                          uint32_t vocabulary_size = 256)
      : NativeRLBWTState(batch_size, max_length, 0, 0,
                         checked_vocabulary(vocabulary_size)),
        vocabulary_size_(vocabulary_size) {}

  uint32_t vocabulary_size() const noexcept { return vocabulary_size_; }

private:
  static uint32_t checked_vocabulary(uint32_t vocabulary_size) {
    if (vocabulary_size < 1 || vocabulary_size > 256)
      throw py::value_error("vocabulary_size must be in [1, 256]");
    return vocabulary_size;
  }

  uint32_t vocabulary_size_;
};

class NativeRLBWTStateMC : public NativeRLBWTState {
public:
  NativeRLBWTStateMC(int64_t batch_size, int64_t max_length, uint32_t lanes,
                     uint64_t seed)
      : NativeRLBWTState(batch_size, max_length, checked_lanes(lanes), seed) {}

private:
  static uint32_t checked_lanes(uint32_t lanes) {
    if (lanes != 2 && lanes != 3)
      throw py::value_error("lanes must be 2 or 3");
    return lanes;
  }
};

// Allocating candidate entry points delegate to their caller-owned variants.
PYBIND11_MODULE(rosa_native_step, m) {
  m.doc() = "Exact CPU SAM+LCT and RLBWT backends (no libtorch calls)";
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
      .def("step_into", &NativeCandidateState::step_into)
      .def("reset", &NativeCandidateState::reset)
      .def("step_masked", &NativeCandidateState::step_masked)
      .def("reset_masked", &NativeCandidateState::reset_masked)
      .def("prefill", &NativeCandidateState::prefill)
      .def("prefill_into", &NativeCandidateState::prefill_into)
      .def_property_readonly("position", &NativeCandidateState::position)
      .def_property_readonly("positions", &NativeCandidateState::positions)
      .def_property_readonly("worker_count",
                             &NativeCandidateState::worker_count);
  py::class_<NativeRLBWTState>(m, "NativeRLBWTState")
      .def(py::init<int64_t, int64_t>(), py::arg("batch_size"),
           py::arg("max_length"))
      .def("step", &NativeRLBWTState::step, py::arg("tokens").noconvert())
      .def("prefill", &NativeRLBWTState::prefill,
           py::arg("tokens").noconvert())
      .def("prefill_append", &NativeRLBWTState::prefill_append,
           py::arg("tokens").noconvert())
      .def("reset", &NativeRLBWTState::reset)
      .def("row_snapshot", &NativeRLBWTState::row_snapshot,
           py::arg("batch_index"))
      .def_property_readonly("position", &NativeRLBWTState::position)
      .def_property_readonly("batch_size", &NativeRLBWTState::batch_size)
      .def_property_readonly("max_length", &NativeRLBWTState::max_length)
      .def_property_readonly("sources", &NativeRLBWTState::sources)
      .def_property_readonly("lrs_lengths", &NativeRLBWTState::lrs_lengths)
      .def_property_readonly("run_counts", &NativeRLBWTState::run_counts)
      .def_property_readonly("storage_bytes", &NativeRLBWTState::storage_bytes)
      .def_property_readonly("storage_breakdown",
                             &NativeRLBWTState::storage_breakdown);
  py::class_<NativeRLBWTStateMC, NativeRLBWTState>(m, "NativeRLBWTStateMC")
      .def(py::init<int64_t, int64_t, uint32_t, uint64_t>(),
           py::arg("batch_size"), py::arg("max_length"), py::arg("lanes"),
           py::arg("seed"))
      .def_property_readonly("lanes", &NativeRLBWTStateMC::lanes)
      .def_property_readonly("seed", &NativeRLBWTStateMC::seed);
  py::class_<NativeRLBWTCompactState, NativeRLBWTState>(
      m, "NativeRLBWTCompactState")
      .def(py::init<int64_t, int64_t, uint32_t>(), py::arg("batch_size"),
           py::arg("max_length"), py::arg("vocabulary_size") = 256)
      .def_property_readonly("vocabulary_size",
                             &NativeRLBWTCompactState::vocabulary_size);
  m.attr("candidate_abi_version") = py::int_(1);
  m.attr("rlbwt_abi_version") = py::int_(1);
  m.attr("rlbwt_mc_abi_version") = py::int_(1);
  m.attr("rlbwt_compact_abi_version") = py::int_(1);
}
