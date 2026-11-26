#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "quality_kd_build_identity.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(_M_X64) || defined(__x86_64__)
#include <emmintrin.h>
#endif
#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace py = pybind11;

namespace {

inline std::uint32_t ctz64(std::uint64_t value) {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
    unsigned long index = 0;
    _BitScanForward64(&index, value);
    return static_cast<std::uint32_t>(index);
#elif defined(__GNUC__) || defined(__clang__)
    return static_cast<std::uint32_t>(__builtin_ctzll(value));
#else
    std::uint32_t index = 0;
    while ((value & 1ULL) == 0ULL) {
        value >>= 1U;
        ++index;
    }
    return index;
#endif
}

constexpr std::size_t kStackCapacity = 64;
constexpr std::uint64_t kUint32Max = std::numeric_limits<std::uint32_t>::max();
constexpr std::uint8_t kDividerAddMarker = 0x40;
constexpr std::uint32_t kPackedCoordinateLimit = 0xffff;
constexpr std::uint64_t kForegroundMatteSparseRecordLimitDivisor = 4;
constexpr std::size_t kGeometryQueryWorkerCap = 8;
constexpr std::size_t kIndexBuildWorkerCap = 8;
constexpr std::size_t kEdgeBandWorkerCap = 8;
constexpr std::size_t kNearestLabelUpsampleWorkerCap = 8;
constexpr std::size_t kFourNeighbourJumpWorkerCap = 8;
constexpr std::size_t kNativeSpawnedWorkerCap = 7;
constexpr std::uint32_t kEdgeBandRowsPerWorker = 128;
constexpr std::uint32_t kNearestLabelUpsampleRowsPerWorker = 128;
constexpr std::uint32_t kFourNeighbourJumpRowsPerWorker = 128;
constexpr py::ssize_t kGeometryQueriesPerWorker = 4096;
constexpr py::ssize_t kGeometryQueryBlockSize = 64;
constexpr double kStableSampleMaxNeighbourJumpPx = 0.25;
constexpr std::uint32_t kLocalGeometryQueryRadius = 16;

std::atomic<std::size_t> g_native_spawned_workers{0};

class NativeWorkerLease {
public:
    explicit NativeWorkerLease(std::size_t requested) {
        std::size_t observed = g_native_spawned_workers.load(std::memory_order_relaxed);
        while (observed < kNativeSpawnedWorkerCap) {
            const std::size_t claimed = std::min(
                requested,
                kNativeSpawnedWorkerCap - observed
            );
            if (claimed == 0) {
                return;
            }
            if (g_native_spawned_workers.compare_exchange_weak(
                observed,
                observed + claimed,
                std::memory_order_acq_rel,
                std::memory_order_relaxed
            )) {
                claimed_ = claimed;
                return;
            }
        }
    }

    NativeWorkerLease(const NativeWorkerLease&) = delete;
    NativeWorkerLease& operator=(const NativeWorkerLease&) = delete;

    ~NativeWorkerLease() {
        if (claimed_ != 0) {
            g_native_spawned_workers.fetch_sub(claimed_, std::memory_order_release);
        }
    }

    std::size_t worker_count() const { return claimed_ + 1; }

private:
    std::size_t claimed_ = 0;
};

class PersistentParallelPool {
public:
    using Invoker = void (*)(const void*, std::size_t);

    static PersistentParallelPool& instance() {
        // Heap-allocated singleton avoids joining threads under the Windows
        // DLL_PROCESS_DETACH loader lock; Python atexit calls shutdown().
        static auto* pool = new PersistentParallelPool();
        return *pool;
    }

    template <typename Fn>
    void run(std::size_t worker_count, Fn&& fn) {
        if (worker_count <= 1) {
            fn(0);
            return;
        }
        using FnType = std::remove_reference_t<Fn>;
        Invoker invoke = [](const void* raw, std::size_t index) {
            (*static_cast<FnType*>(const_cast<void*>(raw)))(index);
        };
        const std::size_t helper_count = worker_count - 1;
        BatchState batch;
        batch.pending = helper_count;
        bool ran_inline = false;
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (stopped_) {
                ran_inline = true;
            } else {
                const std::size_t target_threads = std::min<std::size_t>(
                    kNativeSpawnedWorkerCap,
                    in_flight_tasks_ + helper_count
                );
                try {
                    ensure_threads_locked(target_threads);
                } catch (...) {
                    // Fall back gracefully if OS thread creation fails.
                }
                if (threads_.empty()) {
                    ran_inline = true;
                } else {
                    in_flight_tasks_ += helper_count;
                    for (std::size_t worker = 1; worker < worker_count; ++worker) {
                        queue_.push_back(Task{invoke, &fn, worker, &batch});
                    }
                }
            }
        }
        if (ran_inline) {
            for (std::size_t worker = 0; worker < worker_count; ++worker) {
                fn(worker);
            }
            return;
        }
        queue_cv_.notify_all();
        std::exception_ptr local_error;
        try {
            fn(0);
        } catch (...) {
            local_error = std::current_exception();
        }
        {
            std::unique_lock<std::mutex> lock(batch.mutex);
            batch.cv.wait(lock, [&]() { return batch.pending == 0; });
        }
        if (local_error) {
            std::rethrow_exception(local_error);
        }
        if (batch.error) {
            std::rethrow_exception(batch.error);
        }
    }

    void shutdown() {
        std::vector<std::thread> to_join;
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (stopped_) {
                return;
            }
            stopped_ = true;
            to_join.swap(threads_);
        }
        queue_cv_.notify_all();
        for (auto& thread : to_join) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

private:
    struct BatchState {
        std::mutex mutex;
        std::condition_variable cv;
        std::size_t pending = 0;
        std::exception_ptr error;
    };

    struct Task {
        Invoker invoke;
        const void* context;
        std::size_t worker_index;
        BatchState* batch;
    };

    void ensure_threads_locked(std::size_t needed) {
        while (threads_.size() < needed) {
            threads_.emplace_back([this]() { worker_loop(); });
        }
    }

    void worker_loop() {
        bool completed_one = false;
        while (true) {
            Task task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                if (completed_one) {
                    --in_flight_tasks_;
                    completed_one = false;
                }
                queue_cv_.wait(lock, [this]() {
                    return stopped_ || !queue_.empty();
                });
                if (queue_.empty()) {
                    if (stopped_) {
                        return;
                    }
                    continue;
                }
                task = queue_.front();
                queue_.pop_front();
            }
            std::exception_ptr caught;
            try {
                task.invoke(task.context, task.worker_index);
            } catch (...) {
                caught = std::current_exception();
            }
            {
                std::lock_guard<std::mutex> lock(task.batch->mutex);
                if (caught && !task.batch->error) {
                    task.batch->error = caught;
                }
                if (--task.batch->pending == 0) {
                    task.batch->cv.notify_one();
                }
            }
            completed_one = true;
        }
    }

    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::deque<Task> queue_;
    std::vector<std::thread> threads_;
    std::size_t in_flight_tasks_ = 0;
    bool stopped_ = false;
};

template <typename Fn>
void run_native_parallel(std::size_t worker_count, Fn&& fn) {
    PersistentParallelPool::instance().run(worker_count, std::forward<Fn>(fn));
}

class QualityNativeBudgetError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class U32Divider {
public:
    explicit U32Divider(std::uint32_t divisor) {
        std::uint8_t floor_log_2 = 0;
        for (std::uint32_t value = divisor; value > 1; value >>= 1) {
            ++floor_log_2;
        }
        if ((divisor & (divisor - 1)) == 0) {
            more_ = floor_log_2;
            return;
        }
        const std::uint64_t numerator = 1ULL << (32 + floor_log_2);
        std::uint32_t proposed_magic = static_cast<std::uint32_t>(numerator / divisor);
        const std::uint32_t remainder = static_cast<std::uint32_t>(numerator % divisor);
        if (divisor - remainder < (1U << floor_log_2)) {
            more_ = floor_log_2;
        } else {
            proposed_magic += proposed_magic;
            const std::uint32_t twice_remainder = remainder + remainder;
            if (twice_remainder >= divisor || twice_remainder < remainder) {
                ++proposed_magic;
            }
            more_ = static_cast<std::uint8_t>(floor_log_2 | kDividerAddMarker);
        }
        magic_ = proposed_magic + 1;
    }

    inline std::uint32_t divide(std::uint32_t numerator) const {
        if (magic_ == 0) {
            return numerator >> more_;
        }
        const std::uint32_t quotient = static_cast<std::uint32_t>(
            (static_cast<std::uint64_t>(magic_) * numerator) >> 32
        );
        const std::uint8_t shift = static_cast<std::uint8_t>(more_ & 0x1f);
        if ((more_ & kDividerAddMarker) != 0) {
            return (((numerator - quotient) >> 1) + quotient) >> shift;
        }
        return quotient >> shift;
    }

private:
    std::uint32_t magic_ = 0;
    std::uint8_t more_ = 0;
};

struct BuildFrame {
    std::uint32_t start;
    std::uint32_t end;
    std::uint8_t depth;
};

struct GeometryBest {
    bool found = false;
    double distance2 = 0.0;
    std::uint32_t y = 0;
    std::uint32_t x = 0;
    std::uint32_t sample_index = 0;
};

void select_kth_numeric(
    std::uint32_t* members,
    std::uint32_t left,
    std::uint32_t right,
    std::uint32_t kth
) {
    if (right - left < 96) {
        std::sort(members + left, members + right + 1);
        return;
    }
    // Unique raster keys and recursive child partitioning make the final
    // implicit layout independent of nth_element's internal permutation.
    std::nth_element(
        members + left,
        members + kth,
        members + right + 1
    );
}

std::size_t push_frame(
    std::array<BuildFrame, kStackCapacity>& stack,
    std::size_t top,
    std::uint32_t start,
    std::uint32_t end,
    std::uint8_t depth
) {
    if (top >= stack.size()) {
        throw std::runtime_error("implicit k-d construction stack capacity exceeded");
    }
    stack[top] = BuildFrame{start, end, depth};
    return top + 1;
}

void partition_region_tree(
    std::uint32_t* members,
    std::uint32_t start,
    std::uint32_t end,
    std::uint32_t height,
    std::uint32_t width,
    bool packed_coordinates = false
) {
    if (start == end) {
        return;
    }
    std::array<BuildFrame, kStackCapacity> stack{};
    const U32Divider width_divider(width);
    const U32Divider height_divider(height);
    std::size_t top = push_frame(stack, 0, start, end, 0);
    while (top > 0) {
        const BuildFrame frame = stack[--top];
        const std::uint32_t middle = frame.start + ((frame.end - frame.start - 1) / 2);
        const bool column_axis = (frame.depth & 1) != 0;
        if (column_axis) {
            for (std::uint32_t cursor = frame.start; cursor < frame.end; ++cursor) {
                if (packed_coordinates) {
                    const std::uint32_t packed = members[cursor];
                    members[cursor] = (packed << 16) | (packed >> 16);
                    continue;
                }
                const std::uint32_t sample = members[cursor];
                const std::uint32_t row = width_divider.divide(sample);
                const std::uint32_t column = sample - row * width;
                members[cursor] = column * height + row;
            }
        }
        if (frame.depth != 0) {
            select_kth_numeric(
                members,
                frame.start,
                frame.end - 1,
                middle
            );
        }
        if (column_axis) {
            for (std::uint32_t cursor = frame.start; cursor < frame.end; ++cursor) {
                if (packed_coordinates) {
                    const std::uint32_t packed = members[cursor];
                    members[cursor] = (packed << 16) | (packed >> 16);
                    continue;
                }
                const std::uint32_t column_major = members[cursor];
                const std::uint32_t column = height_divider.divide(column_major);
                const std::uint32_t row = column_major - column * height;
                members[cursor] = row * width + column;
            }
        }
        const std::uint8_t next_depth = static_cast<std::uint8_t>(frame.depth + 1);
        if (middle + 1 < frame.end) {
            top = push_frame(stack, top, middle + 1, frame.end, next_depth);
        }
        if (frame.start < middle) {
            top = push_frame(stack, top, frame.start, middle, next_depth);
        }
    }
}

bool geometry_candidate_is_better(
    double distance2,
    std::uint32_t y,
    std::uint32_t x,
    const GeometryBest& best
) {
    if (!best.found || distance2 != best.distance2) {
        return !best.found || distance2 < best.distance2;
    }
    if (y != best.y) {
        return y < best.y;
    }
    return x < best.x;
}

void query_geometry_range(
    const std::uint32_t* members,
    std::uint32_t start,
    std::uint32_t end,
    std::uint8_t depth,
    std::uint32_t width,
    const U32Divider& width_divider,
    double target_y,
    double target_x,
    std::uint64_t* budget_state,
    std::uint32_t* visited_output,
    GeometryBest& best
) {
    if (start == end) {
        return;
    }
    if (budget_state[0] >= budget_state[1]) {
        throw std::runtime_error("quality geometry visit cap exceeded");
    }
    ++budget_state[0];
    ++(*visited_output);

    const std::uint32_t middle = start + ((end - start - 1) / 2);
    const std::uint32_t sample_index = members[middle];
    const std::uint32_t y = width_divider.divide(sample_index);
    const std::uint32_t x = sample_index - y * width;
    const double dx = target_x - static_cast<double>(x);
    const double dy = target_y - static_cast<double>(y);
    const double distance2 = (dx * dx) + (dy * dy);
    if (geometry_candidate_is_better(distance2, y, x, best)) {
        best = GeometryBest{true, distance2, y, x, sample_index};
    }

    const double delta = (depth & 1) == 0
        ? target_y - static_cast<double>(y)
        : target_x - static_cast<double>(x);
    const double plane2 = delta * delta;
    const std::pair<std::uint32_t, std::uint32_t> lower{start, middle};
    const std::pair<std::uint32_t, std::uint32_t> upper{middle + 1, end};
    const auto near_range = delta <= 0.0 ? lower : upper;
    const auto far_range = delta <= 0.0 ? upper : lower;
    const std::uint8_t next_depth = static_cast<std::uint8_t>(depth + 1);
    query_geometry_range(
        members,
        near_range.first,
        near_range.second,
        next_depth,
        width,
        width_divider,
        target_y,
        target_x,
        budget_state,
        visited_output,
        best
    );
    if (far_range.first < far_range.second && plane2 <= best.distance2) {
        query_geometry_range(
            members,
            far_range.first,
            far_range.second,
            next_depth,
            width,
            width_divider,
            target_y,
            target_x,
            budget_state,
            visited_output,
            best
        );
    }
}

GeometryBest query_geometry_one(
    const std::uint32_t* members,
    std::uint32_t start,
    std::uint32_t end,
    std::uint32_t width,
    const U32Divider& width_divider,
    double target_y,
    double target_x,
    std::uint64_t* budget_state,
    std::uint32_t* visited_output
) {
    if (start == end) {
        throw std::runtime_error("selected geometry region has no indexed member");
    }
    GeometryBest best{};
    *visited_output = 0;
    query_geometry_range(
        members,
        start,
        end,
        0,
        width,
        width_divider,
        target_y,
        target_x,
        budget_state,
        visited_output,
        best
    );
    return best;
}

void validate_array(
    const py::array& values,
    const py::dtype& dtype,
    const char* name
) {
    if (!values.dtype().is(dtype)) {
        throw py::type_error(std::string(name) + " has the wrong dtype");
    }
    if (values.ndim() != 2 || values.shape(0) <= 0 || values.shape(1) <= 0) {
        throw py::value_error(std::string(name) + " must be a non-empty 2D raster");
    }
    if ((values.flags() & py::array::c_style) == 0) {
        throw py::value_error(std::string(name) + " must be C-contiguous");
    }
}

void validate_vector(
    const py::array& values,
    const py::dtype& dtype,
    const char* name,
    py::ssize_t expected_length = -1,
    bool require_writable = false
) {
    if (!values.dtype().is(dtype)) {
        throw py::type_error(std::string(name) + " has the wrong dtype");
    }
    if (values.ndim() != 1) {
        throw py::value_error(std::string(name) + " must be one-dimensional");
    }
    if (expected_length >= 0 && values.shape(0) != expected_length) {
        throw py::value_error(std::string(name) + " has the wrong length");
    }
    if ((values.flags() & py::array::c_style) == 0) {
        throw py::value_error(std::string(name) + " must be C-contiguous");
    }
    if (require_writable && !values.writeable()) {
        throw py::value_error(std::string(name) + " must be writable");
    }
}

void validate_index_arrays(
    const py::array& members,
    const py::array& offsets,
    std::uint64_t height,
    std::uint64_t width
) {
    validate_vector(members, py::dtype::of<std::uint32_t>(), "member_index");
    validate_vector(offsets, py::dtype::of<std::uint32_t>(), "region_offsets");
    if (height == 0 || width == 0 || height * width + 1 > kUint32Max) {
        throw py::value_error("indexed raster shape must fit the uint32 sample contract");
    }
    if (offsets.shape(0) < 2) {
        throw py::value_error("region_offsets must contain one region and a sentinel");
    }
    const auto* offset_values = static_cast<const std::uint32_t*>(offsets.data());
    const auto member_count = static_cast<std::uint64_t>(members.shape(0));
    std::uint32_t previous = 0;
    for (py::ssize_t index = 0; index < offsets.shape(0); ++index) {
        const std::uint32_t value = offset_values[index];
        if (value < previous || value > member_count) {
            throw py::value_error("region_offsets must be monotonic within member_index");
        }
        previous = value;
    }
    if (previous != member_count) {
        throw py::value_error("region_offsets sentinel must equal member_index length");
    }
    const auto* member_values = static_cast<const std::uint32_t*>(members.data());
    const std::uint64_t sample_count = height * width;
    for (std::uint64_t index = 0; index < member_count; ++index) {
        if (member_values[index] >= sample_count) {
            throw py::value_error("member_index contains a sample outside the indexed raster");
        }
    }
}

void validate_budget_state(const py::array& budget_state) {
    validate_vector(
        budget_state,
        py::dtype::of<std::uint64_t>(),
        "budget_state",
        2,
        true
    );
    const auto* values = static_cast<const std::uint64_t*>(budget_state.data());
    if (values[0] > values[1]) {
        throw py::value_error("budget consumption cannot exceed its limit");
    }
}

py::tuple query_geometry_batch(
    const py::array& members,
    const py::array& offsets,
    std::uint64_t height,
    std::uint64_t width,
    const py::array& query_regions,
    const py::array& target_y,
    const py::array& target_x,
    py::array budget_state
) {
    validate_index_arrays(members, offsets, height, width);
    validate_vector(query_regions, py::dtype::of<std::uint32_t>(), "query_regions");
    const py::ssize_t query_count = query_regions.shape(0);
    validate_vector(target_y, py::dtype::of<double>(), "target_y", query_count);
    validate_vector(target_x, py::dtype::of<double>(), "target_x", query_count);
    validate_budget_state(budget_state);

    const auto* member_values = static_cast<const std::uint32_t*>(members.data());
    const auto* offset_values = static_cast<const std::uint32_t*>(offsets.data());
    const auto* region_values = static_cast<const std::uint32_t*>(query_regions.data());
    const auto* y_values = static_cast<const double*>(target_y.data());
    const auto* x_values = static_cast<const double*>(target_x.data());
    auto* budget_values = static_cast<std::uint64_t*>(budget_state.mutable_data());
    const std::uint32_t region_count = static_cast<std::uint32_t>(offsets.shape(0) - 1);
    const std::uint32_t width32 = static_cast<std::uint32_t>(width);
    const U32Divider width_divider(width32);

    py::array_t<std::uint32_t> result_samples(query_count);
    py::array_t<double> result_distances(query_count);
    py::array_t<std::uint32_t> result_visits(query_count);
    auto* sample_output = result_samples.mutable_data();
    auto* distance_output = result_distances.mutable_data();
    auto* visit_output = result_visits.mutable_data();
    {
        py::gil_scoped_release release;
        bool parallel_safe = true;
        for (py::ssize_t query = 0; query < query_count; ++query) {
            const std::uint32_t region = region_values[query];
            if (
                region == 0 ||
                region > region_count ||
                offset_values[region - 1] == offset_values[region] ||
                !std::isfinite(y_values[query]) ||
                !std::isfinite(x_values[query])
            ) {
                parallel_safe = false;
                break;
            }
        }

        const auto sequential_query = [&]() {
            for (py::ssize_t query = 0; query < query_count; ++query) {
                const std::uint32_t region = region_values[query];
                if (region == 0 || region > region_count) {
                    throw std::invalid_argument(
                        "query region exceeds the canonical region count"
                    );
                }
                if (!std::isfinite(y_values[query]) || !std::isfinite(x_values[query])) {
                    throw std::invalid_argument("geometry query coordinates must be finite");
                }
                const GeometryBest best = query_geometry_one(
                    member_values,
                    offset_values[region - 1],
                    offset_values[region],
                    width32,
                    width_divider,
                    y_values[query],
                    x_values[query],
                    budget_values,
                    &visit_output[query]
                );
                sample_output[query] = best.sample_index;
                distance_output[query] = best.distance2;
            }
        };

        const std::size_t hardware_workers = std::max(
            1U,
            std::thread::hardware_concurrency()
        );
        const std::size_t useful_workers = static_cast<std::size_t>(
            (query_count + kGeometryQueriesPerWorker - 1) / kGeometryQueriesPerWorker
        );
        const std::size_t desired_worker_count = std::min(
            {kGeometryQueryWorkerCap, hardware_workers, std::max<std::size_t>(1, useful_workers)}
        );
        NativeWorkerLease worker_lease(
            parallel_safe && desired_worker_count > 1 ? desired_worker_count - 1 : 0
        );
        const std::size_t worker_count = parallel_safe ? worker_lease.worker_count() : 1;
        if (!parallel_safe || worker_count == 1) {
            sequential_query();
        } else {
            std::atomic<py::ssize_t> next_query{0};
            const auto worker = [&]() {
                while (true) {
                    const py::ssize_t start = next_query.fetch_add(
                        kGeometryQueryBlockSize,
                        std::memory_order_relaxed
                    );
                    if (start >= query_count) {
                        return;
                    }
                    const py::ssize_t stop = std::min(
                        query_count,
                        start + kGeometryQueryBlockSize
                    );
                    for (py::ssize_t query = start; query < stop; ++query) {
                        const std::uint32_t region = region_values[query];
                        std::uint64_t local_budget[2] = {
                            0,
                            std::numeric_limits<std::uint64_t>::max(),
                        };
                        const GeometryBest best = query_geometry_one(
                            member_values,
                            offset_values[region - 1],
                            offset_values[region],
                            width32,
                            width_divider,
                            y_values[query],
                            x_values[query],
                            local_budget,
                            &visit_output[query]
                        );
                        sample_output[query] = best.sample_index;
                        distance_output[query] = best.distance2;
                    }
                }
            };

            run_native_parallel(worker_count, [&](std::size_t) { worker(); });

            std::uint64_t consumed = budget_values[0];
            const std::uint64_t limit = budget_values[1];
            for (py::ssize_t query = 0; query < query_count; ++query) {
                const std::uint64_t visits = visit_output[query];
                if (visits > limit - consumed) {
                    budget_values[0] = limit;
                    throw std::runtime_error("quality geometry visit cap exceeded");
                }
                consumed += visits;
            }
            budget_values[0] = consumed;
        }
    }
    return py::make_tuple(
        std::move(result_samples),
        std::move(result_distances),
        std::move(result_visits)
    );
}

py::tuple query_local_geometry_batch(
    const py::array& region_ids,
    const py::array& include,
    const py::array& query_regions,
    const py::array& target_y,
    const py::array& target_x,
    std::uint32_t radius_px,
    py::array budget_state
) {
    validate_array(region_ids, py::dtype::of<std::uint32_t>(), "region_ids");
    validate_array(include, py::dtype::of<bool>(), "include");
    if (
        include.shape(0) != region_ids.shape(0) ||
        include.shape(1) != region_ids.shape(1)
    ) {
        throw py::value_error("include must match region_ids shape");
    }
    validate_vector(query_regions, py::dtype::of<std::uint32_t>(), "query_regions");
    const py::ssize_t query_count = query_regions.shape(0);
    validate_vector(target_y, py::dtype::of<double>(), "target_y", query_count);
    validate_vector(target_x, py::dtype::of<double>(), "target_x", query_count);
    validate_budget_state(budget_state);
    const auto height = static_cast<std::uint32_t>(region_ids.shape(0));
    const auto width = static_cast<std::uint32_t>(region_ids.shape(1));
    const std::uint32_t effective_radius = std::min(
        radius_px,
        std::max(height - 1U, width - 1U)
    );

    const auto* region_values = static_cast<const std::uint32_t*>(region_ids.data());
    const auto* include_values = static_cast<const std::uint8_t*>(include.data());
    const auto* query_region_values =
        static_cast<const std::uint32_t*>(query_regions.data());
    const auto* y_values = static_cast<const double*>(target_y.data());
    const auto* x_values = static_cast<const double*>(target_x.data());
    auto* budget_values = static_cast<std::uint64_t*>(budget_state.mutable_data());
    py::array_t<std::uint32_t> result_samples(query_count);
    py::array_t<double> result_distances(query_count);
    py::array_t<std::uint32_t> result_visits(query_count);
    auto* sample_output = result_samples.mutable_data();
    auto* distance_output = result_distances.mutable_data();
    auto* visit_output = result_visits.mutable_data();
    {
        py::gil_scoped_release release;
        const auto query_one = [&](
            py::ssize_t query,
            std::uint64_t* query_budget
        ) {
            const std::uint32_t selected_region = query_region_values[query];
            const double query_y = y_values[query];
            const double query_x = x_values[query];
            if (selected_region == 0) {
                throw std::invalid_argument("query region must be positive");
            }
            if (!std::isfinite(query_y) || !std::isfinite(query_x)) {
                throw std::invalid_argument("geometry query coordinates must be finite");
            }
            const double clipped_y = std::min(
                std::max(query_y, 0.0),
                static_cast<double>(height - 1)
            );
            const double clipped_x = std::min(
                std::max(query_x, 0.0),
                static_cast<double>(width - 1)
            );
            const auto center_y = static_cast<std::uint32_t>(std::floor(clipped_y));
            const auto center_x = static_cast<std::uint32_t>(std::floor(clipped_x));
            GeometryBest best{};
            std::uint32_t visits = 0;
            bool proven = false;
            const auto evaluate = [&](std::uint32_t row, std::uint32_t column) {
                if (query_budget[0] >= query_budget[1]) {
                    throw std::runtime_error("quality geometry visit cap exceeded");
                }
                ++query_budget[0];
                ++visits;
                const std::uint32_t pixel = row * width + column;
                if (
                    include_values[pixel] == 0 ||
                    region_values[pixel] != selected_region
                ) {
                    return;
                }
                const double dx = query_x - static_cast<double>(column);
                const double dy = query_y - static_cast<double>(row);
                const double distance2 = (dx * dx) + (dy * dy);
                if (geometry_candidate_is_better(distance2, row, column, best)) {
                    best = GeometryBest{true, distance2, row, column, pixel};
                }
            };
            for (std::uint32_t radius = 0; radius <= effective_radius; ++radius) {
                if (radius == 0) {
                    evaluate(center_y, center_x);
                } else {
                    const std::int64_t top =
                        static_cast<std::int64_t>(center_y) - radius;
                    const std::int64_t bottom =
                        static_cast<std::int64_t>(center_y) + radius;
                    const std::int64_t left =
                        static_cast<std::int64_t>(center_x) - radius;
                    const std::int64_t right =
                        static_cast<std::int64_t>(center_x) + radius;
                    const auto column_start = static_cast<std::uint32_t>(std::max<std::int64_t>(
                        0,
                        left
                    ));
                    const auto column_end = static_cast<std::uint32_t>(std::min<std::int64_t>(
                        width - 1,
                        right
                    ));
                    if (top >= 0 && top < height) {
                        for (
                            std::uint32_t column = column_start;
                            column <= column_end;
                            ++column
                        ) {
                            evaluate(static_cast<std::uint32_t>(top), column);
                        }
                    }
                    if (bottom >= 0 && bottom < height && bottom != top) {
                        for (
                            std::uint32_t column = column_start;
                            column <= column_end;
                            ++column
                        ) {
                            evaluate(static_cast<std::uint32_t>(bottom), column);
                        }
                    }
                    const auto row_start = std::max<std::int64_t>(
                        0,
                        static_cast<std::int64_t>(center_y) - radius + 1
                    );
                    const auto row_end = std::min<std::int64_t>(
                        height - 1,
                        static_cast<std::int64_t>(center_y) + radius - 1
                    );
                    for (std::int64_t row = row_start; row <= row_end; ++row) {
                        if (left >= 0 && left < width) {
                            evaluate(
                                static_cast<std::uint32_t>(row),
                                static_cast<std::uint32_t>(left)
                            );
                        }
                        if (right >= 0 && right < width && right != left) {
                            evaluate(
                                static_cast<std::uint32_t>(row),
                                static_cast<std::uint32_t>(right)
                            );
                        }
                    }
                }

                const std::uint64_t next_radius =
                    static_cast<std::uint64_t>(radius) + 1;
                double unseen_distance2 = std::numeric_limits<double>::infinity();
                if (next_radius <= center_y) {
                    const double dy = query_y - static_cast<double>(center_y - next_radius);
                    unseen_distance2 = std::min(unseen_distance2, dy * dy);
                }
                if (static_cast<std::uint64_t>(center_y) + next_radius < height) {
                    const double dy = query_y - static_cast<double>(center_y + next_radius);
                    unseen_distance2 = std::min(unseen_distance2, dy * dy);
                }
                if (next_radius <= center_x) {
                    const double dx = query_x - static_cast<double>(center_x - next_radius);
                    unseen_distance2 = std::min(unseen_distance2, dx * dx);
                }
                if (static_cast<std::uint64_t>(center_x) + next_radius < width) {
                    const double dx = query_x - static_cast<double>(center_x + next_radius);
                    unseen_distance2 = std::min(unseen_distance2, dx * dx);
                }
                if (
                    best.found &&
                    (
                        !std::isfinite(unseen_distance2) ||
                        unseen_distance2 > best.distance2
                    )
                ) {
                    proven = true;
                    break;
                }
            }
            visit_output[query] = visits;
            if (proven) {
                sample_output[query] = best.sample_index;
                distance_output[query] = best.distance2;
            } else {
                sample_output[query] = std::numeric_limits<std::uint32_t>::max();
                distance_output[query] = std::numeric_limits<double>::infinity();
            }
        };

        bool parallel_safe = true;
        for (py::ssize_t query = 0; query < query_count; ++query) {
            if (
                query_region_values[query] == 0 ||
                !std::isfinite(y_values[query]) ||
                !std::isfinite(x_values[query])
            ) {
                parallel_safe = false;
                break;
            }
        }
        const std::size_t hardware_workers = std::max(
            1U,
            std::thread::hardware_concurrency()
        );
        const std::size_t useful_workers = static_cast<std::size_t>(
            (query_count + kGeometryQueriesPerWorker - 1) /
            kGeometryQueriesPerWorker
        );
        const std::size_t desired_worker_count = std::min(
            {
                kGeometryQueryWorkerCap,
                hardware_workers,
                std::max<std::size_t>(1, useful_workers),
            }
        );
        NativeWorkerLease worker_lease(
            parallel_safe && desired_worker_count > 1
                ? desired_worker_count - 1
                : 0
        );
        const std::size_t worker_count = parallel_safe
            ? worker_lease.worker_count()
            : 1;
        if (!parallel_safe || worker_count == 1) {
            for (py::ssize_t query = 0; query < query_count; ++query) {
                query_one(query, budget_values);
            }
        } else {
            std::atomic<py::ssize_t> next_query{0};
            std::vector<std::exception_ptr> errors(worker_count);
            const auto worker = [&](std::size_t worker_index) {
                try {
                    while (true) {
                        const py::ssize_t start = next_query.fetch_add(
                            kGeometryQueryBlockSize,
                            std::memory_order_relaxed
                        );
                        if (start >= query_count) {
                            return;
                        }
                        const py::ssize_t stop = std::min(
                            query_count,
                            start + kGeometryQueryBlockSize
                        );
                        for (py::ssize_t query = start; query < stop; ++query) {
                            std::uint64_t local_budget[2] = {
                                0,
                                std::numeric_limits<std::uint64_t>::max(),
                            };
                            query_one(query, local_budget);
                        }
                    }
                } catch (...) {
                    errors[worker_index] = std::current_exception();
                }
            };
            run_native_parallel(worker_count, worker);
            for (const auto& error : errors) {
                if (error) {
                    std::rethrow_exception(error);
                }
            }
            std::uint64_t consumed = budget_values[0];
            const std::uint64_t limit = budget_values[1];
            for (py::ssize_t query = 0; query < query_count; ++query) {
                const std::uint64_t visits = visit_output[query];
                if (visits > limit - consumed) {
                    budget_values[0] = limit;
                    throw std::runtime_error("quality geometry visit cap exceeded");
                }
                consumed += visits;
            }
            budget_values[0] = consumed;
        }
    }
    return py::make_tuple(
        std::move(result_samples),
        std::move(result_distances),
        std::move(result_visits)
    );
}

// Keep comparison keys contiguous instead of gathering from full-frame rasters.
struct RegionHeapPriority {
    std::uint64_t distance;
    std::uint32_t rank;
    std::uint32_t owner;
};

static_assert(sizeof(RegionHeapPriority) == 16);

class IndexedRegionHeap {
public:
    IndexedRegionHeap(
        std::uint64_t* distance,
        std::uint32_t* owner,
        const std::uint32_t* region_rank_bits,
        std::uint32_t region_rank_count,
        std::int32_t* position,
        std::vector<std::uint32_t>&& pixel_storage
    )
        : distance_(distance),
          owner_(owner),
          region_rank_bits_(region_rank_bits),
          region_rank_count_(region_rank_count),
          position_(position),
          heap_pixel_(std::move(pixel_storage)),
          heap_priority_(heap_pixel_.size()),
          capacity_(static_cast<std::uint32_t>(heap_pixel_.size())) {}

    void initialize_candidates(std::uint64_t initial_decrease_keys) {
        const std::uint32_t source_length = capacity_;
        for (std::uint32_t source_slot = 0; source_slot < source_length; ++source_slot) {
            const std::uint32_t pixel = heap_pixel_[source_slot];
            if (owner_[pixel] != 0) {
                heap_pixel_[length_] = pixel;
                heap_priority_[length_] = priority_for(pixel);
                position_[pixel] = static_cast<std::int32_t>(length_);
                ++length_;
            }
        }
        insertions += length_;
        max_live_entries = std::max(max_live_entries, length_);
        // Compact before heapify so no sift overwrites unread band pixels.
        if (length_ > 1) {
            const std::uint32_t parent_count = (length_ - 2) / kArity + 1;
            for (std::uint32_t parent = parent_count; parent != 0; --parent) {
                sift_down(parent - 1);
            }
        }
        decrease_keys += initial_decrease_keys;
    }

    void push_or_decrease(std::uint32_t pixel) {
        const std::int32_t slot = position_[pixel];
        if (slot == -2) {
            throw std::runtime_error(
                "quality region queue cannot decrease a settled pixel"
            );
        }
        if (slot == -1) {
            insert(pixel);
            return;
        }
        if (
            slot < 0 ||
            static_cast<std::uint32_t>(slot) >= length_ ||
            heap_pixel_[static_cast<std::uint32_t>(slot)] != pixel
        ) {
            throw std::runtime_error("quality region queue position is inconsistent");
        }
        ++decrease_keys;
        sift_up(static_cast<std::uint32_t>(slot));
    }

    std::uint32_t pop() {
        if (length_ == 0) {
            throw std::runtime_error("quality region queue cannot pop an empty heap");
        }
        const std::uint32_t result = heap_pixel_[0];
        position_[result] = -2;
        --length_;
        ++pop_count;
        if (length_ != 0) {
            const std::uint32_t replacement = heap_pixel_[length_];
            heap_pixel_[0] = replacement;
            heap_priority_[0] = heap_priority_[length_];
            sift_down(0);
        }
        return result;
    }

    bool empty() const { return length_ == 0; }
    std::uint32_t capacity() const { return capacity_; }
    std::uint64_t storage_bytes() const {
        return static_cast<std::uint64_t>(heap_pixel_.capacity()) * sizeof(std::uint32_t) +
            static_cast<std::uint64_t>(heap_priority_.capacity()) * sizeof(RegionHeapPriority);
    }

    std::uint32_t max_live_entries = 0;
    std::uint64_t insertions = 0;
    std::uint64_t decrease_keys = 0;
    std::uint64_t pop_count = 0;

private:
    static constexpr std::uint32_t kArity = 4;

    RegionHeapPriority priority_for(std::uint32_t pixel) const {
        const std::uint32_t owner = owner_[pixel];
        if (owner >= region_rank_count_) {
            throw std::runtime_error("quality region queue owner rank is inconsistent");
        }
        return {distance_[pixel], region_rank_bits_[owner], owner};
    }

    static bool less_priority(
        const RegionHeapPriority& left,
        std::uint32_t left_pixel,
        const RegionHeapPriority& right,
        std::uint32_t right_pixel
    ) {
        if (left.distance != right.distance) {
            return left.distance < right.distance;
        }
        if (left.rank != right.rank) {
            return left.rank > right.rank;
        }
        if (left.owner != right.owner) {
            return left.owner < right.owner;
        }
        return left_pixel < right_pixel;
    }

    void sift_up(std::uint32_t slot) {
        const std::uint32_t pixel = heap_pixel_[slot];
        // A decrease may change the owner/rank without changing the distance.
        const RegionHeapPriority priority = priority_for(pixel);
        while (slot != 0) {
            const std::uint32_t parent = (slot - 1) / kArity;
            const std::uint32_t parent_pixel = heap_pixel_[parent];
            if (!less_priority(priority, pixel, heap_priority_[parent], parent_pixel)) {
                break;
            }
            heap_pixel_[slot] = parent_pixel;
            heap_priority_[slot] = heap_priority_[parent];
            position_[parent_pixel] = static_cast<std::int32_t>(slot);
            slot = parent;
        }
        heap_pixel_[slot] = pixel;
        heap_priority_[slot] = priority;
        position_[pixel] = static_cast<std::int32_t>(slot);
    }

    void sift_down(std::uint32_t slot) {
        const std::uint32_t pixel = heap_pixel_[slot];
        const RegionHeapPriority priority = heap_priority_[slot];
        while (true) {
            const std::uint64_t first_child =
                static_cast<std::uint64_t>(slot) * kArity + 1;
            if (first_child >= length_) {
                break;
            }
            std::uint32_t child = static_cast<std::uint32_t>(first_child);
            const std::uint32_t child_end = static_cast<std::uint32_t>(
                std::min(first_child + kArity, static_cast<std::uint64_t>(length_))
            );
            for (std::uint32_t candidate = child + 1; candidate < child_end; ++candidate) {
                if (less_priority(
                    heap_priority_[candidate], heap_pixel_[candidate],
                    heap_priority_[child], heap_pixel_[child]
                )) {
                    child = candidate;
                }
            }
            const std::uint32_t child_pixel = heap_pixel_[child];
            if (!less_priority(heap_priority_[child], child_pixel, priority, pixel)) {
                break;
            }
            heap_pixel_[slot] = child_pixel;
            heap_priority_[slot] = heap_priority_[child];
            position_[child_pixel] = static_cast<std::int32_t>(slot);
            slot = child;
        }
        heap_pixel_[slot] = pixel;
        heap_priority_[slot] = priority;
        position_[pixel] = static_cast<std::int32_t>(slot);
    }

    void insert(std::uint32_t pixel) {
        if (length_ >= capacity_) {
            throw std::runtime_error("quality region queue capacity exceeded");
        }
        const std::uint32_t slot = length_;
        heap_pixel_[slot] = pixel;
        ++length_;
        ++insertions;
        max_live_entries = std::max(max_live_entries, length_);
        sift_up(slot);
    }

    std::uint64_t* distance_;
    std::uint32_t* owner_;
    const std::uint32_t* region_rank_bits_;
    std::uint32_t region_rank_count_;
    std::int32_t* position_;
    std::vector<std::uint32_t> heap_pixel_;
    std::vector<RegionHeapPriority> heap_priority_;
    std::uint32_t capacity_;
    std::uint32_t length_ = 0;
};

std::uint64_t bgr_movement_cost(
    const std::uint8_t* guide,
    std::uint32_t left_pixel,
    std::uint32_t right_pixel,
    std::uint32_t base_cost,
    std::uint32_t edge_scale
) {
    const std::size_t left_offset = static_cast<std::size_t>(left_pixel) * 3;
    const std::size_t right_offset = static_cast<std::size_t>(right_pixel) * 3;
    std::uint32_t maximum = 0;
    for (std::uint32_t channel = 0; channel < 3; ++channel) {
        const int difference =
            static_cast<int>(guide[left_offset + channel]) -
            static_cast<int>(guide[right_offset + channel]);
        const std::uint32_t magnitude = static_cast<std::uint32_t>(
            difference < 0 ? -difference : difference
        );
        maximum = std::max(maximum, magnitude);
    }
    return static_cast<std::uint64_t>(base_cost) +
        static_cast<std::uint64_t>(edge_scale) * maximum;
}

bool region_candidate_is_better(
    std::uint64_t candidate_distance,
    std::uint32_t candidate_owner,
    std::uint64_t current_distance,
    std::uint32_t current_owner,
    const std::uint32_t* ranks
) {
    if (current_owner == 0 || candidate_distance != current_distance) {
        return current_owner == 0 || candidate_distance < current_distance;
    }
    if (ranks[candidate_owner] != ranks[current_owner]) {
        return ranks[candidate_owner] > ranks[current_owner];
    }
    return candidate_owner < current_owner;
}

void relax_region_pixel(
    std::uint32_t pixel,
    std::uint64_t candidate_distance,
    std::uint32_t candidate_owner,
    std::uint64_t* distance,
    std::uint32_t* owner,
    const std::uint32_t* ranks,
    IndexedRegionHeap& heap
) {
    if (!region_candidate_is_better(
        candidate_distance,
        candidate_owner,
        distance[pixel],
        owner[pixel],
        ranks
    )) {
        return;
    }
    distance[pixel] = candidate_distance;
    owner[pixel] = candidate_owner;
    heap.push_or_decrease(pixel);
}

template <typename Callback>
void visit_four_neighbours(
    std::uint32_t pixel,
    std::uint32_t height,
    std::uint32_t width,
    Callback callback
) {
    const std::uint32_t y = pixel / width;
    const std::uint32_t x = pixel % width;
    if (y != 0) {
        callback(pixel - width);
    }
    if (x != 0) {
        callback(pixel - 1);
    }
    if (x + 1 < width) {
        callback(pixel + 1);
    }
    if (y + 1 < height) {
        callback(pixel + width);
    }
}

py::tuple solve_geodesic_regions(
    const py::array& guide,
    const py::array& band_mask,
    const py::array& seed_region_map,
    const py::array& region_rank_bits,
    std::uint32_t movement_base_cost,
    std::uint32_t movement_edge_scale
) {
    if (movement_base_cost == 0 || movement_edge_scale == 0) {
        throw py::value_error("region movement costs must be positive");
    }
    if (
        !guide.dtype().is(py::dtype::of<std::uint8_t>()) ||
        guide.ndim() != 3 ||
        guide.shape(0) <= 0 ||
        guide.shape(1) <= 0 ||
        guide.shape(2) != 3 ||
        (guide.flags() & py::array::c_style) == 0
    ) {
        throw py::type_error("guide must be a C-contiguous uint8 BGR raster");
    }
    validate_array(band_mask, py::dtype::of<bool>(), "band_mask");
    validate_array(
        seed_region_map,
        py::dtype::of<std::uint32_t>(),
        "seed_region_map"
    );
    validate_vector(
        region_rank_bits,
        py::dtype::of<std::uint32_t>(),
        "region_rank_bits"
    );
    if (region_rank_bits.shape(0) < 2) {
        throw py::value_error("region_rank_bits must contain a positive region");
    }
    if (
        band_mask.shape(0) != guide.shape(0) ||
        band_mask.shape(1) != guide.shape(1) ||
        seed_region_map.shape(0) != guide.shape(0) ||
        seed_region_map.shape(1) != guide.shape(1)
    ) {
        throw py::value_error("region solver rasters must have matching shapes");
    }
    const std::uint64_t sample_count64 =
        static_cast<std::uint64_t>(guide.shape(0)) *
        static_cast<std::uint64_t>(guide.shape(1));
    if (sample_count64 > kUint32Max) {
        throw py::value_error("region solver raster exceeds uint32 capacity");
    }

    const auto height = static_cast<std::uint32_t>(guide.shape(0));
    const auto width = static_cast<std::uint32_t>(guide.shape(1));
    const auto sample_count = static_cast<std::uint32_t>(sample_count64);
    const auto rank_count = static_cast<std::uint32_t>(region_rank_bits.shape(0));
    const auto* guide_values = static_cast<const std::uint8_t*>(guide.data());
    const auto* band_values = static_cast<const std::uint8_t*>(band_mask.data());
    const auto* seed_values =
        static_cast<const std::uint32_t*>(seed_region_map.data());
    const auto* rank_values =
        static_cast<const std::uint32_t*>(region_rank_bits.data());

    py::array_t<std::uint32_t> owner({guide.shape(0), guide.shape(1)});
    py::array_t<std::uint64_t> distance({guide.shape(0), guide.shape(1)});
    py::array_t<std::int32_t> position({guide.shape(0), guide.shape(1)});
    auto* owner_values = owner.mutable_data();
    auto* distance_values = distance.mutable_data();
    auto* position_values = position.mutable_data();
    std::uint32_t band_count = 0;
    {
        py::gil_scoped_release release;
        for (std::uint32_t pixel = 0; pixel < sample_count; ++pixel) {
            if (seed_values[pixel] >= rank_count) {
                throw std::invalid_argument(
                    "seed region exceeds region_rank_bits"
                );
            }
            band_count += band_values[pixel] != 0;
        }
    }
    std::vector<std::uint32_t> band_pixels(band_count);
    std::uint32_t max_live_entries = 0;
    std::uint64_t insertions = 0;
    std::uint64_t decrease_keys = 0;
    std::uint64_t pop_count = 0;
    std::uint64_t heap_storage_bytes = 0;
    {
        py::gil_scoped_release release;
        std::uint32_t band_cursor = 0;
        for (std::uint32_t pixel = 0; pixel < sample_count; ++pixel) {
            owner_values[pixel] = seed_values[pixel];
            distance_values[pixel] = std::numeric_limits<std::uint64_t>::max();
            position_values[pixel] = -1;
            if (band_values[pixel] != 0) {
                band_pixels[band_cursor++] = pixel;
            }
        }
        for (const std::uint32_t pixel : band_pixels) {
            if (owner_values[pixel] != 0) {
                distance_values[pixel] = 0;
            }
        }
        std::uint64_t initial_decrease_keys = 0;
        for (const std::uint32_t pixel : band_pixels) {
            if (owner_values[pixel] != 0) {
                continue;
            }
            visit_four_neighbours(pixel, height, width, [&](std::uint32_t neighbour) {
                const std::uint32_t region = owner_values[neighbour];
                if (band_values[neighbour] != 0 || region == 0) {
                    return;
                }
                const std::uint64_t candidate_distance = bgr_movement_cost(
                    guide_values,
                    neighbour,
                    pixel,
                    movement_base_cost,
                    movement_edge_scale
                );
                if (!region_candidate_is_better(
                    candidate_distance,
                    region,
                    distance_values[pixel],
                    owner_values[pixel],
                    rank_values
                )) {
                    return;
                }
                initial_decrease_keys += owner_values[pixel] != 0;
                distance_values[pixel] = candidate_distance;
                owner_values[pixel] = region;
            });
        }
        IndexedRegionHeap heap(
            distance_values,
            owner_values,
            rank_values,
            rank_count,
            position_values,
            std::move(band_pixels)
        );
        heap.initialize_candidates(initial_decrease_keys);
        const std::uint64_t maximum_movement =
            static_cast<std::uint64_t>(movement_base_cost) +
            static_cast<std::uint64_t>(movement_edge_scale) * 255;
        const std::uint64_t maximum_prunable_source =
            std::numeric_limits<std::uint64_t>::max() - maximum_movement;
        while (!heap.empty()) {
            const std::uint32_t pixel = heap.pop();
            const std::uint64_t source_distance = distance_values[pixel];
            const std::uint32_t source_owner = owner_values[pixel];
            // Disable pruning near overflow so the original edge checks still run.
            const std::uint64_t candidate_lower_bound =
                source_distance <= maximum_prunable_source
                    ? source_distance + movement_base_cost
                    : 0;
            visit_four_neighbours(pixel, height, width, [&](std::uint32_t neighbour) {
                if (band_values[neighbour] == 0) {
                    return;
                }
                // Equal distances can still improve the owner's priority.
                if (distance_values[neighbour] < candidate_lower_bound) {
                    return;
                }
                const std::uint64_t movement =
                    bgr_movement_cost(
                        guide_values,
                        pixel,
                        neighbour,
                        movement_base_cost,
                        movement_edge_scale
                    );
                if (
                    source_distance >
                    std::numeric_limits<std::uint64_t>::max() - movement
                ) {
                    throw std::runtime_error("quality region queue distance overflow");
                }
                relax_region_pixel(
                    neighbour,
                    source_distance + movement,
                    source_owner,
                    distance_values,
                    owner_values,
                    rank_values,
                    heap
                );
            });
        }
        if (heap.pop_count != band_count) {
            throw std::invalid_argument("edge-band component has no seed");
        }
        max_live_entries = heap.max_live_entries;
        insertions = heap.insertions;
        decrease_keys = heap.decrease_keys;
        pop_count = heap.pop_count;
        heap_storage_bytes = heap.storage_bytes();
    }

    py::dict statistics;
    statistics["max_live_entries"] = max_live_entries;
    statistics["settled_count"] = pop_count;
    statistics["queue_insertions"] = insertions;
    statistics["queue_decrease_keys"] = decrease_keys;
    statistics["queue_pops"] = pop_count;
    // Preserve the scratch-accounting field while charging the cached keys too.
    statistics["heap_pixel_bytes"] = heap_storage_bytes;
    return py::make_tuple(
        std::move(owner),
        std::move(distance),
        std::move(position),
        std::move(statistics)
    );
}

py::tuple solve_geodesic_regions_owner_only(
    const py::array& guide,
    const py::array& band_mask,
    const py::array& seed_region_map,
    const py::array& region_rank_bits,
    std::uint32_t movement_base_cost,
    std::uint32_t movement_edge_scale
) {
    if (movement_base_cost == 0 || movement_edge_scale == 0) {
        throw py::value_error("region movement costs must be positive");
    }
    if (
        !guide.dtype().is(py::dtype::of<std::uint8_t>()) ||
        guide.ndim() != 3 ||
        guide.shape(0) <= 0 ||
        guide.shape(1) <= 0 ||
        guide.shape(2) != 3 ||
        (guide.flags() & py::array::c_style) == 0
    ) {
        throw py::type_error("guide must be a C-contiguous uint8 BGR raster");
    }
    validate_array(band_mask, py::dtype::of<bool>(), "band_mask");
    validate_array(
        seed_region_map,
        py::dtype::of<std::uint32_t>(),
        "seed_region_map"
    );
    validate_vector(
        region_rank_bits,
        py::dtype::of<std::uint32_t>(),
        "region_rank_bits"
    );
    if (region_rank_bits.shape(0) < 2) {
        throw py::value_error("region_rank_bits must contain a positive region");
    }
    if (
        band_mask.shape(0) != guide.shape(0) ||
        band_mask.shape(1) != guide.shape(1) ||
        seed_region_map.shape(0) != guide.shape(0) ||
        seed_region_map.shape(1) != guide.shape(1)
    ) {
        throw py::value_error("region solver rasters must have matching shapes");
    }
    const std::uint64_t sample_count64 =
        static_cast<std::uint64_t>(guide.shape(0)) *
        static_cast<std::uint64_t>(guide.shape(1));
    if (sample_count64 > kUint32Max) {
        throw py::value_error("region solver raster exceeds uint32 capacity");
    }
    const std::uint64_t maximum_movement =
        static_cast<std::uint64_t>(movement_base_cost) +
        static_cast<std::uint64_t>(movement_edge_scale) * 255ULL;
    constexpr std::uint32_t kDialBuckets = 4096U;
    constexpr std::uint32_t kDialMask = kDialBuckets - 1U;
    if (maximum_movement >= kDialBuckets) {
        py::tuple full = solve_geodesic_regions(
            guide,
            band_mask,
            seed_region_map,
            region_rank_bits,
            movement_base_cost,
            movement_edge_scale
        );
        return py::make_tuple(full[0], full[3]);
    }

    const auto height = static_cast<std::uint32_t>(guide.shape(0));
    const auto width = static_cast<std::uint32_t>(guide.shape(1));
    const auto sample_count = static_cast<std::uint32_t>(sample_count64);
    const auto rank_count = static_cast<std::uint32_t>(region_rank_bits.shape(0));
    const auto* guide_values = static_cast<const std::uint8_t*>(guide.data());
    const auto* band_values = static_cast<const std::uint8_t*>(band_mask.data());
    const auto* seed_values =
        static_cast<const std::uint32_t*>(seed_region_map.data());
    const auto* rank_values =
        static_cast<const std::uint32_t*>(region_rank_bits.data());

    py::array_t<std::uint32_t> owner({guide.shape(0), guide.shape(1)});
    auto* owner_values = owner.mutable_data();
    std::uint32_t band_count = 0;
    std::uint64_t heap_storage_bytes = 0;
    {
        py::gil_scoped_release release;
        constexpr std::uint32_t kThreadLocalSlotCap = 2500000U;
        thread_local std::vector<std::uint32_t> tl_band_slot;
        std::vector<std::uint32_t> local_band_slot;
        std::uint32_t* band_slot = nullptr;
        if (sample_count <= kThreadLocalSlotCap) {
            if (tl_band_slot.size() < sample_count) {
                tl_band_slot.assign(sample_count, kUint32Max);
            }
            band_slot = tl_band_slot.data();
        } else {
            local_band_slot.assign(sample_count, kUint32Max);
            band_slot = local_band_slot.data();
        }

        std::memcpy(
            owner_values,
            seed_values,
            static_cast<std::size_t>(sample_count) * sizeof(std::uint32_t)
        );
        std::vector<std::uint32_t> band_pixels;
        band_pixels.reserve(sample_count / 16U + 64U);
        for (std::uint32_t pixel = 0; pixel < sample_count; ++pixel) {
            if (seed_values[pixel] >= rank_count) {
                throw std::invalid_argument(
                    "seed region exceeds region_rank_bits"
                );
            }
            if (band_values[pixel] != 0) {
                band_slot[pixel] = static_cast<std::uint32_t>(band_pixels.size());
                band_pixels.push_back(pixel);
            }
        }
        band_count = static_cast<std::uint32_t>(band_pixels.size());
        heap_storage_bytes = static_cast<std::uint64_t>(band_count) * 20ULL;

        struct SlotCleaner {
            std::uint32_t* slots;
            const std::vector<std::uint32_t>& pixels;
            bool active;
            ~SlotCleaner() {
                if (active) {
                    for (const std::uint32_t p : pixels) {
                        slots[p] = kUint32Max;
                    }
                }
            }
        } slot_cleaner{band_slot, band_pixels, sample_count <= kThreadLocalSlotCap};

        if (band_count > 0) {
            std::vector<std::uint64_t> band_dist(
                band_count,
                std::numeric_limits<std::uint64_t>::max()
            );
            std::vector<std::uint32_t> band_owner(band_count, 0);
            std::vector<std::uint8_t> band_settled(band_count, 0);

            thread_local std::array<std::vector<std::uint32_t>, kDialBuckets> tl_buckets;
            thread_local std::vector<std::uint32_t> tl_active_bucket;
            std::array<std::uint64_t, kDialBuckets / 64U> non_empty_mask{};

            struct BucketCleaner {
                std::array<std::vector<std::uint32_t>, kDialBuckets>& buckets;
                std::array<std::uint64_t, kDialBuckets / 64U>& mask;
                ~BucketCleaner() {
                    for (std::uint32_t w = 0; w < kDialBuckets / 64U; ++w) {
                        std::uint64_t bits = mask[w];
                        while (bits != 0) {
                            const std::uint32_t bit = ctz64(bits);
                            buckets[(w << 6U) | bit].clear();
                            bits &= bits - 1ULL;
                        }
                    }
                }
            } bucket_cleaner{tl_buckets, non_empty_mask};

            const auto push_bucket = [&](std::uint64_t dist, std::uint32_t slot) {
                const std::uint32_t b = static_cast<std::uint32_t>(dist & kDialMask);
                tl_buckets[b].push_back(slot);
                non_empty_mask[b >> 6U] |= (1ULL << (b & 63U));
            };

            for (std::uint32_t slot = 0; slot < band_count; ++slot) {
                const std::uint32_t pixel = band_pixels[slot];
                const std::uint32_t seed_owner = owner_values[pixel];
                if (seed_owner != 0) {
                    band_dist[slot] = 0;
                    band_owner[slot] = seed_owner;
                    push_bucket(0, slot);
                    continue;
                }
                std::uint64_t best_dist = std::numeric_limits<std::uint64_t>::max();
                std::uint32_t best_owner = 0;
                visit_four_neighbours(pixel, height, width, [&](std::uint32_t neighbour) {
                    if (band_slot[neighbour] != kUint32Max) {
                        return;
                    }
                    const std::uint32_t region = owner_values[neighbour];
                    if (region == 0) {
                        return;
                    }
                    const std::uint64_t cand_dist = bgr_movement_cost(
                        guide_values,
                        neighbour,
                        pixel,
                        movement_base_cost,
                        movement_edge_scale
                    );
                    if (region_candidate_is_better(
                        cand_dist,
                        region,
                        best_dist,
                        best_owner,
                        rank_values
                    )) {
                        best_dist = cand_dist;
                        best_owner = region;
                    }
                });
                if (best_owner != 0) {
                    band_dist[slot] = best_dist;
                    band_owner[slot] = best_owner;
                    push_bucket(best_dist, slot);
                }
            }

            const std::uint64_t maximum_prunable_source =
                std::numeric_limits<std::uint64_t>::max() - maximum_movement;
            std::uint64_t cur_dist = 0;
            std::uint32_t settled = 0;
            while (settled < band_count) {
                const std::uint32_t b = static_cast<std::uint32_t>(cur_dist & kDialMask);
                const std::uint32_t w0 = b >> 6U;
                const std::uint32_t bit0 = b & 63U;
                std::uint32_t next_b = kDialBuckets;
                const std::uint64_t first_masked = non_empty_mask[w0] & (~0ULL << bit0);
                if (first_masked != 0) {
                    next_b = (w0 << 6U) | ctz64(first_masked);
                } else {
                    for (std::uint32_t step = 1; step < kDialBuckets / 64U; ++step) {
                        const std::uint32_t w = (w0 + step) & ((kDialBuckets / 64U) - 1U);
                        const std::uint64_t word = non_empty_mask[w];
                        if (word != 0) {
                            next_b = (w << 6U) | ctz64(word);
                            break;
                        }
                    }
                }
                if (next_b == kDialBuckets) {
                    break;
                }
                const std::uint32_t delta = (next_b - b) & kDialMask;
                if (cur_dist > std::numeric_limits<std::uint64_t>::max() - delta) {
                    throw std::runtime_error("quality region queue distance overflow");
                }
                cur_dist += delta;
                non_empty_mask[next_b >> 6U] &= ~(1ULL << (next_b & 63U));
                tl_active_bucket.clear();
                tl_active_bucket.swap(tl_buckets[next_b]);

                const std::uint64_t candidate_lower_bound =
                    cur_dist <= maximum_prunable_source
                        ? cur_dist + movement_base_cost
                        : 0;
                for (const std::uint32_t slot : tl_active_bucket) {
                    if (band_settled[slot] != 0 || band_dist[slot] != cur_dist) {
                        continue;
                    }
                    band_settled[slot] = 1;
                    ++settled;
                    const std::uint32_t pixel = band_pixels[slot];
                    const std::uint32_t source_owner = band_owner[slot];
                    owner_values[pixel] = source_owner;

                    const std::uint32_t y = pixel / width;
                    const std::uint32_t x = pixel - y * width;
                    const auto relax_nb = [&](std::uint32_t nb) {
                        const std::uint32_t nb_slot = band_slot[nb];
                        if (nb_slot == kUint32Max || band_settled[nb_slot] != 0) {
                            return;
                        }
                        const std::uint64_t nb_dist = band_dist[nb_slot];
                        if (nb_dist < candidate_lower_bound) {
                            return;
                        }
                        const std::uint64_t movement = bgr_movement_cost(
                            guide_values,
                            pixel,
                            nb,
                            movement_base_cost,
                            movement_edge_scale
                        );
                        if (cur_dist > std::numeric_limits<std::uint64_t>::max() - movement) {
                            throw std::runtime_error("quality region queue distance overflow");
                        }
                        const std::uint64_t cand_dist = cur_dist + movement;
                        if (region_candidate_is_better(
                            cand_dist,
                            source_owner,
                            nb_dist,
                            band_owner[nb_slot],
                            rank_values
                        )) {
                            band_dist[nb_slot] = cand_dist;
                            band_owner[nb_slot] = source_owner;
                            push_bucket(cand_dist, nb_slot);
                        }
                    };
                    if (y != 0) {
                        relax_nb(pixel - width);
                    }
                    if (x != 0) {
                        relax_nb(pixel - 1);
                    }
                    if (x + 1 < width) {
                        relax_nb(pixel + 1);
                    }
                    if (y + 1 < height) {
                        relax_nb(pixel + width);
                    }
                }
            }
            if (settled != band_count) {
                throw std::invalid_argument("edge-band component has no seed");
            }
        }
    }

    py::dict statistics;
    statistics["max_live_entries"] = band_count;
    statistics["settled_count"] = band_count;
    statistics["queue_insertions"] = static_cast<std::uint64_t>(band_count);
    statistics["queue_decrease_keys"] = static_cast<std::uint64_t>(0);
    statistics["queue_pops"] = static_cast<std::uint64_t>(band_count);
    statistics["heap_pixel_bytes"] = heap_storage_bytes;
    return py::make_tuple(std::move(owner), std::move(statistics));
}

py::tuple build_index(
    const py::array& region_ids,
    std::uint64_t region_count_value,
    const py::object& include_object,
    const py::object& allowed_regions_object
) {
    validate_array(region_ids, py::dtype::of<std::uint32_t>(), "region_ids");
    if (region_count_value == 0 || region_count_value + 1 > kUint32Max) {
        throw py::value_error("region_count plus its sentinel must fit uint32");
    }
    const std::uint64_t height = static_cast<std::uint64_t>(region_ids.shape(0));
    const std::uint64_t width = static_cast<std::uint64_t>(region_ids.shape(1));
    const std::uint64_t sample_count = height * width;
    if (sample_count + 1 > kUint32Max) {
        throw py::value_error("sample count plus its sentinel must fit uint32");
    }

    const auto* include = static_cast<const std::uint8_t*>(nullptr);
    py::array include_array;
    if (!include_object.is_none()) {
        include_array = py::cast<py::array>(include_object);
        validate_array(include_array, py::dtype::of<bool>(), "include");
        if (
            include_array.shape(0) != region_ids.shape(0) ||
            include_array.shape(1) != region_ids.shape(1)
        ) {
            throw py::value_error("include must match region_ids shape");
        }
        include = static_cast<const std::uint8_t*>(include_array.data());
    }

    const auto region_count = static_cast<std::uint32_t>(region_count_value);
    const auto* allowed_regions = static_cast<const std::uint8_t*>(nullptr);
    py::array allowed_regions_array;
    if (!allowed_regions_object.is_none()) {
        allowed_regions_array = py::cast<py::array>(allowed_regions_object);
        validate_vector(
            allowed_regions_array,
            py::dtype::of<bool>(),
            "allowed_regions",
            static_cast<py::ssize_t>(region_count) + 1
        );
        allowed_regions = static_cast<const std::uint8_t*>(
            allowed_regions_array.data()
        );
    }
    const auto sample_count32 = static_cast<std::uint32_t>(sample_count);
    const auto width32 = static_cast<std::uint32_t>(width);
    const auto* regions = static_cast<const std::uint32_t*>(region_ids.data());
    py::array_t<std::uint32_t> offsets(static_cast<py::ssize_t>(region_count) + 1);
    auto* offset_values = offsets.mutable_data();
    std::fill(offset_values, offset_values + region_count + 1, 0);
    std::vector<std::uint8_t> seen(region_count, 0);
    std::uint32_t included_count = 0;

    {
        py::gil_scoped_release release;
        for (std::uint32_t sample_index = 0; sample_index < sample_count32; ++sample_index) {
            const std::uint32_t region = regions[sample_index];
            if (region == 0) {
                throw std::invalid_argument("every native sample must have a positive region ID");
            }
            if (region > region_count) {
                throw std::invalid_argument("region ID exceeds region_count");
            }
            seen[region - 1] = 1;
            if (
                (include == nullptr || include[sample_index] != 0) &&
                (
                    allowed_regions == nullptr ||
                    allowed_regions[region] != 0
                )
            ) {
                ++offset_values[region - 1];
                ++included_count;
            }
        }
        for (std::uint32_t region_index = 0; region_index < region_count; ++region_index) {
            if (seen[region_index] == 0) {
                throw std::invalid_argument(
                    "positive region IDs must be contiguous through region_count"
                );
            }
        }
        std::uint32_t running = 0;
        for (std::uint32_t region_index = 0; region_index < region_count; ++region_index) {
            running += offset_values[region_index];
            offset_values[region_index] = running;
        }
        offset_values[region_count] = included_count;
    }

    py::array_t<std::uint32_t> members(included_count);
    auto* member_values = members.mutable_data();
    {
        py::gil_scoped_release release;
        const bool packed_coordinates =
            height <= kPackedCoordinateLimit && width <= kPackedCoordinateLimit;
        std::uint32_t row = static_cast<std::uint32_t>(height) - 1;
        std::uint32_t column = width32 - 1;
        for (std::uint32_t cursor = sample_count32; cursor > 0; --cursor) {
            const std::uint32_t sample_index = cursor - 1;
            const std::uint32_t region = regions[sample_index];
            if (
                (include == nullptr || include[sample_index] != 0) &&
                (
                    allowed_regions == nullptr ||
                    allowed_regions[region] != 0
                )
            ) {
                const std::uint32_t region_slot = region - 1;
                const std::uint32_t destination = --offset_values[region_slot];
                member_values[destination] = packed_coordinates
                    ? (row << 16) | column
                    : sample_index;
            }
            if (column == 0) {
                column = width32 - 1;
                if (row > 0) {
                    --row;
                }
            } else {
                --column;
            }
        }
        const std::size_t hardware_workers = std::max<std::size_t>(
            1,
            std::thread::hardware_concurrency()
        );
        const std::size_t desired_worker_count = std::min<std::size_t>(
            {kIndexBuildWorkerCap, hardware_workers, region_count}
        );
        NativeWorkerLease worker_lease(
            desired_worker_count > 1 ? desired_worker_count - 1 : 0
        );
        const std::size_t worker_count = worker_lease.worker_count();
        if (worker_count == 1) {
            for (
                std::uint32_t region_index = 0;
                region_index < region_count;
                ++region_index
            ) {
                partition_region_tree(
                    member_values,
                    offset_values[region_index],
                    offset_values[region_index + 1],
                    static_cast<std::uint32_t>(height),
                    width32,
                    packed_coordinates
                );
            }
        } else {
            std::atomic<std::uint32_t> next_region{0};
            std::vector<std::exception_ptr> errors(worker_count);
            const auto worker = [&](std::size_t worker_index) {
                try {
                    while (true) {
                        const std::uint32_t region_index = next_region.fetch_add(
                            1,
                            std::memory_order_relaxed
                        );
                        if (region_index >= region_count) {
                            return;
                        }
                        partition_region_tree(
                            member_values,
                            offset_values[region_index],
                            offset_values[region_index + 1],
                            static_cast<std::uint32_t>(height),
                            width32,
                            packed_coordinates
                        );
                    }
                } catch (...) {
                    errors[worker_index] = std::current_exception();
                }
            };
            run_native_parallel(worker_count, worker);
            for (const auto& error : errors) {
                if (error) {
                    std::rethrow_exception(error);
                }
            }
        }
        if (packed_coordinates) {
            for (std::uint32_t cursor = 0; cursor < included_count; ++cursor) {
                const std::uint32_t packed = member_values[cursor];
                const std::uint32_t row = packed >> 16;
                const std::uint32_t column = packed & kPackedCoordinateLimit;
                member_values[cursor] = row * width32 + column;
            }
        }
    }
    return py::make_tuple(std::move(members), std::move(offsets));
}

inline std::uint32_t count_u16_bits(std::uint16_t value) {
    std::uint32_t count = 0;
    while (value != 0) {
        value = static_cast<std::uint16_t>(value & (value - 1));
        ++count;
    }
    return count;
}

inline std::uint16_t missing_lane_mask(const std::uint8_t* valid) {
#if defined(_M_X64) || defined(__x86_64__)
    const __m128i values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(valid));
    return static_cast<std::uint16_t>(
        _mm_movemask_epi8(_mm_cmpeq_epi8(values, _mm_setzero_si128()))
    );
#else
    std::uint16_t mask = 0;
    for (std::uint32_t lane = 0; lane < 16; ++lane) {
        mask = static_cast<std::uint16_t>(
            mask | (static_cast<std::uint16_t>(valid[lane] == 0) << lane)
        );
    }
    return mask;
#endif
}

inline std::uint32_t least_u16_bit(std::uint16_t value) {
    std::uint32_t bit = 0;
    while ((value & 1U) == 0U) {
        value = static_cast<std::uint16_t>(value >> 1);
        ++bit;
    }
    return bit;
}

inline std::uint32_t greatest_u16_bit(std::uint16_t value) {
    std::uint32_t bit = 0;
    while (value > 1) {
        value = static_cast<std::uint16_t>(value >> 1);
        ++bit;
    }
    return bit;
}

template <typename MissingMaskAt>
std::uint64_t accumulate_missing_lane_mask_histograms_into(
    py::ssize_t rows,
    std::uint64_t width,
    MissingMaskAt&& missing_mask_at,
    std::uint64_t* lane_bins,
    std::uint64_t* span_bins,
    std::uint8_t* coverage_values
) {
    std::uint64_t run_count = 0;
    for (py::ssize_t row = 0; row < rows; ++row) {
        auto* row_coverage = coverage_values == nullptr
            ? nullptr
            : coverage_values + static_cast<std::uint64_t>(row) * width;
        bool active = false;
        std::uint64_t start = 0;
        std::uint64_t end = 0;
        const auto flush_active = [&]() {
            if (!active) {
                return;
            }
            const std::uint64_t lane_length = end - start;
            const std::uint64_t pixel_span =
                (end - 1ULL) / 16ULL - start / 16ULL + 1ULL;
            if (lane_bins[lane_length] == std::numeric_limits<std::uint64_t>::max()) {
                throw std::overflow_error("hole-run lane histogram exceeds uint64");
            }
            if (span_bins[pixel_span] == std::numeric_limits<std::uint64_t>::max()) {
                throw std::overflow_error("hole-run span histogram exceeds uint64");
            }
            if (run_count == std::numeric_limits<std::uint64_t>::max()) {
                throw std::overflow_error("hole-run count exceeds uint64");
            }
            ++lane_bins[lane_length];
            ++span_bins[pixel_span];
            ++run_count;
            active = false;
        };
        for (std::uint64_t column = 0; column < width; ++column) {
            std::uint16_t mask = missing_mask_at(row, column);
            if (row_coverage != nullptr) {
                row_coverage[column] = static_cast<std::uint8_t>(
                    16U - count_u16_bits(mask)
                );
            }
            while (mask != 0) {
                const std::uint32_t first = least_u16_bit(mask);
                const std::uint16_t shifted = static_cast<std::uint16_t>(mask >> first);
                const std::uint16_t after_run = static_cast<std::uint16_t>(~shifted);
                const std::uint32_t length = after_run == 0
                    ? 16U - first
                    : least_u16_bit(after_run);
                const std::uint64_t run_start = 16ULL * column + first;
                const std::uint64_t run_end = run_start + length;
                if (active && run_start != end) {
                    flush_active();
                }
                if (!active) {
                    active = true;
                    start = run_start;
                }
                end = run_end;
                const std::uint32_t run_bits = ((1U << length) - 1U) << first;
                mask = static_cast<std::uint16_t>(
                    mask & static_cast<std::uint16_t>(~run_bits)
                );
            }
        }
        flush_active();
    }
    return run_count;
}

std::uint64_t accumulate_lane_hole_run_histograms_into_native(
    const py::array& lane_valid,
    py::array lane_histogram,
    py::array span_histogram,
    py::object coverage_output_object
) {
    if (
        !lane_valid.dtype().is(py::dtype::of<bool>()) ||
        lane_valid.ndim() != 3 || lane_valid.shape(2) != 16 ||
        (lane_valid.flags() & py::array::c_style) == 0
    ) {
        throw py::type_error(
            "lane_valid must be a C-contiguous bool [rows,width,16] array"
        );
    }
    const std::uint64_t width = static_cast<std::uint64_t>(lane_valid.shape(1));
    if (width == 0) {
        throw py::value_error("lane validity width must be positive");
    }
    const std::uint64_t expected_lane_bins = 16ULL * (width + 1ULL);
    const std::uint64_t expected_span_bins = width + 1ULL;
    if (
        !lane_histogram.dtype().is(py::dtype::of<std::uint64_t>()) ||
        lane_histogram.ndim() != 1 ||
        (lane_histogram.flags() & py::array::c_style) == 0 ||
        !lane_histogram.writeable() ||
        static_cast<std::uint64_t>(lane_histogram.shape(0)) != expected_lane_bins
    ) {
        throw py::type_error(
            "lane_histogram must be a writable uint64 vector with 16*(width+1) bins"
        );
    }
    if (
        !span_histogram.dtype().is(py::dtype::of<std::uint64_t>()) ||
        span_histogram.ndim() != 1 ||
        (span_histogram.flags() & py::array::c_style) == 0 ||
        !span_histogram.writeable() ||
        static_cast<std::uint64_t>(span_histogram.shape(0)) != expected_span_bins
    ) {
        throw py::type_error(
            "span_histogram must be a writable uint64 vector with width+1 bins"
        );
    }
    py::array coverage_output;
    std::uint8_t* coverage_values = nullptr;
    if (!coverage_output_object.is_none()) {
        coverage_output = py::cast<py::array>(coverage_output_object);
        if (
            !coverage_output.dtype().is(py::dtype::of<std::uint8_t>()) ||
            coverage_output.ndim() != 2 ||
            (coverage_output.flags() & py::array::c_style) == 0 ||
            !coverage_output.writeable() ||
            coverage_output.shape(0) != lane_valid.shape(0) ||
            coverage_output.shape(1) != lane_valid.shape(1)
        ) {
            throw py::type_error(
                "coverage_output must be a writable C-contiguous uint8 [rows,width] array"
            );
        }
        coverage_values = static_cast<std::uint8_t*>(coverage_output.mutable_data());
    }
    const auto* valid = static_cast<const std::uint8_t*>(lane_valid.data());
    auto* lane_bins = static_cast<std::uint64_t*>(lane_histogram.mutable_data());
    auto* span_bins = static_cast<std::uint64_t*>(span_histogram.mutable_data());
    std::uint64_t run_count;
    {
        py::gil_scoped_release release;
        run_count = accumulate_missing_lane_mask_histograms_into(
            lane_valid.shape(0),
            width,
            [valid, width](py::ssize_t row, std::uint64_t column) {
                const std::uint64_t pixel =
                    static_cast<std::uint64_t>(row) * width + column;
                return missing_lane_mask(valid + 16ULL * pixel);
            },
            lane_bins,
            span_bins,
            coverage_values
        );
    }
    return run_count;
}

std::uint64_t accumulate_packed_lane_hole_run_histograms_into_native(
    const py::array& packed_missing_masks,
    py::array lane_histogram,
    py::array span_histogram,
    py::object coverage_output_object
) {
    if (
        !packed_missing_masks.dtype().is(py::dtype::of<std::uint16_t>()) ||
        packed_missing_masks.ndim() != 2 ||
        (packed_missing_masks.flags() & py::array::c_style) == 0
    ) {
        throw py::type_error(
            "packed_missing_masks must be a C-contiguous uint16 [rows,width] array"
        );
    }
    const std::uint64_t width = static_cast<std::uint64_t>(
        packed_missing_masks.shape(1)
    );
    if (width == 0) {
        throw py::value_error("packed missing-mask width must be positive");
    }
    const std::uint64_t expected_lane_bins = 16ULL * (width + 1ULL);
    const std::uint64_t expected_span_bins = width + 1ULL;
    if (
        !lane_histogram.dtype().is(py::dtype::of<std::uint64_t>()) ||
        lane_histogram.ndim() != 1 ||
        (lane_histogram.flags() & py::array::c_style) == 0 ||
        !lane_histogram.writeable() ||
        static_cast<std::uint64_t>(lane_histogram.shape(0)) != expected_lane_bins
    ) {
        throw py::type_error(
            "lane_histogram must be a writable uint64 vector with 16*(width+1) bins"
        );
    }
    if (
        !span_histogram.dtype().is(py::dtype::of<std::uint64_t>()) ||
        span_histogram.ndim() != 1 ||
        (span_histogram.flags() & py::array::c_style) == 0 ||
        !span_histogram.writeable() ||
        static_cast<std::uint64_t>(span_histogram.shape(0)) != expected_span_bins
    ) {
        throw py::type_error(
            "span_histogram must be a writable uint64 vector with width+1 bins"
        );
    }
    py::array coverage_output;
    std::uint8_t* coverage_values = nullptr;
    if (!coverage_output_object.is_none()) {
        coverage_output = py::cast<py::array>(coverage_output_object);
        if (
            !coverage_output.dtype().is(py::dtype::of<std::uint8_t>()) ||
            coverage_output.ndim() != 2 ||
            (coverage_output.flags() & py::array::c_style) == 0 ||
            !coverage_output.writeable() ||
            coverage_output.shape(0) != packed_missing_masks.shape(0) ||
            coverage_output.shape(1) != packed_missing_masks.shape(1)
        ) {
            throw py::type_error(
                "coverage_output must be a writable C-contiguous uint8 [rows,width] array"
            );
        }
        coverage_values = static_cast<std::uint8_t*>(coverage_output.mutable_data());
    }
    const auto* masks = static_cast<const std::uint16_t*>(
        packed_missing_masks.data()
    );
    auto* lane_bins = static_cast<std::uint64_t*>(lane_histogram.mutable_data());
    auto* span_bins = static_cast<std::uint64_t*>(span_histogram.mutable_data());
    std::uint64_t run_count;
    {
        py::gil_scoped_release release;
        run_count = accumulate_missing_lane_mask_histograms_into(
            packed_missing_masks.shape(0),
            width,
            [masks, width](py::ssize_t row, std::uint64_t column) {
                return masks[static_cast<std::uint64_t>(row) * width + column];
            },
            lane_bins,
            span_bins,
            coverage_values
        );
    }
    return run_count;
}

py::tuple count_repair_state_bits_native(const py::array& repair_bits) {
    if (
        !repair_bits.dtype().is(py::dtype::of<std::uint8_t>()) ||
        repair_bits.ndim() != 2 ||
        (repair_bits.flags() & py::array::c_style) == 0
    ) {
        throw py::type_error(
            "repair_bits must be a C-contiguous uint8 [rows,width] array"
        );
    }
    const auto* values = static_cast<const std::uint8_t*>(repair_bits.data());
    const py::ssize_t pixel_count = repair_bits.size();
    std::array<std::uint64_t, 7> counts{};
    {
        py::gil_scoped_release release;
        for (py::ssize_t index = 0; index < pixel_count; ++index) {
            const std::uint8_t value = values[index];
            if (value == 0) {
                continue;
            }
            for (std::size_t bit = 0; bit < counts.size(); ++bit) {
                counts[bit] += (value >> bit) & 1U;
            }
        }
    }
    return py::make_tuple(
        counts[0], counts[1], counts[2], counts[3], counts[4], counts[5], counts[6]
    );
}

void build_relative_eye_offsets_into_native(
    const py::array& near_score,
    double stereo_strength,
    double convergence,
    int direction,
    py::array output,
    py::array scratch
) {
    if (
        !near_score.dtype().is(py::dtype::of<float>()) ||
        near_score.ndim() != 2 ||
        (near_score.flags() & py::array::c_style) == 0 ||
        near_score.shape(0) <= 0 || near_score.shape(1) <= 0
    ) {
        throw py::type_error(
            "near_score must be a nonempty C-contiguous float32 raster"
        );
    }
    if (
        !output.dtype().is(py::dtype::of<std::int32_t>()) ||
        output.ndim() != 2 ||
        (output.flags() & py::array::c_style) == 0 ||
        !output.writeable() ||
        output.shape(0) != near_score.shape(0) ||
        output.shape(1) != near_score.shape(1)
    ) {
        throw py::type_error(
            "output must be a writable matching C-contiguous int32 raster"
        );
    }
    if (
        !scratch.dtype().is(py::dtype::of<double>()) ||
        scratch.ndim() != 1 ||
        (scratch.flags() & py::array::c_style) == 0 ||
        !scratch.writeable() ||
        scratch.shape(0) != near_score.shape(1)
    ) {
        throw py::type_error(
            "scratch must be a writable width-sized C-contiguous float64 vector"
        );
    }
    if (
        !std::isfinite(stereo_strength) || stereo_strength < 0.0 ||
        stereo_strength > 5.0
    ) {
        throw py::value_error("stereo_strength must be finite and within [0,5]");
    }
    if (!std::isfinite(convergence) || convergence < 0.0 || convergence > 1.0) {
        throw py::value_error("convergence must be finite and within [0,1]");
    }
    if (direction != 1 && direction != -1) {
        throw py::value_error("direction must be 1 or -1");
    }
    const auto* near_values = static_cast<const float*>(near_score.data());
    auto* output_values = static_cast<std::int32_t*>(output.mutable_data());
    auto* scratch_values = static_cast<double*>(scratch.mutable_data());
    const std::uint64_t width = static_cast<std::uint64_t>(near_score.shape(1));
    const std::uint64_t height = static_cast<std::uint64_t>(near_score.shape(0));
    double scale = static_cast<double>(width) * stereo_strength;
    scale = scale / 200.0;
    {
        py::gil_scoped_release release;
        for (std::uint64_t row = 0; row < height; ++row) {
            const std::uint64_t row_start = row * width;
            for (std::uint64_t column = 0; column < width; ++column) {
                const std::uint64_t index = row_start + column;
                double value = static_cast<double>(near_values[index]);
                if (!std::isfinite(value)) {
                    throw py::value_error("near_score must contain only finite values");
                }
                value = value - convergence;
                value = value * scale;
                value = value * 16.0;
                if (direction < 0) {
                    value = -value;
                }
                value = value - 0.5;
                value = std::ceil(value);
                if (
                    value < static_cast<double>(std::numeric_limits<std::int32_t>::min()) ||
                    value > static_cast<double>(std::numeric_limits<std::int32_t>::max())
                ) {
                    throw py::value_error(
                        "projected fine-sample offsets exceed int32 range"
                    );
                }
                scratch_values[column] = value;
                output_values[index] = static_cast<std::int32_t>(value);
            }
        }
    }
}

std::uint64_t update_relative_total_inside_band_native(
    py::array total,
    const py::array& near_score,
    const py::array& edge_band,
    double stereo_strength,
    double convergence
) {
    validate_array(total, py::dtype::of<double>(), "total_disparity_fraction");
    validate_array(near_score, py::dtype::of<float>(), "near_score");
    validate_array(edge_band, py::dtype::of<bool>(), "edge_band");
    if (
        total.shape(0) != near_score.shape(0) ||
        total.shape(1) != near_score.shape(1) ||
        edge_band.shape(0) != near_score.shape(0) ||
        edge_band.shape(1) != near_score.shape(1)
    ) {
        throw py::value_error("relative band rasters must have matching shapes");
    }
    if (!total.writeable()) {
        throw py::value_error("total_disparity_fraction must be writeable");
    }
    if (
        !std::isfinite(stereo_strength) || stereo_strength < 0.0 ||
        stereo_strength > 5.0
    ) {
        throw py::value_error("stereo_strength must be finite and within [0,5]");
    }
    if (!std::isfinite(convergence) || convergence < 0.0 || convergence > 1.0) {
        throw py::value_error("convergence must be finite and within [0,1]");
    }

    auto* total_values = static_cast<double*>(total.mutable_data());
    const auto* near_values = static_cast<const float*>(near_score.data());
    const auto* edge_values = static_cast<const std::uint8_t*>(edge_band.data());
    const std::uint64_t sample_count = static_cast<std::uint64_t>(near_score.size());
    std::uint64_t updated = 0;
    {
        py::gil_scoped_release release;
        for (std::uint64_t pixel = 0; pixel < sample_count; ++pixel) {
            if (edge_values[pixel] == 0) {
                continue;
            }
            double value = static_cast<double>(near_values[pixel]);
            if (!std::isfinite(value)) {
                throw std::invalid_argument("near_score must contain only finite values");
            }
            value = value - convergence;
            value = value * stereo_strength;
            value = value / 100.0;
            total_values[pixel] = value;
            ++updated;
        }
    }
    return updated;
}

double max_four_neighbour_jump_native(const py::array& displacement) {
    validate_array(displacement, py::dtype::of<double>(), "displacement");
    const auto height = static_cast<std::uint64_t>(displacement.shape(0));
    const auto width = static_cast<std::uint64_t>(displacement.shape(1));
    const auto* values = static_cast<const double*>(displacement.data());
    double maximum = 0.0;
    {
        py::gil_scoped_release release;
        const std::size_t hardware_workers = std::max(
            1U,
            std::thread::hardware_concurrency()
        );
        const std::size_t useful_workers = static_cast<std::size_t>(
            (height + kFourNeighbourJumpRowsPerWorker - 1) /
            kFourNeighbourJumpRowsPerWorker
        );
        const std::size_t desired_worker_count = std::min(
            {
                kFourNeighbourJumpWorkerCap,
                hardware_workers,
                std::max<std::size_t>(1, useful_workers),
            }
        );
        NativeWorkerLease worker_lease(desired_worker_count - 1);
        const std::size_t worker_count = worker_lease.worker_count();
        const std::uint64_t rows_per_worker =
            (height + worker_count - 1) / worker_count;
        std::vector<double> worker_maxima(worker_count, 0.0);
        std::vector<std::exception_ptr> errors(worker_count);
        const auto worker = [&](std::size_t worker_index) {
            try {
                const std::uint64_t row_start = std::min<std::uint64_t>(
                    height,
                    worker_index * rows_per_worker
                );
                const std::uint64_t row_stop = std::min<std::uint64_t>(
                    height,
                    row_start + rows_per_worker
                );
                double local_maximum = 0.0;
                for (std::uint64_t row = row_start; row < row_stop; ++row) {
                    const std::uint64_t pixel_start = row * width;
                    for (std::uint64_t column = 0; column < width; ++column) {
                        const std::uint64_t pixel = pixel_start + column;
                        const double value = values[pixel];
                        if (!std::isfinite(value)) {
                            throw std::invalid_argument(
                                "displacement must contain only finite values"
                            );
                        }
                        if (column + 1 < width) {
                            local_maximum = std::max(
                                local_maximum,
                                std::abs(values[pixel + 1] - value)
                            );
                        }
                        if (row + 1 < height) {
                            local_maximum = std::max(
                                local_maximum,
                                std::abs(values[pixel + width] - value)
                            );
                        }
                    }
                }
                worker_maxima[worker_index] = local_maximum;
            } catch (...) {
                errors[worker_index] = std::current_exception();
            }
        };
        run_native_parallel(worker_count, worker);
        for (const auto& error : errors) {
            if (error) {
                std::rethrow_exception(error);
            }
        }
        for (double value : worker_maxima) {
            maximum = std::max(maximum, value);
        }
    }
    return maximum;
}

py::tuple build_background_proxy_state_into_native(
    const py::array& coverage,
    py::array repair_bits,
    py::array missing_mask
) {
    if (
        !coverage.dtype().is(py::dtype::of<std::uint8_t>()) ||
        coverage.ndim() != 2 ||
        (coverage.flags() & py::array::c_style) == 0
    ) {
        throw py::type_error("coverage must be a C-contiguous uint8 raster");
    }
    const auto validate_output = [&](const py::array& output, const char* name) {
        if (
            !output.dtype().is(py::dtype::of<std::uint8_t>()) ||
            output.ndim() != 2 ||
            (output.flags() & py::array::c_style) == 0 ||
            !output.writeable() ||
            output.shape(0) != coverage.shape(0) ||
            output.shape(1) != coverage.shape(1)
        ) {
            throw py::type_error(
                std::string(name) +
                " must be a writable matching C-contiguous uint8 raster"
            );
        }
    };
    validate_output(repair_bits, "repair_bits");
    validate_output(missing_mask, "missing_mask");
    const auto* coverage_values = static_cast<const std::uint8_t*>(coverage.data());
    auto* repair_values = static_cast<std::uint8_t*>(repair_bits.mutable_data());
    auto* missing_values = static_cast<std::uint8_t*>(missing_mask.mutable_data());
    const std::uint64_t count = static_cast<std::uint64_t>(coverage.size());
    std::uint64_t missing_lanes = 0;
    std::uint64_t missing_pixels = 0;
    {
        py::gil_scoped_release release;
        for (std::uint64_t index = 0; index < count; ++index) {
            const std::uint8_t value = coverage_values[index];
            if (value > 16U) {
                throw py::value_error("coverage exceeds the horizontal subpixel count");
            }
            const bool missing = value < 16U;
            missing_values[index] = static_cast<std::uint8_t>(missing);
            repair_values[index] = !missing
                ? std::uint8_t{0}
                : (value == 0 ? std::uint8_t{0x2a} : std::uint8_t{0x29});
            missing_lanes += static_cast<std::uint64_t>(16U - value);
            missing_pixels += static_cast<std::uint64_t>(missing);
        }
    }
    return py::make_tuple(missing_lanes, missing_pixels);
}

std::uint32_t low_region_find(
    std::vector<std::uint32_t>& parent,
    std::uint32_t pixel
) {
    std::uint32_t root = pixel;
    while (parent[root] != root) {
        root = parent[root];
    }
    while (parent[pixel] != pixel) {
        const std::uint32_t next = parent[pixel];
        parent[pixel] = root;
        pixel = next;
    }
    return root;
}

void low_region_union(
    std::vector<std::uint32_t>& parent,
    std::vector<std::uint32_t>& canonical,
    std::uint32_t left,
    std::uint32_t right
) {
    std::uint32_t left_root = low_region_find(parent, left);
    std::uint32_t right_root = low_region_find(parent, right);
    if (left_root == right_root) {
        return;
    }
    std::uint32_t left_key = canonical[left_root];
    std::uint32_t right_key = canonical[right_root];
    if (
        right_key < left_key ||
        (right_key == left_key && right_root < left_root)
    ) {
        std::swap(left_root, right_root);
        std::swap(left_key, right_key);
    }
    parent[right_root] = left_root;
    canonical[left_root] = std::min(left_key, right_key);
}

void low_region_union_minimum_root(
    std::vector<std::uint32_t>& parent,
    std::uint32_t left,
    std::uint32_t right
) {
    std::uint32_t left_root = low_region_find(parent, left);
    std::uint32_t right_root = low_region_find(parent, right);
    if (left_root == right_root) {
        return;
    }
    if (right_root < left_root) {
        std::swap(left_root, right_root);
    }
    parent[right_root] = left_root;
}

py::tuple build_low_resolution_regions_native(
    const py::array& displacement,
    const py::array& validity,
    const py::array& near_score
) {
    validate_array(displacement, py::dtype::of<double>(), "one_eye_displacement_px");
    validate_array(validity, py::dtype::of<bool>(), "metric_validity");
    validate_array(near_score, py::dtype::of<float>(), "near_score");
    if (
        validity.shape(0) != displacement.shape(0) ||
        validity.shape(1) != displacement.shape(1) ||
        near_score.shape(0) != displacement.shape(0) ||
        near_score.shape(1) != displacement.shape(1)
    ) {
        throw py::value_error("low-resolution region rasters must match");
    }
    const std::uint64_t sample_count64 =
        static_cast<std::uint64_t>(displacement.shape(0)) *
        static_cast<std::uint64_t>(displacement.shape(1));
    if (sample_count64 + 1 > kUint32Max) {
        throw py::value_error("low-resolution sample count exceeds uint32 contract");
    }
    const auto height = static_cast<std::uint32_t>(displacement.shape(0));
    const auto width = static_cast<std::uint32_t>(displacement.shape(1));
    const auto sample_count = static_cast<std::uint32_t>(sample_count64);
    const auto* displacement_values = static_cast<const double*>(displacement.data());
    const auto* validity_values = static_cast<const std::uint8_t*>(validity.data());
    const auto* score_values = static_cast<const float*>(near_score.data());
    std::vector<std::uint32_t> parent(sample_count);
    std::vector<std::uint32_t> canonical(sample_count);
    for (std::uint32_t pixel = 0; pixel < sample_count; ++pixel) {
        parent[pixel] = pixel;
        canonical[pixel] = pixel;
    }
    py::array_t<std::uint32_t> region_map({height, width});
    std::uint32_t region_count = 0;
    {
        py::gil_scoped_release release;
        for (std::uint32_t row = 0; row < height; ++row) {
            for (std::uint32_t column = 0; column < width; ++column) {
                const std::uint32_t pixel = row * width + column;
                const auto connected = [&](std::uint32_t neighbour) {
                    return
                        validity_values[pixel] == validity_values[neighbour] &&
                        std::abs(
                            displacement_values[pixel] - displacement_values[neighbour]
                        ) < 1.0;
                };
                if (row > 0 && connected(pixel - width)) {
                    low_region_union(parent, canonical, pixel, pixel - width);
                }
                if (column > 0 && connected(pixel - 1)) {
                    low_region_union(parent, canonical, pixel, pixel - 1);
                }
            }
        }
        for (std::uint32_t pixel = 0; pixel < sample_count; ++pixel) {
            parent[pixel] = low_region_find(parent, pixel);
            if (parent[pixel] == pixel) {
                ++region_count;
            }
        }
    }
    py::array_t<std::uint32_t> canonical_keys(region_count + 1);
    py::array_t<std::uint32_t> rank_bits(region_count + 1);
    auto* region_values = region_map.mutable_data();
    auto* key_values = canonical_keys.mutable_data();
    auto* rank_values = rank_bits.mutable_data();
    std::fill(key_values, key_values + region_count + 1, 0);
    std::fill(rank_values, rank_values + region_count + 1, 0);
    std::vector<std::uint32_t> region_for_root(sample_count, 0);
    {
        py::gil_scoped_release release;
        std::uint32_t next_region = 0;
        for (std::uint32_t pixel = 0; pixel < sample_count; ++pixel) {
            if (parent[pixel] != pixel) {
                continue;
            }
            ++next_region;
            region_for_root[pixel] = next_region;
            key_values[next_region] = canonical[pixel];
        }
        for (std::uint32_t pixel = 0; pixel < sample_count; ++pixel) {
            const std::uint32_t region = region_for_root[parent[pixel]];
            region_values[pixel] = region;
            std::uint32_t bits = 0;
            if (score_values[pixel] != 0.0F) {
                std::memcpy(&bits, score_values + pixel, sizeof(bits));
            }
            rank_values[region] = std::max(rank_values[region], bits);
        }
    }
    return py::make_tuple(
        std::move(region_map),
        std::move(canonical_keys),
        std::move(rank_bits)
    );
}

py::tuple build_stable_surface_support_native(
    const py::array& initial_regions,
    const py::array& stable_regions,
    const py::array& displacement,
    const py::array& near_score
) {
    validate_array(initial_regions, py::dtype::of<std::uint32_t>(), "initial_region_map");
    validate_vector(stable_regions, py::dtype::of<bool>(), "stable_regions");
    validate_array(displacement, py::dtype::of<double>(), "one_eye_displacement_px");
    validate_array(near_score, py::dtype::of<float>(), "near_score");
    if (
        displacement.shape(0) != initial_regions.shape(0) ||
        displacement.shape(1) != initial_regions.shape(1) ||
        near_score.shape(0) != initial_regions.shape(0) ||
        near_score.shape(1) != initial_regions.shape(1)
    ) {
        throw py::value_error("stable surface support rasters must match");
    }
    const std::uint64_t sample_count64 =
        static_cast<std::uint64_t>(initial_regions.shape(0)) *
        static_cast<std::uint64_t>(initial_regions.shape(1));
    if (sample_count64 + 1 > kUint32Max) {
        throw py::value_error("stable surface sample count exceeds uint32 contract");
    }
    const auto height = static_cast<std::uint32_t>(initial_regions.shape(0));
    const auto width = static_cast<std::uint32_t>(initial_regions.shape(1));
    const auto sample_count = static_cast<std::uint32_t>(sample_count64);
    const auto* initial_values =
        static_cast<const std::uint32_t*>(initial_regions.data());
    const auto* displacement_values = static_cast<const double*>(displacement.data());
    const auto* score_values = static_cast<const float*>(near_score.data());
    std::uint32_t initial_region_count = 0;
    {
        py::gil_scoped_release release;
        for (std::uint32_t pixel = 0; pixel < sample_count; ++pixel) {
            const std::uint32_t region = initial_values[pixel];
            if (region == 0 || region == kUint32Max) {
                throw std::invalid_argument(
                    "initial_region_map must contain positive finite region identifiers"
                );
            }
            if (
                !std::isfinite(displacement_values[pixel]) ||
                !std::isfinite(score_values[pixel])
            ) {
                throw std::invalid_argument(
                    "stable surface support geometry must contain only finite values"
                );
            }
            initial_region_count = std::max(initial_region_count, region);
        }
    }
    if (
        stable_regions.shape(0) !=
        static_cast<py::ssize_t>(initial_region_count) + 1
    ) {
        throw py::value_error("stable_regions must contain one entry for every region");
    }

    const auto* stable_region_values =
        static_cast<const std::uint8_t*>(stable_regions.data());
    py::array_t<bool> stable_samples({height, width});
    auto* stable_sample_values = stable_samples.mutable_data();
    std::vector<std::uint32_t> parent(sample_count);
    std::uint32_t surface_count = 0;
    {
        py::gil_scoped_release release;
        std::vector<std::uint64_t> eligible_counts(initial_region_count + 1, 0);
        std::vector<std::uint8_t> supported_regions(initial_region_count + 1, 0);
        for (std::uint32_t pixel = 0; pixel < sample_count; ++pixel) {
            stable_sample_values[pixel] =
                stable_region_values[initial_values[pixel]] != 0;
        }
        for (std::uint32_t row = 0; row < height; ++row) {
            const std::uint32_t row_start = row * width;
            for (std::uint32_t column = 0; column < width; ++column) {
                const std::uint32_t pixel = row_start + column;
                if (row > 0 &&
                    std::abs(
                        displacement_values[pixel] - displacement_values[pixel - width]
                    ) >= kStableSampleMaxNeighbourJumpPx
                ) {
                    stable_sample_values[pixel] = false;
                    stable_sample_values[pixel - width] = false;
                }
                if (column > 0 &&
                    std::abs(
                        displacement_values[pixel] - displacement_values[pixel - 1]
                    ) >= kStableSampleMaxNeighbourJumpPx
                ) {
                    stable_sample_values[pixel] = false;
                    stable_sample_values[pixel - 1] = false;
                }
            }
        }
        for (std::uint32_t pixel = 0; pixel < sample_count; ++pixel) {
            if (stable_sample_values[pixel]) {
                ++eligible_counts[initial_values[pixel]];
            }
        }
        bool has_supported_region = false;
        for (std::uint32_t region = 1; region <= initial_region_count; ++region) {
            supported_regions[region] = static_cast<std::uint8_t>(
                stable_region_values[region] != 0 && eligible_counts[region] != 0
            );
            has_supported_region =
                has_supported_region || supported_regions[region] != 0;
        }
        for (std::uint32_t pixel = 0; pixel < sample_count; ++pixel) {
            if (!has_supported_region) {
                stable_sample_values[pixel] =
                    stable_region_values[initial_values[pixel]] != 0;
            } else {
                stable_sample_values[pixel] =
                    stable_sample_values[pixel] &&
                    supported_regions[initial_values[pixel]] != 0;
            }
            parent[pixel] = pixel;
        }
        for (std::uint32_t row = 0; row < height; ++row) {
            const std::uint32_t row_start = row * width;
            for (std::uint32_t column = 0; column < width; ++column) {
                const std::uint32_t pixel = row_start + column;
                const auto connected = [&](std::uint32_t neighbour) {
                    return
                        initial_values[pixel] == initial_values[neighbour] &&
                        stable_sample_values[pixel] == stable_sample_values[neighbour];
                };
                if (
                    row > 0 &&
                    parent[pixel] != parent[pixel - width] &&
                    connected(pixel - width)
                ) {
                    low_region_union_minimum_root(parent, pixel, pixel - width);
                }
                if (
                    column > 0 &&
                    parent[pixel] != parent[pixel - 1] &&
                    connected(pixel - 1)
                ) {
                    low_region_union_minimum_root(parent, pixel, pixel - 1);
                }
            }
        }
        for (std::uint32_t pixel = 0; pixel < sample_count; ++pixel) {
            parent[pixel] = low_region_find(parent, pixel);
            if (parent[pixel] == pixel) {
                ++surface_count;
            }
        }
    }

    py::array_t<std::uint32_t> surface_region_map({height, width});
    py::array_t<std::uint32_t> canonical_keys(surface_count + 1);
    py::array_t<std::uint32_t> rank_bits(surface_count + 1);
    py::array_t<bool> stable_surfaces(surface_count + 1);
    py::array_t<bool> stable_sample_include({height, width});
    auto* surface_values = surface_region_map.mutable_data();
    auto* key_values = canonical_keys.mutable_data();
    auto* rank_values = rank_bits.mutable_data();
    auto* stable_surface_values = stable_surfaces.mutable_data();
    auto* include_values = stable_sample_include.mutable_data();
    std::fill(key_values, key_values + surface_count + 1, 0);
    std::fill(rank_values, rank_values + surface_count + 1, 0);
    std::fill(stable_surface_values, stable_surface_values + surface_count + 1, false);
    std::vector<std::uint32_t> region_for_root(sample_count, 0);
    std::vector<std::uint64_t> surface_eligible_counts(surface_count + 1, 0);
    std::vector<std::uint64_t> surface_interior_counts(surface_count + 1, 0);
    bool has_stable_surface = false;
    {
        py::gil_scoped_release release;
        std::uint32_t next_region = 0;
        for (std::uint32_t pixel = 0; pixel < sample_count; ++pixel) {
            if (parent[pixel] != pixel) {
                continue;
            }
            ++next_region;
            region_for_root[pixel] = next_region;
            key_values[next_region] = pixel;
        }
        for (std::uint32_t row = 0; row < height; ++row) {
            const std::uint32_t row_start = row * width;
            for (std::uint32_t column = 0; column < width; ++column) {
                const std::uint32_t pixel = row_start + column;
                const std::uint32_t region = region_for_root[parent[pixel]];
                surface_values[pixel] = region;
                std::uint32_t bits = 0;
                if (score_values[pixel] != 0.0F) {
                    std::memcpy(&bits, score_values + pixel, sizeof(bits));
                }
                rank_values[region] = std::max(rank_values[region], bits);
                if (stable_sample_values[pixel]) {
                    stable_surface_values[region] = true;
                    has_stable_surface = true;
                }
                bool included = stable_sample_values[pixel];
                if (included && row > 0) {
                    included = region_for_root[parent[pixel - width]] == region;
                }
                if (included && row + 1 < height) {
                    included = region_for_root[parent[pixel + width]] == region;
                }
                if (included && column > 0) {
                    included = region_for_root[parent[pixel - 1]] == region;
                }
                if (included && column + 1 < width) {
                    included = region_for_root[parent[pixel + 1]] == region;
                }
                include_values[pixel] = included;
                if (stable_sample_values[pixel]) {
                    ++surface_eligible_counts[region];
                }
                if (included) {
                    ++surface_interior_counts[region];
                }
            }
        }
        std::vector<std::uint32_t>().swap(parent);
        std::vector<std::uint32_t>().swap(region_for_root);
        std::vector<std::uint8_t> fallback_surfaces(surface_count + 1, 0);
        for (std::uint32_t region = 1; region <= surface_count; ++region) {
            fallback_surfaces[region] = static_cast<std::uint8_t>(
                surface_eligible_counts[region] != 0 &&
                surface_interior_counts[region] == 0
            );
        }
        for (std::uint32_t pixel = 0; pixel < sample_count; ++pixel) {
            include_values[pixel] =
                include_values[pixel] ||
                (
                    stable_sample_values[pixel] &&
                    fallback_surfaces[surface_values[pixel]] != 0
                );
        }
    }
    if (!has_stable_surface) {
        throw std::runtime_error("stable surface partition has no eligible component");
    }
    return py::make_tuple(
        std::move(surface_region_map),
        std::move(canonical_keys),
        std::move(rank_bits),
        std::move(stable_samples),
        std::move(stable_surfaces),
        std::move(stable_sample_include)
    );
}

py::array select_stable_region_support_native(
    const py::array& region_map,
    std::uint32_t region_count,
    std::uint32_t area_threshold
) {
    validate_array(region_map, py::dtype::of<std::uint32_t>(), "region_map");
    if (region_count == 0 || area_threshold == 0) {
        throw py::value_error("region_count and area_threshold must be positive");
    }
    const auto* regions = static_cast<const std::uint32_t*>(region_map.data());
    const auto sample_count = static_cast<std::uint64_t>(region_map.size());
    if (static_cast<std::uint64_t>(region_count) > sample_count) {
        throw py::value_error("region_count cannot exceed the sample count");
    }
    std::vector<std::uint64_t> counts(region_count + 1, 0);
    py::array_t<bool> output(region_count + 1);
    auto* stable = output.mutable_data();
    {
        py::gil_scoped_release release;
        for (std::uint64_t pixel = 0; pixel < sample_count; ++pixel) {
            const std::uint32_t region = regions[pixel];
            if (region == 0 || region > region_count) {
                throw std::invalid_argument(
                    "region_map exceeds the declared region_count"
                );
            }
            ++counts[region];
        }
        stable[0] = false;
        bool any_stable = false;
        for (std::uint32_t region = 1; region <= region_count; ++region) {
            stable[region] = counts[region] >= area_threshold;
            any_stable = any_stable || stable[region];
        }
        if (!any_stable) {
            for (std::uint32_t region = 1; region <= region_count; ++region) {
                stable[region] = counts[region] != 0;
            }
        }
    }
    return output;
}

py::array nearest_label_upsample_native(
    const py::array& labels,
    std::uint32_t destination_height,
    std::uint32_t destination_width
) {
    validate_array(labels, py::dtype::of<std::uint32_t>(), "low_resolution_regions");
    if (destination_height == 0 || destination_width == 0) {
        throw py::value_error("render dimensions must be positive");
    }
    const auto source_height = static_cast<std::uint32_t>(labels.shape(0));
    const auto source_width = static_cast<std::uint32_t>(labels.shape(1));
    const auto* input = static_cast<const std::uint32_t*>(labels.data());
    py::array_t<std::uint32_t> output({destination_height, destination_width});
    auto* result = output.mutable_data();
    std::vector<std::uint32_t> source_rows(destination_height);
    std::vector<std::uint32_t> source_columns(destination_width);
    const auto map_coordinate = [](
        std::uint32_t destination_index,
        std::uint32_t source_size,
        std::uint32_t destination_size
    ) {
        const double source_coordinate =
            ((static_cast<double>(destination_index) + 0.5) * source_size) /
            destination_size - 0.5;
        const auto rounded = static_cast<std::int64_t>(
            std::floor(source_coordinate + 0.5)
        );
        return static_cast<std::uint32_t>(std::min(
            std::max<std::int64_t>(rounded, 0),
            static_cast<std::int64_t>(source_size - 1)
        ));
    };
    {
        py::gil_scoped_release release;
        for (std::uint32_t row = 0; row < destination_height; ++row) {
            source_rows[row] = map_coordinate(row, source_height, destination_height);
        }
        for (std::uint32_t column = 0; column < destination_width; ++column) {
            source_columns[column] = map_coordinate(
                column,
                source_width,
                destination_width
            );
        }
        const std::size_t hardware_workers = std::max(
            1U,
            std::thread::hardware_concurrency()
        );
        const std::size_t useful_workers = std::max<std::size_t>(
            1,
            (destination_height + kNearestLabelUpsampleRowsPerWorker - 1) /
                kNearestLabelUpsampleRowsPerWorker
        );
        const std::size_t desired_worker_count = std::min(
            {kNearestLabelUpsampleWorkerCap, hardware_workers, useful_workers}
        );
        NativeWorkerLease worker_lease(
            desired_worker_count > 1 ? desired_worker_count - 1 : 0
        );
        const std::size_t worker_count = worker_lease.worker_count();
        const auto run_worker = [&](std::size_t worker) {
            const auto start = static_cast<std::uint32_t>(
                static_cast<std::uint64_t>(destination_height) * worker /
                worker_count
            );
            const auto stop = static_cast<std::uint32_t>(
                static_cast<std::uint64_t>(destination_height) * (worker + 1) /
                worker_count
            );
            for (std::uint32_t row = start; row < stop; ++row) {
                const auto* source = input +
                    static_cast<std::uint64_t>(source_rows[row]) * source_width;
                auto* destination = result +
                    static_cast<std::uint64_t>(row) * destination_width;
                for (std::uint32_t column = 0; column < destination_width; ++column) {
                    destination[column] = source[source_columns[column]];
                }
            }
        };
        run_native_parallel(worker_count, run_worker);
    }
    return output;
}

template <typename DistT>
void build_edge_band_l1_impl(
    const std::uint32_t* regions,
    bool* result,
    std::uint32_t height,
    std::uint32_t width,
    std::uint32_t effective_radius,
    std::size_t hardware_workers,
    bool include_zero
) {
    const auto is_region = [](std::uint32_t value) {
        return value != 0 && value != std::numeric_limits<std::uint32_t>::max();
    };
    const DistT inf_dist = static_cast<DistT>(effective_radius + 1);
    const std::uint64_t sample_count = static_cast<std::uint64_t>(height) * width;
    std::vector<DistT> dist(sample_count);

    // Pass 1: parallel per-row seed detection + 1D horizontal L1 distance sweep.
    const std::size_t useful_row_workers = std::max<std::size_t>(
        1,
        (height + kEdgeBandRowsPerWorker - 1) / kEdgeBandRowsPerWorker
    );
    const std::size_t desired_row_workers = std::min(
        {kEdgeBandWorkerCap, hardware_workers, useful_row_workers}
    );
    {
        NativeWorkerLease row_lease(
            desired_row_workers > 1 ? desired_row_workers - 1 : 0
        );
        const std::size_t worker_count = row_lease.worker_count();
        run_native_parallel(worker_count, [&](std::size_t worker) {
            const auto start_row = static_cast<std::uint32_t>(
                static_cast<std::uint64_t>(height) * worker / worker_count
            );
            const auto stop_row = static_cast<std::uint32_t>(
                static_cast<std::uint64_t>(height) * (worker + 1) / worker_count
            );
            for (std::uint32_t row = start_row; row < stop_row; ++row) {
                const std::uint64_t row_offset = static_cast<std::uint64_t>(row) * width;
                DistT left_dist = inf_dist;
                for (std::uint32_t col = 0; col < width; ++col) {
                    const std::uint64_t pixel = row_offset + col;
                    const std::uint32_t region = regions[pixel];
                    const bool is_seed = (include_zero && region == 0) || (
                        is_region(region) && (
                            (row != 0 && is_region(regions[pixel - width]) &&
                             region != regions[pixel - width]) ||
                            (row + 1 < height && is_region(regions[pixel + width]) &&
                             region != regions[pixel + width]) ||
                            (col != 0 && is_region(regions[pixel - 1]) &&
                             region != regions[pixel - 1]) ||
                            (col + 1 < width && is_region(regions[pixel + 1]) &&
                             region != regions[pixel + 1])
                        )
                    );
                    if (is_seed) {
                        left_dist = 0;
                    } else if (left_dist < inf_dist) {
                        ++left_dist;
                    }
                    dist[pixel] = left_dist;
                }
                if (width > 1) {
                    DistT right_dist = dist[row_offset + width - 1];
                    for (std::uint32_t col = width - 1; col > 0; --col) {
                        const std::uint64_t pixel = row_offset + (col - 1);
                        if (right_dist < inf_dist) {
                            ++right_dist;
                        }
                        if (right_dist < dist[pixel]) {
                            dist[pixel] = right_dist;
                        } else {
                            right_dist = dist[pixel];
                        }
                    }
                }
            }
        });
    }

    // Pass 2: parallel column-strip vertical L1 distance sweep (linear inner loops).
    constexpr std::uint32_t kColsPerWorker = 128;
    const std::size_t useful_col_workers = std::max<std::size_t>(
        1,
        (width + kColsPerWorker - 1) / kColsPerWorker
    );
    const std::size_t desired_col_workers = std::min(
        {kEdgeBandWorkerCap, hardware_workers, useful_col_workers}
    );
    {
        NativeWorkerLease col_lease(
            desired_col_workers > 1 ? desired_col_workers - 1 : 0
        );
        const std::size_t worker_count = col_lease.worker_count();
        run_native_parallel(worker_count, [&](std::size_t worker) {
            const auto col_start = static_cast<std::uint32_t>(
                static_cast<std::uint64_t>(width) * worker / worker_count
            );
            const auto col_stop = static_cast<std::uint32_t>(
                static_cast<std::uint64_t>(width) * (worker + 1) / worker_count
            );
            for (std::uint32_t row = 1; row < height; ++row) {
                const DistT* prev_row =
                    dist.data() + static_cast<std::uint64_t>(row - 1) * width;
                DistT* curr_row =
                    dist.data() + static_cast<std::uint64_t>(row) * width;
                for (std::uint32_t col = col_start; col < col_stop; ++col) {
                    const DistT cand = static_cast<DistT>(prev_row[col] + 1);
                    if (cand < curr_row[col]) {
                        curr_row[col] = cand;
                    }
                }
            }
            {
                const std::uint64_t last_offset =
                    static_cast<std::uint64_t>(height - 1) * width;
                const DistT* last_row = dist.data() + last_offset;
                bool* out_row = result + last_offset;
                for (std::uint32_t col = col_start; col < col_stop; ++col) {
                    out_row[col] = (last_row[col] <= effective_radius);
                }
            }
            for (std::uint32_t row = height - 1; row > 0; --row) {
                const std::uint64_t next_offset =
                    static_cast<std::uint64_t>(row) * width;
                const std::uint64_t curr_offset =
                    static_cast<std::uint64_t>(row - 1) * width;
                const DistT* next_row = dist.data() + next_offset;
                DistT* curr_row = dist.data() + curr_offset;
                bool* out_row = result + curr_offset;
                for (std::uint32_t col = col_start; col < col_stop; ++col) {
                    DistT value = curr_row[col];
                    const DistT cand = static_cast<DistT>(next_row[col] + 1);
                    if (cand < value) {
                        value = cand;
                        curr_row[col] = value;
                    }
                    out_row[col] = (value <= effective_radius);
                }
            }
        });
    }
}

py::array build_four_connected_edge_band_native(
    const py::array& labels,
    std::uint32_t radius,
    bool include_zero = false
) {
    validate_array(labels, py::dtype::of<std::uint32_t>(), "mapped_regions");
    const auto height = static_cast<std::uint32_t>(labels.shape(0));
    const auto width = static_cast<std::uint32_t>(labels.shape(1));
    const auto* regions = static_cast<const std::uint32_t*>(labels.data());
    py::array_t<bool> output({height, width});
    if (height == 0 || width == 0) {
        return output;
    }
    auto* result = output.mutable_data();
    const std::size_t hardware_workers = std::max(
        1U,
        std::thread::hardware_concurrency()
    );
    const auto is_region = [](std::uint32_t value) {
        return value != 0 && value != std::numeric_limits<std::uint32_t>::max();
    };
    {
        py::gil_scoped_release release;
        if (radius == 0) {
            const std::size_t useful_workers = std::max<std::size_t>(
                1,
                (height + kEdgeBandRowsPerWorker - 1) / kEdgeBandRowsPerWorker
            );
            const std::size_t desired_workers = std::min(
                {kEdgeBandWorkerCap, hardware_workers, useful_workers}
            );
            NativeWorkerLease lease(desired_workers > 1 ? desired_workers - 1 : 0);
            const std::size_t worker_count = lease.worker_count();
            run_native_parallel(worker_count, [&](std::size_t worker) {
                const auto start_row = static_cast<std::uint32_t>(
                    static_cast<std::uint64_t>(height) * worker / worker_count
                );
                const auto stop_row = static_cast<std::uint32_t>(
                    static_cast<std::uint64_t>(height) * (worker + 1) / worker_count
                );
                for (std::uint32_t row = start_row; row < stop_row; ++row) {
                    const std::uint64_t row_offset = static_cast<std::uint64_t>(row) * width;
                    for (std::uint32_t col = 0; col < width; ++col) {
                        const std::uint64_t pixel = row_offset + col;
                        const std::uint32_t region = regions[pixel];
                        result[pixel] = (include_zero && region == 0) || (
                            is_region(region) && (
                                (row != 0 && is_region(regions[pixel - width]) &&
                                 region != regions[pixel - width]) ||
                                (row + 1 < height && is_region(regions[pixel + width]) &&
                                 region != regions[pixel + width]) ||
                                (col != 0 && is_region(regions[pixel - 1]) &&
                                 region != regions[pixel - 1]) ||
                                (col + 1 < width && is_region(regions[pixel + 1]) &&
                                 region != regions[pixel + 1])
                            )
                        );
                    }
                }
            });
        } else {
            const auto effective_radius = static_cast<std::uint32_t>(
                std::min<std::uint64_t>(
                    radius,
                    static_cast<std::uint64_t>(height) + width
                )
            );
            if (effective_radius < 65534U) {
                build_edge_band_l1_impl<std::uint16_t>(
                    regions, result, height, width, effective_radius, hardware_workers, include_zero
                );
            } else {
                build_edge_band_l1_impl<std::uint32_t>(
                    regions, result, height, width, effective_radius, hardware_workers, include_zero
                );
            }
        }
    }
    return output;
}

py::tuple label_region_fragments_native(
    const py::array& region_map,
    const py::array& band_mask
) {
    validate_array(region_map, py::dtype::of<std::uint32_t>(), "region_map");
    validate_array(band_mask, py::dtype::of<bool>(), "band_mask");
    if (
        region_map.shape(0) != band_mask.shape(0) ||
        region_map.shape(1) != band_mask.shape(1)
    ) {
        throw py::value_error("region_map and band_mask must match");
    }
    const std::uint64_t sample_count64 =
        static_cast<std::uint64_t>(region_map.shape(0)) *
        static_cast<std::uint64_t>(region_map.shape(1));
    if (sample_count64 + 1 > kUint32Max) {
        throw py::value_error("region fragment raster exceeds uint32 capacity");
    }
    const auto height = static_cast<std::uint32_t>(region_map.shape(0));
    const auto width = static_cast<std::uint32_t>(region_map.shape(1));
    const auto sample_count = static_cast<std::uint32_t>(sample_count64);
    const auto* regions = static_cast<const std::uint32_t*>(region_map.data());
    const auto* band = static_cast<const std::uint8_t*>(band_mask.data());
    std::uint32_t band_count = 0;
    for (std::uint32_t pixel = 0; pixel < sample_count; ++pixel) {
        band_count += band[pixel] != 0;
    }
    py::array_t<std::uint32_t> fragment_map({height, width});
    py::array_t<std::uint32_t> fragment_regions(
        static_cast<py::ssize_t>(band_count) + 1
    );
    py::array_t<std::uint32_t> fragment_bounds(
        {
            static_cast<py::ssize_t>(band_count) + 1,
            static_cast<py::ssize_t>(4),
        }
    );
    auto* labels = fragment_map.mutable_data();
    auto* output_regions = fragment_regions.mutable_data();
    auto* bounds = fragment_bounds.mutable_data();
    std::fill(labels, labels + sample_count, 0);
    std::fill(output_regions, output_regions + band_count + 1, 0);
    std::fill(bounds, bounds + static_cast<std::uint64_t>(band_count + 1) * 4, 0);
    std::vector<std::uint32_t> queue(band_count);
    std::uint32_t fragment_count = 0;
    {
        py::gil_scoped_release release;
        const auto is_region = [](std::uint32_t value) {
            return value != 0 && value != kUint32Max;
        };
        for (std::uint32_t start = 0; start < sample_count; ++start) {
            const std::uint32_t region = regions[start];
            if (band[start] == 0 || labels[start] != 0 || !is_region(region)) {
                continue;
            }
            ++fragment_count;
            std::uint32_t head = 0;
            std::uint32_t tail = 1;
            queue[0] = start;
            labels[start] = fragment_count;
            output_regions[fragment_count] = region;
            const std::uint32_t start_row = start / width;
            const std::uint32_t start_column = start % width;
            std::uint32_t* fragment_bound = bounds + fragment_count * 4;
            fragment_bound[0] = start_row;
            fragment_bound[1] = start_row;
            fragment_bound[2] = start_column;
            fragment_bound[3] = start_column;
            while (head < tail) {
                const std::uint32_t pixel = queue[head++];
                const std::uint32_t row = pixel / width;
                const std::uint32_t column = pixel % width;
                fragment_bound[0] = std::min(fragment_bound[0], row);
                fragment_bound[1] = std::max(fragment_bound[1], row);
                fragment_bound[2] = std::min(fragment_bound[2], column);
                fragment_bound[3] = std::max(fragment_bound[3], column);
                const auto enqueue = [&](std::uint32_t neighbour) {
                    if (
                        band[neighbour] == 0 ||
                        labels[neighbour] != 0 ||
                        regions[neighbour] != region
                    ) {
                        return;
                    }
                    labels[neighbour] = fragment_count;
                    queue[tail++] = neighbour;
                };
                if (row != 0) {
                    enqueue(pixel - width);
                }
                if (column != 0) {
                    enqueue(pixel - 1);
                }
                if (column + 1 < width) {
                    enqueue(pixel + 1);
                }
                if (row + 1 < height) {
                    enqueue(pixel + width);
                }
            }
        }
    }
    return py::make_tuple(
        std::move(fragment_map),
        std::move(fragment_regions),
        std::move(fragment_bounds),
        fragment_count
    );
}

bool cross_erode_fragment_pixels(
    const std::vector<std::uint32_t>& pixels,
    std::uint32_t pixel_count,
    const std::vector<std::uint8_t>& input,
    std::vector<std::uint8_t>& output,
    std::uint32_t height,
    std::uint32_t width
) {
    bool any = false;
    for (std::uint32_t index = 0; index < pixel_count; ++index) {
        const std::uint32_t pixel = pixels[index];
        const std::uint32_t row = pixel / width;
        const std::uint32_t column = pixel % width;
        const std::uint8_t selected =
            row != 0 &&
            row + 1 < height &&
            column != 0 &&
            column + 1 < width &&
            input[pixel] != 0 &&
            input[pixel - width] != 0 &&
            input[pixel + width] != 0 &&
            input[pixel - 1] != 0 &&
            input[pixel + 1] != 0;
        output[pixel] = selected;
        any = any || selected != 0;
    }
    return any;
}

bool cross_dilate_contains_pixel(
    const std::vector<std::uint8_t>& input,
    std::uint32_t pixel,
    std::uint32_t height,
    std::uint32_t width
) {
    const std::uint32_t row = pixel / width;
    const std::uint32_t column = pixel % width;
    return input[pixel] != 0 ||
        (row != 0 && input[pixel - width] != 0) ||
        (row + 1 < height && input[pixel + width] != 0) ||
        (column != 0 && input[pixel - 1] != 0) ||
        (column + 1 < width && input[pixel + 1] != 0);
}

py::array build_sparse_region_seeds_impl(
    const py::array& region_map,
    const py::array& band_mask,
    std::uint32_t radius,
    const std::uint8_t* stable_initial_values,
    const std::uint8_t* stable_surface_values,
    std::uint64_t stable_surface_count
) {
    validate_array(region_map, py::dtype::of<std::uint32_t>(), "mapped_regions");
    validate_array(band_mask, py::dtype::of<bool>(), "edge_band");
    if (
        region_map.shape(0) != band_mask.shape(0) ||
        region_map.shape(1) != band_mask.shape(1)
    ) {
        throw py::value_error("mapped_regions and edge_band must match");
    }
    const std::uint64_t sample_count64 =
        static_cast<std::uint64_t>(region_map.shape(0)) *
        static_cast<std::uint64_t>(region_map.shape(1));
    if (sample_count64 + 1 > kUint32Max) {
        throw py::value_error("sparse seed raster exceeds uint32 capacity");
    }
    const auto height = static_cast<std::uint32_t>(region_map.shape(0));
    const auto width = static_cast<std::uint32_t>(region_map.shape(1));
    const auto sample_count = static_cast<std::uint32_t>(sample_count64);
    const auto* regions = static_cast<const std::uint32_t*>(region_map.data());
    const auto* band = static_cast<const std::uint8_t*>(band_mask.data());
    py::array_t<std::uint32_t> output({height, width});
    auto* seeds = output.mutable_data();
    const bool filter_stable = stable_surface_values != nullptr;
    {
        py::gil_scoped_release release;
        std::vector<std::uint8_t> visited(sample_count, 0);
        std::uint32_t band_count = 0;
        for (std::uint32_t pixel = 0; pixel < sample_count; ++pixel) {
            const std::uint32_t reg = regions[pixel];
            if (band[pixel] != 0) {
                ++band_count;
                seeds[pixel] = 0;
            } else if (filter_stable) {
                if (static_cast<std::uint64_t>(reg) >= stable_surface_count) {
                    throw std::invalid_argument(
                        "sparse seed region exceeds stable_surfaces"
                    );
                }
                seeds[pixel] =
                    (stable_initial_values[pixel] != 0 && stable_surface_values[reg] != 0)
                        ? reg
                        : 0U;
            } else {
                seeds[pixel] = reg;
            }
        }
        std::vector<std::uint32_t> queue(band_count);
        std::vector<std::uint8_t> current(sample_count, 0);
        std::vector<std::uint8_t> eroded(sample_count, 0);
        const auto is_region = [](std::uint32_t value) {
            return value != 0 && value != kUint32Max;
        };
        for (std::uint32_t start = 0; start < sample_count; ++start) {
            const std::uint32_t region = regions[start];
            if (band[start] == 0 || visited[start] != 0 || !is_region(region)) {
                continue;
            }
            if (filter_stable) {
                if (static_cast<std::uint64_t>(region) >= stable_surface_count) {
                    throw std::invalid_argument(
                        "sparse seed region exceeds stable_surfaces"
                    );
                }
                if (stable_surface_values[region] == 0) {
                    visited[start] = 1;
                    continue;
                }
            }
            std::uint32_t head = 0;
            std::uint32_t tail = 1;
            queue[0] = start;
            visited[start] = 1;
            while (head < tail) {
                const std::uint32_t pixel = queue[head++];
                const std::uint32_t row = pixel / width;
                const std::uint32_t column = pixel - row * width;
                const auto enqueue = [&](std::uint32_t neighbour) {
                    if (
                        band[neighbour] == 0 ||
                        visited[neighbour] != 0 ||
                        regions[neighbour] != region
                    ) {
                        return;
                    }
                    visited[neighbour] = 1;
                    queue[tail++] = neighbour;
                };
                if (row != 0) {
                    enqueue(pixel - width);
                }
                if (column != 0) {
                    enqueue(pixel - 1);
                }
                if (column + 1 < width) {
                    enqueue(pixel + 1);
                }
                if (row + 1 < height) {
                    enqueue(pixel + width);
                }
            }

            for (std::uint32_t index = 0; index < tail; ++index) {
                current[queue[index]] = 1;
            }

            bool current_any = true;
            for (std::uint32_t iteration = 0; iteration < radius && current_any; ++iteration) {
                current_any = cross_erode_fragment_pixels(
                    queue,
                    tail,
                    current,
                    eroded,
                    height,
                    width
                );
                current.swap(eroded);
            }

            bool selected_any = false;
            if (current_any) {
                for (std::uint32_t index = 0; index < tail; ++index) {
                    const std::uint32_t pixel = queue[index];
                    if (current[pixel] == 0) {
                        continue;
                    }
                    seeds[pixel] = (!filter_stable || stable_initial_values[pixel] != 0)
                        ? region
                        : 0U;
                    selected_any = true;
                }
            } else {
                for (std::uint32_t index = 0; index < tail; ++index) {
                    const std::uint32_t pixel = queue[index];
                    current[pixel] = 1;
                    eroded[pixel] = 0;
                }
                current_any = true;
                while (current_any) {
                    const bool eroded_any = cross_erode_fragment_pixels(
                        queue,
                        tail,
                        current,
                        eroded,
                        height,
                        width
                    );
                    for (std::uint32_t index = 0; index < tail; ++index) {
                        const std::uint32_t pixel = queue[index];
                        if (
                            current[pixel] != 0 &&
                            !cross_dilate_contains_pixel(
                                eroded,
                                pixel,
                                height,
                                width
                            )
                        ) {
                            seeds[pixel] = (!filter_stable || stable_initial_values[pixel] != 0)
                                ? region
                                : 0U;
                            selected_any = true;
                        }
                    }
                    current.swap(eroded);
                    current_any = eroded_any;
                }
            }

            if (!selected_any) {
                seeds[start] = (!filter_stable || stable_initial_values[start] != 0)
                    ? region
                    : 0U;
            }
            for (std::uint32_t index = 0; index < tail; ++index) {
                const std::uint32_t pixel = queue[index];
                current[pixel] = 0;
                eroded[pixel] = 0;
            }
        }
    }
    return output;
}

py::array build_sparse_region_seeds_native(
    const py::array& region_map,
    const py::array& band_mask,
    std::uint32_t radius
) {
    return build_sparse_region_seeds_impl(region_map, band_mask, radius, nullptr, nullptr, 0);
}

py::array build_sparse_stable_region_seeds_native(
    const py::array& region_map,
    const py::array& band_mask,
    const py::array& stable_initial,
    const py::array& stable_surfaces,
    std::uint32_t radius
) {
    validate_array(stable_initial, py::dtype::of<bool>(), "stable_initial");
    validate_vector(stable_surfaces, py::dtype::of<bool>(), "stable_surfaces");
    if (
        stable_initial.shape(0) != region_map.shape(0) ||
        stable_initial.shape(1) != region_map.shape(1)
    ) {
        throw py::value_error("stable_initial must match mapped_regions");
    }
    if (stable_surfaces.shape(0) == 0) {
        throw py::value_error("stable_surfaces must contain the zero region");
    }
    return build_sparse_region_seeds_impl(
        region_map,
        band_mask,
        radius,
        static_cast<const std::uint8_t*>(stable_initial.data()),
        static_cast<const std::uint8_t*>(stable_surfaces.data()),
        static_cast<std::uint64_t>(stable_surfaces.shape(0))
    );
}

std::pair<std::uint32_t, std::uint32_t> validate_background_proxy_layout(
    const py::array& pre_repair,
    const py::array& coverage,
    const py::array* proxy = nullptr
) {
    if (
        !pre_repair.dtype().is(py::dtype::of<std::uint8_t>()) ||
        pre_repair.ndim() != 3 ||
        pre_repair.shape(0) <= 0 ||
        pre_repair.shape(1) <= 0 ||
        pre_repair.shape(2) != 3 ||
        (pre_repair.flags() & py::array::c_style) == 0
    ) {
        throw py::type_error("pre_repair must be a C-contiguous uint8 BGR raster");
    }
    validate_array(coverage, py::dtype::of<std::uint8_t>(), "coverage");
    if (
        coverage.shape(0) != pre_repair.shape(0) ||
        coverage.shape(1) != pre_repair.shape(1)
    ) {
        throw py::value_error("coverage must match pre_repair");
    }
    if (proxy != nullptr) {
        if (
            !proxy->dtype().is(py::dtype::of<std::uint8_t>()) ||
            proxy->ndim() != 3 ||
            proxy->shape(0) != pre_repair.shape(0) ||
            proxy->shape(1) != pre_repair.shape(1) ||
            proxy->shape(2) != 3 ||
            (proxy->flags() & py::array::c_style) == 0
        ) {
            throw py::type_error("proxy must match the C-contiguous uint8 BGR raster");
        }
    }
    return {
        static_cast<std::uint32_t>(pre_repair.shape(0)),
        static_cast<std::uint32_t>(pre_repair.shape(1)),
    };
}

std::pair<std::uint32_t, std::uint32_t> validate_background_proxy_rasters(
    const py::array& pre_repair,
    const py::array& coverage,
    const py::array* proxy = nullptr
) {
    const auto [height, width] = validate_background_proxy_layout(
        pre_repair,
        coverage,
        proxy
    );
    const std::uint64_t sample_count = static_cast<std::uint64_t>(height) * width;
    const auto* coverage_values = static_cast<const std::uint8_t*>(coverage.data());
    for (std::uint64_t pixel = 0; pixel < sample_count; ++pixel) {
        if (coverage_values[pixel] > 16) {
            throw py::value_error("coverage exceeds the horizontal subpixel count");
        }
    }
    return {height, width};
}

std::uint8_t round_ties_even_clamped_u8(
    std::uint32_t numerator,
    std::uint32_t denominator
) {
    std::uint32_t quotient = numerator / denominator;
    const std::uint32_t remainder = numerator % denominator;
    const std::uint32_t twice_remainder = remainder * 2;
    if (
        twice_remainder > denominator ||
        (twice_remainder == denominator && (quotient & 1U) != 0)
    ) {
        ++quotient;
    }
    return static_cast<std::uint8_t>(std::min<std::uint32_t>(quotient, 255));
}

const std::array<std::uint8_t, 17 * 256>& background_proxy_normalization_lut() {
    static const auto values = [] {
        std::array<std::uint8_t, 17 * 256> result{};
        for (std::uint32_t coverage = 1; coverage <= 16; ++coverage) {
            for (std::uint32_t value = 0; value <= 255; ++value) {
                result[coverage * 256 + value] = round_ties_even_clamped_u8(
                    value * 16,
                    coverage
                );
            }
        }
        return result;
    }();
    return values;
}

py::tuple prepare_background_proxy_native(
    const py::array& pre_repair,
    const py::array& coverage,
    int far_side_direction,
    std::uint32_t foreground_exclusion_px
) {
    const auto [height, width] = validate_background_proxy_layout(
        pre_repair,
        coverage
    );
    if (far_side_direction != -1 && far_side_direction != 1) {
        throw py::value_error("far_side_direction must be -1 or 1");
    }
    if (foreground_exclusion_px < 1 || foreground_exclusion_px > 32) {
        throw py::value_error("foreground_exclusion_px must be within 1..32");
    }

    py::array_t<std::uint8_t> proxy_source({
        pre_repair.shape(0),
        pre_repair.shape(1),
        pre_repair.shape(2),
    });
    py::array_t<std::uint8_t> mask({
        pre_repair.shape(0),
        pre_repair.shape(1),
    });
    py::array_t<std::uint8_t> repair_bits({
        pre_repair.shape(0),
        pre_repair.shape(1),
    });
    py::array_t<std::uint8_t> missing({
        pre_repair.shape(0),
        pre_repair.shape(1),
    });
    const auto* source = static_cast<const std::uint8_t*>(pre_repair.data());
    const auto* coverage_values = static_cast<const std::uint8_t*>(coverage.data());
    auto* proxy_values = proxy_source.mutable_data();
    auto* mask_values = mask.mutable_data();
    auto* repair_values = repair_bits.mutable_data();
    auto* missing_values = missing.mutable_data();
    const std::uint64_t sample_count = static_cast<std::uint64_t>(height) * width;
    std::uint64_t missing_lanes = 0;
    std::uint64_t missing_pixels = 0;
    std::uint64_t masked_pixels = 0;
    std::uint32_t mask_start_row = height;
    std::uint32_t mask_end_row = 0;
    std::uint32_t mask_start_column = width;
    std::uint32_t mask_end_column = 0;
    const auto& normalization = background_proxy_normalization_lut();
    {
        py::gil_scoped_release release;
        for (std::uint64_t pixel = 0; pixel < sample_count; ++pixel) {
            const std::uint32_t lane_count = coverage_values[pixel];
            if (lane_count > 16) {
                throw std::invalid_argument(
                    "coverage exceeds the horizontal subpixel count"
                );
            }
            const bool pixel_missing = lane_count < 16;
            missing_values[pixel] = static_cast<std::uint8_t>(pixel_missing);
            repair_values[pixel] = !pixel_missing
                ? std::uint8_t{0}
                : (lane_count == 0 ? std::uint8_t{0x2a} : std::uint8_t{0x29});
            missing_lanes += 16U - lane_count;
            missing_pixels += static_cast<std::uint64_t>(pixel_missing);
            for (std::uint32_t channel = 0; channel < 3; ++channel) {
                const std::uint64_t sample = pixel * 3 + channel;
                proxy_values[sample] = normalization[
                    lane_count * 256 + source[sample]
                ];
            }
        }

        const std::uint32_t left_extent = far_side_direction > 0
            ? 2U
            : foreground_exclusion_px;
        const std::uint32_t right_extent = far_side_direction > 0
            ? foreground_exclusion_px
            : 2U;
        std::vector<std::uint8_t> previous(width, 0);
        std::vector<std::uint8_t> current(width, 0);
        std::vector<std::uint8_t> next(width, 0);
        const auto fill_horizontal = [&](std::uint32_t row, auto& output) {
            const std::uint64_t row_start = static_cast<std::uint64_t>(row) * width;
            std::uint32_t start = 0;
            std::uint32_t end = std::min<std::uint32_t>(width - 1, right_extent);
            std::uint32_t low_coverage_count = 0;
            for (std::uint32_t column = start; column <= end; ++column) {
                low_coverage_count += coverage_values[row_start + column] <= 8U;
            }
            for (std::uint32_t column = 0; column < width; ++column) {
                output[column] = static_cast<std::uint8_t>(low_coverage_count != 0);
                if (column + 1 == width) {
                    continue;
                }
                const std::uint32_t next_start = column + 1 > left_extent
                    ? column + 1 - left_extent
                    : 0U;
                const std::uint32_t next_end = std::min<std::uint32_t>(
                    width - 1,
                    column + 1 + right_extent
                );
                if (next_start > start) {
                    low_coverage_count -= coverage_values[row_start + start] <= 8U;
                    start = next_start;
                }
                if (next_end > end) {
                    end = next_end;
                    low_coverage_count += coverage_values[row_start + end] <= 8U;
                }
            }
        };
        fill_horizontal(0, current);
        if (height > 1) {
            fill_horizontal(1, next);
        }
        for (std::uint32_t row = 0; row < height; ++row) {
            const std::uint64_t row_start = static_cast<std::uint64_t>(row) * width;
            for (std::uint32_t column = 0; column < width; ++column) {
                const bool excluded =
                    previous[column] != 0 ||
                    current[column] != 0 ||
                    next[column] != 0;
                mask_values[row_start + column] = excluded ? 255U : 0U;
                masked_pixels += static_cast<std::uint64_t>(excluded);
                if (excluded) {
                    mask_start_row = std::min(mask_start_row, row);
                    mask_end_row = std::max(mask_end_row, row + 1);
                    mask_start_column = std::min(mask_start_column, column);
                    mask_end_column = std::max(mask_end_column, column + 1);
                }
            }
            previous.swap(current);
            current.swap(next);
            if (row + 2 < height) {
                fill_horizontal(row + 2, next);
            } else {
                std::fill(next.begin(), next.end(), std::uint8_t{0});
            }
        }
    }
    py::tuple mask_bounds(4);
    mask_bounds[0] = masked_pixels == 0 ? 0U : mask_start_row;
    mask_bounds[1] = masked_pixels == 0 ? 0U : mask_end_row;
    mask_bounds[2] = masked_pixels == 0 ? 0U : mask_start_column;
    mask_bounds[3] = masked_pixels == 0 ? 0U : mask_end_column;
    return py::make_tuple(
        std::move(proxy_source),
        std::move(mask),
        std::move(repair_bits),
        std::move(missing),
        missing_lanes,
        missing_pixels,
        masked_pixels,
        mask_bounds
    );
}

py::array normalize_background_proxy_source_native(
    const py::array& pre_repair,
    const py::array& coverage
) {
    const auto [height, width] = validate_background_proxy_rasters(
        pre_repair,
        coverage
    );
    const std::uint64_t sample_count = static_cast<std::uint64_t>(height) * width;
    const auto* source = static_cast<const std::uint8_t*>(pre_repair.data());
    const auto* coverage_values = static_cast<const std::uint8_t*>(coverage.data());
    py::array_t<std::uint8_t> output({
        pre_repair.shape(0),
        pre_repair.shape(1),
        pre_repair.shape(2),
    });
    auto* destination = output.mutable_data();
    {
        py::gil_scoped_release release;
        for (std::uint64_t pixel = 0; pixel < sample_count; ++pixel) {
            const std::uint32_t lane_count = coverage_values[pixel];
            for (std::uint32_t channel = 0; channel < 3; ++channel) {
                const std::uint64_t sample = pixel * 3 + channel;
                destination[sample] = lane_count == 0
                    ? 0
                    : round_ties_even_clamped_u8(
                        static_cast<std::uint32_t>(source[sample]) * 16,
                        lane_count
                    );
            }
        }
    }
    return output;
}

py::array composite_background_proxy_native(
    const py::array& pre_repair,
    const py::array& coverage,
    const py::array& proxy
) {
    const auto [height, width] = validate_background_proxy_rasters(
        pre_repair,
        coverage,
        &proxy
    );
    const std::uint64_t sample_count = static_cast<std::uint64_t>(height) * width;
    const auto* source = static_cast<const std::uint8_t*>(pre_repair.data());
    const auto* coverage_values = static_cast<const std::uint8_t*>(coverage.data());
    const auto* proxy_values = static_cast<const std::uint8_t*>(proxy.data());
    py::array_t<std::uint8_t> output({
        pre_repair.shape(0),
        pre_repair.shape(1),
        pre_repair.shape(2),
    });
    auto* destination = output.mutable_data();
    {
        py::gil_scoped_release release;
        for (std::uint64_t pixel = 0; pixel < sample_count; ++pixel) {
            const std::uint32_t lane_count = coverage_values[pixel];
            const std::uint32_t missing_lanes = 16 - lane_count;
            for (std::uint32_t channel = 0; channel < 3; ++channel) {
                const std::uint64_t sample = pixel * 3 + channel;
                destination[sample] = missing_lanes == 0
                    ? source[sample]
                    : round_ties_even_clamped_u8(
                        static_cast<std::uint32_t>(source[sample]) * 16 +
                            static_cast<std::uint32_t>(proxy_values[sample]) *
                                missing_lanes,
                        16
                    );
            }
        }
    }
    return output;
}

py::array inpaint_background_proxy_wavefront_native(
    py::array proxy_source,
    const py::array& mask,
    std::uint32_t first_row,
    std::uint32_t end_row,
    std::uint32_t first_col,
    std::uint32_t end_col
) {
    const auto [height, width] = validate_background_proxy_layout(proxy_source, mask);
    if (!proxy_source.writeable()) {
        throw py::value_error("proxy_source must be writeable");
    }
    if (end_row <= first_row || end_col <= first_col) {
        return proxy_source;
    }
    first_row = std::min(first_row, height);
    end_row = std::min(end_row, height);
    first_col = std::min(first_col, width);
    end_col = std::min(end_col, width);
    if (end_row <= first_row || end_col <= first_col) {
        return proxy_source;
    }

    auto* img = static_cast<std::uint8_t*>(proxy_source.mutable_data());
    const auto* msk = static_cast<const std::uint8_t*>(mask.data());
    {
        py::gil_scoped_release release;
        const std::uint32_t y0 = first_row > 0U ? first_row - 1U : 0U;
        const std::uint32_t y1 = std::min(height, end_row + 1U);
        const std::uint32_t x0 = first_col > 0U ? first_col - 1U : 0U;
        const std::uint32_t x1 = std::min(width, end_col + 1U);
        const std::uint32_t rh = y1 - y0;
        const std::uint32_t rw = x1 - x0;
        const std::size_t roi_size = static_cast<std::size_t>(rh) * rw;

        // 0 = unknown masked, 1 = queued in frontier, 2 = known valid/filled
        constexpr std::size_t kThreadLocalRoiCap = 2500000U;
        thread_local std::vector<std::uint8_t> tl_state;
        std::vector<std::uint8_t> local_state;
        std::uint8_t* state = nullptr;
        if (roi_size <= kThreadLocalRoiCap) {
            if (tl_state.size() < roi_size) {
                tl_state.resize(roi_size);
            }
            state = tl_state.data();
        } else {
            local_state.resize(roi_size);
            state = local_state.data();
        }

        thread_local std::vector<std::uint32_t> tl_frontier;
        thread_local std::vector<std::uint32_t> tl_next_frontier;
        thread_local std::vector<std::uint8_t> tl_layer_rgb;
        tl_frontier.clear();
        tl_next_frontier.clear();

        // Populate state for ROI and detect initial 4-connected boundary frontier.
        for (std::uint32_t r = 0; r < rh; ++r) {
            const std::uint64_t g_row = static_cast<std::uint64_t>(y0 + r) * width + x0;
            std::uint8_t* s_row = state + static_cast<std::size_t>(r) * rw;
            const std::uint8_t* m_row = msk + g_row;
            for (std::uint32_t c = 0; c < rw; ++c) {
                s_row[c] = (m_row[c] == 0U) ? 2U : 0U;
            }
        }
        const std::uint32_t r_begin = first_row - y0;
        const std::uint32_t r_end = end_row - y0;
        const std::uint32_t c_begin = first_col - x0;
        const std::uint32_t c_end = end_col - x0;
        for (std::uint32_t r = r_begin; r < r_end; ++r) {
            const std::size_t ro = static_cast<std::size_t>(r) * rw;
            for (std::uint32_t c = c_begin; c < c_end; ++c) {
                const std::size_t idx = ro + c;
                if (state[idx] != 0U) {
                    continue;
                }
                const bool has_known_nb =
                    (r > 0 && state[idx - rw] == 2U) ||
                    (r + 1 < rh && state[idx + rw] == 2U) ||
                    (c > 0 && state[idx - 1] == 2U) ||
                    (c + 1 < rw && state[idx + 1] == 2U);
                if (has_known_nb) {
                    state[idx] = 1U;
                    tl_frontier.push_back(static_cast<std::uint32_t>(idx));
                }
            }
        }

        // Exact isotropic octagonal wavefront BFS (alternating 4/8 expansion + 3x3 weighted avg).
        std::uint32_t step = 0;
        while (!tl_frontier.empty()) {
            const std::size_t fcount = tl_frontier.size();
            tl_layer_rgb.resize(fcount * 3U);
            const bool expand_diag = (step & 1U) != 0U;
            tl_next_frontier.clear();

            for (std::size_t i = 0; i < fcount; ++i) {
                const std::uint32_t idx = tl_frontier[i];
                const std::uint32_t r = idx / rw;
                const std::uint32_t c = idx - r * rw;
                const std::uint32_t gy = y0 + r;
                const std::uint32_t gx = x0 + c;
                const std::uint64_t gp = static_cast<std::uint64_t>(gy) * width + gx;
                std::uint32_t s0 = 0, s1 = 0, s2 = 0, wsum = 0;

                const auto accum = [&](std::size_t nb_idx, std::uint64_t nb_gp, std::uint32_t w) {
                    if (state[nb_idx] == 2U) {
                        const std::uint8_t* p = img + nb_gp * 3ULL;
                        s0 += p[0] * w;
                        s1 += p[1] * w;
                        s2 += p[2] * w;
                        wsum += w;
                    }
                };
                const bool has_up = r > 0;
                const bool has_dn = r + 1 < rh;
                const bool has_lt = c > 0;
                const bool has_rt = c + 1 < rw;

                if (has_up) accum(idx - rw, gp - width, 2U);
                if (has_dn) accum(idx + rw, gp + width, 2U);
                if (has_lt) accum(idx - 1, gp - 1, 2U);
                if (has_rt) accum(idx + 1, gp + 1, 2U);
                if (has_up && has_lt) accum(idx - rw - 1, gp - width - 1, 1U);
                if (has_up && has_rt) accum(idx - rw + 1, gp - width + 1, 1U);
                if (has_dn && has_lt) accum(idx + rw - 1, gp + width - 1, 1U);
                if (has_dn && has_rt) accum(idx + rw + 1, gp + width + 1, 1U);

                if (wsum != 0U) {
                    const std::uint32_t half = wsum >> 1U;
                    tl_layer_rgb[i * 3U] = static_cast<std::uint8_t>((s0 + half) / wsum);
                    tl_layer_rgb[i * 3U + 1U] = static_cast<std::uint8_t>((s1 + half) / wsum);
                    tl_layer_rgb[i * 3U + 2U] = static_cast<std::uint8_t>((s2 + half) / wsum);
                } else {
                    const std::uint8_t* p = img + gp * 3ULL;
                    tl_layer_rgb[i * 3U] = p[0];
                    tl_layer_rgb[i * 3U + 1U] = p[1];
                    tl_layer_rgb[i * 3U + 2U] = p[2];
                }

                const auto try_push = [&](std::size_t nb_idx) {
                    if (state[nb_idx] == 0U) {
                        state[nb_idx] = 1U;
                        tl_next_frontier.push_back(static_cast<std::uint32_t>(nb_idx));
                    }
                };
                if (has_up) try_push(idx - rw);
                if (has_dn) try_push(idx + rw);
                if (has_lt) try_push(idx - 1);
                if (has_rt) try_push(idx + 1);
                if (expand_diag) {
                    if (has_up && has_lt) try_push(idx - rw - 1);
                    if (has_up && has_rt) try_push(idx - rw + 1);
                    if (has_dn && has_lt) try_push(idx + rw - 1);
                    if (has_dn && has_rt) try_push(idx + rw + 1);
                }
            }

            // Commit entire wavefront simultaneously so propagation is strictly isotropic.
            for (std::size_t i = 0; i < fcount; ++i) {
                const std::uint32_t idx = tl_frontier[i];
                const std::uint32_t r = idx / rw;
                const std::uint32_t c = idx - r * rw;
                const std::uint64_t gp =
                    static_cast<std::uint64_t>(y0 + r) * width + (x0 + c);
                std::uint8_t* dst = img + gp * 3ULL;
                dst[0] = tl_layer_rgb[i * 3U];
                dst[1] = tl_layer_rgb[i * 3U + 1U];
                dst[2] = tl_layer_rgb[i * 3U + 2U];
                state[idx] = 2U;
            }
            tl_frontier.swap(tl_next_frontier);
            ++step;
        }
    }
    return proxy_source;
}

struct ForegroundMatteWinner {
    std::uint64_t destination;
    double total_disparity_fraction;
    double displacement;
    float near_score;
    std::uint32_t region;
    bool source_valid;
    bool metric_valid;
};

template <typename T>
void copy_foreground_matte_winners(
    T* values,
    const bool* expanded,
    const std::uint8_t* winner,
    std::uint64_t sample_count,
    std::uint64_t width
) {
    std::vector<T> original(values, values + sample_count);
    for (std::uint64_t pixel = 0; pixel < sample_count; ++pixel) {
        if (!expanded[pixel]) {
            continue;
        }
        std::uint64_t source = pixel;
        switch (winner[pixel]) {
            case 1:
                source -= width;
                break;
            case 2:
                source += width;
                break;
            case 3:
                --source;
                break;
            case 4:
                ++source;
                break;
            default:
                throw std::logic_error("expanded foreground matte has no winner");
        }
        values[pixel] = original[source];
    }
}

py::array expand_foreground_matte_one_pixel_native(
    py::array near_score,
    py::array total_disparity_fraction,
    py::array source_valid,
    py::array final_region_map,
    py::array one_eye_displacement_px,
    const py::object& metric_valid_object,
    const py::object& candidate_mask_object,
    bool update_displacement = false
) {
    validate_array(near_score, py::dtype::of<float>(), "near_score");
    validate_array(
        total_disparity_fraction,
        py::dtype::of<double>(),
        "total_disparity_fraction"
    );
    validate_array(source_valid, py::dtype::of<bool>(), "source_valid");
    validate_array(final_region_map, py::dtype::of<std::uint32_t>(), "final_region_map");
    validate_array(
        one_eye_displacement_px,
        py::dtype::of<double>(),
        "one_eye_displacement_px"
    );
    const auto matches_near_score = [&](const py::array& values) {
        return values.shape(0) == near_score.shape(0) &&
            values.shape(1) == near_score.shape(1);
    };
    if (
        !matches_near_score(total_disparity_fraction) ||
        !matches_near_score(source_valid) ||
        !matches_near_score(final_region_map) ||
        !matches_near_score(one_eye_displacement_px)
    ) {
        throw py::value_error("foreground matte rasters must match near_score");
    }
    if (
        !near_score.writeable() ||
        !total_disparity_fraction.writeable() ||
        !source_valid.writeable() ||
        !final_region_map.writeable() ||
        !one_eye_displacement_px.writeable()
    ) {
        throw py::value_error("foreground matte rasters must be writeable");
    }

    py::array metric_valid;
    const bool has_metric_valid = !metric_valid_object.is_none();
    if (has_metric_valid) {
        metric_valid = py::cast<py::array>(metric_valid_object);
        validate_array(metric_valid, py::dtype::of<bool>(), "metric_valid");
        if (!matches_near_score(metric_valid)) {
            throw py::value_error("metric_valid must match near_score");
        }
        if (!metric_valid.writeable()) {
            throw py::value_error("metric_valid must be writeable");
        }
    }
    py::array candidate_mask;
    const bool has_candidate_mask = !candidate_mask_object.is_none();
    if (has_candidate_mask) {
        candidate_mask = py::cast<py::array>(candidate_mask_object);
        validate_array(candidate_mask, py::dtype::of<bool>(), "candidate_mask");
        if (!matches_near_score(candidate_mask)) {
            throw py::value_error("candidate_mask must match near_score");
        }
    }

    const std::uint64_t height = static_cast<std::uint64_t>(near_score.shape(0));
    const std::uint64_t width = static_cast<std::uint64_t>(near_score.shape(1));
    const std::uint64_t sample_count = height * width;
    auto* near_values = static_cast<float*>(near_score.mutable_data());
    auto* total_values = static_cast<double*>(total_disparity_fraction.mutable_data());
    auto* source_valid_values = static_cast<bool*>(source_valid.mutable_data());
    auto* region_values = static_cast<std::uint32_t*>(final_region_map.mutable_data());
    auto* displacement_values = static_cast<double*>(one_eye_displacement_px.mutable_data());
    auto* metric_valid_values = has_metric_valid
        ? static_cast<bool*>(metric_valid.mutable_data())
        : nullptr;
    const auto* candidate_values = has_candidate_mask
        ? static_cast<const bool*>(candidate_mask.data())
        : nullptr;
    py::array_t<bool> output({near_score.shape(0), near_score.shape(1)});
    auto* expanded = output.mutable_data();
    std::vector<std::uint8_t> winner(sample_count, 0);
    {
        py::gil_scoped_release release;
        for (std::uint64_t pixel = 0; pixel < sample_count; ++pixel) {
            if (
                !std::isfinite(near_values[pixel]) ||
                !std::isfinite(total_values[pixel]) ||
                !std::isfinite(displacement_values[pixel])
            ) {
                throw std::invalid_argument(
                    "foreground matte geometry must contain only finite values"
                );
            }
            if (region_values[pixel] == 0 || region_values[pixel] == kUint32Max) {
                throw std::invalid_argument(
                    "final_region_map must contain positive canonical IDs"
                );
            }
        }
        const std::uint64_t sparse_record_limit = std::max<std::uint64_t>(
            1,
            sample_count / kForegroundMatteSparseRecordLimitDivisor
        );
        bool use_sparse_records = true;
        std::vector<ForegroundMatteWinner> winners;
        for (std::uint64_t row = 0; row < height; ++row) {
            for (std::uint64_t column = 0; column < width; ++column) {
                const std::uint64_t pixel = row * width + column;
                if (candidate_values != nullptr && !candidate_values[pixel]) {
                    expanded[pixel] = false;
                    continue;
                }
                float best_near = near_values[pixel];
                std::uint8_t selected = 0;
                const auto consider = [&](std::uint64_t source, std::uint8_t code) {
                    if (
                        !source_valid_values[source] ||
                        (metric_valid_values != nullptr && !metric_valid_values[source]) ||
                        region_values[source] == region_values[pixel] ||
                        !(near_values[source] > best_near)
                    ) {
                        return;
                    }
                    best_near = near_values[source];
                    selected = code;
                };
                if (row != 0) {
                    consider(pixel - width, 1);
                }
                if (row + 1 < height) {
                    consider(pixel + width, 2);
                }
                if (column != 0) {
                    consider(pixel - 1, 3);
                }
                if (column + 1 < width) {
                    consider(pixel + 1, 4);
                }
                winner[pixel] = selected;

                std::uint64_t source = pixel;
                switch (selected) {
                    case 1:
                        source -= width;
                        break;
                    case 2:
                        source += width;
                        break;
                    case 3:
                        --source;
                        break;
                    case 4:
                        ++source;
                        break;
                    default:
                        expanded[pixel] = false;
                        continue;
                }
                expanded[pixel] = std::abs(
                    displacement_values[source] - displacement_values[pixel]
                ) >= 0.25;
                if (!expanded[pixel] || !use_sparse_records) {
                    continue;
                }
                if (winners.size() >= sparse_record_limit) {
                    use_sparse_records = false;
                    std::vector<ForegroundMatteWinner>().swap(winners);
                    continue;
                }
                winners.push_back(ForegroundMatteWinner{
                    pixel,
                    total_values[source],
                    displacement_values[source],
                    near_values[source],
                    region_values[source],
                    source_valid_values[source],
                    metric_valid_values != nullptr && metric_valid_values[source],
                });
            }
        }
        if (use_sparse_records) {
            for (const ForegroundMatteWinner& value : winners) {
                near_values[value.destination] = value.near_score;
                total_values[value.destination] = value.total_disparity_fraction;
                if (update_displacement) {
                    displacement_values[value.destination] = value.displacement;
                }
                source_valid_values[value.destination] = value.source_valid;
                region_values[value.destination] = value.region;
                if (metric_valid_values != nullptr) {
                    metric_valid_values[value.destination] = value.metric_valid;
                }
            }
        } else {
            copy_foreground_matte_winners(
                near_values,
                expanded,
                winner.data(),
                sample_count,
                width
            );
            copy_foreground_matte_winners(
                total_values,
                expanded,
                winner.data(),
                sample_count,
                width
            );
            if (update_displacement) {
                copy_foreground_matte_winners(
                    displacement_values,
                    expanded,
                    winner.data(),
                    sample_count,
                    width
                );
            }
            copy_foreground_matte_winners(
                source_valid_values,
                expanded,
                winner.data(),
                sample_count,
                width
            );
            copy_foreground_matte_winners(
                region_values,
                expanded,
                winner.data(),
                sample_count,
                width
            );
            if (metric_valid_values != nullptr) {
                copy_foreground_matte_winners(
                    metric_valid_values,
                    expanded,
                    winner.data(),
                    sample_count,
                    width
                );
            }
        }
    }
    return output;
}

double half_pixel_source_coordinate(
    std::uint32_t output_index,
    std::uint32_t source_size,
    std::uint32_t destination_size
) {
    double coordinate = static_cast<double>(output_index);
    coordinate = coordinate + 0.5;
    coordinate = coordinate * static_cast<double>(source_size);
    coordinate = coordinate / static_cast<double>(destination_size);
    return coordinate - 0.5;
}

std::uint32_t nearest_half_pixel_source_index(
    std::uint32_t output_index,
    std::uint32_t source_size,
    std::uint32_t destination_size
) {
    const double coordinate = half_pixel_source_coordinate(
        output_index,
        source_size,
        destination_size
    );
    const auto rounded = static_cast<std::int64_t>(std::floor(coordinate + 0.5));
    return static_cast<std::uint32_t>(std::min(
        std::max<std::int64_t>(rounded, 0),
        static_cast<std::int64_t>(source_size - 1)
    ));
}

py::tuple collect_one_sided_queries_native(
    const py::tuple& source_primitives,
    const py::tuple& output_primitives,
    const py::array& source_region_map,
    const py::array& final_region_map,
    std::uint32_t radius,
    py::array& query_pixels,
    py::array& query_regions,
    py::array& query_y,
    py::array& query_x
) {
    validate_array(
        source_region_map,
        py::dtype::of<std::uint32_t>(),
        "source_region_map"
    );
    validate_array(
        final_region_map,
        py::dtype::of<std::uint32_t>(),
        "final_region_map"
    );
    validate_vector(query_pixels, py::dtype::of<std::uint32_t>(), "query_pixels");
    validate_vector(query_regions, py::dtype::of<std::uint32_t>(), "query_regions");
    validate_vector(query_y, py::dtype::of<double>(), "query_y");
    validate_vector(query_x, py::dtype::of<double>(), "query_x");
    if (
        source_primitives.size() == 0 ||
        source_primitives.size() != output_primitives.size()
    ) {
        throw py::value_error("source and output primitive tuples must match");
    }
    if (
        query_regions.shape(0) != query_pixels.shape(0) ||
        query_y.shape(0) != query_pixels.shape(0) ||
        query_x.shape(0) != query_pixels.shape(0)
    ) {
        throw py::value_error("query arrays must have matching capacities");
    }
    if (
        !query_pixels.writeable() ||
        !query_regions.writeable() ||
        !query_y.writeable() ||
        !query_x.writeable()
    ) {
        throw py::value_error("query arrays must be writeable");
    }
    const auto source_height = static_cast<std::uint32_t>(source_region_map.shape(0));
    const auto source_width = static_cast<std::uint32_t>(source_region_map.shape(1));
    const auto render_height = static_cast<std::uint32_t>(final_region_map.shape(0));
    const auto render_width = static_cast<std::uint32_t>(final_region_map.shape(1));
    const std::uint64_t render_count64 =
        static_cast<std::uint64_t>(render_height) * render_width;
    if (render_count64 > kUint32Max) {
        throw py::value_error("render shape exceeds uint32 capacity");
    }
    std::vector<py::array> source_arrays;
    std::vector<py::array> output_arrays;
    std::vector<const float*> source_values;
    std::vector<float*> output_values;
    source_arrays.reserve(source_primitives.size());
    output_arrays.reserve(output_primitives.size());
    source_values.reserve(source_primitives.size());
    output_values.reserve(output_primitives.size());
    for (std::size_t index = 0; index < source_primitives.size(); ++index) {
        source_arrays.push_back(py::cast<py::array>(source_primitives[index]));
        output_arrays.push_back(py::cast<py::array>(output_primitives[index]));
        auto& source = source_arrays.back();
        auto& output = output_arrays.back();
        validate_array(source, py::dtype::of<float>(), "source primitive");
        validate_array(output, py::dtype::of<float>(), "output primitive");
        if (
            source.shape(0) != source_height ||
            source.shape(1) != source_width ||
            output.shape(0) != render_height ||
            output.shape(1) != render_width
        ) {
            throw py::value_error("primitive rasters do not match their geometry shapes");
        }
        if (!output.writeable()) {
            throw py::value_error("output primitives must be writeable");
        }
        source_values.push_back(static_cast<const float*>(source.data()));
        output_values.push_back(static_cast<float*>(output.mutable_data()));
    }
    const auto* source_regions =
        static_cast<const std::uint32_t*>(source_region_map.data());
    const auto* final_regions =
        static_cast<const std::uint32_t*>(final_region_map.data());
    auto* pixel_values = static_cast<std::uint32_t*>(query_pixels.mutable_data());
    auto* query_region_values =
        static_cast<std::uint32_t*>(query_regions.mutable_data());
    auto* query_y_values = static_cast<double*>(query_y.mutable_data());
    auto* query_x_values = static_cast<double*>(query_x.mutable_data());
    const auto query_capacity = static_cast<std::uint32_t>(query_pixels.shape(0));
    std::vector<std::uint32_t> source_columns(render_width);
    for (std::uint32_t column = 0; column < render_width; ++column) {
        source_columns[column] = nearest_half_pixel_source_index(
            column,
            source_width,
            render_width
        );
    }
    std::vector<std::uint32_t> source_rows(render_height);
    for (std::uint32_t row = 0; row < render_height; ++row) {
        source_rows[row] = nearest_half_pixel_source_index(
            row,
            source_height,
            render_height
        );
    }
    const auto boundary_cache_rows = static_cast<std::uint32_t>(std::min<std::uint64_t>(
        render_height,
        2ULL * radius + 1
    ));
    std::vector<std::uint8_t> boundary_cache(
        static_cast<std::uint64_t>(boundary_cache_rows) * render_width,
        0
    );
    std::vector<std::uint64_t> boundary_cache_tags(
        boundary_cache_rows,
        std::numeric_limits<std::uint64_t>::max()
    );
    std::vector<std::uint8_t> edge(render_width, 0);
    std::uint32_t query_count = 0;
    std::uint32_t edge_count = 0;
    {
        py::gil_scoped_release release;
        const auto mapped_label = [&](std::uint32_t row, std::uint32_t column) {
            return source_regions[
                source_rows[row] * source_width + source_columns[column]
            ];
        };
        const auto boundary_for = [&](std::uint32_t row) {
            const std::uint32_t slot = row % boundary_cache_rows;
            auto* boundary = boundary_cache.data() +
                static_cast<std::uint64_t>(slot) * render_width;
            if (boundary_cache_tags[slot] == row) {
                return boundary;
            }
            std::fill(boundary, boundary + render_width, 0);
            for (std::uint32_t column = 0; column < render_width; ++column) {
                const std::uint32_t label = mapped_label(row, column);
                if (
                    column + 1 < render_width &&
                    label != mapped_label(row, column + 1)
                ) {
                    boundary[column] = 1;
                    boundary[column + 1] = 1;
                }
                if (row != 0 && label != mapped_label(row - 1, column)) {
                    boundary[column] = 1;
                }
                if (
                    row + 1 < render_height &&
                    label != mapped_label(row + 1, column)
                ) {
                    boundary[column] = 1;
                }
            }
            boundary_cache_tags[slot] = row;
            return boundary;
        };
        for (std::uint32_t row = 0; row < render_height; ++row) {
            std::fill(edge.begin(), edge.end(), 0);
            const std::uint32_t row_start = row > radius ? row - radius : 0;
            const std::uint32_t row_end = static_cast<std::uint32_t>(std::min<std::uint64_t>(
                render_height,
                static_cast<std::uint64_t>(row) + radius + 1
            ));
            for (std::uint32_t boundary_row = row_start; boundary_row < row_end; ++boundary_row) {
                const auto* boundary = boundary_for(boundary_row);
                const std::uint32_t vertical_distance =
                    boundary_row > row ? boundary_row - row : row - boundary_row;
                const std::uint32_t horizontal_radius = radius - vertical_distance;
                for (std::uint32_t column = 0; column < render_width; ++column) {
                    if (boundary[column] == 0) {
                        continue;
                    }
                    const std::uint32_t column_start =
                        column > horizontal_radius ? column - horizontal_radius : 0;
                    const std::uint32_t column_end = static_cast<std::uint32_t>(
                        std::min<std::uint64_t>(
                            render_width,
                            static_cast<std::uint64_t>(column) + horizontal_radius + 1
                        )
                    );
                    std::fill(
                        edge.begin() + column_start,
                        edge.begin() + column_end,
                        1
                    );
                }
            }
            for (std::uint32_t column = 0; column < render_width; ++column) {
                if (edge[column] == 0) {
                    continue;
                }
                ++edge_count;
                const std::uint32_t output_pixel = row * render_width + column;
                const std::uint32_t selected_region = final_regions[output_pixel];
                const double source_x = half_pixel_source_coordinate(
                    column,
                    source_width,
                    render_width
                );
                const double source_y = half_pixel_source_coordinate(
                    row,
                    source_height,
                    render_height
                );
                const double clipped_x = std::min(
                    std::max(source_x, 0.0),
                    static_cast<double>(source_width - 1)
                );
                const double clipped_y = std::min(
                    std::max(source_y, 0.0),
                    static_cast<double>(source_height - 1)
                );
                const std::uint32_t x0 = static_cast<std::uint32_t>(std::floor(clipped_x));
                const std::uint32_t x1 = std::min(x0 + 1, source_width - 1);
                const std::uint32_t y0 = static_cast<std::uint32_t>(std::floor(clipped_y));
                const std::uint32_t y1 = std::min(y0 + 1, source_height - 1);
                const double weight_x = clipped_x - static_cast<double>(x0);
                const double weight_y = clipped_y - static_cast<double>(y0);
                const double opposite_x = 1.0 - weight_x;
                const double opposite_y = 1.0 - weight_y;
                const std::array<double, 4> weights{
                    opposite_y * opposite_x,
                    opposite_y * weight_x,
                    weight_y * opposite_x,
                    weight_y * weight_x,
                };
                const std::array<std::uint32_t, 4> samples{
                    y0 * source_width + x0,
                    y0 * source_width + x1,
                    y1 * source_width + x0,
                    y1 * source_width + x1,
                };
                std::array<double, 4> retained_weights{};
                for (std::uint32_t index = 0; index < 4; ++index) {
                    retained_weights[index] = weights[index] *
                        (source_regions[samples[index]] == selected_region ? 1.0 : 0.0);
                }
                double retained = retained_weights[0] + retained_weights[1];
                retained = retained + retained_weights[2];
                retained = retained + retained_weights[3];
                if (retained != 0.0) {
                    for (std::size_t primitive = 0; primitive < source_values.size(); ++primitive) {
                        const auto* source = source_values[primitive];
                        double numerator =
                            static_cast<double>(source[samples[0]]) * retained_weights[0] +
                            static_cast<double>(source[samples[1]]) * retained_weights[1];
                        numerator = numerator +
                            static_cast<double>(source[samples[2]]) * retained_weights[2];
                        numerator = numerator +
                            static_cast<double>(source[samples[3]]) * retained_weights[3];
                        output_values[primitive][output_pixel] = static_cast<float>(
                            numerator / retained
                        );
                    }
                    continue;
                }
                if (query_count >= query_capacity) {
                    throw std::runtime_error("one-sided query capacity exceeded");
                }
                pixel_values[query_count] = output_pixel;
                query_region_values[query_count] = selected_region;
                query_y_values[query_count] = source_y;
                query_x_values[query_count] = source_x;
                ++query_count;
            }
        }
    }
    return py::make_tuple(query_count, edge_count);
}

py::dict build_info() {
    py::dict result;
    result["algorithm"] = "quality-implicit-region-kd-hybrid-nth-element-v2";
    result["kd_stack_capacity"] = kStackCapacity;
    result["runtime_jit"] = false;
    result["index_build_parallelism"] = "deterministic-region-std-thread-v1";
    result["index_build_worker_cap"] = kIndexBuildWorkerCap;
    result["geometry_query_parallelism"] = "deterministic-std-thread-v1";
    result["geometry_query_worker_cap"] = kGeometryQueryWorkerCap;
    result["edge_band_parallelism"] = "deterministic-row-std-thread-v1";
    result["edge_band_worker_cap"] = kEdgeBandWorkerCap;
    result["foreground_matte_copy_strategy"] =
        "sparse-winner-records-dense-fallback-v1";
    result["foreground_matte_sparse_record_limit_divisor"] =
        kForegroundMatteSparseRecordLimitDivisor;
    result["foreground_matte_winner_scan"] = "fused-destination-order-v1";
    result["foreground_matte_candidate_domain"] = "edge-band-v1";
    result["background_proxy_preparation"] =
        "fused-lut-sliding-window-mask-bounds-v2";
    result["background_proxy_inpaint_domain"] = "exact-padded-mask-roi-v1";
    result["stable_region_selection"] = "known-cardinality-native-v1";
    result["stable_surface_support"] = "min-root-guarded-scan-fused-v2";
    result["local_geometry_query"] = "exact-bounded-grid-kd-fallback-v1";
    result["local_geometry_query_radius"] = kLocalGeometryQueryRadius;
    result["local_geometry_query_parallelism"] =
        "deterministic-block-std-thread-v1";
    result["nearest_label_upsample_parallelism"] =
        "precomputed-axis-deterministic-row-std-thread-v1";
    result["nearest_label_upsample_worker_cap"] =
        kNearestLabelUpsampleWorkerCap;
    result["sparse_seed_filtering"] = "fused-stability-v1";
    result["four_neighbour_jump_parallelism"] =
        "deterministic-row-std-thread-v1";
    result["four_neighbour_jump_worker_cap"] = kFourNeighbourJumpWorkerCap;
    result["native_spawned_worker_cap"] = kNativeSpawnedWorkerCap;
    result["sparse_seed_morphology"] = "active-fragment-pixels-v1";
    result["geodesic_workset"] = "band-pixel-cached-4ary-heap-storage-v2";
    result["source_sha256"] = QUALITY_KD_SOURCE_SHA256;
    return result;
}

}  // namespace

PYBIND11_MODULE(_quality_kd_native, module) {
    module.doc() = "Prebuilt deterministic Quality geometry and stereo native extension";
    py::register_exception<QualityNativeBudgetError>(module, "QualityBudgetError");
    module.def(
        "build_index",
        &build_index,
        py::arg("region_ids"),
        py::arg("region_count"),
        py::arg("include") = py::none(),
        py::arg("allowed_regions") = py::none()
    );
    module.def(
        "query_geometry_batch",
        &query_geometry_batch,
        py::arg("member_index"),
        py::arg("region_offsets"),
        py::arg("height"),
        py::arg("width"),
        py::arg("query_regions"),
        py::arg("target_y"),
        py::arg("target_x"),
        py::arg("budget_state")
    );
    module.def(
        "query_local_geometry_batch",
        &query_local_geometry_batch,
        py::arg("region_ids"),
        py::arg("include"),
        py::arg("query_regions"),
        py::arg("target_y"),
        py::arg("target_x"),
        py::arg("radius_px"),
        py::arg("budget_state")
    );
    module.def(
        "solve_geodesic_regions",
        &solve_geodesic_regions,
        py::arg("guide"),
        py::arg("band_mask"),
        py::arg("seed_region_map"),
        py::arg("region_rank_bits"),
        py::arg("movement_base_cost") = 256,
        py::arg("movement_edge_scale") = 8
    );
    module.def(
        "solve_geodesic_regions_owner_only",
        &solve_geodesic_regions_owner_only,
        py::arg("guide"),
        py::arg("band_mask"),
        py::arg("seed_region_map"),
        py::arg("region_rank_bits"),
        py::arg("movement_base_cost") = 256,
        py::arg("movement_edge_scale") = 8
    );
    module.def(
        "accumulate_lane_hole_run_histograms_into",
        &accumulate_lane_hole_run_histograms_into_native,
        py::arg("lane_valid"),
        py::arg("lane_histogram"),
        py::arg("span_histogram"),
        py::arg("coverage_output") = py::none()
    );
    module.def(
        "accumulate_packed_lane_hole_run_histograms_into",
        &accumulate_packed_lane_hole_run_histograms_into_native,
        py::arg("packed_missing_masks"),
        py::arg("lane_histogram"),
        py::arg("span_histogram"),
        py::arg("coverage_output") = py::none()
    );
    module.def(
        "count_repair_state_bits",
        &count_repair_state_bits_native,
        py::arg("repair_bits")
    );
    module.def(
        "build_relative_eye_offsets_into",
        &build_relative_eye_offsets_into_native,
        py::arg("near_score"),
        py::arg("stereo_strength"),
        py::arg("convergence"),
        py::arg("direction"),
        py::arg("output"),
        py::arg("scratch")
    );
    module.def(
        "update_relative_total_inside_band",
        &update_relative_total_inside_band_native,
        py::arg("total_disparity_fraction"),
        py::arg("near_score"),
        py::arg("edge_band"),
        py::arg("stereo_strength"),
        py::arg("convergence")
    );
    module.def(
        "max_four_neighbour_jump",
        &max_four_neighbour_jump_native,
        py::arg("displacement")
    );
    module.def(
        "build_background_proxy_state_into",
        &build_background_proxy_state_into_native,
        py::arg("coverage"),
        py::arg("repair_bits"),
        py::arg("missing_mask")
    );
    module.def(
        "build_low_resolution_regions",
        &build_low_resolution_regions_native,
        py::arg("one_eye_displacement_px"),
        py::arg("metric_validity"),
        py::arg("near_score")
    );
    module.def(
        "build_stable_surface_support",
        &build_stable_surface_support_native,
        py::arg("initial_region_map"),
        py::arg("stable_regions"),
        py::arg("one_eye_displacement_px"),
        py::arg("near_score")
    );
    module.def(
        "select_stable_region_support",
        &select_stable_region_support_native,
        py::arg("region_map"),
        py::arg("region_count"),
        py::arg("area_threshold")
    );
    module.def(
        "nearest_label_upsample",
        &nearest_label_upsample_native,
        py::arg("low_resolution_regions"),
        py::arg("render_height"),
        py::arg("render_width")
    );
    module.def(
        "build_four_connected_edge_band",
        &build_four_connected_edge_band_native,
        py::arg("mapped_regions"),
        py::arg("radius"),
        py::arg("include_zero") = false
    );
    module.def(
        "label_region_fragments",
        &label_region_fragments_native,
        py::arg("region_map"),
        py::arg("band_mask")
    );
    module.def(
        "build_sparse_region_seeds",
        &build_sparse_region_seeds_native,
        py::arg("mapped_regions"),
        py::arg("edge_band"),
        py::arg("radius")
    );
    module.def(
        "build_sparse_stable_region_seeds",
        &build_sparse_stable_region_seeds_native,
        py::arg("mapped_regions"),
        py::arg("edge_band"),
        py::arg("stable_initial"),
        py::arg("stable_surfaces"),
        py::arg("radius")
    );
    module.def(
        "prepare_background_proxy",
        &prepare_background_proxy_native,
        py::arg("pre_repair"),
        py::arg("coverage"),
        py::arg("far_side_direction"),
        py::arg("foreground_exclusion_px")
    );
    module.def(
        "normalize_background_proxy_source",
        &normalize_background_proxy_source_native,
        py::arg("pre_repair"),
        py::arg("coverage")
    );
    module.def(
        "composite_background_proxy",
        &composite_background_proxy_native,
        py::arg("pre_repair"),
        py::arg("coverage"),
        py::arg("proxy")
    );
    module.def(
        "inpaint_background_proxy_wavefront",
        &inpaint_background_proxy_wavefront_native,
        py::arg("proxy_source"),
        py::arg("mask"),
        py::arg("first_row"),
        py::arg("end_row"),
        py::arg("first_col"),
        py::arg("end_col")
    );
    module.def(
        "expand_foreground_matte_one_pixel",
        &expand_foreground_matte_one_pixel_native,
        py::arg("near_score"),
        py::arg("total_disparity_fraction"),
        py::arg("source_valid"),
        py::arg("final_region_map"),
        py::arg("one_eye_displacement_px"),
        py::arg("metric_valid") = py::none(),
        py::arg("candidate_mask") = py::none(),
        py::arg("update_displacement") = false
    );
    module.def(
        "collect_one_sided_queries",
        &collect_one_sided_queries_native,
        py::arg("source_primitives"),
        py::arg("output_primitives"),
        py::arg("source_region_map"),
        py::arg("final_region_map"),
        py::arg("radius"),
        py::arg("query_pixels"),
        py::arg("query_regions"),
        py::arg("query_y"),
        py::arg("query_x")
    );
    module.def("build_info", &build_info);
    py::module_::import("atexit").attr("register")(
        py::cpp_function([]() {
            PersistentParallelPool::instance().shutdown();
        })
    );
}
