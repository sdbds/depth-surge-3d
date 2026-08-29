#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "quality_kd_build_identity.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_M_X64) || defined(__x86_64__)
#include <emmintrin.h>
#endif

namespace py = pybind11;

namespace {

constexpr std::size_t kStackCapacity = 64;
constexpr std::uint64_t kUint32Max = std::numeric_limits<std::uint32_t>::max();
constexpr std::uint64_t kRepairRecordCap = (64ULL * 1024ULL * 1024ULL) / 16ULL;
constexpr std::uint8_t kDividerAddMarker = 0x40;
constexpr std::uint32_t kPackedCoordinateLimit = 0xffff;
constexpr std::size_t kGeometryQueryWorkerCap = 8;
constexpr std::size_t kIndexBuildWorkerCap = 8;
constexpr std::size_t kEdgeBandWorkerCap = 8;
constexpr std::size_t kNativeSpawnedWorkerCap = 7;
constexpr std::uint32_t kEdgeBandRowsPerWorker = 128;
constexpr py::ssize_t kGeometryQueriesPerWorker = 4096;
constexpr py::ssize_t kGeometryQueryBlockSize = 64;

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

struct RepairFrame {
    std::uint32_t start;
    std::uint32_t end;
    std::uint8_t depth;
    bool check_far;
    std::uint64_t plane2;
};

struct RepairBest {
    bool found = false;
    std::uint64_t distance2 = 0;
    std::uint32_t y = 0;
    std::uint32_t x = 0;
    std::uint32_t sample_index = 0;
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

std::size_t push_repair_frame(
    std::array<RepairFrame, kStackCapacity>& stack,
    std::size_t top,
    RepairFrame frame
) {
    if (top >= stack.size()) {
        throw std::runtime_error("implicit k-d query stack capacity exceeded");
    }
    stack[top] = frame;
    return top + 1;
}

std::uint64_t squared_integer_delta(std::int64_t delta) {
    const std::uint64_t magnitude = delta < 0
        ? static_cast<std::uint64_t>(-(delta + 1)) + 1
        : static_cast<std::uint64_t>(delta);
    return magnitude * magnitude;
}

bool repair_candidate_is_better(
    std::uint64_t distance2,
    std::uint32_t y,
    std::uint32_t x,
    const RepairBest& best
) {
    if (!best.found || distance2 != best.distance2) {
        return !best.found || distance2 < best.distance2;
    }
    if (y != best.y) {
        return y < best.y;
    }
    return x < best.x;
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

RepairBest query_repair_one(
    const std::uint32_t* members,
    std::uint32_t start,
    std::uint32_t end,
    std::uint32_t width,
    const U32Divider& width_divider,
    std::uint32_t target_y,
    std::uint32_t target_x,
    std::uint64_t radius2,
    std::uint64_t* budget_state,
    std::uint32_t* visited_output
) {
    RepairBest best{};
    *visited_output = 0;
    if (start == end) {
        return best;
    }
    std::array<RepairFrame, kStackCapacity> stack{};
    std::size_t top = push_repair_frame(
        stack,
        0,
        RepairFrame{start, end, 0, false, 0}
    );
    while (top > 0) {
        const RepairFrame frame = stack[--top];
        if (frame.check_far) {
            const std::uint64_t bound = best.found && best.distance2 < radius2
                ? best.distance2
                : radius2;
            if (frame.plane2 <= bound) {
                top = push_repair_frame(
                    stack,
                    top,
                    RepairFrame{frame.start, frame.end, frame.depth, false, 0}
                );
            }
            continue;
        }
        if (budget_state[0] >= budget_state[1]) {
            throw std::runtime_error("quality repair fallback visit cap exceeded");
        }
        ++budget_state[0];
        ++(*visited_output);

        const std::uint32_t middle = frame.start + ((frame.end - frame.start - 1) / 2);
        const std::uint32_t sample_index = members[middle];
        const std::uint32_t y = width_divider.divide(sample_index);
        const std::uint32_t x = sample_index - y * width;
        const std::int64_t dy = static_cast<std::int64_t>(target_y) - y;
        const std::int64_t dx = static_cast<std::int64_t>(target_x) - x;
        const std::uint64_t distance2 = squared_integer_delta(dx) + squared_integer_delta(dy);
        if (
            distance2 <= radius2 &&
            repair_candidate_is_better(distance2, y, x, best)
        ) {
            best = RepairBest{true, distance2, y, x, sample_index};
        }

        const bool row_axis = (frame.depth & 1) == 0;
        const std::int64_t delta = row_axis
            ? static_cast<std::int64_t>(target_y) - y
            : static_cast<std::int64_t>(target_x) - x;
        const std::uint64_t plane2 = squared_integer_delta(delta);
        const std::pair<std::uint32_t, std::uint32_t> lower{frame.start, middle};
        const std::pair<std::uint32_t, std::uint32_t> upper{middle + 1, frame.end};
        const auto near_range = delta <= 0 ? lower : upper;
        const auto far_range = delta <= 0 ? upper : lower;
        const std::uint8_t next_depth = static_cast<std::uint8_t>(frame.depth + 1);
        if (far_range.first < far_range.second) {
            top = push_repair_frame(
                stack,
                top,
                RepairFrame{
                    far_range.first,
                    far_range.second,
                    next_depth,
                    true,
                    plane2,
                }
            );
        }
        if (near_range.first < near_range.second) {
            top = push_repair_frame(
                stack,
                top,
                RepairFrame{
                    near_range.first,
                    near_range.second,
                    next_depth,
                    false,
                    0,
                }
            );
        }
    }
    return best;
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

py::tuple query_repair_batch(
    const py::array& members,
    const py::array& offsets,
    std::uint64_t height,
    std::uint64_t width,
    const py::array& query_regions,
    const py::array& target_y,
    const py::array& target_x,
    std::uint64_t radius_px,
    py::array budget_state
) {
    validate_index_arrays(members, offsets, height, width);
    if (height * width > kUint32Max / 16) {
        throw py::value_error("repair queries require 16*H*W to fit uint32");
    }
    validate_vector(query_regions, py::dtype::of<std::uint32_t>(), "query_regions");
    const py::ssize_t query_count = query_regions.shape(0);
    validate_vector(target_y, py::dtype::of<std::uint32_t>(), "target_y", query_count);
    validate_vector(target_x, py::dtype::of<std::uint32_t>(), "target_x", query_count);
    validate_budget_state(budget_state);
    if (radius_px > std::numeric_limits<std::uint32_t>::max()) {
        throw py::value_error("radius_px must fit uint32");
    }

    const auto* member_values = static_cast<const std::uint32_t*>(members.data());
    const auto* offset_values = static_cast<const std::uint32_t*>(offsets.data());
    const auto* region_values = static_cast<const std::uint32_t*>(query_regions.data());
    const auto* y_values = static_cast<const std::uint32_t*>(target_y.data());
    const auto* x_values = static_cast<const std::uint32_t*>(target_x.data());
    auto* budget_values = static_cast<std::uint64_t*>(budget_state.mutable_data());
    const std::uint32_t region_count = static_cast<std::uint32_t>(offsets.shape(0) - 1);
    const std::uint32_t height32 = static_cast<std::uint32_t>(height);
    const std::uint32_t width32 = static_cast<std::uint32_t>(width);
    const U32Divider width_divider(width32);
    const std::uint64_t radius2 = radius_px * radius_px;

    py::array_t<std::uint32_t> result_samples(query_count);
    py::array_t<std::uint64_t> result_distances(query_count);
    py::array_t<std::uint32_t> result_visits(query_count);
    auto* sample_output = result_samples.mutable_data();
    auto* distance_output = result_distances.mutable_data();
    auto* visit_output = result_visits.mutable_data();
    {
        py::gil_scoped_release release;
        for (py::ssize_t query = 0; query < query_count; ++query) {
            const std::uint32_t region = region_values[query];
            if (region == 0 || region > region_count) {
                throw std::invalid_argument("query region exceeds the canonical region count");
            }
            if (y_values[query] >= height32 || x_values[query] >= width32) {
                throw std::invalid_argument("repair query target lies outside the indexed raster");
            }
            const RepairBest best = query_repair_one(
                member_values,
                offset_values[region - 1],
                offset_values[region],
                width32,
                width_divider,
                y_values[query],
                x_values[query],
                radius2,
                budget_values,
                &visit_output[query]
            );
            sample_output[query] = best.found ? best.sample_index : kUint32Max;
            distance_output[query] = best.found
                ? best.distance2
                : std::numeric_limits<std::uint64_t>::max();
        }
    }
    return py::make_tuple(
        std::move(result_samples),
        std::move(result_distances),
        std::move(result_visits)
    );
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

            std::vector<std::thread> workers;
            workers.reserve(worker_count - 1);
            try {
                for (std::size_t index = 1; index < worker_count; ++index) {
                    workers.emplace_back(worker);
                }
            } catch (...) {
                for (auto& thread : workers) {
                    thread.join();
                }
                throw;
            }
            worker();
            for (auto& thread : workers) {
                thread.join();
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
          capacity_(static_cast<std::uint32_t>(heap_pixel_.size())) {}

    void initialize_candidates(std::uint64_t initial_decrease_keys) {
        const std::uint32_t source_length = capacity_;
        for (std::uint32_t source_slot = 0; source_slot < source_length; ++source_slot) {
            const std::uint32_t pixel = heap_pixel_[source_slot];
            if (owner_[pixel] != 0) {
                insert(pixel);
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
            position_[replacement] = 0;
            sift_down(0);
        }
        return result;
    }

    bool empty() const { return length_ == 0; }
    std::uint32_t capacity() const { return capacity_; }

    std::uint32_t max_live_entries = 0;
    std::uint64_t insertions = 0;
    std::uint64_t decrease_keys = 0;
    std::uint64_t pop_count = 0;

private:
    bool less(std::uint32_t left_slot, std::uint32_t right_slot) const {
        const std::uint32_t left_pixel = heap_pixel_[left_slot];
        const std::uint32_t right_pixel = heap_pixel_[right_slot];
        if (distance_[left_pixel] != distance_[right_pixel]) {
            return distance_[left_pixel] < distance_[right_pixel];
        }
        const std::uint32_t left_region = owner_[left_pixel];
        const std::uint32_t right_region = owner_[right_pixel];
        if (left_region >= region_rank_count_ || right_region >= region_rank_count_) {
            throw std::runtime_error("quality region queue owner rank is inconsistent");
        }
        const std::uint32_t left_rank = region_rank_bits_[left_region];
        const std::uint32_t right_rank = region_rank_bits_[right_region];
        if (left_rank != right_rank) {
            return left_rank > right_rank;
        }
        if (left_region != right_region) {
            return left_region < right_region;
        }
        return left_pixel < right_pixel;
    }

    void swap_slots(std::uint32_t left_slot, std::uint32_t right_slot) {
        const std::uint32_t left_pixel = heap_pixel_[left_slot];
        const std::uint32_t right_pixel = heap_pixel_[right_slot];
        heap_pixel_[left_slot] = right_pixel;
        heap_pixel_[right_slot] = left_pixel;
        position_[left_pixel] = static_cast<std::int32_t>(right_slot);
        position_[right_pixel] = static_cast<std::int32_t>(left_slot);
    }

    void sift_up(std::uint32_t slot) {
        while (slot != 0) {
            const std::uint32_t parent = (slot - 1) / 2;
            if (!less(slot, parent)) {
                return;
            }
            swap_slots(slot, parent);
            slot = parent;
        }
    }

    void sift_down(std::uint32_t slot) {
        while (true) {
            const std::uint32_t left = slot * 2 + 1;
            if (left >= length_) {
                return;
            }
            const std::uint32_t right = left + 1;
            const std::uint32_t child =
                right < length_ && less(right, left) ? right : left;
            if (!less(child, slot)) {
                return;
            }
            swap_slots(slot, child);
            slot = child;
        }
    }

    void insert(std::uint32_t pixel) {
        if (length_ >= capacity_) {
            throw std::runtime_error("quality region queue capacity exceeded");
        }
        const std::uint32_t slot = length_;
        heap_pixel_[slot] = pixel;
        position_[pixel] = static_cast<std::int32_t>(slot);
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
        while (!heap.empty()) {
            const std::uint32_t pixel = heap.pop();
            const std::uint64_t source_distance = distance_values[pixel];
            const std::uint32_t source_owner = owner_values[pixel];
            visit_four_neighbours(pixel, height, width, [&](std::uint32_t neighbour) {
                if (band_values[neighbour] == 0) {
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
    }

    py::dict statistics;
    statistics["max_live_entries"] = max_live_entries;
    statistics["settled_count"] = pop_count;
    statistics["queue_insertions"] = insertions;
    statistics["queue_decrease_keys"] = decrease_keys;
    statistics["queue_pops"] = pop_count;
    statistics["heap_pixel_bytes"] =
        static_cast<std::uint64_t>(band_count) * sizeof(std::uint32_t);
    return py::make_tuple(
        std::move(owner),
        std::move(distance),
        std::move(position),
        std::move(statistics)
    );
}

struct RepairRecordStatistics {
    std::uint64_t record_count = 0;
    std::uint64_t prefill_run_count = 0;
    std::uint64_t repair_run_count = 0;
    std::uint64_t invalid_lane_count = 0;
};

template <typename ValidAt, typename ScoreAt, typename RegionAt, typename Callback>
void visit_repair_runs_with_accessors(
    std::uint32_t height,
    std::uint32_t fine_width,
    RepairRecordStatistics* statistics,
    ValidAt valid_at,
    ScoreAt score_at,
    RegionAt region_at,
    Callback callback
) {
    for (std::uint32_t row = 0; row < height; ++row) {
        const std::uint32_t row_start = row * fine_width;
        std::uint32_t fine_column = 0;
        while (fine_column < fine_width) {
            if (valid_at(row_start + fine_column)) {
                ++fine_column;
                continue;
            }
            const std::uint32_t run_start = fine_column;
            while (
                fine_column + 1 < fine_width &&
                !valid_at(row_start + fine_column + 1)
            ) {
                ++fine_column;
            }
            const std::uint32_t run_end = fine_column;
            ++statistics->prefill_run_count;
            statistics->invalid_lane_count += run_end - run_start + 1;
            const bool has_left = run_start != 0;
            const bool has_right = run_end + 1 < fine_width;
            if (!has_left && !has_right) {
                throw std::invalid_argument("row-wide pre-fill run has no donor");
            }

            const auto emit = [&](std::uint32_t start,
                                  std::uint32_t end,
                                  std::uint32_t region,
                                  std::uint8_t far_side) {
                ++statistics->repair_run_count;
                statistics->record_count += end / 16 - start / 16 + 1;
                if (statistics->record_count > kRepairRecordCap) {
                    throw QualityNativeBudgetError("64 MiB repair record arena exceeded");
                }
                callback(row, start, end, region, far_side);
            };

            if (!has_left) {
                emit(
                    run_start,
                    run_end,
                    region_at(row_start + run_end + 1),
                    1
                );
            } else if (!has_right) {
                emit(
                    run_start,
                    run_end,
                    region_at(row_start + run_start - 1),
                    0
                );
            } else {
                const float left_score = score_at(row_start + run_start - 1);
                const float right_score = score_at(row_start + run_end + 1);
                if (left_score < right_score) {
                    emit(
                        run_start,
                        run_end,
                        region_at(row_start + run_start - 1),
                        0
                    );
                } else if (right_score < left_score) {
                    emit(
                        run_start,
                        run_end,
                        region_at(row_start + run_end + 1),
                        1
                    );
                } else {
                    const std::uint32_t left_length =
                        (run_end - run_start + 2) / 2;
                    const std::uint32_t left_end = run_start + left_length - 1;
                    emit(
                        run_start,
                        left_end,
                        region_at(row_start + run_start - 1),
                        0
                    );
                    if (left_end < run_end) {
                        emit(
                            left_end + 1,
                            run_end,
                            region_at(row_start + run_end + 1),
                            1
                        );
                    }
                }
            }
            ++fine_column;
        }
    }
}

template <typename Callback>
void visit_repair_runs(
    const std::uint8_t* valid,
    const float* near_score,
    const std::uint32_t* region_id,
    std::uint32_t height,
    std::uint32_t fine_width,
    RepairRecordStatistics* statistics,
    Callback callback
) {
    visit_repair_runs_with_accessors(
        height,
        fine_width,
        statistics,
        [&](std::uint32_t sample) { return valid[sample] != 0; },
        [&](std::uint32_t sample) { return near_score[sample]; },
        [&](std::uint32_t sample) { return region_id[sample]; },
        callback
    );
}

void write_little_u16(std::uint8_t* destination, std::uint16_t value) {
    destination[0] = static_cast<std::uint8_t>(value & 0xffU);
    destination[1] = static_cast<std::uint8_t>((value >> 8) & 0xffU);
}

void write_little_u32(std::uint8_t* destination, std::uint32_t value) {
    destination[0] = static_cast<std::uint8_t>(value & 0xffU);
    destination[1] = static_cast<std::uint8_t>((value >> 8) & 0xffU);
    destination[2] = static_cast<std::uint8_t>((value >> 16) & 0xffU);
    destination[3] = static_cast<std::uint8_t>((value >> 24) & 0xffU);
}

void write_repair_run_records(
    std::uint8_t* records,
    std::uint64_t* cursor,
    std::uint32_t row,
    std::uint32_t run_start,
    std::uint32_t run_end,
    std::uint32_t region,
    std::uint8_t far_side,
    std::uint32_t width,
    std::uint32_t fine_width,
    const float* winner_bgr
) {
    const std::uint32_t anchor = far_side == 0 ? run_start - 1 : run_end + 1;
    std::uint32_t segment_start = run_start;
    while (segment_start <= run_end) {
        const std::uint32_t column = segment_start / 16;
        const std::uint32_t segment_end = std::min(
            run_end,
            column * 16 + 15
        );
        const std::uint32_t first_lane = segment_start % 16;
        const std::uint32_t lane_count = segment_end - segment_start + 1;
        const std::uint32_t base_mask = lane_count == 16
            ? 0xffffU
            : ((1U << lane_count) - 1U);
        const std::uint16_t mask = static_cast<std::uint16_t>(
            base_mask << first_lane
        );
        std::uint8_t* record = records + (*cursor * 16);
        write_little_u32(record, row * width + column);
        write_little_u16(record + 4, mask);
        write_little_u32(record + 6, region);
        if (winner_bgr != nullptr) {
            const std::uint64_t sample =
                (static_cast<std::uint64_t>(row) * fine_width + anchor) * 3;
            record[10] = static_cast<std::uint8_t>(winner_bgr[sample]);
            record[11] = static_cast<std::uint8_t>(winner_bgr[sample + 1]);
            record[12] = static_cast<std::uint8_t>(winner_bgr[sample + 2]);
        }
        record[14] = far_side;
        ++(*cursor);
        segment_start = segment_end + 1;
    }
}

void write_repair_run_records_from_sources(
    std::uint8_t* records,
    std::uint64_t* cursor,
    std::uint32_t row,
    std::uint32_t run_start,
    std::uint32_t run_end,
    std::uint32_t region,
    std::uint8_t far_side,
    std::uint32_t width,
    std::uint32_t fine_width,
    const std::uint32_t* winner_source_index,
    const std::uint8_t* source_bgr
) {
    const std::uint32_t anchor = far_side == 0 ? run_start - 1 : run_end + 1;
    const std::uint32_t source =
        winner_source_index[static_cast<std::uint64_t>(row) * fine_width + anchor];
    std::uint32_t segment_start = run_start;
    while (segment_start <= run_end) {
        const std::uint32_t column = segment_start / 16;
        const std::uint32_t segment_end = std::min(
            run_end,
            column * 16 + 15
        );
        const std::uint32_t first_lane = segment_start % 16;
        const std::uint32_t lane_count = segment_end - segment_start + 1;
        const std::uint32_t base_mask = lane_count == 16
            ? 0xffffU
            : ((1U << lane_count) - 1U);
        const std::uint16_t mask = static_cast<std::uint16_t>(
            base_mask << first_lane
        );
        std::uint8_t* record = records + (*cursor * 16);
        write_little_u32(record, row * width + column);
        write_little_u16(record + 4, mask);
        write_little_u32(record + 6, region);
        record[10] = source_bgr[static_cast<std::uint64_t>(source) * 3];
        record[11] = source_bgr[static_cast<std::uint64_t>(source) * 3 + 1];
        record[12] = source_bgr[static_cast<std::uint64_t>(source) * 3 + 2];
        record[14] = far_side;
        ++(*cursor);
        segment_start = segment_end + 1;
    }
}

py::tuple build_repair_records(
    const py::array& valid,
    const py::array& winner_near_score,
    const py::array& winner_region_id,
    const py::object& winner_bgr_object
) {
    validate_array(valid, py::dtype::of<bool>(), "valid");
    validate_array(
        winner_near_score,
        py::dtype::of<float>(),
        "winner_near_score"
    );
    validate_array(
        winner_region_id,
        py::dtype::of<std::uint32_t>(),
        "winner_region_id"
    );
    if (
        winner_near_score.shape(0) != valid.shape(0) ||
        winner_near_score.shape(1) != valid.shape(1) ||
        winner_region_id.shape(0) != valid.shape(0) ||
        winner_region_id.shape(1) != valid.shape(1)
    ) {
        throw py::value_error("winner rasters must have matching shapes");
    }
    if (valid.shape(1) % 16 != 0) {
        throw py::value_error("valid width must be a multiple of 16");
    }
    const std::uint64_t sample_count =
        static_cast<std::uint64_t>(valid.shape(0)) *
        static_cast<std::uint64_t>(valid.shape(1));
    if (sample_count > kUint32Max) {
        throw py::value_error("fine-grid sample count must fit uint32");
    }

    const auto height = static_cast<std::uint32_t>(valid.shape(0));
    const auto fine_width = static_cast<std::uint32_t>(valid.shape(1));
    const auto width = fine_width / 16;
    const auto* valid_values = static_cast<const std::uint8_t*>(valid.data());
    const auto* score_values = static_cast<const float*>(winner_near_score.data());
    const auto* region_values =
        static_cast<const std::uint32_t*>(winner_region_id.data());
    py::array winner_bgr;
    const float* colour_values = nullptr;
    if (!winner_bgr_object.is_none()) {
        winner_bgr = py::cast<py::array>(winner_bgr_object);
        if (
            !winner_bgr.dtype().is(py::dtype::of<float>()) ||
            winner_bgr.ndim() != 3 ||
            winner_bgr.shape(0) != valid.shape(0) ||
            winner_bgr.shape(1) != valid.shape(1) ||
            winner_bgr.shape(2) != 3 ||
            (winner_bgr.flags() & py::array::c_style) == 0
        ) {
            throw py::type_error(
                "winner_bgr must be a matching C-contiguous float32 [H,W,3] raster"
            );
        }
        colour_values = static_cast<const float*>(winner_bgr.data());
    }
    RepairRecordStatistics statistics{};
    {
        py::gil_scoped_release release;
        for (std::uint64_t sample = 0; sample < sample_count; ++sample) {
            if (valid_values[sample] == 0) {
                continue;
            }
            if (!std::isfinite(score_values[sample]) || score_values[sample] < 0.0F) {
                throw std::invalid_argument(
                    "valid winner near scores must be finite and nonnegative"
                );
            }
            if (region_values[sample] == 0 || region_values[sample] == kUint32Max) {
                throw std::invalid_argument(
                    "valid winners must have positive canonical regions"
                );
            }
            if (colour_values != nullptr) {
                for (std::uint32_t channel = 0; channel < 3; ++channel) {
                    const float value = colour_values[sample * 3 + channel];
                    if (
                        !std::isfinite(value) || value < 0.0F || value > 255.0F ||
                        std::floor(value) != value
                    ) {
                        throw std::invalid_argument(
                            "valid winner colours must be integer-valued uint8-range float32"
                        );
                    }
                }
            }
        }
        visit_repair_runs(
            valid_values,
            score_values,
            region_values,
            height,
            fine_width,
            &statistics,
            [](std::uint32_t,
               std::uint32_t,
               std::uint32_t,
               std::uint32_t,
               std::uint8_t) {}
        );
    }

    py::array_t<std::uint8_t> records(
        {
            static_cast<py::ssize_t>(statistics.record_count),
            static_cast<py::ssize_t>(16),
        }
    );
    auto* record_values = records.mutable_data();
    std::fill(
        record_values,
        record_values + statistics.record_count * 16,
        0
    );
    std::uint64_t cursor = 0;
    {
        py::gil_scoped_release release;
        RepairRecordStatistics second_pass{};
        visit_repair_runs(
            valid_values,
            score_values,
            region_values,
            height,
            fine_width,
            &second_pass,
            [&](std::uint32_t row,
                std::uint32_t start,
                std::uint32_t end,
                std::uint32_t region,
                std::uint8_t far_side) {
                write_repair_run_records(
                    record_values,
                    &cursor,
                    row,
                    start,
                    end,
                    region,
                    far_side,
                    width,
                    fine_width,
                    colour_values
                );
            }
        );
        if (
            cursor != statistics.record_count ||
            second_pass.prefill_run_count != statistics.prefill_run_count ||
            second_pass.repair_run_count != statistics.repair_run_count ||
            second_pass.invalid_lane_count != statistics.invalid_lane_count
        ) {
            throw std::runtime_error("repair record preflight count changed");
        }
    }
    py::dict result_statistics;
    result_statistics["prefill_run_count"] = statistics.prefill_run_count;
    result_statistics["repair_run_count"] = statistics.repair_run_count;
    result_statistics["invalid_lane_count"] = statistics.invalid_lane_count;
    return py::make_tuple(std::move(records), std::move(result_statistics));
}

std::uint8_t rounded_u8_mean(std::uint32_t sum, std::uint32_t count) {
    std::uint32_t quotient = sum / count;
    const std::uint32_t remainder = sum % count;
    const std::uint32_t twice_remainder = remainder * 2;
    if (twice_remainder > count || (twice_remainder == count && (quotient & 1U) != 0)) {
        ++quotient;
    }
    return static_cast<std::uint8_t>(quotient);
}

py::tuple analyze_splat_band(
    const py::array& winner_source_index,
    const py::array& source_bgr,
    const py::array& source_near_score,
    const py::array& source_region_map,
    py::array& coverage_count,
    py::array& pure_region_id,
    py::array& blue,
    py::array& green,
    py::array& red,
    py::object record_output_object
) {
    validate_array(
        winner_source_index,
        py::dtype::of<std::uint32_t>(),
        "winner_source_index"
    );
    validate_array(
        source_near_score,
        py::dtype::of<float>(),
        "source_near_score"
    );
    validate_array(
        source_region_map,
        py::dtype::of<std::uint32_t>(),
        "source_region_map"
    );
    validate_array(
        coverage_count,
        py::dtype::of<std::uint8_t>(),
        "coverage_count"
    );
    validate_array(
        pure_region_id,
        py::dtype::of<std::uint32_t>(),
        "pure_region_id"
    );
    validate_array(blue, py::dtype::of<std::uint8_t>(), "blue");
    validate_array(green, py::dtype::of<std::uint8_t>(), "green");
    validate_array(red, py::dtype::of<std::uint8_t>(), "red");
    if (
        !source_bgr.dtype().is(py::dtype::of<std::uint8_t>()) ||
        source_bgr.ndim() != 3 ||
        source_bgr.shape(2) != 3 ||
        (source_bgr.flags() & py::array::c_style) == 0
    ) {
        throw py::type_error(
            "source_bgr must be a C-contiguous uint8 [H,W,3] raster"
        );
    }
    if (
        source_bgr.shape(0) != source_region_map.shape(0) ||
        source_bgr.shape(1) != source_region_map.shape(1) ||
        source_near_score.shape(0) != source_region_map.shape(0) ||
        source_near_score.shape(1) != source_region_map.shape(1)
    ) {
        throw py::value_error("source rasters must have matching shapes");
    }
    if (winner_source_index.shape(1) % 16 != 0) {
        throw py::value_error("winner width must be a multiple of 16");
    }
    const auto height = static_cast<std::uint32_t>(winner_source_index.shape(0));
    const auto fine_width = static_cast<std::uint32_t>(winner_source_index.shape(1));
    const auto width = fine_width / 16;
    for (py::array* output : {
             &coverage_count,
             &pure_region_id,
             &blue,
             &green,
             &red,
         }) {
        if (output->shape(0) != height || output->shape(1) != width) {
            throw py::value_error("analysis outputs must match the coarse band shape");
        }
        if (!output->writeable()) {
            throw py::value_error("analysis outputs must be writeable");
        }
    }
    const std::uint64_t sample_count64 =
        static_cast<std::uint64_t>(height) * fine_width;
    const std::uint64_t source_count64 =
        static_cast<std::uint64_t>(source_region_map.shape(0)) *
        static_cast<std::uint64_t>(source_region_map.shape(1));
    if (sample_count64 > kUint32Max || source_count64 > kUint32Max) {
        throw py::value_error("splat analysis shapes must fit uint32");
    }
    const auto sample_count = static_cast<std::uint32_t>(sample_count64);
    const auto source_count = static_cast<std::uint32_t>(source_count64);
    const auto* winner_values =
        static_cast<const std::uint32_t*>(winner_source_index.data());
    const auto* colour_values = static_cast<const std::uint8_t*>(source_bgr.data());
    const auto* source_scores = static_cast<const float*>(source_near_score.data());
    const auto* source_regions =
        static_cast<const std::uint32_t*>(source_region_map.data());
    auto* coverage_values = static_cast<std::uint8_t*>(coverage_count.mutable_data());
    auto* pure_region_values =
        static_cast<std::uint32_t*>(pure_region_id.mutable_data());
    auto* blue_values = static_cast<std::uint8_t*>(blue.mutable_data());
    auto* green_values = static_cast<std::uint8_t*>(green.mutable_data());
    auto* red_values = static_cast<std::uint8_t*>(red.mutable_data());
    const auto valid_at = [&](std::uint32_t sample) {
        return winner_values[sample] != kUint32Max;
    };
    const auto score_at = [&](std::uint32_t sample) {
        return source_scores[winner_values[sample]];
    };
    const auto region_at = [&](std::uint32_t sample) {
        return source_regions[winner_values[sample]];
    };
    RepairRecordStatistics statistics{};
    {
        py::gil_scoped_release release;
        for (std::uint32_t pixel = 0; pixel < height * width; ++pixel) {
            const std::uint32_t first_sample = pixel * 16;
            std::uint32_t lane_count = 0;
            std::uint32_t first_region = kUint32Max;
            bool pure = true;
            std::array<std::uint32_t, 3> sums{0, 0, 0};
            for (std::uint32_t lane = 0; lane < 16; ++lane) {
                const std::uint32_t sample = first_sample + lane;
                const std::uint32_t source = winner_values[sample];
                if (source == kUint32Max) {
                    continue;
                }
                if (source >= source_count) {
                    throw std::invalid_argument("valid winner source lies outside source_region_map");
                }
                const std::uint32_t region = source_regions[source];
                if (region == 0 || region == kUint32Max) {
                    throw std::invalid_argument(
                        "valid winners must have positive canonical regions"
                    );
                }
                const float score = source_scores[source];
                if (!std::isfinite(score) || score < 0.0F) {
                    throw std::invalid_argument(
                        "valid winner near scores must be finite and nonnegative"
                    );
                }
                if (lane_count == 0) {
                    first_region = region;
                } else if (region != first_region) {
                    pure = false;
                }
                for (std::uint32_t channel = 0; channel < 3; ++channel) {
                    sums[channel] += colour_values[
                        static_cast<std::uint64_t>(source) * 3 + channel
                    ];
                }
                ++lane_count;
            }
            coverage_values[pixel] = static_cast<std::uint8_t>(lane_count);
            pure_region_values[pixel] = kUint32Max;
            blue_values[pixel] = 0;
            green_values[pixel] = 0;
            red_values[pixel] = 0;
            if (lane_count != 0 && pure) {
                pure_region_values[pixel] = first_region;
                blue_values[pixel] = rounded_u8_mean(sums[0], lane_count);
                green_values[pixel] = rounded_u8_mean(sums[1], lane_count);
                red_values[pixel] = rounded_u8_mean(sums[2], lane_count);
            }
        }
        visit_repair_runs_with_accessors(
            height,
            fine_width,
            &statistics,
            valid_at,
            score_at,
            region_at,
            [](std::uint32_t,
               std::uint32_t,
               std::uint32_t,
               std::uint32_t,
               std::uint8_t) {}
        );
    }
    const bool caller_owned_output = !record_output_object.is_none();
    py::array records;
    if (caller_owned_output) {
        records = py::cast<py::array>(record_output_object);
        if (
            !records.dtype().is(py::dtype::of<std::uint8_t>()) ||
            records.ndim() != 2 || records.shape(1) != 16 ||
            (records.flags() & py::array::c_style) == 0 || !records.writeable()
        ) {
            throw py::type_error(
                "record_output must be a writable C-contiguous uint8 [capacity,16] array"
            );
        }
        if (static_cast<std::uint64_t>(records.shape(0)) < statistics.record_count) {
            throw QualityNativeBudgetError("64 MiB repair record arena exceeded");
        }
    } else {
        records = py::array_t<std::uint8_t>({
            static_cast<py::ssize_t>(statistics.record_count),
            static_cast<py::ssize_t>(16),
        });
    }
    auto* record_values = static_cast<std::uint8_t*>(records.mutable_data());
    std::fill(record_values, record_values + statistics.record_count * 16, 0);
    std::uint64_t cursor = 0;
    {
        py::gil_scoped_release release;
        RepairRecordStatistics second_pass{};
        visit_repair_runs_with_accessors(
            height,
            fine_width,
            &second_pass,
            valid_at,
            score_at,
            region_at,
            [&](std::uint32_t row,
                std::uint32_t start,
                std::uint32_t end,
                std::uint32_t region,
                std::uint8_t far_side) {
                write_repair_run_records_from_sources(
                    record_values,
                    &cursor,
                    row,
                    start,
                    end,
                    region,
                    far_side,
                    width,
                    fine_width,
                    winner_values,
                    colour_values
                );
            }
        );
        if (
            cursor != statistics.record_count ||
            second_pass.prefill_run_count != statistics.prefill_run_count ||
            second_pass.repair_run_count != statistics.repair_run_count ||
            second_pass.invalid_lane_count != statistics.invalid_lane_count
        ) {
            throw std::runtime_error("repair record preflight count changed");
        }
    }
    py::dict result_statistics;
    result_statistics["prefill_run_count"] = statistics.prefill_run_count;
    result_statistics["repair_run_count"] = statistics.repair_run_count;
    result_statistics["invalid_lane_count"] = statistics.invalid_lane_count;
    if (caller_owned_output) {
        return py::make_tuple(statistics.record_count, std::move(result_statistics));
    }
    return py::make_tuple(std::move(records), std::move(result_statistics));
}

bool is_safe_donor_pixel(
    const std::uint8_t* coverage,
    const std::uint32_t* regions,
    std::uint32_t height,
    std::uint32_t width,
    std::uint32_t pixel,
    std::uint32_t region,
    std::uint32_t radius
) {
    const std::uint32_t row = pixel / width;
    const std::uint32_t column = pixel % width;
    const std::uint32_t row_start = row > radius ? row - radius : 0;
    const std::uint32_t column_start = column > radius ? column - radius : 0;
    const std::uint32_t row_end = static_cast<std::uint32_t>(std::min<std::uint64_t>(
        height,
        static_cast<std::uint64_t>(row) + radius + 1
    ));
    const std::uint32_t column_end = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(
            width,
            static_cast<std::uint64_t>(column) + radius + 1
        )
    );
    for (std::uint32_t sample_row = row_start; sample_row < row_end; ++sample_row) {
        for (
            std::uint32_t sample_column = column_start;
            sample_column < column_end;
            ++sample_column
        ) {
            const std::uint32_t sample = sample_row * width + sample_column;
            if (coverage[sample] != 16 || regions[sample] != region) {
                return false;
            }
        }
    }
    return true;
}

py::tuple build_safe_donor_index(
    const py::array& coverage_count,
    const py::array& pure_region_id,
    const py::array& referenced_regions,
    bool include_full_pure_for_empty_regions
) {
    validate_array(
        coverage_count,
        py::dtype::of<std::uint8_t>(),
        "coverage_count"
    );
    validate_array(
        pure_region_id,
        py::dtype::of<std::uint32_t>(),
        "pure_region_id"
    );
    validate_vector(
        referenced_regions,
        py::dtype::of<bool>(),
        "referenced_regions"
    );
    if (
        pure_region_id.shape(0) != coverage_count.shape(0) ||
        pure_region_id.shape(1) != coverage_count.shape(1)
    ) {
        throw py::value_error("safe-donor analysis rasters must match");
    }
    if (referenced_regions.shape(0) < 2) {
        throw py::value_error("referenced_regions must include a positive region");
    }
    const std::uint64_t sample_count64 =
        static_cast<std::uint64_t>(coverage_count.shape(0)) *
        static_cast<std::uint64_t>(coverage_count.shape(1));
    const std::uint64_t region_count64 =
        static_cast<std::uint64_t>(referenced_regions.shape(0) - 1);
    if (sample_count64 + 1 > kUint32Max || region_count64 + 1 > kUint32Max) {
        throw py::value_error("safe-donor index shape must fit uint32");
    }

    const auto height = static_cast<std::uint32_t>(coverage_count.shape(0));
    const auto width = static_cast<std::uint32_t>(coverage_count.shape(1));
    const auto sample_count = static_cast<std::uint32_t>(sample_count64);
    const auto region_count = static_cast<std::uint32_t>(region_count64);
    const auto* coverage =
        static_cast<const std::uint8_t*>(coverage_count.data());
    const auto* regions =
        static_cast<const std::uint32_t*>(pure_region_id.data());
    const auto* referenced =
        static_cast<const std::uint8_t*>(referenced_regions.data());
    const auto radius = static_cast<std::uint32_t>(std::max(
        1.0,
        std::floor(static_cast<double>(height) / 1080.0 + 0.5)
    ));
    py::array_t<std::uint32_t> offsets(
        static_cast<py::ssize_t>(region_count) + 1
    );
    auto* offset_values = offsets.mutable_data();
    std::fill(offset_values, offset_values + region_count + 1, 0);
    std::vector<std::uint8_t> use_full_pure(region_count + 1, 0);
    std::uint32_t donor_count = 0;
    {
        py::gil_scoped_release release;
        for (std::uint32_t pixel = 0; pixel < sample_count; ++pixel) {
            const std::uint32_t region = regions[pixel];
            if (
                region == 0 ||
                region > region_count ||
                referenced[region] == 0 ||
                coverage[pixel] != 16
            ) {
                continue;
            }
            if (!is_safe_donor_pixel(
                coverage,
                regions,
                height,
                width,
                pixel,
                region,
                radius
            )) {
                continue;
            }
            ++offset_values[region - 1];
            ++donor_count;
        }
        if (include_full_pure_for_empty_regions) {
            for (std::uint32_t region = 1; region <= region_count; ++region) {
                if (referenced[region] != 0 && offset_values[region - 1] == 0) {
                    use_full_pure[region] = 1;
                }
            }
            for (std::uint32_t pixel = 0; pixel < sample_count; ++pixel) {
                const std::uint32_t region = regions[pixel];
                if (
                    region == 0 || region > region_count ||
                    use_full_pure[region] == 0 || coverage[pixel] != 16
                ) {
                    continue;
                }
                ++offset_values[region - 1];
                ++donor_count;
            }
        }
        std::uint32_t running = 0;
        for (std::uint32_t region_index = 0; region_index < region_count; ++region_index) {
            running += offset_values[region_index];
            offset_values[region_index] = running;
        }
        offset_values[region_count] = donor_count;
    }
    const std::uint64_t index_bytes =
        static_cast<std::uint64_t>(donor_count) * sizeof(std::uint32_t) +
        static_cast<std::uint64_t>(region_count + 1) * sizeof(std::uint32_t);
    if (index_bytes > 48ULL * 1024ULL * 1024ULL) {
        throw std::runtime_error("safe-donor index exceeds its 48 MiB partition");
    }

    py::array_t<std::uint32_t> members(donor_count);
    auto* member_values = members.mutable_data();
    {
        py::gil_scoped_release release;
        const bool packed_coordinates =
            height <= kPackedCoordinateLimit && width <= kPackedCoordinateLimit;
        for (std::uint32_t cursor = sample_count; cursor > 0; --cursor) {
            const std::uint32_t pixel = cursor - 1;
            const std::uint32_t region = regions[pixel];
            if (
                region == 0 ||
                region > region_count ||
                referenced[region] == 0 ||
                coverage[pixel] != 16
            ) {
                continue;
            }
            if (
                use_full_pure[region] == 0 &&
                !is_safe_donor_pixel(
                    coverage,
                    regions,
                    height,
                    width,
                    pixel,
                    region,
                    radius
                )
            ) {
                continue;
            }
            const std::uint32_t destination = --offset_values[region - 1];
            if (packed_coordinates) {
                const std::uint32_t member_row = pixel / width;
                const std::uint32_t member_column = pixel - member_row * width;
                member_values[destination] = (member_row << 16) | member_column;
            } else {
                member_values[destination] = pixel;
            }
        }
        for (std::uint32_t region_index = 0; region_index < region_count; ++region_index) {
            partition_region_tree(
                member_values,
                offset_values[region_index],
                offset_values[region_index + 1],
                height,
                width,
                packed_coordinates
            );
        }
        if (packed_coordinates) {
            for (std::uint32_t cursor = 0; cursor < donor_count; ++cursor) {
                const std::uint32_t packed = member_values[cursor];
                const std::uint32_t member_row = packed >> 16;
                const std::uint32_t member_column = packed & kPackedCoordinateLimit;
                member_values[cursor] = member_row * width + member_column;
            }
        }
    }
    return py::make_tuple(std::move(members), std::move(offsets));
}

py::tuple build_safe_donor_index_into(
    const py::array& coverage_count,
    const py::array& pure_region_id,
    const py::array& referenced_regions,
    bool include_full_pure_for_empty_regions,
    py::array members_output,
    py::array offsets_output,
    py::array use_full_output,
    const py::object& safe_cache_metadata_object
) {
    validate_array(coverage_count, py::dtype::of<std::uint8_t>(), "coverage_count");
    validate_array(pure_region_id, py::dtype::of<std::uint32_t>(), "pure_region_id");
    validate_vector(referenced_regions, py::dtype::of<bool>(), "referenced_regions");
    auto require_vector = [](py::array& value, const py::dtype& dtype, const char* name) {
        if (
            !value.dtype().is(dtype) || value.ndim() != 1 ||
            (value.flags() & py::array::c_style) == 0 || !value.writeable()
        ) {
            throw py::type_error(std::string(name) + " must be a writable contiguous vector");
        }
    };
    require_vector(members_output, py::dtype::of<std::uint32_t>(), "members_output");
    require_vector(offsets_output, py::dtype::of<std::uint32_t>(), "offsets_output");
    require_vector(use_full_output, py::dtype::of<bool>(), "use_full_output");
    if (
        pure_region_id.shape(0) != coverage_count.shape(0) ||
        pure_region_id.shape(1) != coverage_count.shape(1) ||
        referenced_regions.shape(0) < 2
    ) {
        throw py::value_error("safe-donor index inputs are inconsistent");
    }
    const std::uint64_t sample_count64 =
        static_cast<std::uint64_t>(coverage_count.shape(0)) * coverage_count.shape(1);
    const std::uint64_t region_count64 = referenced_regions.shape(0) - 1;
    if (sample_count64 + 1 > kUint32Max || region_count64 + 1 > kUint32Max) {
        throw py::value_error("safe-donor index shape must fit uint32");
    }
    const auto height = static_cast<std::uint32_t>(coverage_count.shape(0));
    const auto width = static_cast<std::uint32_t>(coverage_count.shape(1));
    const auto sample_count = static_cast<std::uint32_t>(sample_count64);
    const auto region_count = static_cast<std::uint32_t>(region_count64);
    if (
        offsets_output.shape(0) < static_cast<py::ssize_t>(region_count) + 1 ||
        use_full_output.shape(0) < static_cast<py::ssize_t>(region_count) + 1
    ) {
        throw std::runtime_error("safe-donor region descriptors exceed repair arena");
    }
    const auto* coverage = static_cast<const std::uint8_t*>(coverage_count.data());
    const auto* regions = static_cast<const std::uint32_t*>(pure_region_id.data());
    const auto* referenced = static_cast<const std::uint8_t*>(referenced_regions.data());
    auto* members = static_cast<std::uint32_t*>(members_output.mutable_data());
    auto* offsets = static_cast<std::uint32_t*>(offsets_output.mutable_data());
    auto* use_full = static_cast<std::uint8_t*>(use_full_output.mutable_data());
    std::uint64_t* safe_cache_metadata = nullptr;
    if (!safe_cache_metadata_object.is_none()) {
        py::array metadata = py::cast<py::array>(safe_cache_metadata_object);
        if (
            !metadata.dtype().is(py::dtype::of<std::uint64_t>()) ||
            metadata.ndim() != 1 || metadata.shape(0) != 2 ||
            (metadata.flags() & py::array::c_style) == 0 || !metadata.writeable()
        ) {
            throw py::type_error(
                "safe_cache_metadata must be a writable contiguous uint64[2] vector"
            );
        }
        safe_cache_metadata = static_cast<std::uint64_t*>(metadata.mutable_data());
        safe_cache_metadata[0] = std::numeric_limits<std::uint64_t>::max();
        safe_cache_metadata[1] = 0;
    }
    std::fill(offsets, offsets + region_count + 1, 0);
    std::fill(use_full, use_full + region_count + 1, 0);
    const std::uint64_t member_capacity = static_cast<std::uint64_t>(
        members_output.shape(0)
    );
    const std::uint64_t safe_word_count = (
        static_cast<std::uint64_t>(sample_count) + 31ULL
    ) / 32ULL;
    const bool can_stage_safe_bits =
        safe_cache_metadata != nullptr && safe_word_count <= member_capacity;
    auto* safe_words = can_stage_safe_bits
        ? members + (member_capacity - safe_word_count)
        : nullptr;
    if (can_stage_safe_bits) {
        std::fill(safe_words, safe_words + safe_word_count, 0U);
    }
    const auto radius = static_cast<std::uint32_t>(std::max(
        1.0,
        std::floor(static_cast<double>(height) / 1080.0 + 0.5)
    ));
    std::uint32_t donor_count = 0;
    {
        py::gil_scoped_release release;
        for (std::uint32_t pixel = 0; pixel < sample_count; ++pixel) {
            const std::uint32_t region = regions[pixel];
            if (
                region == 0 || region > region_count || referenced[region] == 0 ||
                coverage[pixel] != 16 || !is_safe_donor_pixel(
                    coverage, regions, height, width, pixel, region, radius
                )
            ) {
                continue;
            }
            if (can_stage_safe_bits) {
                safe_words[pixel / 32U] |= 1U << (pixel % 32U);
            }
            ++offsets[region - 1];
            ++donor_count;
        }
        if (include_full_pure_for_empty_regions) {
            for (std::uint32_t region = 1; region <= region_count; ++region) {
                if (referenced[region] != 0 && offsets[region - 1] == 0) {
                    use_full[region] = 1;
                }
            }
            for (std::uint32_t pixel = 0; pixel < sample_count; ++pixel) {
                const std::uint32_t region = regions[pixel];
                if (
                    region == 0 || region > region_count || use_full[region] == 0 ||
                    coverage[pixel] != 16
                ) {
                    continue;
                }
                ++offsets[region - 1];
                ++donor_count;
            }
        }
        const std::uint64_t index_bytes =
            static_cast<std::uint64_t>(donor_count) * sizeof(std::uint32_t) +
            static_cast<std::uint64_t>(region_count + 1) * sizeof(std::uint32_t) +
            static_cast<std::uint64_t>(region_count + 1);
        if (
            index_bytes > 48ULL * 1024ULL * 1024ULL ||
            members_output.shape(0) < donor_count
        ) {
            throw std::runtime_error("safe-donor index exceeds its 48 MiB partition");
        }
        const bool retain_safe_bits =
            can_stage_safe_bits &&
            static_cast<std::uint64_t>(donor_count) + safe_word_count <= member_capacity;
        if (retain_safe_bits) {
            safe_cache_metadata[0] = member_capacity - safe_word_count;
            safe_cache_metadata[1] = safe_word_count;
        }
        std::uint32_t running = 0;
        for (std::uint32_t region_index = 0; region_index < region_count; ++region_index) {
            running += offsets[region_index];
            offsets[region_index] = running;
        }
        offsets[region_count] = donor_count;
        const bool packed_coordinates =
            height <= kPackedCoordinateLimit && width <= kPackedCoordinateLimit;
        for (std::uint32_t cursor = sample_count; cursor > 0; --cursor) {
            const std::uint32_t pixel = cursor - 1;
            const std::uint32_t region = regions[pixel];
            if (
                region == 0 || region > region_count || referenced[region] == 0 ||
                coverage[pixel] != 16
            ) {
                continue;
            }
            if (
                use_full[region] == 0 &&
                !(
                    retain_safe_bits
                        ? (safe_words[pixel / 32U] & (1U << (pixel % 32U))) != 0
                        : is_safe_donor_pixel(
                            coverage, regions, height, width, pixel, region, radius
                        )
                )
            ) {
                continue;
            }
            const std::uint32_t destination = --offsets[region - 1];
            if (packed_coordinates) {
                const std::uint32_t row = pixel / width;
                members[destination] = (row << 16) | (pixel - row * width);
            } else {
                members[destination] = pixel;
            }
        }
        for (std::uint32_t region_index = 0; region_index < region_count; ++region_index) {
            partition_region_tree(
                members,
                offsets[region_index],
                offsets[region_index + 1],
                height,
                width,
                packed_coordinates
            );
        }
        if (packed_coordinates) {
            for (std::uint32_t cursor = 0; cursor < donor_count; ++cursor) {
                const std::uint32_t packed = members[cursor];
                members[cursor] = (packed >> 16) * width + (packed & kPackedCoordinateLimit);
            }
        }
    }
    const std::uint64_t reported_index_bytes =
        static_cast<std::uint64_t>(donor_count) * sizeof(std::uint32_t) +
        static_cast<std::uint64_t>(region_count + 1) * sizeof(std::uint32_t);
    return py::make_tuple(donor_count, reported_index_bytes);
}

std::uint64_t build_referenced_unplanned_regions_into_native(
    const py::array& raw_records,
    std::uint32_t pixel_count,
    std::uint32_t region_count,
    py::array referenced_output
) {
    if (
        !raw_records.dtype().is(py::dtype::of<std::uint8_t>()) ||
        raw_records.ndim() != 2 || raw_records.shape(1) != 16 ||
        (raw_records.flags() & py::array::c_style) == 0
    ) {
        throw py::type_error("raw_records must be a C-contiguous uint8 [n,16] array");
    }
    if (pixel_count == 0 || region_count == 0 || region_count == kUint32Max) {
        throw py::value_error("unplanned repair bounds must be positive uint32 values");
    }
    if (
        !referenced_output.dtype().is(py::dtype::of<bool>()) ||
        referenced_output.ndim() != 1 ||
        (referenced_output.flags() & py::array::c_style) == 0 ||
        !referenced_output.writeable() ||
        static_cast<std::uint64_t>(referenced_output.shape(0)) !=
            static_cast<std::uint64_t>(region_count) + 1ULL
    ) {
        throw py::type_error(
            "referenced_output must be a writable bool[region_count+1] vector"
        );
    }
    const auto* records = static_cast<const std::uint8_t*>(raw_records.data());
    auto* referenced = static_cast<std::uint8_t*>(referenced_output.mutable_data());
    std::uint64_t query_count = 0;
    {
        py::gil_scoped_release release;
        std::fill(referenced, referenced + region_count + 1, 0U);
        for (py::ssize_t index = 0; index < raw_records.shape(0); ++index) {
            const std::uint8_t* record = records + 16 * index;
            if (record[13] != 0) {
                continue;
            }
            std::uint32_t pixel = 0;
            std::uint32_t region = 0;
            std::memcpy(&pixel, record, sizeof(pixel));
            std::memcpy(&region, record + 6, sizeof(region));
            if (region == 0 || region > region_count) {
                throw py::value_error("unplanned record region exceeds region_count");
            }
            if (pixel >= pixel_count) {
                throw py::value_error("record pixel lies outside render_shape");
            }
            referenced[region] = 1U;
            ++query_count;
        }
    }
    return query_count;
}

void expand_safe_donor_cache_into_native(
    const py::array& safe_words,
    const py::array& pure_region_id,
    py::array safe_output,
    py::array region_output
) {
    if (
        !safe_words.dtype().is(py::dtype::of<std::uint32_t>()) ||
        safe_words.ndim() != 1 ||
        (safe_words.flags() & py::array::c_style) == 0
    ) {
        throw py::type_error("safe_words must be a contiguous uint32 vector");
    }
    if (
        !pure_region_id.dtype().is(py::dtype::of<std::uint32_t>()) ||
        pure_region_id.ndim() != 2 ||
        (pure_region_id.flags() & py::array::c_style) == 0
    ) {
        throw py::type_error("pure_region_id must be a contiguous uint32 image");
    }
    if (
        !safe_output.dtype().is(py::dtype::of<bool>()) || safe_output.ndim() != 2 ||
        (safe_output.flags() & py::array::c_style) == 0 || !safe_output.writeable() ||
        safe_output.shape(0) != pure_region_id.shape(0) ||
        safe_output.shape(1) != pure_region_id.shape(1)
    ) {
        throw py::type_error(
            "safe_output must be a matching writable contiguous bool image"
        );
    }
    if (
        !region_output.dtype().is(py::dtype::of<bool>()) ||
        region_output.ndim() != 1 || region_output.shape(0) < 2 ||
        (region_output.flags() & py::array::c_style) == 0 ||
        !region_output.writeable()
    ) {
        throw py::type_error(
            "region_output must be a writable contiguous bool region vector"
        );
    }
    const std::uint64_t pixel_count =
        static_cast<std::uint64_t>(pure_region_id.shape(0)) *
        static_cast<std::uint64_t>(pure_region_id.shape(1));
    const std::uint64_t expected_words = (pixel_count + 31ULL) / 32ULL;
    if (static_cast<std::uint64_t>(safe_words.shape(0)) != expected_words) {
        throw py::value_error("safe_words length does not match pure_region_id");
    }
    const auto* words = static_cast<const std::uint32_t*>(safe_words.data());
    const auto* regions = static_cast<const std::uint32_t*>(pure_region_id.data());
    auto* safe = static_cast<std::uint8_t*>(safe_output.mutable_data());
    auto* region_has_safe = static_cast<std::uint8_t*>(region_output.mutable_data());
    const auto overlaps = [](
        const void* left,
        std::uint64_t left_bytes,
        const void* right,
        std::uint64_t right_bytes
    ) {
        const auto left_start = reinterpret_cast<std::uintptr_t>(left);
        const auto right_start = reinterpret_cast<std::uintptr_t>(right);
        return left_start < right_start + right_bytes &&
            right_start < left_start + left_bytes;
    };
    if (
        overlaps(words, expected_words * 4ULL, safe, pixel_count) ||
        overlaps(
            words,
            expected_words * 4ULL,
            region_has_safe,
            static_cast<std::uint64_t>(region_output.shape(0))
        ) ||
        overlaps(
            safe,
            pixel_count,
            region_has_safe,
            static_cast<std::uint64_t>(region_output.shape(0))
        )
    ) {
        throw py::value_error("safe-donor cache expansion arrays overlap");
    }
    const std::uint64_t region_count =
        static_cast<std::uint64_t>(region_output.shape(0) - 1);
    {
        py::gil_scoped_release release;
        std::fill(region_has_safe, region_has_safe + region_output.shape(0), 0U);
        for (std::uint64_t pixel = 0; pixel < pixel_count; ++pixel) {
            const bool is_safe =
                (words[pixel / 32ULL] & (1U << (pixel % 32ULL))) != 0;
            safe[pixel] = static_cast<std::uint8_t>(is_safe);
            if (!is_safe) {
                continue;
            }
            const std::uint32_t region = regions[pixel];
            if (region == 0 || region > region_count) {
                throw py::value_error("safe-donor cache references an invalid region");
            }
            region_has_safe[region] = 1U;
        }
    }
}

py::tuple build_safe_donor_mask(
    const py::array& coverage_count,
    const py::array& pure_region_id,
    const py::array& referenced_regions,
    py::object safe_output_object,
    py::object region_output_object
) {
    validate_array(
        coverage_count,
        py::dtype::of<std::uint8_t>(),
        "coverage_count"
    );
    validate_array(
        pure_region_id,
        py::dtype::of<std::uint32_t>(),
        "pure_region_id"
    );
    validate_vector(
        referenced_regions,
        py::dtype::of<bool>(),
        "referenced_regions"
    );
    if (
        pure_region_id.shape(0) != coverage_count.shape(0) ||
        pure_region_id.shape(1) != coverage_count.shape(1)
    ) {
        throw py::value_error("safe-donor analysis rasters must match");
    }
    if (referenced_regions.shape(0) < 2) {
        throw py::value_error("referenced_regions must include a positive region");
    }
    const std::uint64_t sample_count64 =
        static_cast<std::uint64_t>(coverage_count.shape(0)) *
        static_cast<std::uint64_t>(coverage_count.shape(1));
    const std::uint64_t region_count64 =
        static_cast<std::uint64_t>(referenced_regions.shape(0) - 1);
    if (sample_count64 > kUint32Max || region_count64 + 1 > kUint32Max) {
        throw py::value_error("safe-donor mask shape must fit uint32");
    }
    const auto height = static_cast<std::uint32_t>(coverage_count.shape(0));
    const auto width = static_cast<std::uint32_t>(coverage_count.shape(1));
    const auto sample_count = static_cast<std::uint32_t>(sample_count64);
    const auto region_count = static_cast<std::uint32_t>(region_count64);
    const auto* coverage =
        static_cast<const std::uint8_t*>(coverage_count.data());
    const auto* regions =
        static_cast<const std::uint32_t*>(pure_region_id.data());
    const auto* referenced =
        static_cast<const std::uint8_t*>(referenced_regions.data());
    const auto radius = static_cast<std::uint32_t>(std::max(
        1.0,
        std::floor(static_cast<double>(height) / 1080.0 + 0.5)
    ));
    if (safe_output_object.is_none() != region_output_object.is_none()) {
        throw py::value_error("safe donor outputs must be supplied together");
    }
    py::array safe;
    py::array region_has_safe_donor;
    if (safe_output_object.is_none()) {
        safe = py::array_t<bool>({coverage_count.shape(0), coverage_count.shape(1)});
        region_has_safe_donor = py::array_t<bool>(referenced_regions.shape(0));
    } else {
        safe = py::cast<py::array>(safe_output_object);
        region_has_safe_donor = py::cast<py::array>(region_output_object);
        if (
            !safe.dtype().is(py::dtype::of<bool>()) || safe.ndim() != 2 ||
            safe.shape(0) != coverage_count.shape(0) ||
            safe.shape(1) != coverage_count.shape(1) ||
            (safe.flags() & py::array::c_style) == 0 || !safe.writeable()
        ) {
            throw py::type_error("safe_output must be a writable matching bool raster");
        }
        if (
            !region_has_safe_donor.dtype().is(py::dtype::of<bool>()) ||
            region_has_safe_donor.ndim() != 1 ||
            region_has_safe_donor.shape(0) < referenced_regions.shape(0) ||
            (region_has_safe_donor.flags() & py::array::c_style) == 0 ||
            !region_has_safe_donor.writeable()
        ) {
            throw py::type_error("region_output must cover every referenced region");
        }
    }
    auto* safe_values = static_cast<bool*>(safe.mutable_data());
    auto* region_values = static_cast<bool*>(region_has_safe_donor.mutable_data());
    {
        py::gil_scoped_release release;
        std::fill(safe_values, safe_values + sample_count, false);
        std::fill(region_values, region_values + region_count + 1, false);
        for (std::uint32_t pixel = 0; pixel < sample_count; ++pixel) {
            const std::uint32_t region = regions[pixel];
            if (
                region == 0 ||
                region > region_count ||
                referenced[region] == 0 ||
                coverage[pixel] != 16 ||
                !is_safe_donor_pixel(
                    coverage,
                    regions,
                    height,
                    width,
                    pixel,
                    region,
                    radius
                )
            ) {
                continue;
            }
            safe_values[pixel] = true;
            region_values[region] = true;
        }
    }
    return py::make_tuple(std::move(safe), std::move(region_has_safe_donor));
}

std::uint32_t read_little_u32(const std::uint8_t* source) {
    return
        static_cast<std::uint32_t>(source[0]) |
        (static_cast<std::uint32_t>(source[1]) << 8) |
        (static_cast<std::uint32_t>(source[2]) << 16) |
        (static_cast<std::uint32_t>(source[3]) << 24);
}

py::tuple query_fallback_records(
    const py::array& members,
    const py::array& offsets,
    std::uint64_t height,
    std::uint64_t width,
    py::array raw_records,
    const py::array& blue,
    const py::array& green,
    const py::array& red,
    std::uint64_t radius_px,
    py::array budget_state,
    bool allow_seeded_missing,
    py::object visits_output_object
) {
    validate_index_arrays(members, offsets, height, width);
    if (
        !raw_records.dtype().is(py::dtype::of<std::uint8_t>()) ||
        raw_records.ndim() != 2 ||
        raw_records.shape(1) != 16 ||
        (raw_records.flags() & py::array::c_style) == 0 ||
        !raw_records.writeable()
    ) {
        throw py::value_error("raw_records must be a writable uint8 [N,16] array");
    }
    for (const auto* plane : {&blue, &green, &red}) {
        validate_array(*plane, py::dtype::of<std::uint8_t>(), "pure_bgr plane");
        if (
            plane->shape(0) != static_cast<py::ssize_t>(height) ||
            plane->shape(1) != static_cast<py::ssize_t>(width)
        ) {
            throw py::value_error("pure_bgr planes must match the indexed raster");
        }
    }
    validate_budget_state(budget_state);
    if (radius_px > std::numeric_limits<std::uint32_t>::max()) {
        throw py::value_error("fallback radius must fit uint32");
    }
    std::uint32_t query_count = 0;
    const auto* input_records =
        static_cast<const std::uint8_t*>(raw_records.data());
    for (py::ssize_t index = 0; index < raw_records.shape(0); ++index) {
        if (input_records[index * 16 + 13] == 0) {
            ++query_count;
        }
    }
    if (
        static_cast<std::uint64_t>(query_count) * sizeof(std::uint32_t) >
        16ULL * 1024ULL * 1024ULL
    ) {
        throw std::runtime_error(
            "fallback visit counts exceed their 16 MiB partition"
        );
    }

    const bool caller_owned_visits = !visits_output_object.is_none();
    py::array visits;
    if (caller_owned_visits) {
        visits = py::cast<py::array>(visits_output_object);
        if (
            !visits.dtype().is(py::dtype::of<std::uint32_t>()) || visits.ndim() != 1 ||
            (visits.flags() & py::array::c_style) == 0 || !visits.writeable() ||
            visits.shape(0) < query_count
        ) {
            throw py::type_error(
                "visits_output must be a writable uint32 vector covering every query"
            );
        }
    } else {
        visits = py::array_t<std::uint32_t>(query_count);
    }
    auto* visit_values = static_cast<std::uint32_t*>(visits.mutable_data());
    auto* record_values =
        static_cast<std::uint8_t*>(raw_records.mutable_data());
    const auto* member_values =
        static_cast<const std::uint32_t*>(members.data());
    const auto* offset_values =
        static_cast<const std::uint32_t*>(offsets.data());
    const auto* blue_values = static_cast<const std::uint8_t*>(blue.data());
    const auto* green_values = static_cast<const std::uint8_t*>(green.data());
    const auto* red_values = static_cast<const std::uint8_t*>(red.data());
    auto* budget_values =
        static_cast<std::uint64_t*>(budget_state.mutable_data());
    const auto region_count = static_cast<std::uint32_t>(offsets.shape(0) - 1);
    const auto width32 = static_cast<std::uint32_t>(width);
    const U32Divider width_divider(width32);
    const std::uint64_t radius2 = radius_px * radius_px;
    std::uint32_t query = 0;
    std::uint32_t seeded_missing = 0;
    {
        py::gil_scoped_release release;
        for (py::ssize_t index = 0; index < raw_records.shape(0); ++index) {
            std::uint8_t* record = record_values + index * 16;
            if (record[13] != 0) {
                continue;
            }
            const std::uint32_t pixel = read_little_u32(record);
            const std::uint32_t region = read_little_u32(record + 6);
            if (pixel >= height * width || region == 0 || region > region_count) {
                throw std::invalid_argument("fallback record identity is invalid");
            }
            const std::uint32_t pixel_y = width_divider.divide(pixel);
            const std::uint32_t pixel_x = pixel - pixel_y * width32;
            const RepairBest best = query_repair_one(
                member_values,
                offset_values[region - 1],
                offset_values[region],
                width32,
                width_divider,
                pixel_y,
                pixel_x,
                radius2,
                budget_values,
                &visit_values[query]
            );
            if (!best.found) {
                if (!allow_seeded_missing) {
                    throw std::runtime_error(
                        "repair record has no safe donor within fallback limit"
                    );
                }
                ++seeded_missing;
                ++query;
                continue;
            }
            record[10] = blue_values[best.sample_index];
            record[11] = green_values[best.sample_index];
            record[12] = red_values[best.sample_index];
            ++query;
        }
    }
    if (caller_owned_visits) {
        return py::make_tuple(query_count, seeded_missing);
    }
    return py::make_tuple(std::move(visits), seeded_missing);
}

py::tuple build_index(
    const py::array& region_ids,
    std::uint64_t region_count_value,
    const py::object& include_object
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
            if (include == nullptr || include[sample_index] != 0) {
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
            if (include == nullptr || include[sample_index] != 0) {
                const std::uint32_t region_slot = regions[sample_index] - 1;
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
            std::vector<std::thread> workers;
            workers.reserve(worker_count - 1);
            try {
                for (std::size_t index = 1; index < worker_count; ++index) {
                    workers.emplace_back(worker, index);
                }
            } catch (...) {
                for (auto& thread : workers) {
                    thread.join();
                }
                throw;
            }
            worker(0);
            for (auto& thread : workers) {
                thread.join();
            }
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

inline std::int64_t reflect_101(std::int64_t index, std::int64_t length) {
    if (length == 1) {
        return 0;
    }
    const std::int64_t period = 2 * length - 2;
    std::int64_t reflected = index % period;
    if (reflected < 0) {
        reflected += period;
    }
    return reflected < length ? reflected : period - reflected;
}

inline std::int32_t exemplar_luma(const std::uint8_t* pixel) {
    return (
        29 * static_cast<std::int32_t>(pixel[0]) +
        150 * static_cast<std::int32_t>(pixel[1]) +
        77 * static_cast<std::int32_t>(pixel[2]) +
        128
    ) >> 8;
}

std::pair<std::int32_t, std::int32_t> exemplar_sobel(
    const std::uint8_t* working,
    std::int64_t height,
    std::int64_t width,
    std::int64_t row,
    std::int64_t column,
    bool reflected
) {
    std::array<std::int32_t, 9> luma{};
    std::size_t cursor = 0;
    for (std::int64_t delta_row = -1; delta_row <= 1; ++delta_row) {
        const std::int64_t sample_row = reflected
            ? reflect_101(row + delta_row, height)
            : row + delta_row;
        for (std::int64_t delta_column = -1; delta_column <= 1; ++delta_column) {
            const std::int64_t sample_column = reflected
                ? reflect_101(column + delta_column, width)
                : column + delta_column;
            const auto sample = static_cast<std::uint64_t>(sample_row * width + sample_column);
            luma[cursor++] = exemplar_luma(working + 3 * sample);
        }
    }
    const std::int32_t gx =
        -luma[0] + luma[2] - 2 * luma[3] + 2 * luma[5] - luma[6] + luma[8];
    const std::int32_t gy =
        -luma[0] - 2 * luma[1] - luma[2] + luma[6] + 2 * luma[7] + luma[8];
    return {gx, gy};
}

py::tuple build_exemplar_sobel_planes(
    const py::array& working_bgr,
    py::object gradient_x_object,
    py::object gradient_y_object
) {
    if (
        !working_bgr.dtype().is(py::dtype::of<std::uint8_t>()) ||
        working_bgr.ndim() != 3 ||
        working_bgr.shape(0) <= 0 ||
        working_bgr.shape(1) <= 0 ||
        working_bgr.shape(2) != 3 ||
        (working_bgr.flags() & py::array::c_style) == 0
    ) {
        throw py::type_error("working_bgr must be a C-contiguous uint8 [h,w,3] array");
    }
    const std::int64_t height = working_bgr.shape(0);
    const std::int64_t width = working_bgr.shape(1);
    const auto* working = static_cast<const std::uint8_t*>(working_bgr.data());
    if (gradient_x_object.is_none() != gradient_y_object.is_none()) {
        throw py::value_error("exemplar gradient outputs must be supplied together");
    }
    py::array gradient_x;
    py::array gradient_y;
    if (gradient_x_object.is_none()) {
        gradient_x = py::array_t<std::int16_t>({height, width});
        gradient_y = py::array_t<std::int16_t>({height, width});
    } else {
        gradient_x = py::cast<py::array>(gradient_x_object);
        gradient_y = py::cast<py::array>(gradient_y_object);
        for (py::array* output : {&gradient_x, &gradient_y}) {
            if (
                !output->dtype().is(py::dtype::of<std::int16_t>()) ||
                output->ndim() != 2 || output->shape(0) != height ||
                output->shape(1) != width ||
                (output->flags() & py::array::c_style) == 0 || !output->writeable()
            ) {
                throw py::type_error(
                    "exemplar gradient outputs must be writable matching int16 rasters"
                );
            }
        }
    }
    auto* output_x = static_cast<std::int16_t*>(gradient_x.mutable_data());
    auto* output_y = static_cast<std::int16_t*>(gradient_y.mutable_data());
    {
        py::gil_scoped_release release;
        for (std::int64_t row = 0; row < height; ++row) {
            for (std::int64_t column = 0; column < width; ++column) {
                const auto gradient = exemplar_sobel(
                    working,
                    height,
                    width,
                    row,
                    column,
                    true
                );
                const auto sample = static_cast<std::uint64_t>(row * width + column);
                output_x[sample] = static_cast<std::int16_t>(gradient.first);
                output_y[sample] = static_cast<std::int16_t>(gradient.second);
            }
        }
    }
    return py::make_tuple(std::move(gradient_x), std::move(gradient_y));
}

struct ExemplarTargetSample {
    std::int64_t delta_row = 0;
    std::int64_t delta_column = 0;
    std::uint64_t sample = 0;
    std::array<std::uint8_t, 3> bgr{};
    bool gradient_known = false;
    std::int32_t gradient_x = 0;
    std::int32_t gradient_y = 0;
};

py::tuple select_exemplar_donor(
    const py::array& working_bgr,
    const py::array& processed_mask,
    const py::array& barrier_mask,
    std::int64_t target_row,
    std::int64_t target_column,
    const py::array& candidates,
    const py::array& selected_indexes,
    const py::object& donor_gradient_x_object,
    const py::object& donor_gradient_y_object
) {
    if (
        !working_bgr.dtype().is(py::dtype::of<std::uint8_t>()) ||
        working_bgr.ndim() != 3 ||
        working_bgr.shape(0) <= 0 ||
        working_bgr.shape(1) <= 0 ||
        working_bgr.shape(2) != 3 ||
        (working_bgr.flags() & py::array::c_style) == 0
    ) {
        throw py::type_error("working_bgr must be a C-contiguous uint8 [h,w,3] array");
    }
    validate_array(processed_mask, py::dtype::of<bool>(), "processed_mask");
    validate_array(barrier_mask, py::dtype::of<bool>(), "barrier_mask");
    const std::int64_t height = working_bgr.shape(0);
    const std::int64_t width = working_bgr.shape(1);
    if (
        processed_mask.shape(0) != height || processed_mask.shape(1) != width ||
        barrier_mask.shape(0) != height || barrier_mask.shape(1) != width
    ) {
        throw py::value_error("exemplar masks must match working_bgr");
    }
    if (
        !candidates.dtype().is(py::dtype::of<std::uint32_t>()) ||
        candidates.ndim() != 2 || candidates.shape(1) != 2 ||
        (candidates.flags() & py::array::c_style) == 0
    ) {
        throw py::type_error("candidates must be a C-contiguous uint32 [n,2] array");
    }
    validate_vector(
        selected_indexes,
        py::dtype::of<std::uint32_t>(),
        "selected_indexes"
    );
    if (
        target_row < 3 || target_row >= height - 3 ||
        target_column < 3 || target_column >= width - 3
    ) {
        throw py::value_error("target centre has incomplete exemplar support");
    }
    if (selected_indexes.shape(0) == 0) {
        throw py::value_error("selected_indexes cannot be empty");
    }

    py::array donor_gradient_x;
    py::array donor_gradient_y;
    const std::int16_t* donor_gradient_x_values = nullptr;
    const std::int16_t* donor_gradient_y_values = nullptr;
    if (donor_gradient_x_object.is_none() != donor_gradient_y_object.is_none()) {
        throw py::value_error("both donor gradient planes must be supplied together");
    }
    if (!donor_gradient_x_object.is_none()) {
        donor_gradient_x = py::cast<py::array>(donor_gradient_x_object);
        donor_gradient_y = py::cast<py::array>(donor_gradient_y_object);
        validate_array(
            donor_gradient_x,
            py::dtype::of<std::int16_t>(),
            "donor_gradient_x"
        );
        validate_array(
            donor_gradient_y,
            py::dtype::of<std::int16_t>(),
            "donor_gradient_y"
        );
        if (
            donor_gradient_x.shape(0) != height || donor_gradient_x.shape(1) != width ||
            donor_gradient_y.shape(0) != height || donor_gradient_y.shape(1) != width
        ) {
            throw py::value_error("donor gradient planes must match working_bgr");
        }
        donor_gradient_x_values = static_cast<const std::int16_t*>(
            donor_gradient_x.data()
        );
        donor_gradient_y_values = static_cast<const std::int16_t*>(
            donor_gradient_y.data()
        );
    }

    const auto* working = static_cast<const std::uint8_t*>(working_bgr.data());
    const auto* processed = static_cast<const bool*>(processed_mask.data());
    const auto* barrier = static_cast<const bool*>(barrier_mask.data());
    const auto* candidate_values = static_cast<const std::uint32_t*>(candidates.data());
    const auto* selected_values = static_cast<const std::uint32_t*>(selected_indexes.data());
    const auto candidate_count = static_cast<std::uint64_t>(candidates.shape(0));
    bool found = false;
    std::uint64_t best_score = 0;
    std::uint32_t best_row = 0;
    std::uint32_t best_column = 0;

    std::array<ExemplarTargetSample, 49> target_samples{};
    std::size_t target_sample_count = 0;
    for (std::int64_t delta_row = -3; delta_row <= 3; ++delta_row) {
        for (std::int64_t delta_column = -3; delta_column <= 3; ++delta_column) {
            const std::int64_t row = target_row + delta_row;
            const std::int64_t column = target_column + delta_column;
            const auto target_sample = static_cast<std::uint64_t>(row * width + column);
            if (!processed[target_sample]) {
                continue;
            }
            ExemplarTargetSample& target = target_samples[target_sample_count++];
            target.delta_row = delta_row;
            target.delta_column = delta_column;
            target.sample = target_sample;
            for (std::size_t channel = 0; channel < 3; ++channel) {
                target.bgr[channel] = working[3 * target_sample + channel];
            }
            target.gradient_known = true;
            for (std::int64_t gradient_row = -1; gradient_row <= 1; ++gradient_row) {
                const std::int64_t mapped_row = reflect_101(row + gradient_row, height);
                for (
                    std::int64_t gradient_column = -1;
                    gradient_column <= 1;
                    ++gradient_column
                ) {
                    const std::int64_t mapped_column = reflect_101(
                        column + gradient_column,
                        width
                    );
                    const auto mapped = static_cast<std::uint64_t>(
                        mapped_row * width + mapped_column
                    );
                    if (barrier[mapped] || !processed[mapped]) {
                        target.gradient_known = false;
                    }
                }
            }
            if (target.gradient_known) {
                const auto gradient = exemplar_sobel(
                    working,
                    height,
                    width,
                    row,
                    column,
                    true
                );
                target.gradient_x = gradient.first;
                target.gradient_y = gradient.second;
            }
        }
    }

    {
        py::gil_scoped_release release;
        for (py::ssize_t ordinal = 0; ordinal < selected_indexes.shape(0); ++ordinal) {
            const std::uint32_t candidate_index = selected_values[ordinal];
            if (candidate_index >= candidate_count) {
                throw py::value_error("selected exemplar candidate index is out of range");
            }
            const std::uint32_t donor_row = candidate_values[2 * candidate_index];
            const std::uint32_t donor_column = candidate_values[2 * candidate_index + 1];
            if (
                donor_row < 4 || donor_row >= static_cast<std::uint64_t>(height - 4) ||
                donor_column < 4 || donor_column >= static_cast<std::uint64_t>(width - 4)
            ) {
                throw py::value_error("donor centre has incomplete exemplar support");
            }
            std::uint64_t score = 0;
            for (std::size_t index = 0; index < target_sample_count; ++index) {
                const ExemplarTargetSample& target = target_samples[index];
                const auto donor_sample = static_cast<std::uint64_t>(
                    (static_cast<std::int64_t>(donor_row) + target.delta_row) * width +
                    static_cast<std::int64_t>(donor_column) + target.delta_column
                );
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    score += 2ULL * static_cast<std::uint64_t>(std::abs(
                        static_cast<std::int32_t>(target.bgr[channel]) -
                        static_cast<std::int32_t>(working[3 * donor_sample + channel])
                    ));
                }
                if (target.gradient_known) {
                    std::int32_t donor_x = 0;
                    std::int32_t donor_y = 0;
                    if (donor_gradient_x_values != nullptr) {
                        donor_x = donor_gradient_x_values[donor_sample];
                        donor_y = donor_gradient_y_values[donor_sample];
                    } else {
                        const auto donor_gradient = exemplar_sobel(
                            working,
                            height,
                            width,
                            static_cast<std::int64_t>(donor_row) + target.delta_row,
                            static_cast<std::int64_t>(donor_column) + target.delta_column,
                            false
                        );
                        donor_x = donor_gradient.first;
                        donor_y = donor_gradient.second;
                    }
                    score += static_cast<std::uint64_t>(std::abs(
                        target.gradient_x - donor_x
                    ));
                    score += static_cast<std::uint64_t>(std::abs(
                        target.gradient_y - donor_y
                    ));
                }
                const bool donor_precedes_best =
                    donor_row < best_row ||
                    (donor_row == best_row && donor_column < best_column);
                if (
                    found &&
                    (score > best_score || (score == best_score && !donor_precedes_best))
                ) {
                    break;
                }
            }
            if (
                !found || score < best_score ||
                (score == best_score && donor_row < best_row) ||
                (score == best_score && donor_row == best_row && donor_column < best_column)
            ) {
                found = true;
                best_score = score;
                best_row = donor_row;
                best_column = donor_column;
            }
        }
    }
    return py::make_tuple(best_score, best_row, best_column);
}

inline std::uint32_t load_u32_le(const std::uint8_t* source) {
    std::uint32_t value = 0;
    std::memcpy(&value, source, sizeof(value));
    return value;
}

inline std::uint16_t load_u16_le(const std::uint8_t* source) {
    std::uint16_t value = 0;
    std::memcpy(&value, source, sizeof(value));
    return value;
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

inline bool is_contiguous_u16_mask(std::uint16_t value) {
    if (value == 0) {
        return false;
    }
    const std::uint16_t shifted = static_cast<std::uint16_t>(
        value >> least_u16_bit(value)
    );
    return (shifted & static_cast<std::uint16_t>(shifted + 1U)) == 0;
}

std::uint64_t collect_exemplar_targets_into_native(
    const py::array& raw_records,
    const py::array& record_indexes,
    std::uint32_t region,
    std::uint32_t frame_height,
    std::uint32_t frame_width,
    py::array target_pixels,
    const py::object& target_colours_object
) {
    if (
        !raw_records.dtype().is(py::dtype::of<std::uint8_t>()) ||
        raw_records.ndim() != 2 || raw_records.shape(1) != 16 ||
        (raw_records.flags() & py::array::c_style) == 0
    ) {
        throw py::type_error("raw_records must be a C-contiguous uint8 [n,16] array");
    }
    validate_vector(record_indexes, py::dtype::of<std::uint32_t>(), "record_indexes");
    if (
        !target_pixels.dtype().is(py::dtype::of<std::uint32_t>()) ||
        target_pixels.ndim() != 1 ||
        (target_pixels.flags() & py::array::c_style) == 0 ||
        !target_pixels.writeable() ||
        target_pixels.shape(0) < record_indexes.shape(0)
    ) {
        throw py::type_error(
            "target_pixels must be a writable uint32 vector with record capacity"
        );
    }
    if (frame_height == 0 || frame_width == 0 || region == 0 || region == kUint32Max) {
        throw py::value_error("exemplar target identity is invalid");
    }
    const std::uint64_t pixel_count =
        static_cast<std::uint64_t>(frame_height) * frame_width;
    if (pixel_count > kUint32Max) {
        throw py::value_error("exemplar frame exceeds uint32 pixels");
    }
    py::array target_colours;
    std::uint8_t* colour_values = nullptr;
    if (!target_colours_object.is_none()) {
        target_colours = py::cast<py::array>(target_colours_object);
        if (
            !target_colours.dtype().is(py::dtype::of<std::uint8_t>()) ||
            target_colours.ndim() != 2 || target_colours.shape(1) != 3 ||
            target_colours.shape(0) < record_indexes.shape(0) ||
            (target_colours.flags() & py::array::c_style) == 0 ||
            !target_colours.writeable()
        ) {
            throw py::type_error(
                "target_colours must be a writable uint8 [record_capacity,3] array"
            );
        }
        colour_values = static_cast<std::uint8_t*>(target_colours.mutable_data());
    }
    const auto* records = static_cast<const std::uint8_t*>(raw_records.data());
    const auto* indexes = static_cast<const std::uint32_t*>(record_indexes.data());
    auto* pixels = static_cast<std::uint32_t*>(target_pixels.mutable_data());
    std::uint64_t target_count = 0;
    bool have_previous = false;
    std::uint32_t previous_pixel = 0;
    {
        py::gil_scoped_release release;
        for (py::ssize_t ordinal = 0; ordinal < record_indexes.shape(0); ++ordinal) {
            const std::uint32_t record_index = indexes[ordinal];
            if (record_index >= static_cast<std::uint64_t>(raw_records.shape(0))) {
                throw std::runtime_error("exemplar record index lies outside its arena");
            }
            const std::uint8_t* record = records + 16ULL * record_index;
            const std::uint32_t pixel = load_u32_le(record);
            if (pixel >= pixel_count) {
                throw std::runtime_error("exemplar target pixel lies outside the frame");
            }
            if (have_previous && pixel < previous_pixel) {
                throw std::runtime_error("exemplar component records are not row-major");
            }
            previous_pixel = pixel;
            have_previous = true;
            if (record[13] != 0) {
                continue;
            }
            if (load_u32_le(record + 6) != region) {
                throw std::runtime_error("exemplar component crosses a region");
            }
            if (target_count != 0 && pixels[target_count - 1] == pixel) {
                if (
                    colour_values != nullptr &&
                    (
                        colour_values[3 * (target_count - 1)] != record[10] ||
                        colour_values[3 * (target_count - 1) + 1] != record[11] ||
                        colour_values[3 * (target_count - 1) + 2] != record[12]
                    )
                ) {
                    throw std::runtime_error(
                        "records for one exemplar target disagree on colour"
                    );
                }
                continue;
            }
            pixels[target_count] = pixel;
            if (colour_values != nullptr) {
                colour_values[3 * target_count] = record[10];
                colour_values[3 * target_count + 1] = record[11];
                colour_values[3 * target_count + 2] = record[12];
            }
            ++target_count;
        }
    }
    return target_count;
}

std::uint64_t prepare_exemplar_full_level_into_native(
    const py::array& coverage_count,
    const py::array& pure_region_id,
    const py::array& blue,
    const py::array& green,
    const py::array& red,
    const py::array& safe_donor_mask,
    const py::array& target_pixels,
    const py::array& target_colours,
    const py::array& completed,
    const py::array& completed_colours,
    std::uint32_t region,
    std::uint32_t y0,
    std::uint32_t x0,
    std::uint32_t interior_y0,
    std::uint32_t interior_y1,
    std::uint32_t interior_x0,
    std::uint32_t interior_x1,
    py::array working,
    py::array donor,
    py::array target,
    py::array processed,
    py::array barrier
) {
    validate_array(coverage_count, py::dtype::of<std::uint8_t>(), "coverage_count");
    validate_array(pure_region_id, py::dtype::of<std::uint32_t>(), "pure_region_id");
    validate_array(blue, py::dtype::of<std::uint8_t>(), "blue");
    validate_array(green, py::dtype::of<std::uint8_t>(), "green");
    validate_array(red, py::dtype::of<std::uint8_t>(), "red");
    validate_array(safe_donor_mask, py::dtype::of<bool>(), "safe_donor_mask");
    validate_vector(target_pixels, py::dtype::of<std::uint32_t>(), "target_pixels");
    validate_array(target_colours, py::dtype::of<std::uint8_t>(), "target_colours");
    validate_vector(completed, py::dtype::of<bool>(), "completed");
    validate_array(completed_colours, py::dtype::of<std::uint8_t>(), "completed_colours");
    const py::ssize_t frame_height = coverage_count.shape(0);
    const py::ssize_t frame_width = coverage_count.shape(1);
    if (
        frame_height <= 0 || frame_width <= 0 ||
        static_cast<std::uint64_t>(frame_height) > kUint32Max ||
        static_cast<std::uint64_t>(frame_width) > kUint32Max ||
        static_cast<std::uint64_t>(frame_height) *
            static_cast<std::uint64_t>(frame_width) > kUint32Max
    ) {
        throw py::value_error("exemplar source shape must fit uint32 pixels");
    }
    const auto frame_height_u32 = static_cast<std::uint32_t>(frame_height);
    const auto frame_width_u32 = static_cast<std::uint32_t>(frame_width);
    for (const py::array* raster : {
        &pure_region_id, &blue, &green, &red, &safe_donor_mask
    }) {
        if (raster->shape(0) != frame_height || raster->shape(1) != frame_width) {
            throw py::value_error("exemplar source rasters must match");
        }
    }
    const py::ssize_t target_count = target_pixels.shape(0);
    if (
        target_colours.ndim() != 2 || target_colours.shape(0) != target_count ||
        target_colours.shape(1) != 3 || completed.shape(0) != target_count ||
        completed_colours.ndim() != 2 ||
        completed_colours.shape(0) != target_count ||
        completed_colours.shape(1) != 3
    ) {
        throw py::value_error("exemplar target arrays must match");
    }
    if (
        !working.dtype().is(py::dtype::of<std::uint8_t>()) ||
        working.ndim() != 3 || working.shape(2) != 3 ||
        (working.flags() & py::array::c_style) == 0 || !working.writeable()
    ) {
        throw py::type_error("working must be a writable uint8 [h,w,3] array");
    }
    const py::ssize_t local_height = working.shape(0);
    const py::ssize_t local_width = working.shape(1);
    if (
        local_height <= 0 || local_width <= 0 ||
        static_cast<std::uint64_t>(local_height) > kUint32Max ||
        static_cast<std::uint64_t>(local_width) > kUint32Max
    ) {
        throw py::value_error("exemplar working shape must fit uint32 bounds");
    }
    for (py::array* mask : {&donor, &target, &processed, &barrier}) {
        if (
            !mask->dtype().is(py::dtype::of<bool>()) || mask->ndim() != 2 ||
            mask->shape(0) != local_height || mask->shape(1) != local_width ||
            (mask->flags() & py::array::c_style) == 0 || !mask->writeable()
        ) {
            throw py::type_error("exemplar outputs must be writable matching bool rasters");
        }
    }
    const auto local_width_u64 = static_cast<std::uint64_t>(local_width);
    const std::uint64_t y1_64 =
        static_cast<std::uint64_t>(y0) + static_cast<std::uint64_t>(local_height);
    const std::uint64_t x1_64 =
        static_cast<std::uint64_t>(x0) + local_width_u64;
    if (
        y1_64 > frame_height_u32 || x1_64 > frame_width_u32 ||
        !(y0 <= interior_y0 && interior_y0 < interior_y1 && interior_y1 <= y1_64) ||
        !(x0 <= interior_x0 && interior_x0 < interior_x1 && interior_x1 <= x1_64) ||
        region == 0 || region == kUint32Max
    ) {
        throw py::value_error("exemplar working bounds are invalid");
    }
    const auto y1 = static_cast<std::uint32_t>(y1_64);
    const auto x1 = static_cast<std::uint32_t>(x1_64);
    const auto* coverage = static_cast<const std::uint8_t*>(coverage_count.data());
    const auto* regions = static_cast<const std::uint32_t*>(pure_region_id.data());
    const auto* blue_values = static_cast<const std::uint8_t*>(blue.data());
    const auto* green_values = static_cast<const std::uint8_t*>(green.data());
    const auto* red_values = static_cast<const std::uint8_t*>(red.data());
    const auto* safe = static_cast<const bool*>(safe_donor_mask.data());
    const auto* pixels = static_cast<const std::uint32_t*>(target_pixels.data());
    const auto* colours = static_cast<const std::uint8_t*>(target_colours.data());
    const auto* completed_values = static_cast<const bool*>(completed.data());
    const auto* completed_colour_values = static_cast<const std::uint8_t*>(
        completed_colours.data()
    );
    auto* working_values = static_cast<std::uint8_t*>(working.mutable_data());
    auto* donor_values = static_cast<bool*>(donor.mutable_data());
    auto* target_values = static_cast<bool*>(target.mutable_data());
    auto* processed_values = static_cast<bool*>(processed.mutable_data());
    auto* barrier_values = static_cast<bool*>(barrier.mutable_data());
    std::uint64_t owned_count = 0;
    {
        py::gil_scoped_release release;
        for (std::uint32_t row = y0; row < y1; ++row) {
            for (std::uint32_t column = x0; column < x1; ++column) {
                const std::uint64_t source =
                    static_cast<std::uint64_t>(row) * frame_width_u32 + column;
                const std::uint64_t local =
                    static_cast<std::uint64_t>(row - y0) * local_width_u64 + column - x0;
                const bool same = coverage[source] > 0 && regions[source] == region;
                donor_values[local] = same && safe[source];
                target_values[local] = false;
                processed_values[local] = same;
                barrier_values[local] = !same;
                working_values[3 * local] = same ? blue_values[source] : 0;
                working_values[3 * local + 1] = same ? green_values[source] : 0;
                working_values[3 * local + 2] = same ? red_values[source] : 0;
            }
        }
        const std::uint64_t frame_pixels =
            static_cast<std::uint64_t>(frame_height_u32) * frame_width_u32;
        for (py::ssize_t index = 0; index < target_count; ++index) {
            const std::uint32_t pixel = pixels[index];
            if (pixel >= frame_pixels) {
                throw std::runtime_error("exemplar target lies outside the frame");
            }
            const std::uint32_t row = pixel / frame_width_u32;
            const std::uint32_t column = pixel % frame_width_u32;
            if (row < y0 || row >= y1 || column < x0 || column >= x1) {
                continue;
            }
            const std::uint64_t local =
                static_cast<std::uint64_t>(row - y0) * local_width_u64 + column - x0;
            donor_values[local] = false;
            if (
                row >= interior_y0 && row < interior_y1 &&
                column >= interior_x0 && column < interior_x1
            ) {
                target_values[local] = true;
                processed_values[local] = false;
                barrier_values[local] = false;
                working_values[3 * local] = colours[3 * index];
                working_values[3 * local + 1] = colours[3 * index + 1];
                working_values[3 * local + 2] = colours[3 * index + 2];
                ++owned_count;
            } else if (completed_values[index]) {
                processed_values[local] = true;
                barrier_values[local] = false;
                working_values[3 * local] = completed_colour_values[3 * index];
                working_values[3 * local + 1] = completed_colour_values[3 * index + 1];
                working_values[3 * local + 2] = completed_colour_values[3 * index + 2];
            } else {
                processed_values[local] = false;
                barrier_values[local] = true;
                working_values[3 * local] = 0;
                working_values[3 * local + 1] = 0;
                working_values[3 * local + 2] = 0;
            }
        }
    }
    return owned_count;
}

void downsample_exemplar_level_into_native(
    const py::array& child_working,
    const py::array& child_donor,
    const py::array& child_target,
    const py::array& child_processed,
    const py::array& child_barrier,
    py::array parent_working,
    py::array parent_donor,
    py::array parent_target,
    py::array parent_processed,
    py::array parent_barrier
) {
    if (
        !child_working.dtype().is(py::dtype::of<std::uint8_t>()) ||
        child_working.ndim() != 3 || child_working.shape(2) != 3 ||
        (child_working.flags() & py::array::c_style) == 0
    ) {
        throw py::type_error("child_working must be C-contiguous uint8 [h,w,3]");
    }
    const py::ssize_t child_height = child_working.shape(0);
    const py::ssize_t child_width = child_working.shape(1);
    for (const py::array* mask : {
        &child_donor, &child_target, &child_processed, &child_barrier
    }) {
        if (
            !mask->dtype().is(py::dtype::of<bool>()) || mask->ndim() != 2 ||
            mask->shape(0) != child_height || mask->shape(1) != child_width ||
            (mask->flags() & py::array::c_style) == 0
        ) {
            throw py::type_error("child exemplar masks must match");
        }
    }
    const py::ssize_t parent_height = (child_height + 1) / 2;
    const py::ssize_t parent_width = (child_width + 1) / 2;
    if (
        !parent_working.dtype().is(py::dtype::of<std::uint8_t>()) ||
        parent_working.ndim() != 3 || parent_working.shape(0) != parent_height ||
        parent_working.shape(1) != parent_width || parent_working.shape(2) != 3 ||
        (parent_working.flags() & py::array::c_style) == 0 ||
        !parent_working.writeable()
    ) {
        throw py::type_error("parent_working must be a writable matching uint8 raster");
    }
    for (py::array* mask : {
        &parent_donor, &parent_target, &parent_processed, &parent_barrier
    }) {
        if (
            !mask->dtype().is(py::dtype::of<bool>()) || mask->ndim() != 2 ||
            mask->shape(0) != parent_height || mask->shape(1) != parent_width ||
            (mask->flags() & py::array::c_style) == 0 || !mask->writeable()
        ) {
            throw py::type_error("parent exemplar masks must be writable and matching");
        }
    }
    const auto* child_colours = static_cast<const std::uint8_t*>(child_working.data());
    const auto* child_donor_values = static_cast<const bool*>(child_donor.data());
    const auto* child_target_values = static_cast<const bool*>(child_target.data());
    const auto* child_processed_values = static_cast<const bool*>(child_processed.data());
    const auto* child_barrier_values = static_cast<const bool*>(child_barrier.data());
    auto* colours = static_cast<std::uint8_t*>(parent_working.mutable_data());
    auto* donor = static_cast<bool*>(parent_donor.mutable_data());
    auto* target = static_cast<bool*>(parent_target.mutable_data());
    auto* processed = static_cast<bool*>(parent_processed.mutable_data());
    auto* barrier = static_cast<bool*>(parent_barrier.mutable_data());
    {
        py::gil_scoped_release release;
        for (py::ssize_t row = 0; row < parent_height; ++row) {
            for (py::ssize_t column = 0; column < parent_width; ++column) {
                std::array<std::uint32_t, 3> sums{};
                std::uint32_t count = 0;
                bool donor_value = true;
                bool target_value = false;
                bool processed_value = true;
                bool barrier_value = false;
                for (py::ssize_t delta_row = 0; delta_row < 2; ++delta_row) {
                    const py::ssize_t child_row = 2 * row + delta_row;
                    if (child_row >= child_height) {
                        continue;
                    }
                    for (py::ssize_t delta_column = 0; delta_column < 2; ++delta_column) {
                        const py::ssize_t child_column = 2 * column + delta_column;
                        if (child_column >= child_width) {
                            continue;
                        }
                        const std::uint64_t child =
                            static_cast<std::uint64_t>(child_row) * child_width + child_column;
                        ++count;
                        for (std::size_t channel = 0; channel < 3; ++channel) {
                            sums[channel] += child_colours[3 * child + channel];
                        }
                        donor_value = donor_value && child_donor_values[child];
                        target_value = target_value || child_target_values[child];
                        processed_value = processed_value && child_processed_values[child];
                        barrier_value = barrier_value || child_barrier_values[child];
                    }
                }
                const std::uint64_t parent =
                    static_cast<std::uint64_t>(row) * parent_width + column;
                barrier[parent] = barrier_value;
                target[parent] = !barrier_value && target_value;
                donor[parent] = !barrier_value && donor_value;
                processed[parent] = !target_value && !barrier_value && processed_value;
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    if (barrier_value) {
                        colours[3 * parent + channel] = 0;
                        continue;
                    }
                    std::uint32_t quotient = sums[channel] / count;
                    const std::uint32_t remainder = sums[channel] % count;
                    if (
                        2 * remainder > count ||
                        (2 * remainder == count && (quotient & 1U) != 0)
                    ) {
                        ++quotient;
                    }
                    colours[3 * parent + channel] = static_cast<std::uint8_t>(quotient);
                }
            }
        }
    }
}

std::uint64_t enumerate_exemplar_donor_centres_into_native(
    const py::array& donor_mask,
    py::array integral,
    py::array centres
) {
    validate_array(donor_mask, py::dtype::of<bool>(), "donor_mask");
    const py::ssize_t height = donor_mask.shape(0);
    const py::ssize_t width = donor_mask.shape(1);
    if (
        !integral.dtype().is(py::dtype::of<std::uint32_t>()) ||
        integral.ndim() != 2 || integral.shape(0) != height + 1 ||
        integral.shape(1) != width + 1 ||
        (integral.flags() & py::array::c_style) == 0 || !integral.writeable()
    ) {
        throw py::type_error("integral must be a writable uint32 [h+1,w+1] raster");
    }
    const std::uint64_t capacity =
        height >= 9 && width >= 9
        ? static_cast<std::uint64_t>(height - 8) * (width - 8)
        : 0;
    if (
        !centres.dtype().is(py::dtype::of<std::uint32_t>()) ||
        centres.ndim() != 2 || centres.shape(1) != 2 ||
        static_cast<std::uint64_t>(centres.shape(0)) < capacity ||
        (centres.flags() & py::array::c_style) == 0 || !centres.writeable()
    ) {
        throw py::type_error("centres must be a writable uint32 [capacity,2] array");
    }
    const auto* donor = static_cast<const bool*>(donor_mask.data());
    auto* sums = static_cast<std::uint32_t*>(integral.mutable_data());
    auto* output = static_cast<std::uint32_t*>(centres.mutable_data());
    std::uint64_t count = 0;
    {
        py::gil_scoped_release release;
        std::fill(sums, sums + static_cast<std::uint64_t>(height + 1) * (width + 1), 0);
        for (py::ssize_t row = 1; row <= height; ++row) {
            std::uint32_t row_sum = 0;
            for (py::ssize_t column = 1; column <= width; ++column) {
                row_sum += donor[(row - 1) * width + column - 1] ? 1U : 0U;
                sums[row * (width + 1) + column] =
                    sums[(row - 1) * (width + 1) + column] + row_sum;
            }
        }
        for (py::ssize_t row = 4; row + 4 < height; ++row) {
            const py::ssize_t top = row - 4;
            const py::ssize_t bottom = row + 5;
            for (py::ssize_t column = 4; column + 4 < width; ++column) {
                const py::ssize_t left = column - 4;
                const py::ssize_t right = column + 5;
                const std::uint32_t total =
                    sums[bottom * (width + 1) + right] -
                    sums[top * (width + 1) + right] -
                    sums[bottom * (width + 1) + left] +
                    sums[top * (width + 1) + left];
                if (total == 81) {
                    output[2 * count] = static_cast<std::uint32_t>(row);
                    output[2 * count + 1] = static_cast<std::uint32_t>(column);
                    ++count;
                }
            }
        }
    }
    return count;
}

void subsample_exemplar_indexes_into_native(
    std::uint64_t candidate_count,
    std::uint64_t selected_count,
    py::array output
) {
    if (
        !output.dtype().is(py::dtype::of<std::uint32_t>()) || output.ndim() != 1 ||
        (output.flags() & py::array::c_style) == 0 || !output.writeable() ||
        static_cast<std::uint64_t>(output.shape(0)) < selected_count
    ) {
        throw py::type_error("selected output must be a writable uint32 vector");
    }
    if (selected_count > candidate_count || candidate_count > kUint32Max) {
        throw py::value_error("exemplar candidate counts are invalid");
    }
    auto* values = static_cast<std::uint32_t*>(output.mutable_data());
    py::gil_scoped_release release;
    for (std::uint64_t ordinal = 0; ordinal < selected_count; ++ordinal) {
        values[ordinal] = static_cast<std::uint32_t>(
            (ordinal * candidate_count) / selected_count
        );
    }
}

void replicate_exemplar_targets_into_native(
    const py::array& coarse_working,
    const py::array& coarse_target,
    const py::array& coarse_barrier,
    py::array fine_working,
    const py::array& fine_target
) {
    if (
        !coarse_working.dtype().is(py::dtype::of<std::uint8_t>()) ||
        coarse_working.ndim() != 3 || coarse_working.shape(2) != 3 ||
        (coarse_working.flags() & py::array::c_style) == 0 ||
        !fine_working.dtype().is(py::dtype::of<std::uint8_t>()) ||
        fine_working.ndim() != 3 || fine_working.shape(2) != 3 ||
        (fine_working.flags() & py::array::c_style) == 0 || !fine_working.writeable()
    ) {
        throw py::type_error("exemplar working rasters are invalid");
    }
    validate_array(coarse_target, py::dtype::of<bool>(), "coarse_target");
    validate_array(coarse_barrier, py::dtype::of<bool>(), "coarse_barrier");
    validate_array(fine_target, py::dtype::of<bool>(), "fine_target");
    const py::ssize_t coarse_height = coarse_working.shape(0);
    const py::ssize_t coarse_width = coarse_working.shape(1);
    const py::ssize_t fine_height = fine_working.shape(0);
    const py::ssize_t fine_width = fine_working.shape(1);
    if (
        coarse_target.shape(0) != coarse_height || coarse_target.shape(1) != coarse_width ||
        coarse_barrier.shape(0) != coarse_height || coarse_barrier.shape(1) != coarse_width ||
        fine_target.shape(0) != fine_height || fine_target.shape(1) != fine_width ||
        coarse_height != (fine_height + 1) / 2 ||
        coarse_width != (fine_width + 1) / 2
    ) {
        throw py::value_error("exemplar pyramid levels do not align");
    }
    const auto* coarse_colours = static_cast<const std::uint8_t*>(coarse_working.data());
    const auto* coarse_targets = static_cast<const bool*>(coarse_target.data());
    const auto* coarse_barriers = static_cast<const bool*>(coarse_barrier.data());
    auto* fine_colours = static_cast<std::uint8_t*>(fine_working.mutable_data());
    const auto* fine_targets = static_cast<const bool*>(fine_target.data());
    py::gil_scoped_release release;
    for (py::ssize_t row = 0; row < fine_height; ++row) {
        for (py::ssize_t column = 0; column < fine_width; ++column) {
            const std::uint64_t fine = static_cast<std::uint64_t>(row) * fine_width + column;
            if (!fine_targets[fine]) {
                continue;
            }
            const std::uint64_t coarse =
                static_cast<std::uint64_t>(row / 2) * coarse_width + column / 2;
            if (coarse_targets[coarse] && !coarse_barriers[coarse]) {
                fine_colours[3 * fine] = coarse_colours[3 * coarse];
                fine_colours[3 * fine + 1] = coarse_colours[3 * coarse + 1];
                fine_colours[3 * fine + 2] = coarse_colours[3 * coarse + 2];
            }
        }
    }
}

void copy_exemplar_patch_into_native(
    py::array working,
    const py::array& target,
    py::array processed,
    std::uint32_t target_row,
    std::uint32_t target_column,
    std::uint32_t donor_row,
    std::uint32_t donor_column
) {
    if (
        !working.dtype().is(py::dtype::of<std::uint8_t>()) || working.ndim() != 3 ||
        working.shape(2) != 3 || (working.flags() & py::array::c_style) == 0 ||
        !working.writeable()
    ) {
        throw py::type_error("working must be a writable uint8 exemplar raster");
    }
    validate_array(target, py::dtype::of<bool>(), "target");
    if (
        !processed.dtype().is(py::dtype::of<bool>()) || processed.ndim() != 2 ||
        (processed.flags() & py::array::c_style) == 0 || !processed.writeable()
    ) {
        throw py::type_error("processed must be a writable bool exemplar raster");
    }
    const std::uint32_t height = static_cast<std::uint32_t>(working.shape(0));
    const std::uint32_t width = static_cast<std::uint32_t>(working.shape(1));
    if (
        target.shape(0) != height || target.shape(1) != width ||
        processed.shape(0) != height || processed.shape(1) != width ||
        target_row < 3 || target_row + 3 >= height || target_column < 3 ||
        target_column + 3 >= width || donor_row < 3 || donor_row + 3 >= height ||
        donor_column < 3 || donor_column + 3 >= width
    ) {
        throw py::value_error("exemplar copy patch lies outside its level");
    }
    auto* colours = static_cast<std::uint8_t*>(working.mutable_data());
    const auto* targets = static_cast<const bool*>(target.data());
    auto* processed_values = static_cast<bool*>(processed.mutable_data());
    py::gil_scoped_release release;
    for (std::int32_t delta_row = -3; delta_row <= 3; ++delta_row) {
        for (std::int32_t delta_column = -3; delta_column <= 3; ++delta_column) {
            const std::uint64_t destination =
                static_cast<std::uint64_t>(target_row + delta_row) * width +
                target_column + delta_column;
            if (!targets[destination] || processed_values[destination]) {
                continue;
            }
            const std::uint64_t source =
                static_cast<std::uint64_t>(donor_row + delta_row) * width +
                donor_column + delta_column;
            colours[3 * destination] = colours[3 * source];
            colours[3 * destination + 1] = colours[3 * source + 1];
            colours[3 * destination + 2] = colours[3 * source + 2];
            processed_values[destination] = true;
        }
    }
}

std::uint64_t record_exemplar_completions_into_native(
    const py::array& full_working,
    const py::array& full_target,
    const py::array& full_processed,
    const py::array& target_pixels,
    std::uint32_t frame_width,
    std::uint32_t y0,
    std::uint32_t x0,
    py::array completed,
    py::array completed_colours
) {
    if (
        !full_working.dtype().is(py::dtype::of<std::uint8_t>()) ||
        full_working.ndim() != 3 || full_working.shape(2) != 3 ||
        (full_working.flags() & py::array::c_style) == 0
    ) {
        throw py::type_error("full_working must be a C-contiguous uint8 raster");
    }
    validate_array(full_target, py::dtype::of<bool>(), "full_target");
    validate_array(full_processed, py::dtype::of<bool>(), "full_processed");
    validate_vector(target_pixels, py::dtype::of<std::uint32_t>(), "target_pixels");
    if (
        !completed.dtype().is(py::dtype::of<bool>()) || completed.ndim() != 1 ||
        (completed.flags() & py::array::c_style) == 0 || !completed.writeable() ||
        !completed_colours.dtype().is(py::dtype::of<std::uint8_t>()) ||
        completed_colours.ndim() != 2 || completed_colours.shape(1) != 3 ||
        (completed_colours.flags() & py::array::c_style) == 0 ||
        !completed_colours.writeable() ||
        completed.shape(0) != target_pixels.shape(0) ||
        completed_colours.shape(0) != target_pixels.shape(0)
    ) {
        throw py::type_error("completion outputs must match target_pixels");
    }
    const std::uint32_t local_height = static_cast<std::uint32_t>(full_working.shape(0));
    const std::uint32_t local_width = static_cast<std::uint32_t>(full_working.shape(1));
    if (
        full_target.shape(0) != local_height || full_target.shape(1) != local_width ||
        full_processed.shape(0) != local_height ||
        full_processed.shape(1) != local_width || frame_width == 0
    ) {
        throw py::value_error("completion level masks do not match");
    }
    const auto* colours = static_cast<const std::uint8_t*>(full_working.data());
    const auto* targets = static_cast<const bool*>(full_target.data());
    const auto* processed = static_cast<const bool*>(full_processed.data());
    const auto* pixels = static_cast<const std::uint32_t*>(target_pixels.data());
    auto* completed_values = static_cast<bool*>(completed.mutable_data());
    auto* output_colours = static_cast<std::uint8_t*>(completed_colours.mutable_data());
    std::uint64_t count = 0;
    py::gil_scoped_release release;
    for (py::ssize_t index = 0; index < target_pixels.shape(0); ++index) {
        const std::uint32_t row = pixels[index] / frame_width;
        const std::uint32_t column = pixels[index] % frame_width;
        if (
            row < y0 || row >= static_cast<std::uint64_t>(y0) + local_height ||
            column < x0 || column >= static_cast<std::uint64_t>(x0) + local_width
        ) {
            continue;
        }
        const std::uint64_t local =
            static_cast<std::uint64_t>(row - y0) * local_width + column - x0;
        if (targets[local] && processed[local]) {
            if (!completed_values[index]) {
                ++count;
            }
            completed_values[index] = true;
            output_colours[3 * index] = colours[3 * local];
            output_colours[3 * index + 1] = colours[3 * local + 1];
            output_colours[3 * index + 2] = colours[3 * local + 2];
        }
    }
    return count;
}

std::uint64_t commit_exemplar_records_into_native(
    py::array raw_records,
    const py::array& record_indexes,
    const py::array& target_pixels,
    const py::array& completed,
    const py::array& completed_colours
) {
    if (
        !raw_records.dtype().is(py::dtype::of<std::uint8_t>()) ||
        raw_records.ndim() != 2 || raw_records.shape(1) != 16 ||
        (raw_records.flags() & py::array::c_style) == 0 || !raw_records.writeable()
    ) {
        throw py::type_error("raw_records must be writable C-contiguous uint8 [n,16]");
    }
    validate_vector(record_indexes, py::dtype::of<std::uint32_t>(), "record_indexes");
    validate_vector(target_pixels, py::dtype::of<std::uint32_t>(), "target_pixels");
    validate_vector(completed, py::dtype::of<bool>(), "completed");
    validate_array(completed_colours, py::dtype::of<std::uint8_t>(), "completed_colours");
    if (
        completed.shape(0) != target_pixels.shape(0) ||
        completed_colours.ndim() != 2 ||
        completed_colours.shape(0) != target_pixels.shape(0) ||
        completed_colours.shape(1) != 3
    ) {
        throw py::value_error("completed exemplar arrays must match target pixels");
    }
    auto* records = static_cast<std::uint8_t*>(raw_records.mutable_data());
    const auto* indexes = static_cast<const std::uint32_t*>(record_indexes.data());
    const auto* pixels = static_cast<const std::uint32_t*>(target_pixels.data());
    const auto* completed_values = static_cast<const bool*>(completed.data());
    const auto* colours = static_cast<const std::uint8_t*>(completed_colours.data());
    const std::uint32_t target_count = static_cast<std::uint32_t>(target_pixels.shape(0));
    std::uint64_t committed = 0;
    py::gil_scoped_release release;
    for (py::ssize_t ordinal = 0; ordinal < record_indexes.shape(0); ++ordinal) {
        const std::uint32_t record_index = indexes[ordinal];
        if (record_index >= static_cast<std::uint64_t>(raw_records.shape(0))) {
            throw std::runtime_error("exemplar commit index lies outside record arena");
        }
        std::uint8_t* record = records + 16ULL * record_index;
        if (record[13] != 0) {
            continue;
        }
        const std::uint32_t pixel = load_u32_le(record);
        const auto* found = std::lower_bound(pixels, pixels + target_count, pixel);
        if (found == pixels + target_count || *found != pixel) {
            throw std::runtime_error("exemplar record has no target identity");
        }
        const std::uint64_t target_index = found - pixels;
        if (!completed_values[target_index]) {
            continue;
        }
        record[10] = colours[3 * target_index];
        record[11] = colours[3 * target_index + 1];
        record[12] = colours[3 * target_index + 2];
        record[13] = 2;
        ++committed;
    }
    return committed;
}

std::uint64_t validate_repair_record_coverage_native(
    const py::array& raw_records,
    const py::array& invalid_mask
) {
    if (
        !raw_records.dtype().is(py::dtype::of<std::uint8_t>()) ||
        raw_records.ndim() != 2 || raw_records.shape(1) != 16 ||
        (raw_records.flags() & py::array::c_style) == 0
    ) {
        throw py::type_error("raw_records must be a C-contiguous uint8 [n,16] array");
    }
    validate_array(invalid_mask, py::dtype::of<bool>(), "invalid_mask");
    const std::int64_t height = invalid_mask.shape(0);
    const std::int64_t fine_width = invalid_mask.shape(1);
    if (fine_width % 16 != 0) {
        throw py::value_error("invalid_mask width must be a multiple of 16");
    }
    if (static_cast<std::uint64_t>(raw_records.shape(0)) > kRepairRecordCap) {
        throw py::value_error("repair record count exceeds the fixed arena");
    }
    const std::int64_t width = fine_width / 16;
    const std::uint64_t pixel_count = static_cast<std::uint64_t>(height * width);
    const auto* records = static_cast<const std::uint8_t*>(raw_records.data());
    const auto* invalid = static_cast<const bool*>(invalid_mask.data());
    std::vector<std::uint16_t> accumulated(static_cast<std::size_t>(pixel_count), 0);
    std::uint64_t record_popcount = 0;
    bool has_previous = false;
    std::uint32_t previous_pixel = 0;
    std::uint32_t previous_first_bit = 0;
    std::uint32_t previous_region = 0;
    std::uint8_t previous_far_side = 0;
    {
        py::gil_scoped_release release;
        for (py::ssize_t index = 0; index < raw_records.shape(0); ++index) {
            const std::uint8_t* record = records + 16 * index;
            const std::uint32_t pixel = load_u32_le(record);
            const std::uint16_t mask = load_u16_le(record + 4);
            const std::uint32_t region = load_u32_le(record + 6);
            const std::uint8_t far_side = record[14];
            if (pixel >= pixel_count) {
                throw std::runtime_error("record pixel lies outside the frame");
            }
            if (!is_contiguous_u16_mask(mask)) {
                throw std::runtime_error("record lane mask must be nonzero and contiguous");
            }
            if (region == 0 || region == std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error("record region must be positive");
            }
            if (far_side > 1 || record[15] != 0) {
                throw std::runtime_error("record side/reserved byte is invalid");
            }
            const std::uint32_t first_bit = least_u16_bit(mask);
            const bool out_of_order = has_previous && (
                pixel < previous_pixel ||
                (pixel == previous_pixel && first_bit < previous_first_bit) ||
                (
                    pixel == previous_pixel && first_bit == previous_first_bit &&
                    region < previous_region
                ) ||
                (
                    pixel == previous_pixel && first_bit == previous_first_bit &&
                    region == previous_region && far_side < previous_far_side
                )
            );
            if (out_of_order) {
                throw std::runtime_error("records are not canonically ordered");
            }
            has_previous = true;
            previous_pixel = pixel;
            previous_first_bit = first_bit;
            previous_region = region;
            previous_far_side = far_side;
            if ((accumulated[pixel] & mask) != 0) {
                throw std::runtime_error("record lane masks overlap");
            }
            accumulated[pixel] = static_cast<std::uint16_t>(accumulated[pixel] | mask);
            record_popcount += count_u16_bits(mask);
        }

        std::uint64_t expected_popcount = 0;
        for (std::uint64_t pixel = 0; pixel < pixel_count; ++pixel) {
            const std::uint64_t row = pixel / static_cast<std::uint64_t>(width);
            const std::uint64_t column = pixel % static_cast<std::uint64_t>(width);
            std::uint16_t expected_mask = 0;
            const std::uint64_t first_sample = row * static_cast<std::uint64_t>(fine_width) +
                column * 16ULL;
            for (std::uint32_t lane = 0; lane < 16; ++lane) {
                if (invalid[first_sample + lane]) {
                    expected_mask = static_cast<std::uint16_t>(expected_mask | (1U << lane));
                }
            }
            expected_popcount += count_u16_bits(expected_mask);
            if (accumulated[pixel] != expected_mask) {
                throw std::runtime_error("record mask OR does not equal the invalid mask");
            }
        }
        if (record_popcount != expected_popcount) {
            throw std::runtime_error("record popcount does not equal the invalid-lane count");
        }
    }
    return record_popcount;
}

struct NativeRepairRun {
    std::uint32_t record_start = 0;
    std::uint32_t record_end = 0;
    std::uint32_t row = 0;
    std::uint32_t start_fine = 0;
    std::uint32_t end_fine = 0;
    std::uint32_t region = 0;
    std::uint8_t far_side = 0;
};

inline std::uint32_t greatest_u16_bit(std::uint16_t value) {
    std::uint32_t bit = 0;
    while (value > 1) {
        value = static_cast<std::uint16_t>(value >> 1);
        ++bit;
    }
    return bit;
}

inline void store_u32_le(std::uint8_t* destination, std::uint32_t value) {
    std::memcpy(destination, &value, sizeof(value));
}

py::array reconstruct_repair_runs_native(
    const py::array& raw_records,
    std::uint32_t height,
    std::uint32_t width
) {
    if (
        !raw_records.dtype().is(py::dtype::of<std::uint8_t>()) ||
        raw_records.ndim() != 2 || raw_records.shape(1) != 16 ||
        (raw_records.flags() & py::array::c_style) == 0
    ) {
        throw py::type_error("raw_records must be a C-contiguous uint8 [n,16] array");
    }
    if (height == 0 || width == 0) {
        throw py::value_error("render shape must be positive");
    }
    if (static_cast<std::uint64_t>(raw_records.shape(0)) > kRepairRecordCap) {
        throw py::value_error("repair record count exceeds the fixed arena");
    }
    const std::uint64_t pixel_count = static_cast<std::uint64_t>(height) * width;
    if (pixel_count * 16ULL > kUint32Max) {
        throw py::value_error("render shape must fit uint32 fine indexes");
    }
    const auto* records = static_cast<const std::uint8_t*>(raw_records.data());
    std::vector<NativeRepairRun> runs;
    runs.reserve(static_cast<std::size_t>(raw_records.shape(0)));
    bool has_previous_key = false;
    std::uint32_t previous_pixel = 0;
    std::uint32_t previous_first_bit = 0;
    std::uint32_t previous_region = 0;
    std::uint8_t previous_far_side = 0;
    {
        py::gil_scoped_release release;
        for (py::ssize_t index = 0; index < raw_records.shape(0); ++index) {
            const std::uint8_t* record = records + 16 * index;
            const std::uint32_t pixel = load_u32_le(record);
            const std::uint16_t mask = load_u16_le(record + 4);
            const std::uint32_t region = load_u32_le(record + 6);
            const std::uint8_t far_side = record[14];
            if (pixel >= pixel_count) {
                throw std::runtime_error("record pixel lies outside render_shape");
            }
            if (!is_contiguous_u16_mask(mask)) {
                throw std::runtime_error("record lane mask must be nonzero and contiguous");
            }
            if (region == 0 || region == std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error("record region must be positive");
            }
            if (far_side > 1 || record[15] != 0) {
                throw std::runtime_error("record side/reserved byte is invalid");
            }
            const std::uint32_t first_bit = least_u16_bit(mask);
            const bool out_of_order = has_previous_key && (
                pixel < previous_pixel ||
                (pixel == previous_pixel && first_bit < previous_first_bit) ||
                (
                    pixel == previous_pixel && first_bit == previous_first_bit &&
                    region < previous_region
                ) ||
                (
                    pixel == previous_pixel && first_bit == previous_first_bit &&
                    region == previous_region && far_side < previous_far_side
                )
            );
            if (out_of_order) {
                throw std::runtime_error("records are not canonically ordered");
            }
            has_previous_key = true;
            previous_pixel = pixel;
            previous_first_bit = first_bit;
            previous_region = region;
            previous_far_side = far_side;

            const std::uint32_t row = pixel / width;
            const std::uint32_t column = pixel % width;
            const std::uint32_t start_fine = 16U * column + first_bit;
            const std::uint32_t end_fine = 16U * column + greatest_u16_bit(mask);
            if (
                !runs.empty() && runs.back().row == row &&
                runs.back().end_fine + 1U == start_fine &&
                runs.back().region == region && runs.back().far_side == far_side
            ) {
                runs.back().record_end = static_cast<std::uint32_t>(index + 1);
                runs.back().end_fine = end_fine;
            } else {
                runs.push_back(NativeRepairRun{
                    static_cast<std::uint32_t>(index),
                    static_cast<std::uint32_t>(index + 1),
                    row,
                    start_fine,
                    end_fine,
                    region,
                    far_side,
                });
            }
        }
    }
    py::array_t<std::uint8_t> output({
        static_cast<py::ssize_t>(runs.size()),
        static_cast<py::ssize_t>(25),
    });
    auto* destination = output.mutable_data();
    for (std::size_t index = 0; index < runs.size(); ++index) {
        const NativeRepairRun& run = runs[index];
        std::uint8_t* row = destination + 25 * index;
        store_u32_le(row, run.record_start);
        store_u32_le(row + 4, run.record_end);
        store_u32_le(row + 8, run.row);
        store_u32_le(row + 12, run.start_fine);
        store_u32_le(row + 16, run.end_fine);
        store_u32_le(row + 20, run.region);
        row[24] = run.far_side;
    }
    return output;
}

inline std::uint32_t component_find(
    std::vector<std::uint32_t>& parent,
    std::uint32_t index
) {
    std::uint32_t root = index;
    while (parent[root] != root) {
        root = parent[root];
    }
    while (parent[index] != index) {
        const std::uint32_t next = parent[index];
        parent[index] = root;
        index = next;
    }
    return root;
}

inline void component_union(
    std::vector<std::uint32_t>& parent,
    std::vector<std::uint8_t>& rank,
    std::uint32_t left,
    std::uint32_t right
) {
    std::uint32_t left_root = component_find(parent, left);
    std::uint32_t right_root = component_find(parent, right);
    if (left_root == right_root) {
        return;
    }
    if (rank[left_root] < rank[right_root]) {
        std::swap(left_root, right_root);
    }
    parent[right_root] = left_root;
    if (rank[left_root] == rank[right_root]) {
        ++rank[left_root];
    }
}

enum class ComponentRelation : std::uint8_t {
    Same,
    Horizontal,
    Vertical,
};

void union_component_groups(
    const std::uint8_t* records,
    std::vector<std::uint32_t>& parent,
    std::vector<std::uint8_t>& rank,
    std::uint32_t left_start,
    std::uint32_t left_end,
    std::uint32_t right_start,
    std::uint32_t right_end,
    ComponentRelation relation
) {
    for (std::uint32_t left = left_start; left < left_end; ++left) {
        const std::uint8_t* left_record = records + 16ULL * left;
        const std::uint32_t left_region = load_u32_le(left_record + 6);
        const std::uint16_t left_mask = load_u16_le(left_record + 4);
        for (std::uint32_t right = right_start; right < right_end; ++right) {
            const std::uint8_t* right_record = records + 16ULL * right;
            if (left_region != load_u32_le(right_record + 6)) {
                continue;
            }
            const std::uint16_t right_mask = load_u16_le(right_record + 4);
            bool adjacent = false;
            if (relation == ComponentRelation::Same) {
                adjacent = (
                    (
                        (static_cast<std::uint32_t>(left_mask) << 1U) & right_mask
                    ) |
                    (
                        (static_cast<std::uint32_t>(right_mask) << 1U) & left_mask
                    )
                ) != 0;
            } else if (relation == ComponentRelation::Horizontal) {
                adjacent = (left_mask & 0x8000U) != 0 && (right_mask & 0x0001U) != 0;
            } else {
                adjacent = (left_mask & right_mask) != 0;
            }
            if (adjacent) {
                component_union(parent, rank, left, right);
            }
        }
    }
}

py::tuple build_record_component_ids_native(
    const py::array& raw_records,
    std::uint32_t height,
    std::uint32_t width
) {
    if (
        !raw_records.dtype().is(py::dtype::of<std::uint8_t>()) ||
        raw_records.ndim() != 2 || raw_records.shape(1) != 16 ||
        (raw_records.flags() & py::array::c_style) == 0
    ) {
        throw py::type_error("raw_records must be a C-contiguous uint8 [n,16] array");
    }
    if (height == 0 || width == 0) {
        throw py::value_error("render shape must be positive");
    }
    const std::uint64_t count64 = static_cast<std::uint64_t>(raw_records.shape(0));
    if (count64 > kRepairRecordCap) {
        throw py::value_error("repair record count exceeds the fixed arena");
    }
    const std::uint64_t pixel_count = static_cast<std::uint64_t>(height) * width;
    if (pixel_count * 16ULL > kUint32Max) {
        throw py::value_error("render shape must fit uint32 fine indexes");
    }
    const std::uint32_t count = static_cast<std::uint32_t>(count64);
    py::array_t<std::uint32_t> component_ids({raw_records.shape(0)});
    if (count == 0) {
        return py::make_tuple(std::move(component_ids), std::uint32_t(0));
    }
    const auto* records = static_cast<const std::uint8_t*>(raw_records.data());
    auto* output = component_ids.mutable_data();
    std::vector<std::uint32_t> parent(count);
    std::vector<std::uint8_t> rank(count, 0);
    for (std::uint32_t index = 0; index < count; ++index) {
        parent[index] = index;
    }
    std::vector<std::int32_t> previous_starts(width, -1);
    std::vector<std::int32_t> previous_ends(width, -1);
    std::vector<std::int32_t> current_starts(width, -1);
    std::vector<std::int32_t> current_ends(width, -1);
    std::uint32_t cursor = 0;
    std::uint32_t previous_pixel = 0;
    std::uint32_t previous_first_bit = 0;
    std::uint32_t previous_region = 0;
    std::uint8_t previous_far_side = 0;
    bool has_previous_key = false;
    std::uint32_t component_count = 0;
    {
        py::gil_scoped_release release;
        for (std::uint32_t row = 0; row < height; ++row) {
            std::fill(current_starts.begin(), current_starts.end(), -1);
            std::fill(current_ends.begin(), current_ends.end(), -1);
            const std::uint64_t row_start_pixel = static_cast<std::uint64_t>(row) * width;
            const std::uint64_t row_end_pixel = row_start_pixel + width;
            while (cursor < count) {
                const std::uint32_t pixel = load_u32_le(records + 16ULL * cursor);
                if (pixel >= row_end_pixel) {
                    break;
                }
                if (pixel < row_start_pixel || pixel >= pixel_count) {
                    throw std::runtime_error("records are not ordered by output pixel");
                }
                if (has_previous_key && pixel < previous_pixel) {
                    throw std::runtime_error("records are not ordered by output pixel");
                }
                const std::uint32_t column = pixel - static_cast<std::uint32_t>(row_start_pixel);
                const std::uint32_t start = cursor;
                while (
                    cursor < count && load_u32_le(records + 16ULL * cursor) == pixel
                ) {
                    const std::uint8_t* record = records + 16ULL * cursor;
                    const std::uint16_t mask = load_u16_le(record + 4);
                    const std::uint32_t region = load_u32_le(record + 6);
                    const std::uint8_t far_side = record[14];
                    if (!is_contiguous_u16_mask(mask)) {
                        throw std::runtime_error(
                            "record lane mask must be nonzero and contiguous"
                        );
                    }
                    if (
                        region == 0 || region == std::numeric_limits<std::uint32_t>::max()
                    ) {
                        throw std::runtime_error("record region must be positive");
                    }
                    if (far_side > 1 || record[15] != 0) {
                        throw std::runtime_error("record side/reserved byte is invalid");
                    }
                    const std::uint32_t first_bit = least_u16_bit(mask);
                    const bool out_of_order = has_previous_key && (
                        pixel < previous_pixel ||
                        (pixel == previous_pixel && first_bit < previous_first_bit) ||
                        (
                            pixel == previous_pixel && first_bit == previous_first_bit &&
                            region < previous_region
                        ) ||
                        (
                            pixel == previous_pixel && first_bit == previous_first_bit &&
                            region == previous_region && far_side < previous_far_side
                        )
                    );
                    if (out_of_order) {
                        throw std::runtime_error("records are not canonically ordered");
                    }
                    has_previous_key = true;
                    previous_pixel = pixel;
                    previous_first_bit = first_bit;
                    previous_region = region;
                    previous_far_side = far_side;
                    ++cursor;
                }
                const std::uint32_t end = cursor;
                current_starts[column] = static_cast<std::int32_t>(start);
                current_ends[column] = static_cast<std::int32_t>(end);
                union_component_groups(
                    records,
                    parent,
                    rank,
                    start,
                    end,
                    start,
                    end,
                    ComponentRelation::Same
                );
                if (column > 0 && current_starts[column - 1] >= 0) {
                    union_component_groups(
                        records,
                        parent,
                        rank,
                        static_cast<std::uint32_t>(current_starts[column - 1]),
                        static_cast<std::uint32_t>(current_ends[column - 1]),
                        start,
                        end,
                        ComponentRelation::Horizontal
                    );
                }
                if (previous_starts[column] >= 0) {
                    union_component_groups(
                        records,
                        parent,
                        rank,
                        static_cast<std::uint32_t>(previous_starts[column]),
                        static_cast<std::uint32_t>(previous_ends[column]),
                        start,
                        end,
                        ComponentRelation::Vertical
                    );
                }
            }
            previous_starts.swap(current_starts);
            previous_ends.swap(current_ends);
        }
        if (cursor != count) {
            throw std::runtime_error("record pixel lies outside render_shape");
        }
        for (std::uint32_t index = 0; index < count; ++index) {
            parent[index] = component_find(parent, index);
        }
        std::vector<std::uint32_t> root_component(
            count,
            std::numeric_limits<std::uint32_t>::max()
        );
        for (std::uint32_t index = 0; index < count; ++index) {
            const std::uint32_t root = parent[index];
            if (root_component[root] == std::numeric_limits<std::uint32_t>::max()) {
                root_component[root] = component_count++;
            }
            output[index] = root_component[root];
        }
    }
    return py::make_tuple(std::move(component_ids), component_count);
}

inline std::uint32_t fixed_component_find(
    std::uint32_t* parent,
    std::uint32_t index
) {
    std::uint32_t root = index;
    while (parent[root] != root) {
        root = parent[root];
    }
    while (parent[index] != index) {
        const std::uint32_t next = parent[index];
        parent[index] = root;
        index = next;
    }
    return root;
}

inline void fixed_component_union(
    std::uint32_t* parent,
    std::uint8_t* rank,
    std::uint32_t* keys,
    std::uint32_t left,
    std::uint32_t right
) {
    std::uint32_t left_root = fixed_component_find(parent, left);
    std::uint32_t right_root = fixed_component_find(parent, right);
    if (left_root == right_root) {
        return;
    }
    if (
        rank[left_root] < rank[right_root] ||
        (rank[left_root] == rank[right_root] && keys[right_root] < keys[left_root])
    ) {
        std::swap(left_root, right_root);
    }
    parent[right_root] = left_root;
    keys[left_root] = std::min(keys[left_root], keys[right_root]);
    if (rank[left_root] == rank[right_root]) {
        if (rank[left_root] == std::numeric_limits<std::uint8_t>::max()) {
            throw std::runtime_error("repair component union rank overflow");
        }
        ++rank[left_root];
    }
}

void fixed_union_component_groups(
    const std::uint8_t* records,
    std::uint32_t* parent,
    std::uint8_t* rank,
    std::uint32_t* keys,
    std::uint32_t left_start,
    std::uint32_t left_end,
    std::uint32_t right_start,
    std::uint32_t right_end,
    ComponentRelation relation
) {
    for (std::uint32_t left = left_start; left < left_end; ++left) {
        const std::uint8_t* left_record = records + 16ULL * left;
        const std::uint32_t left_region = load_u32_le(left_record + 6);
        const std::uint16_t left_mask = load_u16_le(left_record + 4);
        for (std::uint32_t right = right_start; right < right_end; ++right) {
            const std::uint8_t* right_record = records + 16ULL * right;
            if (left_region != load_u32_le(right_record + 6)) {
                continue;
            }
            const std::uint16_t right_mask = load_u16_le(right_record + 4);
            bool adjacent = false;
            if (relation == ComponentRelation::Same) {
                adjacent = (
                    ((static_cast<std::uint32_t>(left_mask) << 1U) & right_mask) |
                    ((static_cast<std::uint32_t>(right_mask) << 1U) & left_mask)
                ) != 0;
            } else if (relation == ComponentRelation::Horizontal) {
                adjacent = (left_mask & 0x8000U) != 0 && (right_mask & 0x0001U) != 0;
            } else {
                adjacent = (left_mask & right_mask) != 0;
            }
            if (adjacent) {
                fixed_component_union(parent, rank, keys, left, right);
            }
        }
    }
}

inline bool component_member_less(
    std::uint32_t left,
    std::uint32_t right,
    const std::uint32_t* keys
) {
    return keys[left] < keys[right] || (keys[left] == keys[right] && left < right);
}

void sift_component_member_heap(
    std::uint32_t* members,
    std::uint32_t start,
    std::uint32_t end,
    const std::uint32_t* keys
) {
    std::uint32_t root = start;
    while (true) {
        const std::uint64_t child64 = 2ULL * root + 1;
        if (child64 >= end) {
            return;
        }
        std::uint32_t child = static_cast<std::uint32_t>(child64);
        if (
            child + 1 < end &&
            component_member_less(members[child], members[child + 1], keys)
        ) {
            ++child;
        }
        if (!component_member_less(members[root], members[child], keys)) {
            return;
        }
        std::swap(members[root], members[child]);
        root = child;
    }
}

std::uint32_t build_record_components_into_native(
    const py::array& raw_records,
    std::uint32_t height,
    std::uint32_t width,
    py::array parent_array,
    py::array rank_array,
    py::array member_array,
    py::array key_array,
    py::array row_workspace
) {
    if (
        !raw_records.dtype().is(py::dtype::of<std::uint8_t>()) ||
        raw_records.ndim() != 2 || raw_records.shape(1) != 16 ||
        (raw_records.flags() & py::array::c_style) == 0
    ) {
        throw py::type_error("raw_records must be a C-contiguous uint8 [n,16] array");
    }
    auto require_output = [](py::array& value, const py::dtype& dtype, const char* name) {
        if (
            !value.dtype().is(dtype) || value.ndim() != 1 ||
            (value.flags() & py::array::c_style) == 0 || !value.writeable()
        ) {
            throw py::type_error(std::string(name) + " must be a writable contiguous vector");
        }
    };
    require_output(parent_array, py::dtype::of<std::uint32_t>(), "parent_output");
    require_output(rank_array, py::dtype::of<std::uint8_t>(), "rank_output");
    require_output(member_array, py::dtype::of<std::uint32_t>(), "member_output");
    require_output(key_array, py::dtype::of<std::uint32_t>(), "key_output");
    require_output(row_workspace, py::dtype::of<std::int32_t>(), "row_workspace");
    if (height == 0 || width == 0) {
        throw py::value_error("render shape must be positive");
    }
    const std::uint64_t count64 = static_cast<std::uint64_t>(raw_records.shape(0));
    if (count64 > kRepairRecordCap) {
        throw py::value_error("repair record count exceeds the fixed arena");
    }
    if (
        static_cast<std::uint64_t>(parent_array.shape(0)) < count64 ||
        static_cast<std::uint64_t>(rank_array.shape(0)) < count64 ||
        static_cast<std::uint64_t>(member_array.shape(0)) < count64 ||
        static_cast<std::uint64_t>(key_array.shape(0)) < count64 ||
        static_cast<std::uint64_t>(row_workspace.shape(0)) < 4ULL * width
    ) {
        throw QualityNativeBudgetError("64 MiB graph arena partition exceeded");
    }
    const std::uint64_t pixel_count = static_cast<std::uint64_t>(height) * width;
    if (pixel_count * 16ULL > kUint32Max) {
        throw py::value_error("render shape must fit uint32 fine indexes");
    }
    const std::uint32_t count = static_cast<std::uint32_t>(count64);
    if (count == 0) {
        return 0;
    }
    const auto* records = static_cast<const std::uint8_t*>(raw_records.data());
    auto* parent = static_cast<std::uint32_t*>(parent_array.mutable_data());
    auto* rank = static_cast<std::uint8_t*>(rank_array.mutable_data());
    auto* members = static_cast<std::uint32_t*>(member_array.mutable_data());
    auto* keys = static_cast<std::uint32_t*>(key_array.mutable_data());
    auto* rows = static_cast<std::int32_t*>(row_workspace.mutable_data());
    auto* previous_starts = rows;
    auto* previous_ends = rows + width;
    auto* current_starts = rows + 2ULL * width;
    auto* current_ends = rows + 3ULL * width;
    std::fill(previous_starts, previous_starts + 2ULL * width, -1);
    std::fill(rank, rank + count, 0);
    for (std::uint32_t index = 0; index < count; ++index) {
        parent[index] = index;
        members[index] = index;
    }
    std::uint32_t cursor = 0;
    std::uint32_t previous_pixel = 0;
    std::uint32_t previous_first_bit = 0;
    std::uint32_t previous_region = 0;
    std::uint8_t previous_far_side = 0;
    bool has_previous_key = false;
    {
        py::gil_scoped_release release;
        for (std::uint32_t row = 0; row < height; ++row) {
            std::fill(current_starts, current_starts + width, -1);
            std::fill(current_ends, current_ends + width, -1);
            const std::uint64_t row_start_pixel = static_cast<std::uint64_t>(row) * width;
            const std::uint64_t row_end_pixel = row_start_pixel + width;
            while (cursor < count) {
                const std::uint32_t pixel = load_u32_le(records + 16ULL * cursor);
                if (pixel >= row_end_pixel) {
                    break;
                }
                if (pixel < row_start_pixel || pixel >= pixel_count) {
                    throw std::runtime_error("records are not ordered by output pixel");
                }
                const std::uint32_t column = pixel - static_cast<std::uint32_t>(row_start_pixel);
                const std::uint32_t start = cursor;
                while (cursor < count && load_u32_le(records + 16ULL * cursor) == pixel) {
                    const std::uint8_t* record = records + 16ULL * cursor;
                    const std::uint16_t mask = load_u16_le(record + 4);
                    const std::uint32_t region = load_u32_le(record + 6);
                    const std::uint8_t far_side = record[14];
                    if (
                        !is_contiguous_u16_mask(mask) || region == 0 || region == kUint32Max ||
                        far_side > 1 || record[15] != 0
                    ) {
                        throw std::runtime_error("repair record identity is invalid");
                    }
                    const std::uint32_t first_bit = least_u16_bit(mask);
                    const bool out_of_order = has_previous_key && (
                        pixel < previous_pixel ||
                        (pixel == previous_pixel && first_bit < previous_first_bit) ||
                        (
                            pixel == previous_pixel && first_bit == previous_first_bit &&
                            region < previous_region
                        ) ||
                        (
                            pixel == previous_pixel && first_bit == previous_first_bit &&
                            region == previous_region && far_side < previous_far_side
                        )
                    );
                    if (out_of_order) {
                        throw std::runtime_error("records are not canonically ordered");
                    }
                    has_previous_key = true;
                    previous_pixel = pixel;
                    previous_first_bit = first_bit;
                    previous_region = region;
                    previous_far_side = far_side;
                    keys[cursor] = static_cast<std::uint32_t>(
                        row_start_pixel * 16ULL + 16ULL * column + first_bit
                    );
                    ++cursor;
                }
                const std::uint32_t end = cursor;
                current_starts[column] = static_cast<std::int32_t>(start);
                current_ends[column] = static_cast<std::int32_t>(end);
                fixed_union_component_groups(
                    records, parent, rank, keys, start, end, start, end,
                    ComponentRelation::Same
                );
                if (column > 0 && current_starts[column - 1] >= 0) {
                    fixed_union_component_groups(
                        records, parent, rank, keys,
                        static_cast<std::uint32_t>(current_starts[column - 1]),
                        static_cast<std::uint32_t>(current_ends[column - 1]),
                        start, end, ComponentRelation::Horizontal
                    );
                }
                if (previous_starts[column] >= 0) {
                    fixed_union_component_groups(
                        records, parent, rank, keys,
                        static_cast<std::uint32_t>(previous_starts[column]),
                        static_cast<std::uint32_t>(previous_ends[column]),
                        start, end, ComponentRelation::Vertical
                    );
                }
            }
            std::swap(previous_starts, current_starts);
            std::swap(previous_ends, current_ends);
        }
        if (cursor != count) {
            throw std::runtime_error("record pixel lies outside render_shape");
        }
        for (std::uint32_t index = 0; index < count; ++index) {
            const std::uint32_t root = fixed_component_find(parent, index);
            parent[index] = root;
            keys[index] = keys[root];
        }
        for (std::int64_t start = static_cast<std::int64_t>(count) / 2 - 1; start >= 0; --start) {
            sift_component_member_heap(
                members,
                static_cast<std::uint32_t>(start),
                count,
                keys
            );
        }
        for (std::uint32_t end = count; end > 1; --end) {
            std::swap(members[0], members[end - 1]);
            sift_component_member_heap(members, 0, end - 1, keys);
        }
        std::uint32_t component_count = 0;
        std::uint32_t previous_key = kUint32Max;
        for (std::uint32_t position = 0; position < count; ++position) {
            const std::uint32_t index = members[position];
            const std::uint32_t key = keys[index];
            if (position == 0 || key != previous_key) {
                previous_key = key;
                ++component_count;
            }
            parent[index] = component_count - 1;
        }
        return component_count;
    }
}

py::tuple build_component_member_order_native(
    const py::array& component_ids,
    std::uint32_t component_count
) {
    validate_vector(
        component_ids,
        py::dtype::of<std::uint32_t>(),
        "component_ids"
    );
    const std::uint64_t count = static_cast<std::uint64_t>(component_ids.shape(0));
    if (count > kRepairRecordCap) {
        throw py::value_error("component id count exceeds the fixed arena");
    }
    if (
        (count == 0 && component_count != 0) ||
        (count != 0 && (component_count == 0 || component_count > count))
    ) {
        throw py::value_error("component_count is inconsistent with component_ids");
    }
    py::array_t<std::uint32_t> members({component_ids.shape(0)});
    py::array_t<std::uint32_t> offsets({
        static_cast<py::ssize_t>(component_count) + 1,
    });
    auto* member_values = members.mutable_data();
    auto* offset_values = offsets.mutable_data();
    std::fill(offset_values, offset_values + component_count + 1ULL, 0U);
    const auto* ids = static_cast<const std::uint32_t*>(component_ids.data());
    {
        py::gil_scoped_release release;
        for (std::uint64_t index = 0; index < count; ++index) {
            const std::uint32_t component = ids[index];
            if (component >= component_count) {
                throw std::runtime_error("component id lies outside component_count");
            }
            ++offset_values[component + 1];
        }
        for (std::uint32_t component = 0; component < component_count; ++component) {
            offset_values[component + 1] += offset_values[component];
        }
        std::vector<std::uint32_t> cursors(
            offset_values,
            offset_values + component_count
        );
        for (std::uint64_t index = 0; index < count; ++index) {
            const std::uint32_t component = ids[index];
            member_values[cursors[component]++] = static_cast<std::uint32_t>(index);
        }
    }
    return py::make_tuple(std::move(members), std::move(offsets));
}

py::tuple expand_repair_scatter_native(
    const py::array& raw_records,
    std::uint32_t height,
    std::uint32_t width
) {
    if (
        !raw_records.dtype().is(py::dtype::of<std::uint8_t>()) ||
        raw_records.ndim() != 2 || raw_records.shape(1) != 16 ||
        (raw_records.flags() & py::array::c_style) == 0
    ) {
        throw py::type_error("raw_records must be a C-contiguous uint8 [n,16] array");
    }
    if (height == 0 || width == 0) {
        throw py::value_error("render shape must be positive");
    }
    const std::uint64_t pixel_count = static_cast<std::uint64_t>(height) * width;
    if (pixel_count * 16ULL > kUint32Max) {
        throw py::value_error("render shape must fit uint32 fine indexes");
    }
    const auto* records = static_cast<const std::uint8_t*>(raw_records.data());
    std::uint64_t lane_count = 0;
    for (py::ssize_t index = 0; index < raw_records.shape(0); ++index) {
        const std::uint8_t* record = records + 16 * index;
        const std::uint32_t pixel = load_u32_le(record);
        const std::uint16_t mask = load_u16_le(record + 4);
        if (pixel >= pixel_count) {
            throw py::value_error("repair scatter pixel lies outside the band");
        }
        if (!is_contiguous_u16_mask(mask)) {
            throw py::value_error("repair scatter mask must be nonzero and contiguous");
        }
        if (record[13] < 1 || record[13] > 3) {
            throw py::value_error("repair scatter backend must be final");
        }
        lane_count += count_u16_bits(mask);
    }
    if (lane_count > static_cast<std::uint64_t>(std::numeric_limits<py::ssize_t>::max())) {
        throw py::value_error("repair scatter lane count exceeds platform size");
    }
    const auto output_count = static_cast<py::ssize_t>(lane_count);
    py::array_t<std::int64_t> indexes({output_count});
    py::array_t<float> colours({output_count, static_cast<py::ssize_t>(3)});
    auto* output_indexes = indexes.mutable_data();
    auto* output_colours = colours.mutable_data();
    std::uint64_t cursor = 0;
    {
        py::gil_scoped_release release;
        for (py::ssize_t index = 0; index < raw_records.shape(0); ++index) {
            const std::uint8_t* record = records + 16 * index;
            const std::uint32_t pixel = load_u32_le(record);
            const std::uint16_t mask = load_u16_le(record + 4);
            for (std::uint32_t lane = 0; lane < 16; ++lane) {
                if ((mask & (1U << lane)) == 0) {
                    continue;
                }
                output_indexes[cursor] = static_cast<std::int64_t>(
                    static_cast<std::uint64_t>(pixel) * 16ULL + lane
                );
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    output_colours[3 * cursor + channel] = static_cast<float>(
                        record[10 + channel]
                    );
                }
                ++cursor;
            }
        }
    }
    return py::make_tuple(std::move(indexes), std::move(colours));
}

std::uint64_t build_background_repair_bits_into_native(
    const py::array& raw_records,
    const py::array& coverage_count,
    py::array output,
    py::array backend_lane_counts
) {
    if (
        !raw_records.dtype().is(py::dtype::of<std::uint8_t>()) ||
        raw_records.ndim() != 2 || raw_records.shape(1) != 16 ||
        (raw_records.flags() & py::array::c_style) == 0
    ) {
        throw py::type_error("raw_records must be a C-contiguous uint8 [n,16] array");
    }
    if (
        !coverage_count.dtype().is(py::dtype::of<std::uint8_t>()) ||
        coverage_count.ndim() != 2 ||
        (coverage_count.flags() & py::array::c_style) == 0
    ) {
        throw py::type_error("coverage_count must be a C-contiguous uint8 image");
    }
    if (
        !output.dtype().is(py::dtype::of<std::uint8_t>()) || output.ndim() != 2 ||
        (output.flags() & py::array::c_style) == 0 || !output.writeable() ||
        output.shape(0) != coverage_count.shape(0) ||
        output.shape(1) != coverage_count.shape(1)
    ) {
        throw py::type_error(
            "output must be a matching writable C-contiguous uint8 image"
        );
    }
    if (
        !backend_lane_counts.dtype().is(py::dtype::of<std::uint64_t>()) ||
        backend_lane_counts.ndim() != 1 || backend_lane_counts.shape(0) != 3 ||
        (backend_lane_counts.flags() & py::array::c_style) == 0 ||
        !backend_lane_counts.writeable()
    ) {
        throw py::type_error(
            "backend_lane_counts must be a writable C-contiguous uint64[3] vector"
        );
    }
    if (coverage_count.shape(0) <= 0 || coverage_count.shape(1) <= 0) {
        throw py::value_error("coverage_count shape must be positive");
    }
    const std::uint64_t height = static_cast<std::uint64_t>(coverage_count.shape(0));
    const std::uint64_t width = static_cast<std::uint64_t>(coverage_count.shape(1));
    const std::uint64_t pixel_count = height * width;
    if (pixel_count * 16ULL > kUint32Max) {
        throw py::value_error("coverage_count shape must fit uint32 fine indexes");
    }
    const auto* records = static_cast<const std::uint8_t*>(raw_records.data());
    const auto* coverage = static_cast<const std::uint8_t*>(coverage_count.data());
    auto* bits = static_cast<std::uint8_t*>(output.mutable_data());
    auto* lane_counts = static_cast<std::uint64_t*>(backend_lane_counts.mutable_data());
    std::uint64_t planned_lane_count = 0;
    {
        py::gil_scoped_release release;
        std::fill(lane_counts, lane_counts + 3, 0ULL);
        for (std::uint64_t pixel = 0; pixel < pixel_count; ++pixel) {
            const std::uint8_t count = coverage[pixel];
            if (count > 16) {
                throw py::value_error("coverage_count contains a value above 16");
            }
            bits[pixel] = count == 0 ? 0x02U : (count < 16 ? 0x01U : 0x00U);
        }
        for (py::ssize_t index = 0; index < raw_records.shape(0); ++index) {
            const std::uint8_t* record = records + 16 * index;
            const std::uint32_t pixel = load_u32_le(record);
            const std::uint16_t mask = load_u16_le(record + 4);
            const std::uint8_t backend = record[13];
            if (pixel >= pixel_count) {
                throw py::value_error("repair diagnostic pixel lies outside coverage_count");
            }
            if (!is_contiguous_u16_mask(mask)) {
                throw py::value_error(
                    "repair diagnostic mask must be nonzero and contiguous"
                );
            }
            if (backend < 1 || backend > 3) {
                throw py::value_error("repair diagnostic backend must be final");
            }
            const std::uint64_t lanes = count_u16_bits(mask);
            if (lanes > std::numeric_limits<std::uint64_t>::max() - planned_lane_count) {
                throw std::overflow_error("repair diagnostic lane count exceeds uint64");
            }
            planned_lane_count += lanes;
            if (lanes > std::numeric_limits<std::uint64_t>::max() - lane_counts[backend - 1]) {
                throw std::overflow_error("repair backend lane count exceeds uint64");
            }
            lane_counts[backend - 1] += lanes;
            if (backend == 1) {
                bits[pixel] |= 0x04U;
            } else if (backend == 2) {
                bits[pixel] |= 0x18U;
            } else {
                bits[pixel] |= 0x28U;
            }
        }
    }
    return planned_lane_count;
}

std::uint64_t build_hole_run_histograms_into_native(
    const py::array& raw_records,
    std::uint32_t width,
    py::array lane_histogram,
    py::array span_histogram
) {
    if (
        !raw_records.dtype().is(py::dtype::of<std::uint8_t>()) ||
        raw_records.ndim() != 2 || raw_records.shape(1) != 16 ||
        (raw_records.flags() & py::array::c_style) == 0
    ) {
        throw py::type_error("raw_records must be a C-contiguous uint8 [n,16] array");
    }
    if (width == 0) {
        throw py::value_error("histogram width must be positive");
    }
    const std::uint64_t expected_lane_bins = 16ULL * (
        static_cast<std::uint64_t>(width) + 1ULL
    );
    const std::uint64_t expected_span_bins = static_cast<std::uint64_t>(width) + 1ULL;
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
    const auto* records = static_cast<const std::uint8_t*>(raw_records.data());
    auto* lane_bins = static_cast<std::uint64_t*>(lane_histogram.mutable_data());
    auto* span_bins = static_cast<std::uint64_t*>(span_histogram.mutable_data());
    std::uint64_t planned_lane_count = 0;
    bool active = false;
    std::uint64_t active_row = 0;
    std::uint64_t active_start = 0;
    std::uint64_t active_end = 0;
    {
        py::gil_scoped_release release;
        std::fill(lane_bins, lane_bins + lane_histogram.shape(0), 0ULL);
        std::fill(span_bins, span_bins + span_histogram.shape(0), 0ULL);
        const auto flush_active = [&]() {
            if (!active) {
                return;
            }
            const std::uint64_t lane_length = active_end - active_start + 1ULL;
            const std::uint64_t pixel_span = active_end / 16ULL - active_start / 16ULL + 1ULL;
            if (lane_bins[lane_length] == std::numeric_limits<std::uint64_t>::max()) {
                throw std::overflow_error("hole-run lane histogram exceeds uint64");
            }
            if (span_bins[pixel_span] == std::numeric_limits<std::uint64_t>::max()) {
                throw std::overflow_error("hole-run span histogram exceeds uint64");
            }
            ++lane_bins[lane_length];
            ++span_bins[pixel_span];
        };
        for (py::ssize_t index = 0; index < raw_records.shape(0); ++index) {
            const std::uint8_t* record = records + 16 * index;
            const std::uint32_t pixel = load_u32_le(record);
            const std::uint16_t mask = load_u16_le(record + 4);
            if (!is_contiguous_u16_mask(mask)) {
                throw py::value_error(
                    "hole-run mask must be nonzero and contiguous"
                );
            }
            if (record[13] < 1 || record[13] > 3) {
                throw py::value_error("hole-run backend must be final");
            }
            const std::uint64_t lanes = count_u16_bits(mask);
            if (lanes > std::numeric_limits<std::uint64_t>::max() - planned_lane_count) {
                throw std::overflow_error("hole-run lane count exceeds uint64");
            }
            planned_lane_count += lanes;
            const std::uint64_t row = static_cast<std::uint64_t>(pixel) / width;
            const std::uint64_t column = static_cast<std::uint64_t>(pixel) % width;
            const std::uint64_t start = 16ULL * column + least_u16_bit(mask);
            const std::uint64_t end = start + lanes - 1ULL;
            if (
                active &&
                (row < active_row || (row == active_row && start <= active_end))
            ) {
                throw py::value_error("hole-run records are not in canonical order");
            }
            if (active && row == active_row && start == active_end + 1ULL) {
                active_end = end;
                continue;
            }
            flush_active();
            active = true;
            active_row = row;
            active_start = start;
            active_end = end;
        }
        flush_active();
    }
    return planned_lane_count;
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

std::uint64_t scatter_repair_records_float_into_native(
    const py::array& raw_records,
    std::uint32_t pixel_offset,
    py::array output
) {
    if (
        !raw_records.dtype().is(py::dtype::of<std::uint8_t>()) ||
        raw_records.ndim() != 2 || raw_records.shape(1) != 16 ||
        (raw_records.flags() & py::array::c_style) == 0
    ) {
        throw py::type_error("raw_records must be a C-contiguous uint8 [n,16] array");
    }
    if (
        !output.dtype().is(py::dtype::of<float>()) || output.ndim() != 2 ||
        output.shape(1) != 3 || (output.flags() & py::array::c_style) == 0 ||
        !output.writeable() || output.shape(0) % 16 != 0
    ) {
        throw py::type_error(
            "output must be a writable C-contiguous float32 [pixel_count*16,3] array"
        );
    }
    const std::uint64_t pixel_count =
        static_cast<std::uint64_t>(output.shape(0)) / 16ULL;
    const std::uint64_t pixel_end = static_cast<std::uint64_t>(pixel_offset) + pixel_count;
    if (pixel_end > static_cast<std::uint64_t>(kUint32Max) + 1ULL) {
        throw py::value_error("repair scatter pixel window exceeds uint32");
    }
    const auto* records = static_cast<const std::uint8_t*>(raw_records.data());
    auto* output_values = static_cast<float*>(output.mutable_data());
    std::uint64_t lane_count = 0;
    {
        py::gil_scoped_release release;
        for (py::ssize_t index = 0; index < raw_records.shape(0); ++index) {
            const std::uint8_t* record = records + 16 * index;
            const std::uint32_t pixel = load_u32_le(record);
            const std::uint16_t mask = load_u16_le(record + 4);
            if (pixel < pixel_offset || pixel >= pixel_end) {
                throw std::runtime_error("repair scatter record lies outside its pixel window");
            }
            if (!is_contiguous_u16_mask(mask)) {
                throw py::value_error("repair scatter mask must be nonzero and contiguous");
            }
            if (record[13] < 1 || record[13] > 3) {
                throw py::value_error("repair scatter backend must be final");
            }
            const std::uint64_t local_pixel = pixel - pixel_offset;
            for (std::uint32_t lane = 0; lane < 16; ++lane) {
                if ((mask & (1U << lane)) == 0) {
                    continue;
                }
                const std::uint64_t destination = 3ULL * (16ULL * local_pixel + lane);
                output_values[destination] = static_cast<float>(record[10]);
                output_values[destination + 1] = static_cast<float>(record[11]);
                output_values[destination + 2] = static_cast<float>(record[12]);
                ++lane_count;
            }
        }
    }
    return lane_count;
}

std::uint64_t materialize_repair_scatter_u8_into_native(
    const py::array& raw_records,
    std::uint32_t pixel_offset,
    py::array colour_output,
    py::array mask_output
) {
    if (
        !raw_records.dtype().is(py::dtype::of<std::uint8_t>()) ||
        raw_records.ndim() != 2 || raw_records.shape(1) != 16 ||
        (raw_records.flags() & py::array::c_style) == 0
    ) {
        throw py::type_error("raw_records must be a C-contiguous uint8 [n,16] array");
    }
    if (
        !colour_output.dtype().is(py::dtype::of<std::uint8_t>()) ||
        colour_output.ndim() != 2 || colour_output.shape(1) != 3 ||
        (colour_output.flags() & py::array::c_style) == 0 ||
        !colour_output.writeable()
    ) {
        throw py::type_error(
            "colour_output must be a writable C-contiguous uint8 [lane_count,3] array"
        );
    }
    if (
        !mask_output.dtype().is(py::dtype::of<bool>()) || mask_output.ndim() != 1 ||
        (mask_output.flags() & py::array::c_style) == 0 ||
        !mask_output.writeable() || mask_output.shape(0) != colour_output.shape(0) ||
        mask_output.shape(0) % 16 != 0
    ) {
        throw py::type_error(
            "mask_output must be a matching writable C-contiguous bool vector"
        );
    }
    const std::uint64_t pixel_count =
        static_cast<std::uint64_t>(mask_output.shape(0)) / 16ULL;
    const std::uint64_t pixel_end = static_cast<std::uint64_t>(pixel_offset) + pixel_count;
    if (pixel_end > static_cast<std::uint64_t>(kUint32Max) + 1ULL) {
        throw py::value_error("repair scatter pixel window exceeds uint32");
    }
    auto* colours = static_cast<std::uint8_t*>(colour_output.mutable_data());
    auto* masks = static_cast<bool*>(mask_output.mutable_data());
    std::fill(colours, colours + 3ULL * mask_output.shape(0), 0);
    std::fill(masks, masks + mask_output.shape(0), false);
    const auto* records = static_cast<const std::uint8_t*>(raw_records.data());
    std::uint64_t lane_count = 0;
    {
        py::gil_scoped_release release;
        for (py::ssize_t index = 0; index < raw_records.shape(0); ++index) {
            const std::uint8_t* record = records + 16 * index;
            const std::uint32_t pixel = load_u32_le(record);
            const std::uint16_t mask = load_u16_le(record + 4);
            if (pixel < pixel_offset || pixel >= pixel_end) {
                throw std::runtime_error("repair scatter record lies outside its pixel window");
            }
            if (!is_contiguous_u16_mask(mask)) {
                throw py::value_error("repair scatter mask must be nonzero and contiguous");
            }
            if (record[13] < 1 || record[13] > 3) {
                throw py::value_error("repair scatter backend must be final");
            }
            const std::uint64_t local_pixel = pixel - pixel_offset;
            for (std::uint32_t lane = 0; lane < 16; ++lane) {
                if ((mask & (1U << lane)) == 0) {
                    continue;
                }
                const std::uint64_t destination = 16ULL * local_pixel + lane;
                if (masks[destination]) {
                    throw std::runtime_error("repair scatter records overlap");
                }
                masks[destination] = true;
                colours[3ULL * destination] = record[10];
                colours[3ULL * destination + 1] = record[11];
                colours[3ULL * destination + 2] = record[12];
                ++lane_count;
            }
        }
    }
    return lane_count;
}

inline std::uint64_t checked_local_add(std::uint64_t left, std::uint64_t right) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error("local-strip counter exceeds uint64");
    }
    return left + right;
}

inline std::uint64_t checked_local_mul(std::uint64_t left, std::uint64_t right) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error("local-strip product exceeds uint64");
    }
    return left * right;
}

struct LocalStatistics {
    std::uint64_t evaluated_slots = 0;
    std::uint64_t physical_sample_reads = 0;
    std::uint64_t filled_runs = 0;
    std::uint64_t eligible_slots = 0;
    std::uint64_t unsafe_donor_slots = 0;
    std::uint64_t slot_consumed = 0;
    std::uint64_t sample_consumed = 0;
    std::uint64_t budget_skipped_runs = 0;
};

using LocalContext = std::array<std::array<std::uint8_t, 3>, 3>;

bool read_local_context(
    const std::uint8_t* coverage,
    const std::uint32_t* regions,
    const std::array<const std::uint8_t*, 3>& colours,
    std::int64_t height,
    std::int64_t width,
    std::int64_t row,
    const std::array<std::int64_t, 3>& columns,
    std::uint32_t region,
    LocalStatistics& statistics,
    LocalContext& output
) {
    if (row < 0 || row >= height) {
        return false;
    }
    for (std::size_t index = 0; index < columns.size(); ++index) {
        const std::int64_t column = columns[index];
        if (column < 0 || column >= width) {
            return false;
        }
        statistics.physical_sample_reads = checked_local_add(
            statistics.physical_sample_reads,
            1
        );
        const auto sample = static_cast<std::uint64_t>(row * width + column);
        if (coverage[sample] == 0 || regions[sample] != region) {
            return false;
        }
        for (std::size_t channel = 0; channel < 3; ++channel) {
            output[index][channel] = colours[channel][sample];
        }
    }
    return true;
}

bool local_donor_is_safe(
    const std::uint8_t* coverage,
    const std::uint32_t* regions,
    std::int64_t height,
    std::int64_t width,
    std::int64_t row,
    std::int64_t column,
    std::uint32_t region,
    std::int64_t radius,
    LocalStatistics& statistics
) {
    const std::int64_t start_row = std::max<std::int64_t>(0, row - radius);
    const std::int64_t end_row = std::min<std::int64_t>(height, row + radius + 1);
    const std::int64_t start_column = std::max<std::int64_t>(0, column - radius);
    const std::int64_t end_column = std::min<std::int64_t>(width, column + radius + 1);
    for (std::int64_t sample_row = start_row; sample_row < end_row; ++sample_row) {
        for (
            std::int64_t sample_column = start_column;
            sample_column < end_column;
            ++sample_column
        ) {
            statistics.physical_sample_reads = checked_local_add(
                statistics.physical_sample_reads,
                1
            );
            const auto sample = static_cast<std::uint64_t>(
                sample_row * width + sample_column
            );
            if (coverage[sample] != 16 || regions[sample] != region) {
                return false;
            }
        }
    }
    return true;
}

inline std::int32_t local_luma(const std::array<std::uint8_t, 3>& bgr) {
    return (
        29 * static_cast<std::int32_t>(bgr[0]) +
        150 * static_cast<std::int32_t>(bgr[1]) +
        77 * static_cast<std::int32_t>(bgr[2]) +
        128
    ) >> 8;
}

std::uint64_t local_context_score(
    const LocalContext& actual,
    const LocalContext& candidate
) {
    std::uint64_t bgr_l1 = 0;
    std::array<std::int32_t, 3> actual_luma{};
    std::array<std::int32_t, 3> candidate_luma{};
    for (std::size_t index = 0; index < 3; ++index) {
        for (std::size_t channel = 0; channel < 3; ++channel) {
            bgr_l1 += static_cast<std::uint64_t>(std::abs(
                static_cast<std::int32_t>(actual[index][channel]) -
                static_cast<std::int32_t>(candidate[index][channel])
            ));
        }
        actual_luma[index] = local_luma(actual[index]);
        candidate_luma[index] = local_luma(candidate[index]);
    }
    std::uint64_t difference_l1 = 0;
    for (std::size_t index = 0; index < 2; ++index) {
        difference_l1 += static_cast<std::uint64_t>(std::abs(
            (actual_luma[index + 1] - actual_luma[index]) -
            (candidate_luma[index + 1] - candidate_luma[index])
        ));
    }
    return 2 * bgr_l1 + difference_l1;
}

void plan_one_local_run_in_place(
    std::uint8_t* output_records,
    std::uint64_t record_count,
    const NativeRepairRun& run,
    const std::uint8_t* coverage,
    const std::uint32_t* regions,
    const std::array<const std::uint8_t*, 3>& colours,
    std::int64_t height,
    std::int64_t width,
    std::uint32_t local_limit_px,
    std::uint32_t search_limit,
    std::uint32_t safe_radius,
    std::uint64_t slot_remaining,
    std::uint64_t sample_remaining,
    LocalStatistics& statistics
) {
    if (
        run.record_start >= run.record_end || run.record_end > record_count ||
        run.row >= height || run.start_fine > run.end_fine || run.far_side > 1
    ) {
        throw py::value_error("local-strip run is malformed");
    }
    const std::uint64_t lane_count =
        static_cast<std::uint64_t>(run.end_fine) - run.start_fine + 1;
    if (lane_count > 16ULL * local_limit_px) {
        return;
    }
    const std::uint32_t first_pixel = run.start_fine / 16;
    const std::uint32_t last_pixel = run.end_fine / 16;
    const std::uint32_t donor_count = last_pixel - first_pixel + 1;
    if (run.record_end - run.record_start != donor_count) {
        throw py::value_error("local-strip run/record cardinality differs");
    }
    const std::uint64_t slot_charge = checked_local_mul(5, search_limit);
    const std::uint64_t support = checked_local_mul(
        2ULL * safe_radius + 1,
        2ULL * safe_radius + 1
    );
    const std::uint64_t per_slot = checked_local_add(
        3,
        checked_local_mul(donor_count, support)
    );
    const std::uint64_t sample_charge = checked_local_add(
        3,
        checked_local_mul(slot_charge, per_slot)
    );
    if (
        slot_charge > slot_remaining - statistics.slot_consumed ||
        sample_charge > sample_remaining - statistics.sample_consumed
    ) {
        statistics.budget_skipped_runs = checked_local_add(
            statistics.budget_skipped_runs,
            1
        );
        return;
    }
    statistics.slot_consumed = checked_local_add(statistics.slot_consumed, slot_charge);
    statistics.sample_consumed = checked_local_add(statistics.sample_consumed, sample_charge);
    statistics.evaluated_slots = checked_local_add(statistics.evaluated_slots, slot_charge);
    const std::int64_t direction = run.far_side == 0 ? -1 : 1;
    const std::int64_t anchor_fine = run.far_side == 0
        ? static_cast<std::int64_t>(run.start_fine) - 1
        : static_cast<std::int64_t>(run.end_fine) + 1;
    const std::int64_t boundary_column = anchor_fine >= 0
        ? anchor_fine / 16
        : -((-anchor_fine + 15) / 16);
    const std::array<std::int64_t, 3> actual_columns{
        boundary_column + 2 * direction,
        boundary_column + direction,
        boundary_column,
    };
    LocalContext actual{};
    if (!read_local_context(
        coverage,
        regions,
        colours,
        height,
        width,
        run.row,
        actual_columns,
        run.region,
        statistics,
        actual
    )) {
        return;
    }
    constexpr std::array<std::int64_t, 5> row_deltas{0, -1, 1, -2, 2};
    bool best_found = false;
    std::uint64_t best_score = 0;
    std::uint64_t best_ordinal = 0;
    std::int64_t best_row = 0;
    std::int64_t best_start = 0;
    for (std::uint32_t offset = 1; offset <= search_limit; ++offset) {
        for (std::size_t row_index = 0; row_index < row_deltas.size(); ++row_index) {
            const std::uint64_t ordinal = (offset - 1ULL) * 5 + row_index;
            const std::int64_t candidate_row =
                static_cast<std::int64_t>(run.row) + row_deltas[row_index];
            const std::int64_t candidate_start =
                boundary_column + direction * static_cast<std::int64_t>(offset);
            const std::array<std::int64_t, 3> candidate_columns{
                candidate_start + 3 * direction,
                candidate_start + 2 * direction,
                candidate_start + direction,
            };
            bool legal = candidate_row >= 0 && candidate_row < height;
            for (const std::int64_t column : candidate_columns) {
                legal = legal && column >= 0 && column < width;
            }
            for (std::uint32_t donor_index = 0; donor_index < donor_count; ++donor_index) {
                const std::int64_t column =
                    candidate_start - static_cast<std::int64_t>(donor_index) * direction;
                legal = legal && column >= 0 && column < width;
                legal = legal && (column - boundary_column) * direction > 0;
            }
            if (!legal) {
                continue;
            }
            LocalContext candidate{};
            if (!read_local_context(
                coverage,
                regions,
                colours,
                height,
                width,
                candidate_row,
                candidate_columns,
                run.region,
                statistics,
                candidate
            )) {
                continue;
            }
            bool safe = true;
            for (std::uint32_t donor_index = 0; donor_index < donor_count; ++donor_index) {
                const std::int64_t donor_column =
                    candidate_start - static_cast<std::int64_t>(donor_index) * direction;
                if (!local_donor_is_safe(
                    coverage,
                    regions,
                    height,
                    width,
                    candidate_row,
                    donor_column,
                    run.region,
                    safe_radius,
                    statistics
                )) {
                    safe = false;
                    break;
                }
            }
            if (!safe) {
                statistics.unsafe_donor_slots = checked_local_add(
                    statistics.unsafe_donor_slots,
                    1
                );
                continue;
            }
            statistics.eligible_slots = checked_local_add(statistics.eligible_slots, 1);
            const std::uint64_t score = local_context_score(actual, candidate);
            if (
                !best_found || score < best_score ||
                (score == best_score && ordinal < best_ordinal)
            ) {
                best_found = true;
                best_score = score;
                best_ordinal = ordinal;
                best_row = candidate_row;
                best_start = candidate_start;
            }
        }
    }
    if (!best_found) {
        return;
    }
    for (std::uint32_t donor_index = 0; donor_index < donor_count; ++donor_index) {
        const std::uint32_t record_index = run.far_side == 1
            ? run.record_end - 1 - donor_index
            : run.record_start + donor_index;
        std::uint8_t* record = output_records + 16ULL * record_index;
        const std::int64_t donor_column =
            best_start - static_cast<std::int64_t>(donor_index) * direction;
        const auto sample = static_cast<std::uint64_t>(best_row * width + donor_column);
        record[10] = colours[0][sample];
        record[11] = colours[1][sample];
        record[12] = colours[2][sample];
        record[13] = 1;
    }
    statistics.filled_runs = checked_local_add(statistics.filled_runs, 1);
}

py::dict local_statistics_object(const LocalStatistics& statistics) {
    py::dict result;
    result["evaluated_slots"] = statistics.evaluated_slots;
    result["physical_sample_reads"] = statistics.physical_sample_reads;
    result["filled_runs"] = statistics.filled_runs;
    result["eligible_slots"] = statistics.eligible_slots;
    result["unsafe_donor_slots"] = statistics.unsafe_donor_slots;
    result["slot_consumed"] = statistics.slot_consumed;
    result["sample_consumed"] = statistics.sample_consumed;
    result["budget_skipped_runs"] = statistics.budget_skipped_runs;
    return result;
}

py::dict plan_local_strips_in_place_native(
    py::array raw_records,
    const py::array& coverage_count,
    const py::array& pure_region_id,
    const py::array& blue,
    const py::array& green,
    const py::array& red,
    std::uint32_t local_limit_px,
    std::uint32_t search_limit,
    std::uint32_t safe_radius,
    std::uint64_t slot_remaining,
    std::uint64_t sample_remaining
) {
    if (
        !raw_records.dtype().is(py::dtype::of<std::uint8_t>()) ||
        raw_records.ndim() != 2 || raw_records.shape(1) != 16 ||
        (raw_records.flags() & py::array::c_style) == 0 || !raw_records.writeable()
    ) {
        throw py::type_error(
            "raw_records must be a writable C-contiguous uint8 [n,16] array"
        );
    }
    validate_array(coverage_count, py::dtype::of<std::uint8_t>(), "coverage_count");
    validate_array(pure_region_id, py::dtype::of<std::uint32_t>(), "pure_region_id");
    validate_array(blue, py::dtype::of<std::uint8_t>(), "blue");
    validate_array(green, py::dtype::of<std::uint8_t>(), "green");
    validate_array(red, py::dtype::of<std::uint8_t>(), "red");
    const std::int64_t height = coverage_count.shape(0);
    const std::int64_t width = coverage_count.shape(1);
    for (const py::array* plane : {&pure_region_id, &blue, &green, &red}) {
        if (plane->shape(0) != height || plane->shape(1) != width) {
            throw py::value_error("local-strip rasters must have one shape");
        }
    }
    if (local_limit_px == 0 || search_limit == 0 || safe_radius == 0) {
        throw py::value_error("local-strip limits must be positive");
    }
    const std::uint64_t record_count = static_cast<std::uint64_t>(raw_records.shape(0));
    if (record_count > kRepairRecordCap) {
        throw py::value_error("repair record count exceeds the fixed arena");
    }
    auto* records = static_cast<std::uint8_t*>(raw_records.mutable_data());
    const auto* coverage = static_cast<const std::uint8_t*>(coverage_count.data());
    const auto* regions = static_cast<const std::uint32_t*>(pure_region_id.data());
    const std::array<const std::uint8_t*, 3> colours{
        static_cast<const std::uint8_t*>(blue.data()),
        static_cast<const std::uint8_t*>(green.data()),
        static_cast<const std::uint8_t*>(red.data()),
    };
    LocalStatistics statistics;
    NativeRepairRun active{};
    bool has_active = false;
    bool has_previous_key = false;
    std::uint32_t previous_pixel = 0;
    std::uint32_t previous_first_bit = 0;
    std::uint32_t previous_region = 0;
    std::uint8_t previous_far_side = 0;
    for (std::uint32_t index = 0; index < record_count; ++index) {
        const std::uint8_t* record = records + 16ULL * index;
        const std::uint32_t pixel = load_u32_le(record);
        const std::uint16_t mask = load_u16_le(record + 4);
        const std::uint32_t region = load_u32_le(record + 6);
        const std::uint8_t far_side = record[14];
        if (pixel >= static_cast<std::uint64_t>(height) * width) {
            throw std::runtime_error("record pixel lies outside render_shape");
        }
        if (!is_contiguous_u16_mask(mask)) {
            throw std::runtime_error("record lane mask must be nonzero and contiguous");
        }
        if (region == 0 || region == kUint32Max || far_side > 1 || record[15] != 0) {
            throw std::runtime_error("record region/side identity is invalid");
        }
        const std::uint32_t first_bit = least_u16_bit(mask);
        const bool out_of_order = has_previous_key && (
            pixel < previous_pixel ||
            (pixel == previous_pixel && first_bit < previous_first_bit) ||
            (
                pixel == previous_pixel && first_bit == previous_first_bit &&
                region < previous_region
            ) ||
            (
                pixel == previous_pixel && first_bit == previous_first_bit &&
                region == previous_region && far_side < previous_far_side
            )
        );
        if (out_of_order) {
            throw std::runtime_error("records are not canonically ordered");
        }
        has_previous_key = true;
        previous_pixel = pixel;
        previous_first_bit = first_bit;
        previous_region = region;
        previous_far_side = far_side;
        const std::uint32_t row = pixel / static_cast<std::uint32_t>(width);
        const std::uint32_t column = pixel % static_cast<std::uint32_t>(width);
        const std::uint32_t start_fine = 16U * column + first_bit;
        const std::uint32_t end_fine = 16U * column + greatest_u16_bit(mask);
        if (
            has_active && active.row == row && active.end_fine + 1U == start_fine &&
            active.region == region && active.far_side == far_side
        ) {
            active.record_end = index + 1;
            active.end_fine = end_fine;
            continue;
        }
        if (has_active) {
            plan_one_local_run_in_place(
                records,
                record_count,
                active,
                coverage,
                regions,
                colours,
                height,
                width,
                local_limit_px,
                search_limit,
                safe_radius,
                slot_remaining,
                sample_remaining,
                statistics
            );
        }
        active = NativeRepairRun{index, index + 1, row, start_fine, end_fine, region, far_side};
        has_active = true;
    }
    if (has_active) {
        plan_one_local_run_in_place(
            records,
            record_count,
            active,
            coverage,
            regions,
            colours,
            height,
            width,
            local_limit_px,
            search_limit,
            safe_radius,
            slot_remaining,
            sample_remaining,
            statistics
        );
    }
    return local_statistics_object(statistics);
}

py::tuple plan_local_strips_native(
    const py::array& raw_records,
    const py::array& raw_runs,
    const py::array& coverage_count,
    const py::array& pure_region_id,
    const py::array& blue,
    const py::array& green,
    const py::array& red,
    std::uint32_t local_limit_px,
    std::uint32_t search_limit,
    std::uint32_t safe_radius,
    std::uint64_t slot_remaining,
    std::uint64_t sample_remaining
) {
    if (
        !raw_records.dtype().is(py::dtype::of<std::uint8_t>()) ||
        raw_records.ndim() != 2 || raw_records.shape(1) != 16 ||
        (raw_records.flags() & py::array::c_style) == 0
    ) {
        throw py::type_error("raw_records must be a C-contiguous uint8 [n,16] array");
    }
    if (
        !raw_runs.dtype().is(py::dtype::of<std::uint8_t>()) ||
        raw_runs.ndim() != 2 || raw_runs.shape(1) != 25 ||
        (raw_runs.flags() & py::array::c_style) == 0
    ) {
        throw py::type_error("raw_runs must be a C-contiguous uint8 [n,25] array");
    }
    validate_array(coverage_count, py::dtype::of<std::uint8_t>(), "coverage_count");
    validate_array(pure_region_id, py::dtype::of<std::uint32_t>(), "pure_region_id");
    validate_array(blue, py::dtype::of<std::uint8_t>(), "blue");
    validate_array(green, py::dtype::of<std::uint8_t>(), "green");
    validate_array(red, py::dtype::of<std::uint8_t>(), "red");
    const std::int64_t height = coverage_count.shape(0);
    const std::int64_t width = coverage_count.shape(1);
    for (const py::array* plane : {&pure_region_id, &blue, &green, &red}) {
        if (plane->shape(0) != height || plane->shape(1) != width) {
            throw py::value_error("local-strip rasters must have one shape");
        }
    }
    if (local_limit_px == 0 || search_limit == 0 || safe_radius == 0) {
        throw py::value_error("local-strip limits must be positive");
    }
    py::array_t<std::uint8_t> output({raw_records.shape(0), raw_records.shape(1)});
    std::memcpy(output.mutable_data(), raw_records.data(), static_cast<std::size_t>(raw_records.nbytes()));
    auto* output_records = output.mutable_data();
    const auto* runs = static_cast<const std::uint8_t*>(raw_runs.data());
    const auto* coverage = static_cast<const std::uint8_t*>(coverage_count.data());
    const auto* regions = static_cast<const std::uint32_t*>(pure_region_id.data());
    const std::array<const std::uint8_t*, 3> colours{
        static_cast<const std::uint8_t*>(blue.data()),
        static_cast<const std::uint8_t*>(green.data()),
        static_cast<const std::uint8_t*>(red.data()),
    };
    constexpr std::array<std::int64_t, 5> row_deltas{0, -1, 1, -2, 2};
    LocalStatistics statistics;
    const std::uint64_t record_count = static_cast<std::uint64_t>(raw_records.shape(0));

    for (py::ssize_t run_index = 0; run_index < raw_runs.shape(0); ++run_index) {
        const std::uint8_t* run = runs + 25 * run_index;
        const std::uint32_t record_start = load_u32_le(run);
        const std::uint32_t record_end = load_u32_le(run + 4);
        const std::uint32_t row = load_u32_le(run + 8);
        const std::uint32_t start_fine = load_u32_le(run + 12);
        const std::uint32_t end_fine = load_u32_le(run + 16);
        const std::uint32_t region = load_u32_le(run + 20);
        const std::uint8_t far_side = run[24];
        if (
            record_start >= record_end || record_end > record_count || row >= height ||
            start_fine > end_fine || far_side > 1
        ) {
            throw py::value_error("local-strip run is malformed");
        }
        const std::uint64_t lane_count =
            static_cast<std::uint64_t>(end_fine) - start_fine + 1;
        if (lane_count > 16ULL * local_limit_px) {
            continue;
        }
        const std::uint32_t first_pixel = start_fine / 16;
        const std::uint32_t last_pixel = end_fine / 16;
        const std::uint32_t donor_count = last_pixel - first_pixel + 1;
        if (record_end - record_start != donor_count) {
            throw py::value_error("local-strip run/record cardinality differs");
        }
        const std::uint64_t slot_charge = checked_local_mul(5, search_limit);
        const std::uint64_t support = checked_local_mul(2ULL * safe_radius + 1, 2ULL * safe_radius + 1);
        const std::uint64_t per_slot = checked_local_add(3, checked_local_mul(donor_count, support));
        const std::uint64_t sample_charge = checked_local_add(3, checked_local_mul(slot_charge, per_slot));
        if (
            slot_charge > slot_remaining - statistics.slot_consumed ||
            sample_charge > sample_remaining - statistics.sample_consumed
        ) {
            statistics.budget_skipped_runs = checked_local_add(
                statistics.budget_skipped_runs,
                1
            );
            continue;
        }
        statistics.slot_consumed = checked_local_add(statistics.slot_consumed, slot_charge);
        statistics.sample_consumed = checked_local_add(statistics.sample_consumed, sample_charge);
        statistics.evaluated_slots = checked_local_add(statistics.evaluated_slots, slot_charge);
        const std::int64_t direction = far_side == 0 ? -1 : 1;
        const std::int64_t anchor_fine = far_side == 0
            ? static_cast<std::int64_t>(start_fine) - 1
            : static_cast<std::int64_t>(end_fine) + 1;
        const std::int64_t boundary_column = anchor_fine >= 0
            ? anchor_fine / 16
            : -((-anchor_fine + 15) / 16);
        const std::array<std::int64_t, 3> actual_columns{
            boundary_column + 2 * direction,
            boundary_column + direction,
            boundary_column,
        };
        LocalContext actual{};
        if (!read_local_context(
            coverage,
            regions,
            colours,
            height,
            width,
            row,
            actual_columns,
            region,
            statistics,
            actual
        )) {
            continue;
        }
        bool best_found = false;
        std::uint64_t best_score = 0;
        std::uint64_t best_ordinal = 0;
        std::int64_t best_row = 0;
        std::int64_t best_start = 0;
        for (std::uint32_t offset = 1; offset <= search_limit; ++offset) {
            for (std::size_t row_index = 0; row_index < row_deltas.size(); ++row_index) {
                const std::uint64_t ordinal = (offset - 1ULL) * 5 + row_index;
                const std::int64_t candidate_row =
                    static_cast<std::int64_t>(row) + row_deltas[row_index];
                const std::int64_t candidate_start =
                    boundary_column + direction * static_cast<std::int64_t>(offset);
                const std::array<std::int64_t, 3> candidate_columns{
                    candidate_start + 3 * direction,
                    candidate_start + 2 * direction,
                    candidate_start + direction,
                };
                bool legal = candidate_row >= 0 && candidate_row < height;
                for (const std::int64_t column : candidate_columns) {
                    legal = legal && column >= 0 && column < width;
                }
                for (std::uint32_t donor_index = 0; donor_index < donor_count; ++donor_index) {
                    const std::int64_t column =
                        candidate_start - static_cast<std::int64_t>(donor_index) * direction;
                    legal = legal && column >= 0 && column < width;
                    legal = legal && (column - boundary_column) * direction > 0;
                }
                if (!legal) {
                    continue;
                }
                LocalContext candidate{};
                if (!read_local_context(
                    coverage,
                    regions,
                    colours,
                    height,
                    width,
                    candidate_row,
                    candidate_columns,
                    region,
                    statistics,
                    candidate
                )) {
                    continue;
                }
                bool safe = true;
                for (std::uint32_t donor_index = 0; donor_index < donor_count; ++donor_index) {
                    const std::int64_t donor_column =
                        candidate_start - static_cast<std::int64_t>(donor_index) * direction;
                    if (!local_donor_is_safe(
                        coverage,
                        regions,
                        height,
                        width,
                        candidate_row,
                        donor_column,
                        region,
                        safe_radius,
                        statistics
                    )) {
                        safe = false;
                        break;
                    }
                }
                if (!safe) {
                    statistics.unsafe_donor_slots = checked_local_add(
                        statistics.unsafe_donor_slots,
                        1
                    );
                    continue;
                }
                statistics.eligible_slots = checked_local_add(statistics.eligible_slots, 1);
                const std::uint64_t score = local_context_score(actual, candidate);
                if (
                    !best_found || score < best_score ||
                    (score == best_score && ordinal < best_ordinal)
                ) {
                    best_found = true;
                    best_score = score;
                    best_ordinal = ordinal;
                    best_row = candidate_row;
                    best_start = candidate_start;
                }
            }
        }
        if (!best_found) {
            continue;
        }
        for (std::uint32_t donor_index = 0; donor_index < donor_count; ++donor_index) {
            const std::uint32_t record_index = far_side == 1
                ? record_end - 1 - donor_index
                : record_start + donor_index;
            std::uint8_t* record = output_records + 16ULL * record_index;
            const std::int64_t donor_column =
                best_start - static_cast<std::int64_t>(donor_index) * direction;
            const auto sample = static_cast<std::uint64_t>(best_row * width + donor_column);
            record[10] = colours[0][sample];
            record[11] = colours[1][sample];
            record[12] = colours[2][sample];
            record[13] = 1;
        }
        statistics.filled_runs = checked_local_add(statistics.filled_runs, 1);
    }
    py::dict result;
    result["evaluated_slots"] = statistics.evaluated_slots;
    result["physical_sample_reads"] = statistics.physical_sample_reads;
    result["filled_runs"] = statistics.filled_runs;
    result["eligible_slots"] = statistics.eligible_slots;
    result["unsafe_donor_slots"] = statistics.unsafe_donor_slots;
    result["slot_consumed"] = statistics.slot_consumed;
    result["sample_consumed"] = statistics.sample_consumed;
    result["budget_skipped_runs"] = statistics.budget_skipped_runs;
    return py::make_tuple(std::move(output), std::move(result));
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
    {
        py::gil_scoped_release release;
        for (std::uint32_t row = 0; row < destination_height; ++row) {
            const double source_coordinate =
                ((static_cast<double>(row) + 0.5) * source_height) /
                destination_height - 0.5;
            const auto rounded = static_cast<std::int64_t>(
                std::floor(source_coordinate + 0.5)
            );
            const std::uint32_t source_row = static_cast<std::uint32_t>(std::min(
                std::max<std::int64_t>(rounded, 0),
                static_cast<std::int64_t>(source_height - 1)
            ));
            for (std::uint32_t column = 0; column < destination_width; ++column) {
                const double column_coordinate =
                    ((static_cast<double>(column) + 0.5) * source_width) /
                    destination_width - 0.5;
                const auto column_rounded = static_cast<std::int64_t>(
                    std::floor(column_coordinate + 0.5)
                );
                const std::uint32_t source_column = static_cast<std::uint32_t>(std::min(
                    std::max<std::int64_t>(column_rounded, 0),
                    static_cast<std::int64_t>(source_width - 1)
                ));
                result[row * destination_width + column] =
                    input[source_row * source_width + source_column];
            }
        }
    }
    return output;
}

py::array build_four_connected_edge_band_native(
    const py::array& labels,
    std::uint32_t radius
) {
    validate_array(labels, py::dtype::of<std::uint32_t>(), "mapped_regions");
    const auto height = static_cast<std::uint32_t>(labels.shape(0));
    const auto width = static_cast<std::uint32_t>(labels.shape(1));
    const auto sample_count = static_cast<std::uint64_t>(height) * width;
    const auto* regions = static_cast<const std::uint32_t*>(labels.data());
    std::vector<std::uint8_t> current(sample_count, 0);
    std::vector<std::uint8_t> next(sample_count, 0);
    const auto is_region = [](std::uint32_t value) {
        return value != 0 && value != std::numeric_limits<std::uint32_t>::max();
    };
    const auto parallel_rows = [&](auto callback) {
        const std::size_t hardware_workers = std::max(
            1U,
            std::thread::hardware_concurrency()
        );
        const std::size_t useful_workers = std::max<std::size_t>(
            1,
            (height + kEdgeBandRowsPerWorker - 1) / kEdgeBandRowsPerWorker
        );
        const std::size_t desired_worker_count = std::min(
            {kEdgeBandWorkerCap, hardware_workers, useful_workers}
        );
        NativeWorkerLease worker_lease(
            desired_worker_count > 1 ? desired_worker_count - 1 : 0
        );
        const std::size_t worker_count = worker_lease.worker_count();
        if (worker_count == 1) {
            callback(0, height);
            return;
        }
        std::vector<std::thread> workers;
        workers.reserve(worker_count - 1);
        const auto run_worker = [&](std::size_t worker) {
            const auto start = static_cast<std::uint32_t>(
                static_cast<std::uint64_t>(height) * worker / worker_count
            );
            const auto stop = static_cast<std::uint32_t>(
                static_cast<std::uint64_t>(height) * (worker + 1) / worker_count
            );
            callback(start, stop);
        };
        try {
            for (std::size_t worker = 1; worker < worker_count; ++worker) {
                workers.emplace_back(run_worker, worker);
            }
        } catch (...) {
            for (auto& thread : workers) {
                thread.join();
            }
            throw;
        }
        run_worker(0);
        for (auto& thread : workers) {
            thread.join();
        }
    };
    {
        py::gil_scoped_release release;
        parallel_rows([&](std::uint32_t start_row, std::uint32_t stop_row) {
            for (std::uint32_t row = start_row; row < stop_row; ++row) {
                for (std::uint32_t column = 0; column < width; ++column) {
                    const std::uint64_t pixel =
                        static_cast<std::uint64_t>(row) * width + column;
                    const std::uint32_t region = regions[pixel];
                    current[pixel] = is_region(region) && (
                        (row != 0 && is_region(regions[pixel - width]) &&
                         region != regions[pixel - width]) ||
                        (row + 1 < height && is_region(regions[pixel + width]) &&
                         region != regions[pixel + width]) ||
                        (column != 0 && is_region(regions[pixel - 1]) &&
                         region != regions[pixel - 1]) ||
                        (column + 1 < width && is_region(regions[pixel + 1]) &&
                         region != regions[pixel + 1])
                    );
                }
            }
        });
        for (std::uint32_t iteration = 0; iteration < radius; ++iteration) {
            parallel_rows([&](std::uint32_t start_row, std::uint32_t stop_row) {
                for (std::uint32_t row = start_row; row < stop_row; ++row) {
                    for (std::uint32_t column = 0; column < width; ++column) {
                        const std::uint64_t pixel =
                            static_cast<std::uint64_t>(row) * width + column;
                        next[pixel] = current[pixel] ||
                            (row > 0 && current[pixel - width]) ||
                            (column > 0 && current[pixel - 1]) ||
                            (column + 1 < width && current[pixel + 1]) ||
                            (row + 1 < height && current[pixel + width]);
                    }
                }
            });
            current.swap(next);
        }
    }
    py::array_t<bool> output({height, width});
    auto* result = output.mutable_data();
    for (std::uint64_t pixel = 0; pixel < sample_count; ++pixel) {
        result[pixel] = current[pixel] != 0;
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

py::array build_sparse_region_seeds_native(
    const py::array& region_map,
    const py::array& band_mask,
    std::uint32_t radius
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
    std::vector<std::uint8_t> visited(sample_count, 0);
    std::uint32_t band_count = 0;
    for (std::uint32_t pixel = 0; pixel < sample_count; ++pixel) {
        band_count += band[pixel] != 0;
    }
    std::vector<std::uint32_t> queue(band_count);
    std::vector<std::uint8_t> current(sample_count, 0);
    std::vector<std::uint8_t> eroded(sample_count, 0);
    {
        py::gil_scoped_release release;
        for (std::uint32_t pixel = 0; pixel < sample_count; ++pixel) {
            seeds[pixel] = band[pixel] != 0 ? 0 : regions[pixel];
        }
        const auto is_region = [](std::uint32_t value) {
            return value != 0 && value != kUint32Max;
        };
        for (std::uint32_t start = 0; start < sample_count; ++start) {
            const std::uint32_t region = regions[start];
            if (band[start] == 0 || visited[start] != 0 || !is_region(region)) {
                continue;
            }
            std::uint32_t head = 0;
            std::uint32_t tail = 1;
            queue[0] = start;
            visited[start] = 1;
            while (head < tail) {
                const std::uint32_t pixel = queue[head++];
                const std::uint32_t row = pixel / width;
                const std::uint32_t column = pixel % width;
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
                    seeds[pixel] = region;
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
                            seeds[pixel] = region;
                            selected_any = true;
                        }
                    }
                    current.swap(eroded);
                    current_any = eroded_any;
                }
            }

            if (!selected_any) {
                seeds[start] = region;
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

std::pair<std::uint32_t, std::uint32_t> validate_background_proxy_rasters(
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
    const auto height = static_cast<std::uint32_t>(pre_repair.shape(0));
    const auto width = static_cast<std::uint32_t>(pre_repair.shape(1));
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
    const py::object& metric_valid_object
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
    py::array_t<bool> output({near_score.shape(0), near_score.shape(1)});
    auto* expanded = output.mutable_data();
    std::vector<float> best_near(near_values, near_values + sample_count);
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
        const auto consider = [&](std::uint64_t destination, std::uint64_t source, int code) {
            if (
                !source_valid_values[source] ||
                (metric_valid_values != nullptr && !metric_valid_values[source]) ||
                region_values[source] == region_values[destination] ||
                !(near_values[source] > best_near[destination])
            ) {
                return;
            }
            best_near[destination] = near_values[source];
            winner[destination] = static_cast<std::uint8_t>(code);
        };
        for (std::uint64_t row = 1; row < height; ++row) {
            for (std::uint64_t column = 0; column < width; ++column) {
                const std::uint64_t destination = row * width + column;
                consider(destination, destination - width, 1);
            }
        }
        for (std::uint64_t row = 0; row + 1 < height; ++row) {
            for (std::uint64_t column = 0; column < width; ++column) {
                const std::uint64_t destination = row * width + column;
                consider(destination, destination + width, 2);
            }
        }
        for (std::uint64_t row = 0; row < height; ++row) {
            for (std::uint64_t column = 1; column < width; ++column) {
                const std::uint64_t destination = row * width + column;
                consider(destination, destination - 1, 3);
            }
        }
        for (std::uint64_t row = 0; row < height; ++row) {
            for (std::uint64_t column = 0; column + 1 < width; ++column) {
                const std::uint64_t destination = row * width + column;
                consider(destination, destination + 1, 4);
            }
        }

        bool any_expanded = false;
        for (std::uint64_t pixel = 0; pixel < sample_count; ++pixel) {
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
                    expanded[pixel] = false;
                    continue;
            }
            expanded[pixel] = std::abs(
                displacement_values[source] - displacement_values[pixel]
            ) >= 0.25;
            any_expanded = any_expanded || expanded[pixel];
        }
        if (any_expanded) {
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

py::array build_exemplar_candidate_mask_native(
    const py::array& safe_donor_mask,
    const py::array& pure_region_id,
    py::object output_object,
    py::object workspace_object
) {
    validate_array(safe_donor_mask, py::dtype::of<bool>(), "safe_donor_mask");
    validate_array(pure_region_id, py::dtype::of<std::uint32_t>(), "pure_region_id");
    if (
        safe_donor_mask.shape(0) != pure_region_id.shape(0) ||
        safe_donor_mask.shape(1) != pure_region_id.shape(1)
    ) {
        throw py::value_error("exemplar candidate rasters must match");
    }
    const py::ssize_t height_value = safe_donor_mask.shape(0);
    const py::ssize_t width_value = safe_donor_mask.shape(1);
    if (
        height_value <= 0 || width_value <= 0 ||
        static_cast<std::uint64_t>(height_value) > kUint32Max ||
        static_cast<std::uint64_t>(width_value) > kUint32Max
    ) {
        throw py::value_error("exemplar candidate shape must fit uint32 bounds");
    }
    const auto height = static_cast<std::uint32_t>(height_value);
    const auto width = static_cast<std::uint32_t>(width_value);
    const auto* safe = static_cast<const std::uint8_t*>(safe_donor_mask.data());
    const auto* regions = static_cast<const std::uint32_t*>(pure_region_id.data());
    if (output_object.is_none() != workspace_object.is_none()) {
        throw py::value_error("candidate output and workspace must be supplied together");
    }
    py::array output;
    py::array workspace;
    if (output_object.is_none()) {
        output = py::array_t<bool>({height, width});
        workspace = py::array_t<std::uint32_t>(2ULL * width);
    } else {
        output = py::cast<py::array>(output_object);
        workspace = py::cast<py::array>(workspace_object);
        if (
            !output.dtype().is(py::dtype::of<bool>()) || output.ndim() != 2 ||
            output.shape(0) != height || output.shape(1) != width ||
            (output.flags() & py::array::c_style) == 0 || !output.writeable()
        ) {
            throw py::type_error("candidate output must be a writable matching bool raster");
        }
        if (
            !workspace.dtype().is(py::dtype::of<std::uint32_t>()) ||
            workspace.ndim() != 1 ||
            static_cast<std::uint64_t>(workspace.shape(0)) < 2ULL * width ||
            (workspace.flags() & py::array::c_style) == 0 || !workspace.writeable()
        ) {
            throw py::type_error("candidate workspace must contain two uint32 image rows");
        }
    }
    auto* result = static_cast<bool*>(output.mutable_data());
    std::fill(result, result + static_cast<std::uint64_t>(height) * width, false);
    if (height < 9 || width < 9) {
        return output;
    }
    {
        py::gil_scoped_release release;
        auto* vertical_run = static_cast<std::uint32_t*>(workspace.mutable_data());
        auto* vertical_region = vertical_run + width;
        std::fill(vertical_run, vertical_run + width, 0);
        std::fill(
            vertical_region,
            vertical_region + width,
            static_cast<std::uint32_t>(kUint32Max)
        );
        for (std::uint32_t row = 0; row < height; ++row) {
            std::uint32_t horizontal_run = 0;
            std::uint32_t horizontal_region = kUint32Max;
            for (std::uint32_t column = 0; column < width; ++column) {
                const std::uint32_t pixel = row * width + column;
                const std::uint32_t region = regions[pixel];
                const bool usable =
                    safe[pixel] != 0 && region != 0 && region != kUint32Max;
                if (!usable) {
                    horizontal_run = 0;
                    horizontal_region = kUint32Max;
                } else if (horizontal_region == region) {
                    ++horizontal_run;
                } else {
                    horizontal_run = 1;
                    horizontal_region = region;
                }
                if (column < 8) {
                    continue;
                }
                const std::uint32_t center_column = column - 4;
                const bool horizontal_support = horizontal_run >= 9;
                if (!horizontal_support) {
                    vertical_run[center_column] = 0;
                    vertical_region[center_column] = kUint32Max;
                    continue;
                }
                if (vertical_region[center_column] == region) {
                    ++vertical_run[center_column];
                } else {
                    vertical_run[center_column] = 1;
                    vertical_region[center_column] = region;
                }
                if (vertical_run[center_column] >= 9) {
                    const std::uint32_t center_row = row - 4;
                    result[center_row * width + center_column] = true;
                }
            }
        }
    }
    return output;
}

py::object select_exemplar_frontier_native(
    const py::array& target_mask,
    const py::array& processed_mask,
    const py::array& barrier_mask
) {
    validate_array(target_mask, py::dtype::of<bool>(), "target_mask");
    validate_array(processed_mask, py::dtype::of<bool>(), "processed_mask");
    validate_array(barrier_mask, py::dtype::of<bool>(), "barrier_mask");
    if (
        processed_mask.shape(0) != target_mask.shape(0) ||
        processed_mask.shape(1) != target_mask.shape(1) ||
        barrier_mask.shape(0) != target_mask.shape(0) ||
        barrier_mask.shape(1) != target_mask.shape(1)
    ) {
        throw py::value_error("exemplar frontier masks must match");
    }
    const auto height = static_cast<std::uint32_t>(target_mask.shape(0));
    const auto width = static_cast<std::uint32_t>(target_mask.shape(1));
    const auto* target = static_cast<const std::uint8_t*>(target_mask.data());
    const auto* processed = static_cast<const std::uint8_t*>(processed_mask.data());
    const auto* barrier = static_cast<const std::uint8_t*>(barrier_mask.data());
    bool found = false;
    std::uint32_t best_count = 0;
    std::uint32_t best_row = 0;
    std::uint32_t best_column = 0;
    if (height >= 7 && width >= 7) {
        py::gil_scoped_release release;
        for (std::uint32_t row = 3; row + 3 < height; ++row) {
            for (std::uint32_t column = 3; column + 3 < width; ++column) {
                const std::uint32_t pixel = row * width + column;
                if (target[pixel] == 0 || processed[pixel] != 0) {
                    continue;
                }
                const bool has_neighbour =
                    (processed[pixel - width] != 0 && barrier[pixel - width] == 0) ||
                    (processed[pixel - 1] != 0 && barrier[pixel - 1] == 0) ||
                    (processed[pixel + 1] != 0 && barrier[pixel + 1] == 0) ||
                    (processed[pixel + width] != 0 && barrier[pixel + width] == 0);
                if (!has_neighbour) {
                    continue;
                }
                bool has_barrier = false;
                std::uint32_t processed_count = 0;
                for (std::uint32_t sample_row = row - 3; sample_row <= row + 3; ++sample_row) {
                    for (
                        std::uint32_t sample_column = column - 3;
                        sample_column <= column + 3;
                        ++sample_column
                    ) {
                        const std::uint32_t sample = sample_row * width + sample_column;
                        if (barrier[sample] != 0) {
                            has_barrier = true;
                            break;
                        }
                        processed_count += processed[sample] != 0;
                    }
                    if (has_barrier) {
                        break;
                    }
                }
                if (has_barrier || processed_count < 8) {
                    continue;
                }
                if (!found || processed_count > best_count) {
                    found = true;
                    best_count = processed_count;
                    best_row = row;
                    best_column = column;
                }
            }
        }
    }
    if (!found) {
        return py::none();
    }
    return py::make_tuple(best_row, best_column, best_count);
}

struct ExemplarMaskLevelView {
    std::uint32_t height;
    std::uint32_t width;
    std::uint8_t* donor;
    std::uint8_t* target;
    std::uint8_t* processed;
    std::uint8_t* barrier;
};

bool exemplar_level_has_candidate(const ExemplarMaskLevelView& level) {
    if (level.height < 9 || level.width < 9) {
        return false;
    }
    for (std::uint32_t row = 4; row + 4 < level.height; ++row) {
        for (std::uint32_t column = 4; column + 4 < level.width; ++column) {
            bool complete = true;
            for (std::int32_t delta_row = -4; delta_row <= 4 && complete; ++delta_row) {
                for (std::int32_t delta_column = -4; delta_column <= 4; ++delta_column) {
                    const auto sample_row = static_cast<std::uint32_t>(
                        static_cast<std::int32_t>(row) + delta_row
                    );
                    const auto sample_column = static_cast<std::uint32_t>(
                        static_cast<std::int32_t>(column) + delta_column
                    );
                    if (level.donor[sample_row * level.width + sample_column] == 0) {
                        complete = false;
                        break;
                    }
                }
            }
            if (complete) {
                return true;
            }
        }
    }
    return false;
}

bool exemplar_level_has_frontier(const ExemplarMaskLevelView& level) {
    if (level.height < 7 || level.width < 7) {
        return false;
    }
    constexpr std::array<std::pair<std::int32_t, std::int32_t>, 4> neighbours{
        std::pair<std::int32_t, std::int32_t>{-1, 0},
        {0, -1},
        {0, 1},
        {1, 0},
    };
    for (std::uint32_t row = 3; row + 3 < level.height; ++row) {
        for (std::uint32_t column = 3; column + 3 < level.width; ++column) {
            const std::uint32_t pixel = row * level.width + column;
            if (level.target[pixel] == 0 || level.processed[pixel] != 0) {
                continue;
            }
            bool has_processed_neighbour = false;
            for (const auto& neighbour : neighbours) {
                const auto neighbour_row = static_cast<std::uint32_t>(
                    static_cast<std::int32_t>(row) + neighbour.first
                );
                const auto neighbour_column = static_cast<std::uint32_t>(
                    static_cast<std::int32_t>(column) + neighbour.second
                );
                const std::uint32_t neighbour_pixel =
                    neighbour_row * level.width + neighbour_column;
                if (
                    level.processed[neighbour_pixel] != 0 &&
                    level.barrier[neighbour_pixel] == 0
                ) {
                    has_processed_neighbour = true;
                    break;
                }
            }
            if (!has_processed_neighbour) {
                continue;
            }
            std::uint32_t processed_count = 0;
            bool contains_barrier = false;
            for (std::int32_t delta_row = -3; delta_row <= 3 && !contains_barrier; ++delta_row) {
                for (std::int32_t delta_column = -3; delta_column <= 3; ++delta_column) {
                    const auto sample_row = static_cast<std::uint32_t>(
                        static_cast<std::int32_t>(row) + delta_row
                    );
                    const auto sample_column = static_cast<std::uint32_t>(
                        static_cast<std::int32_t>(column) + delta_column
                    );
                    const std::uint32_t sample = sample_row * level.width + sample_column;
                    if (level.barrier[sample] != 0) {
                        contains_barrier = true;
                        break;
                    }
                    processed_count += level.processed[sample] != 0 ? 1U : 0U;
                }
            }
            if (!contains_barrier && processed_count >= 8) {
                return true;
            }
        }
    }
    return false;
}

void downsample_exemplar_masks(
    const ExemplarMaskLevelView& child,
    ExemplarMaskLevelView& parent
) {
    for (std::uint32_t row = 0; row < parent.height; ++row) {
        for (std::uint32_t column = 0; column < parent.width; ++column) {
            bool donor = true;
            bool target = false;
            bool processed = true;
            bool barrier = false;
            for (std::uint32_t delta_row = 0; delta_row < 2; ++delta_row) {
                const std::uint32_t child_row = 2 * row + delta_row;
                if (child_row >= child.height) {
                    continue;
                }
                for (std::uint32_t delta_column = 0; delta_column < 2; ++delta_column) {
                    const std::uint32_t child_column = 2 * column + delta_column;
                    if (child_column >= child.width) {
                        continue;
                    }
                    const std::uint32_t child_pixel =
                        child_row * child.width + child_column;
                    donor = donor && child.donor[child_pixel] != 0;
                    target = target || child.target[child_pixel] != 0;
                    processed = processed && child.processed[child_pixel] != 0;
                    barrier = barrier || child.barrier[child_pixel] != 0;
                }
            }
            const std::uint32_t pixel = row * parent.width + column;
            parent.barrier[pixel] = barrier;
            parent.target[pixel] = !barrier && target;
            parent.donor[pixel] = !barrier && donor;
            parent.processed[pixel] = !target && !barrier && processed;
        }
    }
}

bool component_may_score_exemplar_native(
    const py::array& coverage_count,
    const py::array& pure_region_id,
    const py::array& safe_donor_mask,
    const py::array& target_pixels,
    std::uint32_t region,
    py::array workspace
) {
    validate_array(coverage_count, py::dtype::of<std::uint8_t>(), "coverage_count");
    validate_array(pure_region_id, py::dtype::of<std::uint32_t>(), "pure_region_id");
    validate_array(safe_donor_mask, py::dtype::of<bool>(), "safe_donor_mask");
    validate_vector(target_pixels, py::dtype::of<std::uint32_t>(), "target_pixels");
    if (
        !workspace.dtype().is(py::dtype::of<std::uint8_t>()) ||
        workspace.ndim() != 1 || (workspace.flags() & py::array::c_style) == 0 ||
        !workspace.writeable()
    ) {
        throw py::type_error("exemplar mask workspace must be a writable uint8 vector");
    }
    if (
        pure_region_id.shape(0) != coverage_count.shape(0) ||
        pure_region_id.shape(1) != coverage_count.shape(1) ||
        safe_donor_mask.shape(0) != coverage_count.shape(0) ||
        safe_donor_mask.shape(1) != coverage_count.shape(1)
    ) {
        throw py::value_error("component exemplar rasters must match");
    }
    if (target_pixels.shape(0) == 0 || region == 0 || region == kUint32Max) {
        throw py::value_error("component exemplar identity is invalid");
    }
    const auto height = static_cast<std::uint32_t>(coverage_count.shape(0));
    const auto width = static_cast<std::uint32_t>(coverage_count.shape(1));
    const auto* targets = static_cast<const std::uint32_t*>(target_pixels.data());
    std::uint32_t min_row = height;
    std::uint32_t max_row = 0;
    std::uint32_t min_column = width;
    std::uint32_t max_column = 0;
    for (py::ssize_t index = 0; index < target_pixels.shape(0); ++index) {
        if (targets[index] >= static_cast<std::uint64_t>(height) * width) {
            throw py::value_error("component target pixel lies outside analysis");
        }
        const std::uint32_t row = targets[index] / width;
        const std::uint32_t column = targets[index] % width;
        min_row = std::min(min_row, row);
        max_row = std::max(max_row, row);
        min_column = std::min(min_column, column);
        max_column = std::max(max_column, column);
    }
    if (max_row - min_row + 1 > 384 || max_column - min_column + 1 > 384) {
        return true;
    }
    const std::uint32_t y0 = min_row > 64 ? min_row - 64 : 0;
    const std::uint32_t y1 = static_cast<std::uint32_t>(std::min<std::uint64_t>(
        height,
        static_cast<std::uint64_t>(max_row) + 65
    ));
    const std::uint32_t x0 = min_column > 64 ? min_column - 64 : 0;
    const std::uint32_t x1 = static_cast<std::uint32_t>(std::min<std::uint64_t>(
        width,
        static_cast<std::uint64_t>(max_column) + 65
    ));
    std::array<std::pair<std::uint32_t, std::uint32_t>, 3> shapes{};
    shapes[0] = {y1 - y0, x1 - x0};
    std::size_t level_count = 1;
    while (level_count < shapes.size()) {
        const auto [current_height, current_width] = shapes[level_count - 1];
        const std::uint32_t next_height = (current_height + 1) / 2;
        const std::uint32_t next_width = (current_width + 1) / 2;
        if (std::min(next_height, next_width) < 32) {
            break;
        }
        shapes[level_count] = {next_height, next_width};
        ++level_count;
    }
    std::uint64_t required_bytes = 0;
    for (std::size_t index = 0; index < level_count; ++index) {
        required_bytes += 4ULL * shapes[index].first * shapes[index].second;
    }
    if (required_bytes > static_cast<std::uint64_t>(workspace.shape(0))) {
        throw QualityNativeBudgetError("exemplar mask workspace exceeded");
    }
    auto* workspace_values = static_cast<std::uint8_t*>(workspace.mutable_data());
    std::array<ExemplarMaskLevelView, 3> levels{};
    std::uint64_t workspace_cursor = 0;
    for (std::size_t index = 0; index < level_count; ++index) {
        const auto [level_height, level_width] = shapes[index];
        const std::uint64_t count =
            static_cast<std::uint64_t>(level_height) * level_width;
        auto& level = levels[index];
        level.height = level_height;
        level.width = level_width;
        level.donor = workspace_values + workspace_cursor;
        workspace_cursor += count;
        level.target = workspace_values + workspace_cursor;
        workspace_cursor += count;
        level.processed = workspace_values + workspace_cursor;
        workspace_cursor += count;
        level.barrier = workspace_values + workspace_cursor;
        workspace_cursor += count;
    }
    auto& level = levels[0];
    const std::uint64_t local_count =
        static_cast<std::uint64_t>(level.height) * level.width;
    std::fill(level.donor, level.donor + local_count, 0);
    std::fill(level.target, level.target + local_count, 0);
    std::fill(level.processed, level.processed + local_count, 0);
    std::fill(level.barrier, level.barrier + local_count, 0);
    const auto* coverage = static_cast<const std::uint8_t*>(coverage_count.data());
    const auto* regions = static_cast<const std::uint32_t*>(pure_region_id.data());
    const auto* safe = static_cast<const std::uint8_t*>(safe_donor_mask.data());
    {
        py::gil_scoped_release release;
        for (std::uint32_t row = y0; row < y1; ++row) {
            for (std::uint32_t column = x0; column < x1; ++column) {
                const std::uint32_t source = row * width + column;
                const std::uint32_t local = (row - y0) * level.width + column - x0;
                const bool same_proxy = coverage[source] > 0 && regions[source] == region;
                level.donor[local] = same_proxy && safe[source] != 0;
                level.processed[local] = same_proxy;
                level.barrier[local] = !same_proxy;
            }
        }
        for (py::ssize_t index = 0; index < target_pixels.shape(0); ++index) {
            const std::uint32_t row = targets[index] / width;
            const std::uint32_t column = targets[index] % width;
            const std::uint32_t local = (row - y0) * level.width + column - x0;
            level.donor[local] = 0;
            level.target[local] = 1;
            level.processed[local] = 0;
            level.barrier[local] = 0;
        }
        for (std::size_t index = 1; index < level_count; ++index) {
            downsample_exemplar_masks(levels[index - 1], levels[index]);
        }
        for (std::size_t index = 0; index < level_count; ++index) {
            if (
                exemplar_level_has_candidate(levels[index]) &&
                exemplar_level_has_frontier(levels[index])
            ) {
                return true;
            }
        }
    }
    return false;
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
    result["native_spawned_worker_cap"] = kNativeSpawnedWorkerCap;
    result["sparse_seed_morphology"] = "active-fragment-pixels-v1";
    result["geodesic_workset"] = "band-pixel-heap-storage-v1";
    result["source_sha256"] = QUALITY_KD_SOURCE_SHA256;
    return result;
}

}  // namespace

PYBIND11_MODULE(_quality_kd_native, module) {
    module.doc() = "Prebuilt deterministic implicit region k-d Task 0 extension";
    py::register_exception<QualityNativeBudgetError>(module, "QualityBudgetError");
    module.def(
        "build_index",
        &build_index,
        py::arg("region_ids"),
        py::arg("region_count"),
        py::arg("include") = py::none()
    );
    module.def(
        "query_repair_batch",
        &query_repair_batch,
        py::arg("member_index"),
        py::arg("region_offsets"),
        py::arg("height"),
        py::arg("width"),
        py::arg("query_regions"),
        py::arg("target_y"),
        py::arg("target_x"),
        py::arg("radius_px"),
        py::arg("budget_state")
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
        "build_repair_records",
        &build_repair_records,
        py::arg("valid"),
        py::arg("winner_near_score"),
        py::arg("winner_region_id"),
        py::arg("winner_bgr") = py::none()
    );
    module.def(
        "analyze_splat_band",
        &analyze_splat_band,
        py::arg("winner_source_index"),
        py::arg("source_bgr"),
        py::arg("source_near_score"),
        py::arg("source_region_map"),
        py::arg("coverage_count"),
        py::arg("pure_region_id"),
        py::arg("blue"),
        py::arg("green"),
        py::arg("red"),
        py::arg("record_output") = py::none()
    );
    module.def(
        "build_safe_donor_index",
        &build_safe_donor_index,
        py::arg("coverage_count"),
        py::arg("pure_region_id"),
        py::arg("referenced_regions"),
        py::arg("include_full_pure_for_empty_regions") = false
    );
    module.def(
        "build_safe_donor_index_into",
        &build_safe_donor_index_into,
        py::arg("coverage_count"),
        py::arg("pure_region_id"),
        py::arg("referenced_regions"),
        py::arg("include_full_pure_for_empty_regions"),
        py::arg("members_output"),
        py::arg("offsets_output"),
        py::arg("use_full_output"),
        py::arg("safe_cache_metadata") = py::none()
    );
    module.def(
        "build_referenced_unplanned_regions_into",
        &build_referenced_unplanned_regions_into_native,
        py::arg("raw_records"),
        py::arg("pixel_count"),
        py::arg("region_count"),
        py::arg("referenced_output")
    );
    module.def(
        "expand_safe_donor_cache_into",
        &expand_safe_donor_cache_into_native,
        py::arg("safe_words"),
        py::arg("pure_region_id"),
        py::arg("safe_output"),
        py::arg("region_output")
    );
    module.def(
        "build_safe_donor_mask",
        &build_safe_donor_mask,
        py::arg("coverage_count"),
        py::arg("pure_region_id"),
        py::arg("referenced_regions"),
        py::arg("safe_output") = py::none(),
        py::arg("region_output") = py::none()
    );
    module.def(
        "query_fallback_records",
        &query_fallback_records,
        py::arg("member_index"),
        py::arg("region_offsets"),
        py::arg("height"),
        py::arg("width"),
        py::arg("raw_records"),
        py::arg("blue"),
        py::arg("green"),
        py::arg("red"),
        py::arg("radius_px"),
        py::arg("budget_state"),
        py::arg("allow_seeded_missing") = false,
        py::arg("visits_output") = py::none()
    );
    module.def(
        "build_exemplar_sobel_planes",
        &build_exemplar_sobel_planes,
        py::arg("working_bgr"),
        py::arg("gradient_x") = py::none(),
        py::arg("gradient_y") = py::none()
    );
    module.def(
        "collect_exemplar_targets_into",
        &collect_exemplar_targets_into_native,
        py::arg("raw_records"),
        py::arg("record_indexes"),
        py::arg("region_id"),
        py::arg("frame_height"),
        py::arg("frame_width"),
        py::arg("target_pixels"),
        py::arg("target_colours") = py::none()
    );
    module.def(
        "prepare_exemplar_full_level_into",
        &prepare_exemplar_full_level_into_native,
        py::arg("coverage_count"),
        py::arg("pure_region_id"),
        py::arg("blue"),
        py::arg("green"),
        py::arg("red"),
        py::arg("safe_donor_mask"),
        py::arg("target_pixels"),
        py::arg("target_colours"),
        py::arg("completed"),
        py::arg("completed_colours"),
        py::arg("region_id"),
        py::arg("y0"),
        py::arg("x0"),
        py::arg("interior_y0"),
        py::arg("interior_y1"),
        py::arg("interior_x0"),
        py::arg("interior_x1"),
        py::arg("working"),
        py::arg("donor"),
        py::arg("target"),
        py::arg("processed"),
        py::arg("barrier")
    );
    module.def(
        "downsample_exemplar_level_into",
        &downsample_exemplar_level_into_native,
        py::arg("child_working"),
        py::arg("child_donor"),
        py::arg("child_target"),
        py::arg("child_processed"),
        py::arg("child_barrier"),
        py::arg("parent_working"),
        py::arg("parent_donor"),
        py::arg("parent_target"),
        py::arg("parent_processed"),
        py::arg("parent_barrier")
    );
    module.def(
        "enumerate_exemplar_donor_centres_into",
        &enumerate_exemplar_donor_centres_into_native,
        py::arg("donor_mask"),
        py::arg("integral"),
        py::arg("centres")
    );
    module.def(
        "subsample_exemplar_indexes_into",
        &subsample_exemplar_indexes_into_native,
        py::arg("candidate_count"),
        py::arg("selected_count"),
        py::arg("output")
    );
    module.def(
        "replicate_exemplar_targets_into",
        &replicate_exemplar_targets_into_native,
        py::arg("coarse_working"),
        py::arg("coarse_target"),
        py::arg("coarse_barrier"),
        py::arg("fine_working"),
        py::arg("fine_target")
    );
    module.def(
        "copy_exemplar_patch_into",
        &copy_exemplar_patch_into_native,
        py::arg("working"),
        py::arg("target"),
        py::arg("processed"),
        py::arg("target_row"),
        py::arg("target_column"),
        py::arg("donor_row"),
        py::arg("donor_column")
    );
    module.def(
        "record_exemplar_completions_into",
        &record_exemplar_completions_into_native,
        py::arg("full_working"),
        py::arg("full_target"),
        py::arg("full_processed"),
        py::arg("target_pixels"),
        py::arg("frame_width"),
        py::arg("y0"),
        py::arg("x0"),
        py::arg("completed"),
        py::arg("completed_colours")
    );
    module.def(
        "commit_exemplar_records_into",
        &commit_exemplar_records_into_native,
        py::arg("raw_records"),
        py::arg("record_indexes"),
        py::arg("target_pixels"),
        py::arg("completed"),
        py::arg("completed_colours")
    );
    module.def(
        "validate_repair_record_coverage",
        &validate_repair_record_coverage_native,
        py::arg("raw_records"),
        py::arg("invalid_mask")
    );
    module.def(
        "reconstruct_repair_runs",
        &reconstruct_repair_runs_native,
        py::arg("raw_records"),
        py::arg("height"),
        py::arg("width")
    );
    module.def(
        "build_record_component_ids",
        &build_record_component_ids_native,
        py::arg("raw_records"),
        py::arg("height"),
        py::arg("width")
    );
    module.def(
        "build_record_components_into",
        &build_record_components_into_native,
        py::arg("raw_records"),
        py::arg("height"),
        py::arg("width"),
        py::arg("parent_output"),
        py::arg("rank_output"),
        py::arg("member_output"),
        py::arg("key_output"),
        py::arg("row_workspace")
    );
    module.def(
        "build_component_member_order",
        &build_component_member_order_native,
        py::arg("component_ids"),
        py::arg("component_count")
    );
    module.def(
        "expand_repair_scatter",
        &expand_repair_scatter_native,
        py::arg("raw_records"),
        py::arg("height"),
        py::arg("width")
    );
    module.def(
        "build_background_repair_bits_into",
        &build_background_repair_bits_into_native,
        py::arg("raw_records"),
        py::arg("coverage_count"),
        py::arg("output"),
        py::arg("backend_lane_counts")
    );
    module.def(
        "build_hole_run_histograms_into",
        &build_hole_run_histograms_into_native,
        py::arg("raw_records"),
        py::arg("width"),
        py::arg("lane_histogram"),
        py::arg("span_histogram")
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
        "build_background_proxy_state_into",
        &build_background_proxy_state_into_native,
        py::arg("coverage"),
        py::arg("repair_bits"),
        py::arg("missing_mask")
    );
    module.def(
        "scatter_repair_records_float_into",
        &scatter_repair_records_float_into_native,
        py::arg("raw_records"),
        py::arg("pixel_offset"),
        py::arg("output")
    );
    module.def(
        "materialize_repair_scatter_u8_into",
        &materialize_repair_scatter_u8_into_native,
        py::arg("raw_records"),
        py::arg("pixel_offset"),
        py::arg("colour_output"),
        py::arg("mask_output")
    );
    module.def(
        "select_exemplar_donor",
        &select_exemplar_donor,
        py::arg("working_bgr"),
        py::arg("processed_mask"),
        py::arg("barrier_mask"),
        py::arg("target_row"),
        py::arg("target_column"),
        py::arg("candidates"),
        py::arg("selected_indexes"),
        py::arg("donor_gradient_x") = py::none(),
        py::arg("donor_gradient_y") = py::none()
    );
    module.def(
        "plan_local_strips_in_place",
        &plan_local_strips_in_place_native,
        py::arg("raw_records"),
        py::arg("coverage_count"),
        py::arg("pure_region_id"),
        py::arg("blue"),
        py::arg("green"),
        py::arg("red"),
        py::arg("local_limit_px"),
        py::arg("search_limit"),
        py::arg("safe_radius"),
        py::arg("slot_remaining"),
        py::arg("sample_remaining")
    );
    module.def(
        "plan_local_strips",
        &plan_local_strips_native,
        py::arg("raw_records"),
        py::arg("raw_runs"),
        py::arg("coverage_count"),
        py::arg("pure_region_id"),
        py::arg("blue"),
        py::arg("green"),
        py::arg("red"),
        py::arg("local_limit_px"),
        py::arg("search_limit"),
        py::arg("safe_radius"),
        py::arg("slot_remaining"),
        py::arg("sample_remaining")
    );
    module.def(
        "build_low_resolution_regions",
        &build_low_resolution_regions_native,
        py::arg("one_eye_displacement_px"),
        py::arg("metric_validity"),
        py::arg("near_score")
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
        py::arg("radius")
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
        "expand_foreground_matte_one_pixel",
        &expand_foreground_matte_one_pixel_native,
        py::arg("near_score"),
        py::arg("total_disparity_fraction"),
        py::arg("source_valid"),
        py::arg("final_region_map"),
        py::arg("one_eye_displacement_px"),
        py::arg("metric_valid") = py::none()
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
    module.def(
        "build_exemplar_candidate_mask",
        &build_exemplar_candidate_mask_native,
        py::arg("safe_donor_mask"),
        py::arg("pure_region_id"),
        py::arg("output") = py::none(),
        py::arg("workspace") = py::none()
    );
    module.def(
        "select_exemplar_frontier",
        &select_exemplar_frontier_native,
        py::arg("target_mask"),
        py::arg("processed_mask"),
        py::arg("barrier_mask")
    );
    module.def(
        "component_may_score_exemplar",
        &component_may_score_exemplar_native,
        py::arg("coverage_count"),
        py::arg("pure_region_id"),
        py::arg("safe_donor_mask"),
        py::arg("target_pixels"),
        py::arg("region_id"),
        py::arg("workspace")
    );
    module.def("build_info", &build_info);
}
