#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

constexpr std::size_t kStackCapacity = 64;
constexpr std::uint64_t kUint32Max = std::numeric_limits<std::uint32_t>::max();

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

struct GeometryFrame {
    std::uint32_t start;
    std::uint32_t end;
    std::uint8_t depth;
    bool check_far;
    double plane2;
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

bool key_less(
    std::uint32_t left,
    std::uint32_t right,
    std::uint32_t width,
    std::uint8_t axis
) {
    if (axis == 0) {
        return left < right;
    }
    const std::uint32_t left_x = left % width;
    const std::uint32_t right_x = right % width;
    if (left_x != right_x) {
        return left_x < right_x;
    }
    const std::uint32_t left_y = left / width;
    const std::uint32_t right_y = right / width;
    if (left_y != right_y) {
        return left_y < right_y;
    }
    return left < right;
}

void insertion_sort(
    std::uint32_t* members,
    std::uint32_t left,
    std::uint32_t right,
    std::uint32_t width,
    std::uint8_t axis
) {
    for (std::uint32_t cursor = left + 1; cursor <= right; ++cursor) {
        const std::uint32_t value = members[cursor];
        std::uint32_t destination = cursor;
        while (
            destination > left &&
            key_less(value, members[destination - 1], width, axis)
        ) {
            members[destination] = members[destination - 1];
            --destination;
        }
        members[destination] = value;
    }
}

std::uint32_t partition_about(
    std::uint32_t* members,
    std::uint32_t left,
    std::uint32_t right,
    std::uint32_t pivot_index,
    std::uint32_t width,
    std::uint8_t axis
) {
    const std::uint32_t pivot = members[pivot_index];
    std::swap(members[pivot_index], members[right]);
    std::uint32_t destination = left;
    for (std::uint32_t cursor = left; cursor < right; ++cursor) {
        if (key_less(members[cursor], pivot, width, axis)) {
            std::swap(members[destination], members[cursor]);
            ++destination;
        }
    }
    std::swap(members[right], members[destination]);
    return destination;
}

void select_kth(
    std::uint32_t* members,
    std::uint32_t left,
    std::uint32_t right,
    std::uint32_t kth,
    std::uint32_t width,
    std::uint8_t axis,
    std::uint8_t selection_depth
);

std::uint32_t median_of_medians(
    std::uint32_t* members,
    std::uint32_t left,
    std::uint32_t right,
    std::uint32_t width,
    std::uint8_t axis,
    std::uint8_t selection_depth
) {
    const std::uint32_t size = right - left + 1;
    if (size <= 5) {
        insertion_sort(members, left, right, width, axis);
        return left + ((size - 1) / 2);
    }
    if (selection_depth >= kStackCapacity) {
        throw std::runtime_error("implicit k-d selection stack capacity exceeded");
    }

    std::uint32_t median_end = left;
    for (std::uint32_t group_start = left; group_start <= right;) {
        const std::uint32_t remaining = right - group_start;
        const std::uint32_t group_end = group_start + (remaining < 4 ? remaining : 4);
        insertion_sort(members, group_start, group_end, width, axis);
        const std::uint32_t group_median = group_start + ((group_end - group_start) / 2);
        std::swap(members[median_end], members[group_median]);
        ++median_end;
        if (remaining < 5) {
            break;
        }
        group_start += 5;
    }

    const std::uint32_t median_kth = left + ((median_end - left - 1) / 2);
    select_kth(
        members,
        left,
        median_end - 1,
        median_kth,
        width,
        axis,
        static_cast<std::uint8_t>(selection_depth + 1)
    );
    return median_kth;
}

void select_kth(
    std::uint32_t* members,
    std::uint32_t left,
    std::uint32_t right,
    std::uint32_t kth,
    std::uint32_t width,
    std::uint8_t axis,
    std::uint8_t selection_depth
) {
    while (left < right) {
        std::uint32_t pivot_index = median_of_medians(
            members,
            left,
            right,
            width,
            axis,
            selection_depth
        );
        pivot_index = partition_about(members, left, right, pivot_index, width, axis);
        if (kth == pivot_index) {
            return;
        }
        if (kth < pivot_index) {
            right = pivot_index - 1;
        } else {
            left = pivot_index + 1;
        }
    }
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
    std::uint32_t width
) {
    if (start == end) {
        return;
    }
    std::array<BuildFrame, kStackCapacity> stack{};
    std::size_t top = push_frame(stack, 0, start, end, 0);
    while (top > 0) {
        const BuildFrame frame = stack[--top];
        const std::uint32_t middle = frame.start + ((frame.end - frame.start - 1) / 2);
        select_kth(
            members,
            frame.start,
            frame.end - 1,
            middle,
            width,
            static_cast<std::uint8_t>(frame.depth & 1),
            0
        );
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

std::size_t push_geometry_frame(
    std::array<GeometryFrame, kStackCapacity>& stack,
    std::size_t top,
    GeometryFrame frame
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
        const std::uint32_t y = sample_index / width;
        const std::uint32_t x = sample_index % width;
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

GeometryBest query_geometry_one(
    const std::uint32_t* members,
    std::uint32_t start,
    std::uint32_t end,
    std::uint32_t width,
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
    std::array<GeometryFrame, kStackCapacity> stack{};
    std::size_t top = push_geometry_frame(
        stack,
        0,
        GeometryFrame{start, end, 0, false, 0.0}
    );
    while (top > 0) {
        const GeometryFrame frame = stack[--top];
        if (frame.check_far) {
            if (!best.found || frame.plane2 <= best.distance2) {
                top = push_geometry_frame(
                    stack,
                    top,
                    GeometryFrame{frame.start, frame.end, frame.depth, false, 0.0}
                );
            }
            continue;
        }
        if (budget_state[0] >= budget_state[1]) {
            throw std::runtime_error("quality geometry visit cap exceeded");
        }
        ++budget_state[0];
        ++(*visited_output);

        const std::uint32_t middle = frame.start + ((frame.end - frame.start - 1) / 2);
        const std::uint32_t sample_index = members[middle];
        const std::uint32_t y = sample_index / width;
        const std::uint32_t x = sample_index % width;
        const double dx = target_x - static_cast<double>(x);
        const double dy = target_y - static_cast<double>(y);
        const double distance2 = (dx * dx) + (dy * dy);
        if (geometry_candidate_is_better(distance2, y, x, best)) {
            best = GeometryBest{true, distance2, y, x, sample_index};
        }

        const bool row_axis = (frame.depth & 1) == 0;
        const double delta = row_axis
            ? target_y - static_cast<double>(y)
            : target_x - static_cast<double>(x);
        const double plane2 = delta * delta;
        const std::pair<std::uint32_t, std::uint32_t> lower{frame.start, middle};
        const std::pair<std::uint32_t, std::uint32_t> upper{middle + 1, frame.end};
        const auto near_range = delta <= 0.0 ? lower : upper;
        const auto far_range = delta <= 0.0 ? upper : lower;
        const std::uint8_t next_depth = static_cast<std::uint8_t>(frame.depth + 1);
        if (far_range.first < far_range.second) {
            top = push_geometry_frame(
                stack,
                top,
                GeometryFrame{
                    far_range.first,
                    far_range.second,
                    next_depth,
                    true,
                    plane2,
                }
            );
        }
        if (near_range.first < near_range.second) {
            top = push_geometry_frame(
                stack,
                top,
                GeometryFrame{
                    near_range.first,
                    near_range.second,
                    next_depth,
                    false,
                    0.0,
                }
            );
        }
    }
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

    py::array_t<std::uint32_t> result_samples(query_count);
    py::array_t<double> result_distances(query_count);
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
            if (!std::isfinite(y_values[query]) || !std::isfinite(x_values[query])) {
                throw std::invalid_argument("geometry query coordinates must be finite");
            }
            const GeometryBest best = query_geometry_one(
                member_values,
                offset_values[region - 1],
                offset_values[region],
                width32,
                y_values[query],
                x_values[query],
                budget_values,
                &visit_output[query]
            );
            sample_output[query] = best.sample_index;
            distance_output[query] = best.distance2;
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
        std::uint32_t capacity
    )
        : distance_(distance),
          owner_(owner),
          region_rank_bits_(region_rank_bits),
          region_rank_count_(region_rank_count),
          position_(position),
          heap_pixel_(capacity),
          capacity_(capacity) {}

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
    std::uint32_t right_pixel
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
    return 256 + 8 * maximum;
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
    const py::array& region_rank_bits
) {
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
    std::uint32_t max_live_entries = 0;
    std::uint64_t insertions = 0;
    std::uint64_t decrease_keys = 0;
    std::uint64_t pop_count = 0;
    {
        py::gil_scoped_release release;
        std::copy(seed_values, seed_values + sample_count, owner_values);
        std::fill(
            distance_values,
            distance_values + sample_count,
            std::numeric_limits<std::uint64_t>::max()
        );
        std::fill(position_values, position_values + sample_count, -1);
        for (std::uint32_t pixel = 0; pixel < sample_count; ++pixel) {
            if (owner_values[pixel] >= rank_count) {
                throw std::invalid_argument(
                    "seed region exceeds region_rank_bits"
                );
            }
            if (band_values[pixel] != 0) {
                ++band_count;
            }
        }

        IndexedRegionHeap heap(
            distance_values,
            owner_values,
            rank_values,
            rank_count,
            position_values,
            band_count
        );
        for (std::uint32_t pixel = 0; pixel < sample_count; ++pixel) {
            if (band_values[pixel] != 0 && owner_values[pixel] != 0) {
                distance_values[pixel] = 0;
                heap.push_or_decrease(pixel);
            }
        }
        for (std::uint32_t pixel = 0; pixel < sample_count; ++pixel) {
            if (band_values[pixel] == 0 || owner_values[pixel] != 0) {
                continue;
            }
            visit_four_neighbours(pixel, height, width, [&](std::uint32_t neighbour) {
                const std::uint32_t region = owner_values[neighbour];
                if (band_values[neighbour] != 0 || region == 0) {
                    return;
                }
                relax_region_pixel(
                    pixel,
                    bgr_movement_cost(guide_values, neighbour, pixel),
                    region,
                    distance_values,
                    owner_values,
                    rank_values,
                    heap
                );
            });
        }
        while (!heap.empty()) {
            const std::uint32_t pixel = heap.pop();
            const std::uint64_t source_distance = distance_values[pixel];
            const std::uint32_t source_owner = owner_values[pixel];
            visit_four_neighbours(pixel, height, width, [&](std::uint32_t neighbour) {
                if (band_values[neighbour] == 0) {
                    return;
                }
                const std::uint64_t movement =
                    bgr_movement_cost(guide_values, pixel, neighbour);
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
        for (std::uint32_t pixel = 0; pixel < sample_count; ++pixel) {
            if (band_values[pixel] != 0 && owner_values[pixel] == 0) {
                throw std::invalid_argument("edge-band component has no seed");
            }
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
        for (std::uint32_t cursor = sample_count32; cursor > 0; --cursor) {
            const std::uint32_t sample_index = cursor - 1;
            if (include != nullptr && include[sample_index] == 0) {
                continue;
            }
            const std::uint32_t region_slot = regions[sample_index] - 1;
            const std::uint32_t destination = --offset_values[region_slot];
            member_values[destination] = sample_index;
        }
        for (std::uint32_t region_index = 0; region_index < region_count; ++region_index) {
            partition_region_tree(
                member_values,
                offset_values[region_index],
                offset_values[region_index + 1],
                width32
            );
        }
    }
    return py::make_tuple(std::move(members), std::move(offsets));
}

py::dict build_info() {
    py::dict result;
    result["algorithm"] = "quality-implicit-region-kd-median-of-medians-v1";
    result["kd_stack_capacity"] = kStackCapacity;
    result["runtime_jit"] = false;
    return result;
}

}  // namespace

PYBIND11_MODULE(_quality_kd_native, module) {
    module.doc() = "Prebuilt deterministic implicit region k-d Task 0 extension";
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
        py::arg("region_rank_bits")
    );
    module.def("build_info", &build_info);
}
