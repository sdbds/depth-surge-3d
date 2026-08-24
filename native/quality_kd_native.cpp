#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

constexpr std::size_t kStackCapacity = 64;
constexpr std::uint64_t kUint32Max = std::numeric_limits<std::uint32_t>::max();
constexpr std::uint64_t kRepairRecordCap = (64ULL * 1024ULL * 1024ULL) / 16ULL;
constexpr std::uint8_t kDividerAddMarker = 0x40;
constexpr std::uint32_t kPackedCoordinateLimit = 0xffff;

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

void insertion_sort_numeric(
    std::uint32_t* members,
    std::uint32_t left,
    std::uint32_t right
) {
    for (std::uint32_t cursor = left + 1; cursor <= right; ++cursor) {
        const std::uint32_t value = members[cursor];
        std::uint32_t destination = cursor;
        while (destination > left && value < members[destination - 1]) {
            members[destination] = members[destination - 1];
            --destination;
        }
        members[destination] = value;
    }
}

inline void compare_swap_numeric(std::uint32_t& left, std::uint32_t& right) {
    if (right < left) {
        std::swap(left, right);
    }
}

std::uint32_t median_of_five_numeric(const std::uint32_t* values) {
    std::uint32_t a = values[0];
    std::uint32_t b = values[1];
    std::uint32_t c = values[2];
    std::uint32_t d = values[3];
    std::uint32_t e = values[4];
    compare_swap_numeric(a, b);
    compare_swap_numeric(c, d);
    compare_swap_numeric(a, c);
    compare_swap_numeric(b, d);
    compare_swap_numeric(b, c);
    return e < b ? b : (c < e ? c : e);
}

std::uint32_t partition_about_numeric(
    std::uint32_t* members,
    std::uint32_t left,
    std::uint32_t right,
    std::uint32_t pivot_index
) {
    const std::uint32_t pivot = members[pivot_index];
    std::swap(members[pivot_index], members[right]);
    std::uint32_t destination = left;
    for (std::uint32_t cursor = left; cursor < right; ++cursor) {
        const std::uint32_t cursor_value = members[cursor];
        const std::uint32_t destination_value = members[destination];
        const std::uint32_t swap_mask =
            0U - static_cast<std::uint32_t>(cursor_value < pivot);
        const std::uint32_t swap_bits =
            (cursor_value ^ destination_value) & swap_mask;
        members[destination] = destination_value ^ swap_bits;
        members[cursor] = cursor_value ^ swap_bits;
        destination += swap_mask & 1U;
    }
    std::swap(members[right], members[destination]);
    return destination;
}

void select_kth_numeric(
    std::uint32_t* members,
    std::uint32_t left,
    std::uint32_t right,
    std::uint32_t kth,
    std::uint8_t selection_depth
);

std::uint32_t median_of_medians_numeric(
    std::uint32_t* members,
    std::uint32_t left,
    std::uint32_t right,
    std::uint8_t selection_depth
) {
    const std::uint32_t size = right - left + 1;
    if (size <= 5) {
        insertion_sort_numeric(members, left, right);
        return left + ((size - 1) / 2);
    }
    if (selection_depth >= kStackCapacity) {
        throw std::runtime_error("implicit k-d selection stack capacity exceeded");
    }

    std::uint32_t median_end = left;
    for (std::uint32_t group_start = left; group_start <= right;) {
        const std::uint32_t remaining = right - group_start;
        const std::uint32_t group_end = group_start + (remaining < 4 ? remaining : 4);
        std::uint32_t group_median = group_start;
        if (remaining >= 4) {
            const std::uint32_t median_value = median_of_five_numeric(members + group_start);
            while (members[group_median] != median_value) {
                ++group_median;
            }
        } else {
            insertion_sort_numeric(members, group_start, group_end);
            group_median = group_start + ((group_end - group_start) / 2);
        }
        std::swap(members[median_end], members[group_median]);
        ++median_end;
        if (remaining < 5) {
            break;
        }
        group_start += 5;
    }

    const std::uint32_t median_kth = left + ((median_end - left - 1) / 2);
    select_kth_numeric(
        members,
        left,
        median_end - 1,
        median_kth,
        static_cast<std::uint8_t>(selection_depth + 1)
    );
    return median_kth;
}

void select_kth_numeric(
    std::uint32_t* members,
    std::uint32_t left,
    std::uint32_t right,
    std::uint32_t kth,
    std::uint8_t selection_depth
) {
    if (right - left < 96) {
        std::sort(members + left, members + right + 1);
        return;
    }
    while (left < right) {
        std::uint32_t pivot_index = median_of_medians_numeric(
            members,
            left,
            right,
            selection_depth
        );
        pivot_index = partition_about_numeric(
            members,
            left,
            right,
            pivot_index
        );
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
                middle,
                0
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
        const std::uint32_t y = width_divider.divide(sample_index);
        const std::uint32_t x = sample_index - y * width;
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
                width_divider,
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

struct RepairRecordStatistics {
    std::uint64_t record_count = 0;
    std::uint64_t prefill_run_count = 0;
    std::uint64_t repair_run_count = 0;
    std::uint64_t invalid_lane_count = 0;
};

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
    for (std::uint32_t row = 0; row < height; ++row) {
        const std::uint32_t row_start = row * fine_width;
        std::uint32_t fine_column = 0;
        while (fine_column < fine_width) {
            if (valid[row_start + fine_column] != 0) {
                ++fine_column;
                continue;
            }
            const std::uint32_t run_start = fine_column;
            while (
                fine_column + 1 < fine_width &&
                valid[row_start + fine_column + 1] == 0
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
                    throw std::runtime_error("64 MiB repair record arena exceeded");
                }
                callback(row, start, end, region, far_side);
            };

            if (!has_left) {
                emit(
                    run_start,
                    run_end,
                    region_id[row_start + run_end + 1],
                    1
                );
            } else if (!has_right) {
                emit(
                    run_start,
                    run_end,
                    region_id[row_start + run_start - 1],
                    0
                );
            } else {
                const float left_score = near_score[row_start + run_start - 1];
                const float right_score = near_score[row_start + run_end + 1];
                if (left_score < right_score) {
                    emit(
                        run_start,
                        run_end,
                        region_id[row_start + run_start - 1],
                        0
                    );
                } else if (right_score < left_score) {
                    emit(
                        run_start,
                        run_end,
                        region_id[row_start + run_end + 1],
                        1
                    );
                } else {
                    const std::uint32_t left_length =
                        (run_end - run_start + 2) / 2;
                    const std::uint32_t left_end = run_start + left_length - 1;
                    emit(
                        run_start,
                        left_end,
                        region_id[row_start + run_start - 1],
                        0
                    );
                    if (left_end < run_end) {
                        emit(
                            left_end + 1,
                            run_end,
                            region_id[row_start + run_end + 1],
                            1
                        );
                    }
                }
            }
            ++fine_column;
        }
    }
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
            member_values[destination] = pixel;
        }
        for (std::uint32_t region_index = 0; region_index < region_count; ++region_index) {
            partition_region_tree(
                member_values,
                offset_values[region_index],
                offset_values[region_index + 1],
                height,
                width
            );
        }
    }
    return py::make_tuple(std::move(members), std::move(offsets));
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
    bool allow_seeded_missing
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

    py::array_t<std::uint32_t> visits(query_count);
    auto* visit_values = visits.mutable_data();
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
        for (std::uint32_t region_index = 0; region_index < region_count; ++region_index) {
            partition_region_tree(
                member_values,
                offset_values[region_index],
                offset_values[region_index + 1],
                static_cast<std::uint32_t>(height),
                width32,
                packed_coordinates
            );
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

py::tuple build_exemplar_sobel_planes(const py::array& working_bgr) {
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
    py::array_t<std::int16_t> gradient_x({height, width});
    py::array_t<std::int16_t> gradient_y({height, width});
    auto* output_x = gradient_x.mutable_data();
    auto* output_y = gradient_y.mutable_data();
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
        std::vector<std::pair<std::uint32_t, std::uint32_t>> best_donors;
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
                std::vector<std::pair<std::uint32_t, std::uint32_t>> donors;
                donors.reserve(donor_count);
                for (std::uint32_t donor_index = 0; donor_index < donor_count; ++donor_index) {
                    const std::int64_t column =
                        candidate_start - static_cast<std::int64_t>(donor_index) * direction;
                    legal = legal && column >= 0 && column < width;
                    legal = legal && (column - boundary_column) * direction > 0;
                    if (legal) {
                        donors.emplace_back(
                            static_cast<std::uint32_t>(candidate_row),
                            static_cast<std::uint32_t>(column)
                        );
                    }
                }
                if (!legal || donors.size() != donor_count) {
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
                for (const auto& donor : donors) {
                    if (!local_donor_is_safe(
                        coverage,
                        regions,
                        height,
                        width,
                        donor.first,
                        donor.second,
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
                    best_donors = std::move(donors);
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
            const auto donor = best_donors[donor_index];
            const auto sample = static_cast<std::uint64_t>(donor.first) * width + donor.second;
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
    {
        py::gil_scoped_release release;
        for (std::uint32_t row = 0; row < height; ++row) {
            for (std::uint32_t column = 0; column < width; ++column) {
                const std::uint64_t pixel = static_cast<std::uint64_t>(row) * width + column;
                const std::uint32_t region = regions[pixel];
                if (!is_region(region)) {
                    continue;
                }
                if (
                    column + 1 < width && is_region(regions[pixel + 1]) &&
                    region != regions[pixel + 1]
                ) {
                    current[pixel] = 1;
                    current[pixel + 1] = 1;
                }
                if (
                    row + 1 < height && is_region(regions[pixel + width]) &&
                    region != regions[pixel + width]
                ) {
                    current[pixel] = 1;
                    current[pixel + width] = 1;
                }
            }
        }
        for (std::uint32_t iteration = 0; iteration < radius; ++iteration) {
            std::fill(next.begin(), next.end(), 0);
            for (std::uint32_t row = 0; row < height; ++row) {
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

py::array build_exemplar_candidate_mask_native(
    const py::array& safe_donor_mask,
    const py::array& pure_region_id
) {
    validate_array(safe_donor_mask, py::dtype::of<bool>(), "safe_donor_mask");
    validate_array(pure_region_id, py::dtype::of<std::uint32_t>(), "pure_region_id");
    if (
        safe_donor_mask.shape(0) != pure_region_id.shape(0) ||
        safe_donor_mask.shape(1) != pure_region_id.shape(1)
    ) {
        throw py::value_error("exemplar candidate rasters must match");
    }
    const auto height = static_cast<std::uint32_t>(safe_donor_mask.shape(0));
    const auto width = static_cast<std::uint32_t>(safe_donor_mask.shape(1));
    const auto* safe = static_cast<const std::uint8_t*>(safe_donor_mask.data());
    const auto* regions = static_cast<const std::uint32_t*>(pure_region_id.data());
    py::array_t<bool> output({height, width});
    auto* result = output.mutable_data();
    std::fill(result, result + static_cast<std::uint64_t>(height) * width, false);
    if (height < 9 || width < 9) {
        return output;
    }
    {
        py::gil_scoped_release release;
        for (std::uint32_t row = 4; row + 4 < height; ++row) {
            for (std::uint32_t column = 4; column + 4 < width; ++column) {
                const std::uint32_t pixel = row * width + column;
                const std::uint32_t region = regions[pixel];
                if (
                    safe[pixel] == 0 || region == 0 ||
                    region == std::numeric_limits<std::uint32_t>::max()
                ) {
                    continue;
                }
                bool candidate = true;
                for (std::int32_t delta_row = -4; delta_row <= 4 && candidate; ++delta_row) {
                    const std::uint32_t sample_row = static_cast<std::uint32_t>(
                        static_cast<std::int32_t>(row) + delta_row
                    );
                    for (std::int32_t delta_column = -4; delta_column <= 4; ++delta_column) {
                        const std::uint32_t sample_column = static_cast<std::uint32_t>(
                            static_cast<std::int32_t>(column) + delta_column
                        );
                        const std::uint32_t sample = sample_row * width + sample_column;
                        if (safe[sample] == 0 || regions[sample] != region) {
                            candidate = false;
                            break;
                        }
                    }
                }
                result[pixel] = candidate;
            }
        }
    }
    return output;
}

struct ExemplarMaskLevel {
    std::uint32_t height;
    std::uint32_t width;
    std::vector<std::uint8_t> donor;
    std::vector<std::uint8_t> target;
    std::vector<std::uint8_t> processed;
    std::vector<std::uint8_t> barrier;
};

bool exemplar_level_has_candidate(const ExemplarMaskLevel& level) {
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

bool exemplar_level_has_frontier(const ExemplarMaskLevel& level) {
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

ExemplarMaskLevel downsample_exemplar_masks(const ExemplarMaskLevel& child) {
    ExemplarMaskLevel parent;
    parent.height = (child.height + 1) / 2;
    parent.width = (child.width + 1) / 2;
    const std::uint64_t count = static_cast<std::uint64_t>(parent.height) * parent.width;
    parent.donor.assign(count, 0);
    parent.target.assign(count, 0);
    parent.processed.assign(count, 0);
    parent.barrier.assign(count, 0);
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
    return parent;
}

bool component_may_score_exemplar_native(
    const py::array& coverage_count,
    const py::array& pure_region_id,
    const py::array& safe_donor_mask,
    const py::array& target_pixels,
    std::uint32_t region
) {
    validate_array(coverage_count, py::dtype::of<std::uint8_t>(), "coverage_count");
    validate_array(pure_region_id, py::dtype::of<std::uint32_t>(), "pure_region_id");
    validate_array(safe_donor_mask, py::dtype::of<bool>(), "safe_donor_mask");
    validate_vector(target_pixels, py::dtype::of<std::uint32_t>(), "target_pixels");
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
    ExemplarMaskLevel level;
    level.height = y1 - y0;
    level.width = x1 - x0;
    const std::uint64_t local_count = static_cast<std::uint64_t>(level.height) * level.width;
    level.donor.assign(local_count, 0);
    level.target.assign(local_count, 0);
    level.processed.assign(local_count, 0);
    level.barrier.assign(local_count, 0);
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
        std::array<ExemplarMaskLevel, 3> levels;
        levels[0] = std::move(level);
        std::size_t level_count = 1;
        while (level_count < levels.size()) {
            const auto& current = levels[level_count - 1];
            const std::uint32_t next_height = (current.height + 1) / 2;
            const std::uint32_t next_width = (current.width + 1) / 2;
            if (std::min(next_height, next_width) < 32) {
                break;
            }
            levels[level_count] = downsample_exemplar_masks(current);
            ++level_count;
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
    module.def(
        "build_repair_records",
        &build_repair_records,
        py::arg("valid"),
        py::arg("winner_near_score"),
        py::arg("winner_region_id"),
        py::arg("winner_bgr") = py::none()
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
        py::arg("allow_seeded_missing") = false
    );
    module.def(
        "build_exemplar_sobel_planes",
        &build_exemplar_sobel_planes,
        py::arg("working_bgr")
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
        "build_exemplar_candidate_mask",
        &build_exemplar_candidate_mask_native,
        py::arg("safe_donor_mask"),
        py::arg("pure_region_id")
    );
    module.def(
        "component_may_score_exemplar",
        &component_may_score_exemplar_native,
        py::arg("coverage_count"),
        py::arg("pure_region_id"),
        py::arg("safe_donor_mask"),
        py::arg("target_pixels"),
        py::arg("region_id")
    );
    module.def("build_info", &build_info);
}
