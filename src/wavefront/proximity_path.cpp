#include "wavefront/proximity_path.hpp"

#include <cmath>
#include <deque>
#include <functional>
#include <limits>
#include <queue>
#include <tuple>

namespace namo {

namespace {

constexpr int kObstacle = -1;

// wavefront.py:571-575, in this order; the gradient descent breaks ties by it.
struct Move {
    int di;
    int dj;
    float cost;
};
constexpr Move kMoves[8] = {
    {-1, -1, 1.414f}, {-1, 0, 1.0f}, {-1, 1, 1.414f},
    {0, -1, 1.0f},                   {0, 1, 1.0f},
    {1, -1, 1.414f},  {1, 0, 1.0f},  {1, 1, 1.414f},
};

struct Grid {
    std::vector<std::vector<int>> cells;  // [x][y]
    int width;
    int height;

    bool in_bounds(int x, int y) const { return x >= 0 && x < width && y >= 0 && y < height; }
    bool free(int x, int y) const { return in_bounds(x, y) && cells[x][y] != kObstacle; }
};

// wavefront.py:240-323: force a blocked start free, and its 5x5 block when no 8-neighbour is free.
void trapped_start_recovery(Grid& g, int sx, int sy) {
    if (!g.in_bounds(sx, sy) || g.cells[sx][sy] != kObstacle) {
        return;
    }
    bool trapped = true;
    for (int dy = -1; dy <= 1 && trapped; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if ((dx != 0 || dy != 0) && g.in_bounds(sx + dx, sy + dy) &&
                g.cells[sx + dx][sy + dy] != kObstacle) {
                trapped = false;
                break;
            }
        }
    }
    const int r = trapped ? 2 : 0;
    for (int dy = -r; dy <= r; ++dy) {
        for (int dx = -r; dx <= r; ++dx) {
            if (g.in_bounds(sx + dx, sy + dy)) {
                g.cells[sx + dx][sy + dy] = 0;
            }
        }
    }
}

// wavefront.py:157-227. Multi-source 4-connected BFS from obstacle cells that touch a free cell,
// then cost = 1 + weight * (1 - d / (max_cells + 1)) within max_cells, all float32.
std::vector<float> cost_grid(const Grid& g, double resolution, double distance_m, double weight) {
    std::vector<float> cost(static_cast<size_t>(g.width) * g.height, 1.0f);
    if (distance_m <= 0.0 || weight <= 0.0) {
        return cost;
    }
    const int max_cells = static_cast<int>(std::ceil(distance_m / resolution));
    std::vector<int> dist(cost.size(), max_cells + 1);
    std::deque<std::pair<int, int>> queue;
    constexpr int kN4[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    for (int y = 0; y < g.height; ++y) {
        for (int x = 0; x < g.width; ++x) {
            if (g.cells[x][y] != kObstacle) {
                continue;
            }
            for (const auto& n : kN4) {
                if (g.in_bounds(x + n[0], y + n[1]) && g.cells[x + n[0]][y + n[1]] != kObstacle) {
                    dist[static_cast<size_t>(x) * g.height + y] = 0;
                    queue.emplace_back(x, y);
                    break;
                }
            }
        }
    }
    while (!queue.empty()) {
        const auto [x, y] = queue.front();
        queue.pop_front();
        const int d = dist[static_cast<size_t>(x) * g.height + y];
        if (d >= max_cells) {
            continue;
        }
        for (const auto& n : kN4) {
            const int nx = x + n[0];
            const int ny = y + n[1];
            if (g.free(nx, ny) && d + 1 < dist[static_cast<size_t>(nx) * g.height + ny]) {
                dist[static_cast<size_t>(nx) * g.height + ny] = d + 1;
                queue.emplace_back(nx, ny);
            }
        }
    }
    const float denom = static_cast<float>(max_cells + 1);
    const float w = static_cast<float>(weight);
    for (int x = 0; x < g.width; ++x) {
        for (int y = 0; y < g.height; ++y) {
            const size_t i = static_cast<size_t>(x) * g.height + y;
            if (g.cells[x][y] != kObstacle && dist[i] <= max_cells) {
                const float frac = 1.0f - static_cast<float>(dist[i]) / denom;
                cost[i] = 1.0f + w * frac;
            }
        }
    }
    return cost;
}

}  // namespace

std::vector<std::array<double, 2>> plan_proximity_path(
    std::vector<std::vector<int>> cells,
    double origin_x,
    double origin_y,
    double resolution,
    const std::array<double, 2>& start,
    const std::array<double, 2>& goal,
    double proximity_distance_m,
    double proximity_weight) {
    Grid g{std::move(cells), 0, 0};
    g.width = static_cast<int>(g.cells.size());
    g.height = g.width > 0 ? static_cast<int>(g.cells[0].size()) : 0;

    // wavefront.py:382-391, floor of the offset over the resolution.
    const int sx = static_cast<int>(std::floor((start[0] - origin_x) / resolution));
    const int sy = static_cast<int>(std::floor((start[1] - origin_y) / resolution));
    const int gx = static_cast<int>(std::floor((goal[0] - origin_x) / resolution));
    const int gy = static_cast<int>(std::floor((goal[1] - origin_y) / resolution));

    trapped_start_recovery(g, sx, sy);
    if (!g.free(sx, sy) || !g.free(gx, gy)) {
        return {};
    }
    const std::vector<float> cost = cost_grid(g, resolution, proximity_distance_m, proximity_weight);
    auto at = [&g](int x, int y) { return static_cast<size_t>(x) * g.height + y; };

    // Dijkstra from the goal. Heap entries order as the Python tuples (d, gi, gj) do.
    std::vector<float> dist(cost.size(), std::numeric_limits<float>::infinity());
    using Entry = std::tuple<float, int, int>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> heap;
    dist[at(gx, gy)] = 0.0f;
    heap.emplace(0.0f, gx, gy);
    while (!heap.empty()) {
        const auto [d, x, y] = heap.top();
        heap.pop();
        if (d > dist[at(x, y)]) {
            continue;
        }
        for (const auto& m : kMoves) {
            const int nx = x + m.di;
            const int ny = y + m.dj;
            if (!g.free(nx, ny)) {
                continue;
            }
            const float cell_cost = m.cost * cost[at(nx, ny)];
            const float new_dist = d + cell_cost;
            if (new_dist < dist[at(nx, ny)]) {
                dist[at(nx, ny)] = new_dist;
                heap.emplace(new_dist, nx, ny);
            }
        }
    }

    // Gradient descent from the start: strictly smaller neighbour, first in kMoves order wins.
    if (std::isinf(dist[at(sx, sy)])) {
        return {};
    }
    std::vector<std::pair<int, int>> cells_path{{sx, sy}};
    int x = sx;
    int y = sy;
    const long max_steps = static_cast<long>(g.width) * g.height;
    for (long step = 0; step < max_steps && !(x == gx && y == gy); ++step) {
        int bx = x;
        int by = y;
        float best = dist[at(x, y)];
        for (const auto& m : kMoves) {
            const int nx = x + m.di;
            const int ny = y + m.dj;
            if (g.in_bounds(nx, ny) && dist[at(nx, ny)] < best) {
                best = dist[at(nx, ny)];
                bx = nx;
                by = ny;
            }
        }
        if (bx == x && by == y) {
            return {};
        }
        x = bx;
        y = by;
        cells_path.emplace_back(x, y);
    }

    std::vector<std::array<double, 2>> path;
    path.reserve(cells_path.size());
    for (const auto& [cx, cy] : cells_path) {
        path.push_back({origin_x + (cx + 0.5) * resolution, origin_y + (cy + 0.5) * resolution});
    }
    return path;
}

}  // namespace namo
