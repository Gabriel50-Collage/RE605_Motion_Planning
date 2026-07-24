import math
import time
import matplotlib.pyplot as plt

show_animation = True


class DijkstraPlanner:

    def __init__(self, ox, oy, resolution, rr):
        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(ox, oy)

    class Node:
        """One grid cell visited during the search."""
        def __init__(self, x, y, cost, parent_index):
            self.x = x
            self.y = y
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y),
                               0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y),
                              0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while True:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(open_set, key=lambda o: open_set[o].cost)

            current = open_set[c_id]

            if show_animation:
                plt.plot(self.calc_grid_position(current.x, self.min_x),
                         self.calc_grid_position(current.y, self.min_y),
                         "xc")
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            del open_set[c_id]
            closed_set[c_id] = current

            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2],
                                 c_id)
                n_id = self.calc_grid_index(node)

                if not self.verify_node(node):
                    continue
                if n_id in closed_set:
                    continue
                if n_id not in open_set:
                    open_set[n_id] = node
                else:
                    if open_set[n_id].cost > node.cost:
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)
        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        """Walk the parent pointers back from the goal to reconstruct the path."""
        rx = [self.calc_grid_position(goal_node.x, self.min_x)]
        ry = [self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index
        return rx, ry

    def calc_grid_position(self, index, min_position):
        return index * self.resolution + min_position

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)
        if px < self.min_x or py < self.min_y or px >= self.max_x or py >= self.max_y:
            return False
        if self.obstacle_map[node.x][node.y]:
            return False
        return True

    def calc_obstacle_map(self, ox, oy):
        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    def in_collision(self, x0, y0, x1, y1):
        """Sample points along a straight segment and check each against the grid map."""
        dist = math.hypot(x1 - x0, y1 - y0)
        steps = max(int(dist / (self.resolution / 2)), 1)
        for s in range(steps + 1):
            t = s / steps
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            ix = self.calc_xy_index(x, self.min_x)
            iy = self.calc_xy_index(y, self.min_y)
            if ix < 0 or iy < 0 or ix >= self.x_width or iy >= self.y_width:
                return True
            if self.obstacle_map[ix][iy]:
                return True
        return False

    @staticmethod
    def get_motion_model():
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]
        return motion


def path_length(rx, ry):
    """Total Euclidean length of a path, used as the evaluation metric."""
    if len(rx) < 2:
        return 0.0
    total = 0.0
    for i in range(len(rx) - 1):
        total += math.hypot(rx[i + 1] - rx[i], ry[i + 1] - ry[i])
    return total


def shortcut_path(rx, ry, planner):
    """
    Simple optimization pass: repeatedly try to connect non-adjacent
    waypoints directly, skipping intermediate points whenever the
    straight line between them is collision-free. This removes the
    zig-zags the 8-direction grid search tends to produce.
    """
    sx_, sy_ = list(rx), list(ry)
    i = 0
    while i < len(sx_) - 2:
        j = len(sx_) - 1
        shortened = False
        while j > i + 1:
            if not planner.in_collision(sx_[i], sy_[i], sx_[j], sy_[j]):
                del sx_[i + 1:j]
                del sy_[i + 1:j]
                shortened = True
                break
            j -= 1
        if not shortened:
            i += 1
    return sx_, sy_


def build_environment():
    """Build the fixed start/goal points and obstacle set used by both main() and the benchmark."""
    sx, sy = 5.0, 5.0
    gx, gy = 25.0, 5.0

    ox, oy = [], []
    yellow_ox, yellow_oy = [], []

    for i in range(0, 10):
        ox.append(i); oy.append(0)
    for i in range(20, 71):
        ox.append(i); oy.append(0)
    for i in range(0, 71):
        ox.append(i); oy.append(60)
    for i in range(0, 61):
        ox.append(0); oy.append(i)
    for i in range(0, 61):
        ox.append(70); oy.append(i)
    for i in range(0, 50):
        ox.append(10); oy.append(i)
    for i in range(0, 10):
        ox.append(20); oy.append(i)
    for i in range(10, 61):
        ox.append(i); oy.append(50)
    for i in range(10, 51):
        ox.append(60); oy.append(i)
    for i in range(20, 60):
        ox.append(i); oy.append(10)

    for i in range(58, 60):
        yellow_ox.append(45); yellow_oy.append(i)
    for i in range(51, 53):
        yellow_ox.append(45); yellow_oy.append(i)
    for i in range(58, 60):
        yellow_ox.append(50); yellow_oy.append(i)
    for i in range(51, 53):
        yellow_ox.append(50); yellow_oy.append(i)

    all_ox = ox + yellow_ox
    all_oy = oy + yellow_oy
    return sx, sy, gx, gy, ox, oy, yellow_ox, yellow_oy, all_ox, all_oy


def main():
    print("Dijkstra path planning start")

    sx, sy, gx, gy, ox, oy, yellow_ox, yellow_oy, all_ox, all_oy = build_environment()
    grid_size = 2.0
    robot_radius = 1.0

    if show_animation:
        plt.plot(ox, oy, ".k")
        plt.plot(yellow_ox, yellow_oy, ".y", markersize=8)
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")

    t0 = time.time()
    dijkstra = DijkstraPlanner(all_ox, all_oy, grid_size, robot_radius)
    rx, ry = dijkstra.planning(sx, sy, gx, gy)
    elapsed = time.time() - t0

    raw_len = path_length(rx, ry)
    rx_s, ry_s = shortcut_path(rx, ry, dijkstra)
    opt_len = path_length(rx_s, ry_s)

    print(f"Computation time      : {elapsed:.3f} s")
    print(f"Raw Dijkstra path length : {raw_len:.2f} units")
    print(f"Optimized path length    : {opt_len:.2f} units")
    print(f"Improvement              : {100 * (raw_len - opt_len) / raw_len:.1f}%")

    if show_animation:
        plt.plot(rx, ry, "-r", label="Dijkstra (grid)")
        plt.plot(rx_s, ry_s, "-g", linewidth=2, label="Shortcut-optimized")
        plt.legend()
        plt.pause(0.001)
        plt.show()


def run_benchmark(param_sets=None):
    """
    Runs the planner with several (grid_size, robot_radius) combinations
    and prints computation time, raw path length, and optimized path
    length for each. This is the data used for the Pengujian section —
    run this instead of main() when you just need numbers, not the plot.
    """
    global show_animation
    show_animation = False

    if param_sets is None:
        param_sets = [(1.0, 1.0), (2.0, 1.0), (5.0, 1.0), (2.0, 0.5), (2.0, 1.5)]

    sx, sy, gx, gy, ox, oy, yellow_ox, yellow_oy, all_ox, all_oy = build_environment()

    print(f"{'grid_size':>10} {'robot_r':>8} {'time(s)':>9} {'raw_len':>9} {'opt_len':>9} {'improve%':>9}")
    for grid_size, robot_radius in param_sets:
        t0 = time.time()
        dijkstra = DijkstraPlanner(all_ox, all_oy, grid_size, robot_radius)
        rx, ry = dijkstra.planning(sx, sy, gx, gy)
        elapsed = time.time() - t0
        raw_len = path_length(rx, ry)
        rx_s, ry_s = shortcut_path(rx, ry, dijkstra)
        opt_len = path_length(rx_s, ry_s)
        improve = 100 * (raw_len - opt_len) / raw_len if raw_len else 0.0
        print(f"{grid_size:>10.1f} {robot_radius:>8.1f} {elapsed:>9.3f} {raw_len:>9.2f} {opt_len:>9.2f} {improve:>9.1f}")


if __name__ == '__main__':
    main()
    # run_benchmark()