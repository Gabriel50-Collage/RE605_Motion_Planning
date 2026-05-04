import matplotlib.pyplot as plt
import math

show_animation = True


class Dijkstra:

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x
            self.y = y
            self.cost = cost
            self.parent_index = parent_index

    def __init__(self, ox, oy, resolution, robot_radius):

        self.min_x = None
        self.min_y = None
        self.max_x = None
        self.max_y = None
        self.x_width = None
        self.y_width = None
        self.resolution = resolution
        self.robot_radius = robot_radius

        self.obstacle_map = None
        self.calc_obstacle_map(ox, oy)
        self.motion = self.get_motion_model()

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
            x = self.calc_position(ix, self.min_x)

            for iy in range(self.y_width):
                y = self.calc_position(iy, self.min_y)

                for iox, ioy in zip(ox, oy):

                    d = math.hypot(iox - x, ioy - y)

                    if d <= self.robot_radius:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():

        motion = [
            [1, 0, 1],
            [0, 1, 1],
            [-1, 0, 1],
            [0, -1, 1],
            [-1, -1, math.sqrt(2)],
            [-1, 1, math.sqrt(2)],
            [1, -1, math.sqrt(2)],
            [1, 1, math.sqrt(2)]
        ]

        return motion

    def planning(self, sx, sy, gx, gy):

        start_node = self.Node(
            self.calc_xy_index(sx, self.min_x),
            self.calc_xy_index(sy, self.min_y),
            0.0, -1)

        goal_node = self.Node(
            self.calc_xy_index(gx, self.min_x),
            self.calc_xy_index(gy, self.min_y),
            0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_index(start_node)] = start_node

        while True:

            if not open_set:
                print("No path found")
                return [], []

            c_id = min(open_set, key=lambda o: open_set[o].cost)
            current = open_set[c_id]

            if show_animation:
                plt.plot(self.calc_position(current.x, self.min_x),
                         self.calc_position(current.y, self.min_y), "xc")

                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.cost = current.cost
                goal_node.parent_index = current.parent_index
                break

            del open_set[c_id]
            closed_set[c_id] = current

            for move_x, move_y, move_cost in self.motion:

                node = self.Node(
                    current.x + move_x,
                    current.y + move_y,
                    current.cost + move_cost,
                    c_id)

                n_id = self.calc_index(node)

                if n_id in closed_set:
                    continue

                if not self.verify_node(node):
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node
                else:
                    if open_set[n_id].cost > node.cost:
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):

        rx = [self.calc_position(goal_node.x, self.min_x)]
        ry = [self.calc_position(goal_node.y, self.min_y)]

        parent_index = goal_node.parent_index

        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_position(n.x, self.min_x))
            ry.append(self.calc_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    def calc_position(self, index, minp):
        return index * self.resolution + minp

    def calc_xy_index(self, position, minp):
        return round((position - minp) / self.resolution)

    def calc_index(self, node):
        return node.y * self.x_width + node.x

    def verify_node(self, node):

        px = self.calc_position(node.x, self.min_x)
        py = self.calc_position(node.y, self.min_y)

        if px < self.min_x or py < self.min_y or px >= self.max_x or py >= self.max_y:
            return False

        if self.obstacle_map[node.x][node.y]:
            return False

        return True


def main():

    sx = -5.0
    sy = -5.0
    gx = 50.0
    gy = 50.0

    grid_size = 2.0
    robot_radius = 1.0

    ox, oy = [], []

    # border map
    for i in range(-10, 61):
        ox.append(i)
        oy.append(-10)

    for i in range(-10, 61):
        ox.append(60)
        oy.append(i)

    for i in range(-10, 61):
        ox.append(i)
        oy.append(60)

    for i in range(-10, 61):
        ox.append(-10)
        oy.append(i)

    # obstacle kiri bawah
    for i in range(-10, 10):
        ox.append(i)
        oy.append(10)

    # obstacle horizontal tengah
    for i in range(0, 20):
        ox.append(i)
        oy.append(30)

    # obstacle vertikal tengah
    for i in range(-10, 40):
        ox.append(20)
        oy.append(i)

    # obstacle vertikal kanan
    for i in range(20, 60):
        ox.append(40)
        oy.append(i)

    if show_animation:
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")

    dijkstra = Dijkstra(ox, oy, grid_size, robot_radius)

    rx, ry = dijkstra.planning(sx, sy, gx, gy)

    if show_animation:
        plt.plot(rx, ry, "-r")
        plt.show()


if __name__ == '__main__':
    main()