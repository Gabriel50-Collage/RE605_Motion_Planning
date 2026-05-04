import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# parameter
N_SAMPLE = 800   # saya naikkan sedikit biar lebih stabil
N_KNN = 10
MAX_EDGE_LEN = 30.0

show_animation = True


class Node:
    def __init__(self, x, y, cost, parent_index):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent_index = parent_index

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + \
               str(self.cost) + "," + str(self.parent_index)


def prm_planning(start_x, start_y, goal_x, goal_y,
                 obstacle_x_list, obstacle_y_list,
                 robot_radius, *, rng=None):

    obstacle_kd_tree = KDTree(np.vstack((obstacle_x_list,
                       obstacle_y_list)).T)

    sample_x, sample_y = sample_points(start_x, start_y,
                         goal_x, goal_y, robot_radius,
                         obstacle_x_list, obstacle_y_list,
                         obstacle_kd_tree, rng)

    if show_animation:
        plt.plot(sample_x, sample_y, ".b")

    road_map = generate_road_map(sample_x, sample_y,
                                 robot_radius, obstacle_kd_tree)

    rx, ry = dijkstra_planning(start_x, start_y, goal_x,
             goal_y, road_map, sample_x, sample_y)

    return rx, ry


def is_collision(sx, sy, gx, gy, rr, obstacle_kd_tree):
    x = sx
    y = sy
    dx = gx - sx
    dy = gy - sy
    yaw = math.atan2(gy - sy, gx - sx)
    d = math.hypot(dx, dy)

    if d >= MAX_EDGE_LEN:
        return True

    D = rr
    n_step = round(d / D)

    for i in range(n_step):
        dist, _ = obstacle_kd_tree.query([x, y])
        if dist <= rr:
            return True
        x += D * math.cos(yaw)
        y += D * math.sin(yaw)

    dist, _ = obstacle_kd_tree.query([gx, gy])
    if dist <= rr:
        return True

    return False


def generate_road_map(sample_x, sample_y, rr, obstacle_kd_tree):
    road_map = []
    n_sample = len(sample_x)
    sample_kd_tree = KDTree(np.vstack((sample_x, sample_y)).T)

    for (i, ix, iy) in zip(range(n_sample), sample_x, sample_y):
        dists, indexes = sample_kd_tree.query([ix, iy], k=n_sample)
        edge_id = []

        for ii in range(1, len(indexes)):
            nx = sample_x[indexes[ii]]
            ny = sample_y[indexes[ii]]

            if not is_collision(ix, iy, nx, ny, rr, obstacle_kd_tree):
                edge_id.append(indexes[ii])

            if len(edge_id) >= N_KNN:
                break

        road_map.append(edge_id)

    return road_map


def dijkstra_planning(sx, sy, gx, gy, road_map, sample_x, sample_y):
    start_node = Node(sx, sy, 0.0, -1)
    goal_node = Node(gx, gy, 0.0, -1)

    open_set, closed_set = dict(), dict()
    open_set[len(road_map) - 2] = start_node   # start node

    path_found = True

    while True:
        if not open_set:
            print("Cannot find path")
            path_found = False
            break

        c_id = min(open_set, key=lambda o: open_set[o].cost)
        current = open_set[c_id]

        if show_animation and len(closed_set.keys()) % 2 == 0:
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(current.x, current.y, "xg")
            plt.pause(0.001)

        if c_id == (len(road_map) - 1):   # goal node
            print("Goal is found!")
            goal_node.parent_index = current.parent_index
            goal_node.cost = current.cost
            break

        del open_set[c_id]
        closed_set[c_id] = current

        for i in range(len(road_map[c_id])):
            n_id = road_map[c_id][i]
            dx = sample_x[n_id] - current.x
            dy = sample_y[n_id] - current.y
            d = math.hypot(dx, dy)
            node = Node(sample_x[n_id], sample_y[n_id],
                        current.cost + d, c_id)

            if n_id in closed_set:
                continue

            if n_id in open_set:
                if open_set[n_id].cost > node.cost:
                    open_set[n_id].cost = node.cost
                    open_set[n_id].parent_index = c_id
            else:
                open_set[n_id] = node

    if not path_found:
        return [], []

    # reconstruct path
    rx, ry = [goal_node.x], [goal_node.y]
    parent_index = goal_node.parent_index
    while parent_index != -1:
        n = closed_set[parent_index]
        rx.append(n.x)
        ry.append(n.y)
        parent_index = n.parent_index

    return rx, ry


def sample_points(sx, sy, gx, gy, rr, ox, oy, obstacle_kd_tree, rng):
    max_x = max(ox)
    max_y = max(oy)
    min_x = min(ox)
    min_y = min(oy)

    sample_x, sample_y = [], []

    if rng is None:
        rng = np.random.default_rng()

    while len(sample_x) <= N_SAMPLE:
        tx = (rng.random() * (max_x - min_x)) + min_x
        ty = (rng.random() * (max_y - min_y)) + min_y

        dist, _ = obstacle_kd_tree.query([tx, ty])

        if dist >= rr:
            sample_x.append(tx)
            sample_y.append(ty)

    # tambahkan start dan goal
    sample_x.append(sx)
    sample_y.append(sy)
    sample_x.append(gx)
    sample_y.append(gy)

    return sample_x, sample_y


def main(rng=None):
    print(__file__ + " start!!")

    # Start dan Goal (sama dengan Dijkstra)
    sx = -5.0
    sy = -5.0
    gx = 50.0
    gy = 50.0
    robot_size = 1.0

    ox, oy = [], []

    # ================== OBSTACLE SAMA PERSIS DENGAN DIJKSTRA ==================
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
    # =======================================================================

    if show_animation:
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "^r")   # start
        plt.plot(gx, gy, "^c")   # goal
        plt.grid(True)
        plt.axis("equal")

    rx, ry = prm_planning(sx, sy, gx, gy, ox, oy, robot_size, rng=rng)

    if rx:
        if show_animation:
            plt.plot(rx, ry, "-r")
            plt.pause(0.001)
            plt.show()
    else:
        print("Tidak bisa menemukan path")


if __name__ == '__main__':
    main()