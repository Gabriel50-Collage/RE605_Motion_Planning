import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

N_SAMPLE = 800
N_KNN = 15
MAX_EDGE_LEN = 5.0
show_animation = True

class Node:
    def __init__(self, x, y, cost, parent_index):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent_index = parent_index

def build_circle_obstacle(cx, cy, radius, resolution=36):
    pts_x, pts_y = [], []
    step = 360 // resolution
    for deg in range(0, 360, step):
        angle = math.radians(deg)
        pts_x.append(cx + radius * math.cos(angle))
        pts_y.append(cy + radius * math.sin(angle))
    return pts_x, pts_y

def draw_circle(cx, cy, radius, color="-b"):
    angles = list(range(0, 360, 5)) + [0]
    xs = [cx + radius * math.cos(np.deg2rad(a)) for a in angles]
    ys = [cy + radius * math.sin(np.deg2rad(a)) for a in angles]
    plt.plot(xs, ys, color)

def edge_in_collision(sx, sy, gx, gy, robot_r, obs_tree):
    dist_total = math.hypot(gx - sx, gy - sy)
    if dist_total >= MAX_EDGE_LEN:
        return True
    heading = math.atan2(gy - sy, gx - sx)
    step = robot_r
    n_steps = round(dist_total / step)
    cx, cy = sx, sy
    for _ in range(n_steps):
        nearest, _ = obs_tree.query([cx, cy])
        if nearest <= robot_r:
            return True
        cx += step * math.cos(heading)
        cy += step * math.sin(heading)
    if obs_tree.query([gx, gy])[0] <= robot_r:
        return True
    return False

def build_road_map(pts_x, pts_y, robot_r, obs_tree):
    total = len(pts_x)
    pts_tree = KDTree(np.vstack((pts_x, pts_y)).T)
    road_map = []
    for idx, (ix, iy) in enumerate(zip(pts_x, pts_y)):
        _, neighbours = pts_tree.query([ix, iy], k=total)
        connections = []
        for ni in range(1, len(neighbours)):
            nx = pts_x[neighbours[ni]]
            ny = pts_y[neighbours[ni]]
            if not edge_in_collision(ix, iy, nx, ny, robot_r, obs_tree):
                connections.append(neighbours[ni])
            if len(connections) >= N_KNN:
                break
        road_map.append(connections)
    return road_map

def find_shortest_path(sx, sy, gx, gy, road_map, pts_x, pts_y):
    start = Node(sx, sy, 0.0, -1)
    goal = Node(gx, gy, 0.0, -1)
    open_set = {len(road_map) - 2: start}
    closed_set = {}
    success = True
    while True:
        if not open_set:
            print("Cannot find path")
            success = False
            break
        cur_id = min(open_set, key=lambda k: open_set[k].cost)
        current = open_set[cur_id]
        if show_animation and len(closed_set) % 2 == 0:
            plt.gcf().canvas.mpl_connect(
                "key_release_event",
                lambda e: [exit(0) if e.key == "escape" else None])
            plt.plot(current.x, current.y, "xg")
            plt.pause(0.001)
        if cur_id == len(road_map) - 1:
            print("goal is found!")
            goal.parent_index = current.parent_index
            goal.cost = current.cost
            break
        del open_set[cur_id]
        closed_set[cur_id] = current
        for neighbour_id in road_map[cur_id]:
            d = math.hypot(pts_x[neighbour_id] - current.x,
                              pts_y[neighbour_id] - current.y)
            next_node = Node(pts_x[neighbour_id], pts_y[neighbour_id],
                             current.cost + d, cur_id)
            if neighbour_id in closed_set:
                continue
            if neighbour_id in open_set:
                if open_set[neighbour_id].cost > next_node.cost:
                    open_set[neighbour_id].cost = next_node.cost
                    open_set[neighbour_id].parent_index = cur_id
            else:
                open_set[neighbour_id] = next_node
    if not success:
        return [], []
    path_x = [goal.x]
    path_y = [goal.y]
    pid = goal.parent_index
    while pid != -1:
        node = closed_set[pid]
        path_x.append(node.x)
        path_y.append(node.y)
        pid = node.parent_index
    return path_x, path_y

def sample_free_space(sx, sy, gx, gy, robot_r,
                      ox, oy, obs_tree, rng):
    x_min, x_max = 0.0, 18.0
    y_min, y_max = 0.0, 12.0
    if rng is None:
        rng = np.random.default_rng()
    pts_x, pts_y = [], []
    while len(pts_x) <= N_SAMPLE:
        tx = rng.random() * (x_max - x_min) + x_min
        ty = rng.random() * (y_max - y_min) + y_min
        if obs_tree.query([tx, ty])[0] > robot_r:
            pts_x.append(tx)
            pts_y.append(ty)
    pts_x += [sx, gx]
    pts_y += [sy, gy]
    return pts_x, pts_y

def prm_planning(sx, sy, gx, gy, ox, oy, robot_r, *, rng=None):
    obs_tree = KDTree(np.vstack((ox, oy)).T)
    pts_x, pts_y = sample_free_space(sx, sy, gx, gy,
                                     robot_r, ox, oy, obs_tree, rng)
    if show_animation:
        plt.plot(pts_x, pts_y, ".b")
    road_map = build_road_map(pts_x, pts_y, robot_r, obs_tree)
    path_x, path_y = find_shortest_path(sx, sy, gx, gy,
                                            road_map, pts_x, pts_y)
    return path_x, path_y

def main(rng=None):
    print(__file__ + " start!!")
    sx, sy = 0.0, 0.0
    gx, gy = 6.0, 10.0
    robot_radius = 0.9
    obstacles = [
        (1, 10, 1),
        (3, 10, 1),
        (3, 8, 1),
        (3, 6, 1),
        (5, 5, 1),
        (7, 5, 1),
        (9, 5, 1),
        (9, 3, 1),
        (8, 10, 1),
        (14, 8, 2),
    ]
    ox, oy = [], []
    for (cx, cy, r) in obstacles:
        px, py = build_circle_obstacle(cx, cy, r)
        ox.extend(px)
        oy.extend(py)
    if show_animation:
        plt.figure()
        for (cx, cy, r) in obstacles:
            draw_circle(cx, cy, r)
        plt.plot(sx, sy, "^r")
        plt.plot(gx, gy, "^c")
        plt.grid(True)
        plt.axis("equal")
        plt.axis([-2, 20, -2, 15])
    rx, ry = prm_planning(sx, sy, gx, gy, ox, oy, robot_radius, rng=rng)
    assert rx, "Cannot find path"
    if show_animation:
        plt.plot(rx, ry, "-r")
        plt.pause(0.001)
        plt.show()

if __name__ == "__main__":
    main()