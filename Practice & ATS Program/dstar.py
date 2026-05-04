import math
import sys
import matplotlib.pyplot as plt

show_animation = True

OFFSET = 10   # coordinate shift so negative positions can exist


class State:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.state = "new"
        self.h = 0
        self.k = 0

    def cost(self, other):
        if self.state == "#" or other.state == "#":
            return math.inf
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def set_state(self, tag):
        if tag not in ["new", "#", "o", "c", "s", "e"]:
            return
        self.state = tag


class Map:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.map = self.init_map()

    def init_map(self):
        map_list = []
        for i in range(self.row):
            tmp = []
            for j in range(self.col):
                tmp.append(State(i, j))
            map_list.append(tmp)
        return map_list

    def get_neighbors(self, state):
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue

                ni = state.x + di
                nj = state.y + dj

                if ni < 0 or ni >= self.row or nj < 0 or nj >= self.col:
                    continue

                neighbors.append(self.map[ni][nj])

        return neighbors

    def set_obstacle(self, point_list):

        for x, y in point_list:

            gx = x + OFFSET
            gy = y + OFFSET

            if 0 <= gx < self.row and 0 <= gy < self.col:
                self.map[gx][gy].set_state("#")


class Dstar:

    def __init__(self, mapp):

        self.map = mapp
        self.open_list = set()

    def process_state(self):

        if not self.open_list:
            return -1

        x = min(self.open_list, key=lambda s: s.k)

        kold = x.k
        self.remove(x)

        if kold < x.h:

            for y in self.map.get_neighbors(x):

                if y.k <= kold and y.h > x.h + x.cost(y):

                    y.parent = x
                    self.insert(y, x.h + x.cost(y))

        elif kold == x.h:

            for y in self.map.get_neighbors(x):

                if (y.state == "new" or
                    (y.parent == x and y.h != x.h + x.cost(y)) or
                    (y.parent != x and y.h > x.h + x.cost(y))):

                    y.parent = x
                    self.insert(y, x.h + x.cost(y))

        else:

            self.insert(x, x.h)

            for y in self.map.get_neighbors(x):

                if y.parent == x:
                    self.insert(y, x.h + x.cost(y))

        return x.k

    def insert(self, state, h_new):

        if state.state == "new":
            state.k = h_new

        elif state.state == "o":
            state.k = min(state.k, h_new)

        elif state.state == "c":
            state.k = min(state.h, h_new)

        state.h = h_new
        state.state = "o"

        self.open_list.add(state)

    def remove(self, state):

        if state.state == "o":
            state.state = "c"

        self.open_list.discard(state)

    def run(self, start, goal):

        self.open_list.clear()

        self.open_list.add(goal)

        goal.state = "o"
        goal.k = 0
        goal.h = 0
        goal.parent = None

        while True:

            if not self.open_list:
                print("No path found")
                return [], []

            self.process_state()

            if start.state == "c":
                break

        rx, ry = [], []

        tmp = start

        while tmp is not None:

            rx.append(tmp.x - OFFSET)
            ry.append(tmp.y - OFFSET)

            tmp = tmp.parent

        rx.reverse()
        ry.reverse()

        return rx, ry


def main():

    print("D* Lite start!!")

    m = Map(100, 100)

    ox, oy = [], []

    # left wall
    for i in range(-10, 61):
        ox.append(-10)
        oy.append(i)

    # right wall
    for i in range(-10, 61):
        ox.append(60)
        oy.append(i)

    # bottom wall
    for i in range(-10, 61):
        ox.append(i)
        oy.append(-10)

    # upper wall
    for i in range(-10, 61):
        ox.append(i)
        oy.append(60)

    # inner obstacles
    for i in range(-10, 41):
        ox.append(20)
        oy.append(i)

    for i in range(20, 60):
        ox.append(40)
        oy.append(i)

    for i in range(0, 20):
        ox.append(i)
        oy.append(30)

    for i in range(-10, 10):
        ox.append(i)
        oy.append(10)

    obstacle_points = list(zip(ox, oy))
    m.set_obstacle(obstacle_points)

    start_pos = [-5, -5]
    goal_pos = [50, 50]

    if show_animation:

        plt.plot(ox, oy, ".k")
        plt.plot(start_pos[0], start_pos[1], "og")
        plt.plot(goal_pos[0], goal_pos[1], "xb")

        plt.grid(True)
        plt.axis("equal")

    start = m.map[start_pos[0] + OFFSET][start_pos[1] + OFFSET]
    end = m.map[goal_pos[0] + OFFSET][goal_pos[1] + OFFSET]

    dstar = Dstar(m)

    rx, ry = dstar.run(start, end)

    if show_animation and rx:

        plt.plot(rx, ry, "-r", linewidth=2)

        plt.show()


if __name__ == '__main__':
    main()