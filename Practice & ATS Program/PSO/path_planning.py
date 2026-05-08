import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import CubicSpline


class Obstacle:
    def __init__(self, center, radius):
        self.center = np.array(center, dtype=float)
        self.radius = radius


class Environment:
    def __init__(self, width, height, robot_radius, start, goal):
        self.width = width
        self.height = height
        self.robot_radius = robot_radius
        self.start = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.obstacles = []

    def add_obstacle(self, obstacle):
        self.obstacles.append(obstacle)


def plot_environment(env):
    ax = plt.gca()
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')
    for obs in env.obstacles:
        circle = plt.Circle(obs.center, obs.radius, color='black')
        ax.add_patch(circle)
    # Plot start (red) and goal (green)
    ax.plot(env.start[0], env.start[1], 's', color='red', markersize=10, zorder=5)
    ax.plot(env.goal[0], env.goal[1], 's', color='green', markersize=10, zorder=5)


def plot_path(sol, color='b'):
    x, y = sol
    line, = plt.gca().plot(x, y, color=color, linewidth=2, zorder=4)
    return line


def update_path(sol, line):
    x, y = sol
    line.set_xdata(x)
    line.set_ydata(y)
    plt.gcf().canvas.draw()
    plt.gcf().canvas.flush_events()


def build_path(position, env, num_control_points, resolution):
    """
    Build a spline path from control points encoded in position vector.
    position is a flat array [x1, x2, ..., xn, y1, y2, ..., yn] normalized 0-1.
    Returns (x_path, y_path) arrays.
    """
    xs = position[:num_control_points] * env.width
    ys = position[num_control_points:] * env.height

    # Full set of points: start -> control points -> goal
    x_points = np.concatenate([[env.start[0]], xs, [env.goal[0]]])
    y_points = np.concatenate([[env.start[1]], ys, [env.goal[1]]])

    t = np.linspace(0, 1, len(x_points))
    t_fine = np.linspace(0, 1, resolution)

    cs_x = CubicSpline(t, x_points)
    cs_y = CubicSpline(t, y_points)

    x_path = cs_x(t_fine)
    y_path = cs_y(t_fine)

    return x_path, y_path


def path_length(x_path, y_path):
    dx = np.diff(x_path)
    dy = np.diff(y_path)
    return np.sum(np.sqrt(dx**2 + dy**2))


def penalty(x_path, y_path, env):
    total_penalty = 0.0
    points = np.stack([x_path, y_path], axis=1)

    # Out of bounds penalty
    out_of_bounds = (
        (x_path < 0) | (x_path > env.width) |
        (y_path < 0) | (y_path > env.height)
    )
    total_penalty += np.sum(out_of_bounds) * 1000.0

    # Obstacle collision penalty
    for obs in env.obstacles:
        dists = np.sqrt((x_path - obs.center[0])**2 + (y_path - obs.center[1])**2)
        min_allowed = obs.radius + env.robot_radius
        violations = min_allowed - dists
        violations = np.maximum(violations, 0)
        total_penalty += np.sum(violations) * 1000.0

    return total_penalty


class EnvCostFunction:
    def __init__(self, env, num_control_points, resolution):
        self.env = env
        self.num_control_points = num_control_points
        self.resolution = resolution

    def __call__(self, position):
        x_path, y_path = build_path(
            position, self.env, self.num_control_points, self.resolution
        )
        length = path_length(x_path, y_path)
        pen = penalty(x_path, y_path, self.env)
        cost = length + pen
        details = {
            'sol': (x_path, y_path),
            'length': length,
            'penalty': pen,
        }
        return cost, details