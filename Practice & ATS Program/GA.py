import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches

START = np.array([0.0, 0.0])
GOAL  = np.array([10.0, 10.0])

OBSTACLES = [
    (2.0, 2.0, 1.0),
    (5.0, 4.0, 1.2),
    (7.0, 7.0, 1.0),
    (3.5, 7.0, 0.8),
    (8.0, 3.0, 1.0),
]

NUM_WAYPOINTS  = 5
MAX_POPULATION = 60
MUTATION_RATE  = 0.2
MAX_COORD      = 10.0

def create_gen():
    waypoints = np.random.uniform(0, MAX_COORD, size=(NUM_WAYPOINTS, 2))
    return waypoints

def path_length(waypoints):
    path = np.vstack([START, waypoints, GOAL])
    total = 0.0
    for i in range(len(path) - 1):
        total += np.linalg.norm(path[i+1] - path[i])
    return total

def obstacle_penalty(waypoints):
    path = np.vstack([START, waypoints, GOAL])
    penalty = 0.0
    for i in range(len(path) - 1):
        p1, p2 = path[i], path[i+1]
        for (ox, oy, r) in OBSTACLES:
            center = np.array([ox, oy])
            d = p2 - p1
            t = np.clip(np.dot(center - p1, d) / (np.dot(d, d) + 1e-9), 0, 1)
            closest = p1 + t * d
            dist = np.linalg.norm(closest - center)
            if dist < r:
                penalty += (r - dist) * 50
    return penalty

def calculate_fitness(waypoints):
    length  = path_length(waypoints)
    penalty = obstacle_penalty(waypoints)
    cost    = length + penalty
    fitness = 1.0 / (cost + 1e-9) * 1000
    return fitness

def create_population():
    population = []
    for _ in range(MAX_POPULATION):
        wp  = create_gen()
        fit = calculate_fitness(wp)
        population.append({'waypoints': wp, 'fitness': fit})
    return population

def selection(population):
    sorted_pop = sorted(population, key=lambda x: x['fitness'], reverse=True)
    return sorted_pop[0], sorted_pop[1]

def crossover(parent1, parent2):
    cp = np.random.randint(1, NUM_WAYPOINTS)
    child1_wp = np.vstack([parent1['waypoints'][:cp], parent2['waypoints'][cp:]])
    child2_wp = np.vstack([parent2['waypoints'][:cp], parent1['waypoints'][cp:]])
    child1 = {'waypoints': child1_wp, 'fitness': calculate_fitness(child1_wp)}
    child2 = {'waypoints': child2_wp, 'fitness': calculate_fitness(child2_wp)}
    return child1, child2

def mutation(individual):
    wp = individual['waypoints'].copy()
    for i in range(NUM_WAYPOINTS):
        if np.random.rand() <= MUTATION_RATE:
            wp[i] += np.random.uniform(-1.5, 1.5, size=2)
            wp[i]  = np.clip(wp[i], 0, MAX_COORD)
    fit = calculate_fitness(wp)
    return {'waypoints': wp, 'fitness': fit}

def regeneration(population, children):
    for child in children:
        worst_idx = min(range(len(population)), key=lambda i: population[i]['fitness'])
        if child['fitness'] > population[worst_idx]['fitness']:
            population[worst_idx] = child
    return population

def display(generation, best, elapsed):
    fit    = best['fitness']
    length = path_length(best['waypoints'])
    pen    = obstacle_penalty(best['waypoints'])
    print(f"Gen {generation:4d} | Fitness: {fit:8.4f} | Length: {length:.4f} | Penalty: {pen:.4f} | Time: {elapsed}")

def plot_result(best_individual, history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.set_title('Quadcopter VTOL Path Planning - GA Result')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    for (ox, oy, r) in OBSTACLES:
        circle = patches.Circle((ox, oy), r, color='red', alpha=0.4, label='Obstacle')
        ax.add_patch(circle)
        ax.plot(ox, oy, 'r+', markersize=8)

    wp   = best_individual['waypoints']
    path = np.vstack([START, wp, GOAL])
    ax.plot(path[:, 0], path[:, 1], 'b-o', linewidth=2, markersize=5, label='Optimal Path')
    ax.plot(START[0], START[1], 'gs', markersize=12, label='Start')
    ax.plot(GOAL[0],  GOAL[1],  'g*', markersize=14, label='Goal')
    ax.plot(wp[:, 0], wp[:, 1], 'bo', markersize=6)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc='upper left')

    ax2 = axes[1]
    ax2.plot(history, 'b-', linewidth=1.5)
    ax2.set_title('Fitness over Generations')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Best Fitness')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

target_fitness = 8.0
MAX_GEN        = 500

print("=" * 65)
print("  GA Path Planning - Autonomous Quadcopter VTOL")
print("=" * 65)
print(f"Start      : {START}")
print(f"Goal       : {GOAL}")
print(f"Waypoints  : {NUM_WAYPOINTS}")
print(f"Population : {MAX_POPULATION}")
print(f"Mutation   : {MUTATION_RATE}")
print(f"Obstacles  : {len(OBSTACLES)}")
print("=" * 65)

startTime  = datetime.datetime.now()
population = create_population()
history    = []

parent1, parent2 = selection(population)
best_individual  = parent1
elapsed          = datetime.datetime.now() - startTime
display(0, best_individual, elapsed)

for generation in range(1, MAX_GEN + 1):
    child1, child2   = crossover(parent1, parent2)
    child1           = mutation(child1)
    child2           = mutation(child2)
    population       = regeneration(population, [child1, child2])
    parent1, parent2 = selection(population)

    if parent1['fitness'] > best_individual['fitness']:
        best_individual = parent1
        elapsed         = datetime.datetime.now() - startTime
        display(generation, best_individual, elapsed)

    history.append(best_individual['fitness'])

    if best_individual['fitness'] >= target_fitness and obstacle_penalty(best_individual['waypoints']) == 0:
        print(f"\nSolution found at generation {generation}!")
        break

print("\n" + "=" * 65)
print("FINAL RESULT")
print("=" * 65)
print(f"Best Fitness  : {best_individual['fitness']:.4f}")
print(f"Path Length   : {path_length(best_individual['waypoints']):.4f} m")
print(f"Penalty       : {obstacle_penalty(best_individual['waypoints']):.4f}")
print(f"Waypoints     :")
for i, wp in enumerate(best_individual['waypoints']):
    print(f"  WP{i+1}: ({wp[0]:.3f}, {wp[1]:.3f})")
print(f"Total Time    : {datetime.datetime.now() - startTime}")

plot_result(best_individual, history)