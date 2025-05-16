import math
import random

# ==================== Maze ====================
maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0]
]

start = (0, 0)
end = (4, 4)

moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def is_valid_move(maze, position):
    y, x = position
    return 0 <= y < len(maze) and 0 <= x < len(maze[0]) and maze[y][x] == 0

def heuristic(cell, end):
    return 1.0 / (math.dist(cell, end) + 1e-6)

def select_next_cell(current, neighbors, pheromone, end):
    alpha = 1.0
    beta = 2.0
    probabilities = []
    total = 0

    for cell in neighbors:
        y, x = cell
        pher = pheromone[y][x] ** alpha
        heur = heuristic(cell, end) ** beta
        prob = pher * heur
        probabilities.append(prob)
        total += prob

    probabilities = [p / total for p in probabilities]
    return random.choices(neighbors, weights=probabilities, k=1)[0]

# ==================== ACO ====================

def ant_walk(maze, pheromone, start, end, max_steps=100):
    path = [start]
    current = start

    for _ in range(max_steps):
        if current == end:
            break

        neighbors = []
        for move in moves:
            new_y, new_x = current[0] + move[0], current[1] + move[1]
            if is_valid_move(maze, (new_y, new_x)) and (new_y, new_x) not in path:
                neighbors.append((new_y, new_x))

        if not neighbors:
            break

        next_cell = select_next_cell(current, neighbors, pheromone, end)
        path.append(next_cell)
        current = next_cell

    return path

def solve_maze_with_aco(maze, start, end):
    num_ants = 10
    num_iterations = 20
    evaporation_rate = 0.5
    pheromone_deposit = 100.0

    pheromone = [[0.1 for _ in range(len(maze[0]))] for _ in range(len(maze))]

    best_path = None
    best_path_length = float('inf')

    for _ in range(num_iterations):
        all_paths = []
        for _ in range(num_ants):
            path = ant_walk(maze, pheromone, start, end)
            if path[-1] == end:
                all_paths.append(path)
                if len(path) < best_path_length:
                    best_path = path
                    best_path_length = len(path)

        for y in range(len(pheromone)):
            for x in range(len(pheromone[0])):
                pheromone[y][x] *= (1 - evaporation_rate)

        for path in all_paths:
            contribution = pheromone_deposit / len(path)
            for cell in path:
                y, x = cell
                pheromone[y][x] += contribution

    return best_path

# ==================== PSO ====================

class Particle:
    def __init__(self, start):
        self.position = start
        self.best_position = start
        self.best_distance = math.dist(start, end)
        self.path = [start]

    def move(self, maze, global_best):
        neighbors = []
        for move in moves:
            new_y, new_x = self.position[0] + move[0], self.position[1] + move[1]
            if is_valid_move(maze, (new_y, new_x)) and (new_y, new_x) not in self.path:
                neighbors.append((new_y, new_x))

        if not neighbors:
            return False

        weights = []
        for cell in neighbors:
            dist_to_global = math.dist(cell, global_best)
            dist_to_personal = math.dist(cell, self.best_position)
            weight = 1 / (dist_to_global + 1e-6) + 1 / (dist_to_personal + 1e-6)
            weights.append(weight)

        next_pos = random.choices(neighbors, weights=weights, k=1)[0]
        self.position = next_pos
        self.path.append(next_pos)

        dist = math.dist(next_pos, end)
        if dist < self.best_distance:
            self.best_distance = dist
            self.best_position = next_pos

        return True

def solve_maze_with_pso(maze, start, end):
    num_particles = 10
    num_iterations = 20

    particles = [Particle(start) for _ in range(num_particles)]
    global_best_position = start
    global_best_distance = math.dist(start, end)
    global_best_path = None

    for _ in range(num_iterations):
        for p in particles:
            moved = p.move(maze, global_best_position)
            if not moved:
                continue

            if p.best_distance < global_best_distance:
                global_best_distance = p.best_distance
                global_best_position = p.best_position
                global_best_path = p.path

    return global_best_path


import streamlit as st

st.title("Maze Solver with AI")

algorithm = st.selectbox("Choose an Algorithm", ["ACO", "PSO"])

if st.button("Solve Maze"):
    if algorithm == "ACO":
        path = solve_maze_with_aco(maze, start, end)
    else:
        path = solve_maze_with_pso(maze, start, end)

    if path:
        st.success(f"Found a path of length {len(path)}!")
        maze_copy = [row[:] for row in maze]
        for x, y in path:
            if maze_copy[x][y] not in ['S', 'E']:
                maze_copy[x][y] = '*'
        maze_str = '\n'.join([' '.join(str(cell) for cell in row) for row in maze_copy])
        st.text(maze_str)
    else:
        st.error("No path found!")
