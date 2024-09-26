from flask import Flask, render_template, request, jsonify
import random
import numpy as np
from queue import Queue
import heapq

app = Flask(__name__)

maze = []
start = (0, 0)
end = (9, 9)
size = 10  # Default size if not specified
initial_energy = 50

# Route to serve the frontend
@app.route('/')
def index():
    return render_template('index.html')

# Generate the maze
@app.route('/generate_maze', methods=['POST'])
def generate_maze():
    global maze, start, end, size, energy_positions
    
    data = request.json
    size = int(data['size'])
    walls = int(data['walls'])
    energy5 = int(data['energy5'])
    energy10 = int(data['energy10'])
    start = (int(data['start_x']) - 1, int(data['start_y']) - 1)
    end = (int(data['exit_x']) - 1, int(data['exit_y']) - 1)
    
    maze = np.zeros((size, size), dtype=int)  # Initialize the maze with 0s
    maze[start] = 1  # Start point is marked as 1
    maze[end] = 2    # Exit point is marked as 2

    # Place walls (-1)
    available_positions = [(i, j) for i in range(size) for j in range(size) if (i, j) != start and (i, j) != end]
    walls_positions = random.sample(available_positions, walls)
    for pos in walls_positions:
        maze[pos] = -1

    # Place energy points
    available_positions = [pos for pos in available_positions if pos not in walls_positions]
    energy_5_positions = random.sample(available_positions, energy5)
    available_positions = [pos for pos in available_positions if pos not in energy_5_positions]
    energy_10_positions = random.sample(available_positions, energy10)

    for pos in energy_5_positions:
        maze[pos] = 5
    for pos in energy_10_positions:
        maze[pos] = 10

    return jsonify({'maze': maze.tolist()})

# BFS Algorithm
@app.route('/solve_maze', methods=['GET'])
def solve_maze():
    algorithm = request.args.get('algorithm')
    
    if algorithm == 'bfs':
        path = bfs_solve()
    elif algorithm == 'astar':
        path = astar_solve()
    
    return jsonify({'maze': maze.tolist(), 'path': path})

# BFS Search Algorithm
def bfs_solve():
    global size, start, end
    visited = np.zeros((size, size), dtype=bool)
    queue = Queue()
    queue.put((start, [start], initial_energy))  # Keep track of the path and energy
    
    directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    
    while not queue.empty():
        (current, path, energy) = queue.get()
        
        if current == end:
            return path
        
        if energy <= 0:
            continue
        
        for direction in directions:
            next_cell = (current[0] + direction[0], current[1] + direction[1])
            
            if 0 <= next_cell[0] < size and 0 <= next_cell[1] < size and not visited[next_cell] and maze[next_cell] != -1:
                next_energy = energy - 1
                if maze[next_cell] == 5:
                    next_energy += 5
                elif maze[next_cell] == 10:
                    next_energy += 10
                next_energy = min(next_energy, initial_energy)  # Cap energy to initial
                
                visited[next_cell] = True
                queue.put((next_cell, path + [next_cell], next_energy))

    return []

# A* Search Algorithm
def astar_solve():
    global size, start, end
    open_list = []
    heapq.heappush(open_list, (0, start, [start], initial_energy))  # (priority, current, path, energy)
    
    g_costs = {start: 0}
    directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    
    while open_list:
        _, current, path, energy = heapq.heappop(open_list)
        
        if current == end:
            return path
        
        if energy <= 0:
            continue
        
        for direction in directions:
            next_cell = (current[0] + direction[0], current[1] + direction[1])
            
            if 0 <= next_cell[0] < size and 0 <= next_cell[1] < size and maze[next_cell] != -1:
                new_g = g_costs[current] + 1
                next_energy = energy - 1
                if maze[next_cell] == 5:
                    next_energy += 5
                elif maze[next_cell] == 10:
                    next_energy += 10
                next_energy = min(next_energy, initial_energy)  # Cap energy to initial
                
                if next_cell not in g_costs or new_g < g_costs[next_cell]:
                    g_costs[next_cell] = new_g
                    f_cost = new_g + heuristic(next_cell, end)
                    heapq.heappush(open_list, (f_cost, next_cell, path + [next_cell], next_energy))

    return []

# Heuristic for A* (Manhattan distance)
def heuristic(cell, goal):
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

if __name__ == '__main__':
    app.run(debug=True)
