import numpy as np

def load_maze_data(maze_file, source_dest_file):
    maze = np.load(maze_file)
    source_dest = np.load(source_dest_file)
    start = tuple(source_dest[0])
    goal = tuple(source_dest[1])
    return maze, start, goal

def astar(maze, start, goal):
    def heuristic(a, b):
        return abs(b[0] - a[0]) + abs(b[1] - a[1])
    
    def get_neighbors(pos):
        neighbors = []
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
            new_pos = (pos[0] + dx, pos[1] + dy)
            if (0 <= new_pos[0] < maze.shape[0] and 
                0 <= new_pos[1] < maze.shape[1] and 
                maze[new_pos] == 0):
                neighbors.append(new_pos)
        return neighbors

    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        current = frontier.pop(0)[1]

        if current == goal:
            break

        for next_pos in get_neighbors(current):
            new_cost = cost_so_far[current] + 1

            if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                cost_so_far[next_pos] = new_cost
                priority = new_cost + heuristic(goal, next_pos)
                frontier.append((priority, next_pos))
                frontier.sort()
                came_from[next_pos] = current

    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from.get(current)
    path.reverse()
    return path

def print(*args, **kwargs):
    pass