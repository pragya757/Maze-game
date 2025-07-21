import pygame
import numpy as np
import json
from maze_env import MazeEnv
from DuelingDQN import DuelingDQNAgent, preprocess_state
from Simulation.Utils.utils import load_maze_data, astar

# Custom print function to format episode information
def print_episode_info(episode, steps, return_val, epsilon, loss, reward):
    print(f"Episode {episode}/10000 - Steps: {steps}, Return: {return_val:.2f}, Epsilon: {epsilon:.3f}, Loss: {loss:.3f}, Reward: {reward:.2f}")

def print(*args, **kwargs):
    if len(args) > 0 and isinstance(args[0], str) and args[0].startswith("Episode "):
        print(args[0])
    else:
        pass

class NoPrint:
    def __repr__(self):
        return ''
    
    def __str__(self):
        return ''

# Override all printable classes
MazeEnv.__repr__ = NoPrint().__repr__
MazeEnv.__str__ = NoPrint().__str__
DuelingDQNAgent.__repr__ = NoPrint().__repr__
DuelingDQNAgent.__str__ = NoPrint().__str__

# Load trained agent
def load_trained_agent():
    maze = np.load('maze_grid.npy')
    start = np.load('source_destination.npy')[0]
    goal = np.load('source_destination.npy')[1]
    env = MazeEnv(maze, start, goal)
    
    # Create agent with same parameters
    state_size = env.maze_size
    action_size = 4
    agent = DuelingDQNAgent(state_size, action_size, env.maze.shape)
    
    # Load trained weights
    try:
        agent.policy_net.load_state_dict(torch.load('dqn_model.pth'))
        pass
    except:
        pass
    
    return agent, env

# Initialize agent and environment
agent, dqn_env = load_trained_agent()

def load_config(config_file="config.json"):
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        pass
        return get_default_config()
    except json.JSONDecodeError:
        pass
        return get_default_config()

def get_default_config():
    return {
        "display": {"width": 600, "height": 600, "caption": "Maze Simulation", "fps": 5},
        "files": {"maze_file": "maze_grid.npy", "source_destination_file": "source_destination.npy"},
        "colors": {
            "background": [255, 255, 255], "obstacle": [0, 0, 0], "free_space": [255, 255, 255],
            "grid_lines": [200, 200, 200], "path": [0, 0, 255], "start": [0, 255, 0],
            "goal": [255, 0, 255], "agent": [255, 100, 0]
        },
        "sizes": {
            "path_line_width": 3, "start_radius": 7, "goal_radius": 7,
            "agent_radius": 10, "grid_line_width": 1
        },
        "simulation": {"auto_reset": True, "loop_path": False}
    }

config = load_config()

pygame.init()

try:
    maze, start, goal = load_maze_data(
        config["files"]["maze_file"], 
        config["files"]["source_destination_file"]
    )
except FileNotFoundError as e:
    pass
    pygame.quit()
    exit()

env = MazeEnv(maze, start, goal)
path = astar(maze, start, goal)

WIDTH = config["display"]["width"]
HEIGHT = config["display"]["height"]
ROWS, COLS = maze.shape
CELL_SIZE = WIDTH // COLS

win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(config["display"]["caption"])

def draw_grid(win, maze, path, dqn_path, agent_pos, config):
    colors = config["colors"]
    sizes = config["sizes"]
    
    for i in range(ROWS):
        for j in range(COLS):
            rect = pygame.Rect(j*CELL_SIZE, i*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if maze[i][j] == 1:
                pygame.draw.rect(win, colors["obstacle"], rect)  # obstacle
            else:
                pygame.draw.rect(win, colors["free_space"], rect)  # free

    if len(path) > 1:
        path_points = []
        for point in path:
            pixel_x = point[1] * CELL_SIZE + CELL_SIZE // 2
            pixel_y = point[0] * CELL_SIZE + CELL_SIZE // 2
            path_points.append((pixel_x, pixel_y))
        
        pygame.draw.lines(win, colors["path"], False, path_points, sizes["path_line_width"])

    pygame.draw.circle(win, colors["start"], 
                      (start[1]*CELL_SIZE + CELL_SIZE//2, start[0]*CELL_SIZE + CELL_SIZE//2), 
                      sizes["start_radius"])
    pygame.draw.circle(win, colors["goal"], 
                      (goal[1]*CELL_SIZE + CELL_SIZE//2, goal[0]*CELL_SIZE + CELL_SIZE//2), 
                      sizes["goal_radius"])

    # Draw A* agent
    pygame.draw.circle(win, colors["agent"], 
                      (agent_pos[1]*CELL_SIZE + CELL_SIZE//2, agent_pos[0]*CELL_SIZE + CELL_SIZE//2), 
                      sizes["agent_radius"])
    
    # Draw DQN agent
    if len(dqn_path) > 0:
        dqn_agent_pos = dqn_path[0]
        pygame.draw.circle(win, colors["agent"], 
                          (dqn_agent_pos[1]*CELL_SIZE + CELL_SIZE//2, dqn_agent_pos[0]*CELL_SIZE + CELL_SIZE//2), 
                          sizes["agent_radius"] - 2)  # Slightly smaller circle to distinguish
        
    # Draw DQN path as dashed line
    if len(dqn_path) > 1:
        dqn_path_points = []
        for point in dqn_path:
            pixel_x = point[1] * CELL_SIZE + CELL_SIZE // 2
            pixel_y = point[0] * CELL_SIZE + CELL_SIZE // 2
            dqn_path_points.append((pixel_x, pixel_y))
        
        # Draw dashed line
        for i in range(len(dqn_path_points) - 1):
            if i % 2 == 0:
                pygame.draw.line(win, (255, 0, 0), dqn_path_points[i], dqn_path_points[i+1], sizes["path_line_width"] - 1)

def main():
    clock = pygame.time.Clock()
    run = True
    episode = 1
    total_steps = 0
    total_reward = 0
    episode_reward = 0
    episode_loss = 0
    
    # Initialize both environments
    env.reset()
    dqn_env.reset()
    
    # Get initial positions
    agent_pos = env._get_state()
    dqn_agent_pos = dqn_env._get_state()
    
    # Convert state indices to coordinates
    agent_pos = (agent_pos // maze.shape[1], agent_pos % maze.shape[1])
    dqn_agent_pos = (dqn_agent_pos // maze.shape[1], dqn_agent_pos % maze.shape[1])
    
    # Initialize paths
    dqn_path = [dqn_agent_pos]
    
    while run:
        clock.tick(config["display"]["fps"])
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # Get DQN action
        state = preprocess_state(maze, dqn_agent_pos)
        action = agent.act(state)
        
        # Take step in environment
        next_state, reward, done = dqn_env.step(action)
        
        # Update DQN agent
        agent.remember(state, action, reward, preprocess_state(maze, next_state), done)
        loss = agent.replay()
        
        # Update episode statistics
        total_steps += 1
        episode_reward += reward
        if loss is not None:
            episode_loss += loss
            
        # Check if episode is done
        if done:
            print_episode_info(episode, total_steps, episode_reward, agent.epsilon, episode_loss/total_steps, episode_reward)
            episode += 1
            total_steps = 0
            episode_reward = 0
            episode_loss = 0
            
            # Reset environments
            dqn_env.reset()
            dqn_agent_pos = dqn_env._get_state()
            dqn_agent_pos = (dqn_agent_pos // maze.shape[1], dqn_agent_pos % maze.shape[1])
            dqn_path = [dqn_agent_pos]

        # Use A* path for A* agent
        if len(path) > 0:
            next_pos = path[0]
            current_x, current_y = agent_pos
            next_x, next_y = next_pos
            
            if next_x > current_x:  # Move down
                action = 2
            elif next_x < current_x:  # Move up
                action = 0
            elif next_y > current_y:  # Move right
                action = 1
            else:  # Move left
                action = 3
            
            state_index, reward, done = env.step(action)
            agent_pos = (state_index // maze.shape[1], state_index % maze.shape[1])
            
            # Only remove from path if we actually moved
            if agent_pos == next_pos:
                path.pop(0)
                if len(path) == 0:  # Check if goal is reached
                    pass
                if len(path) > 0:
                    next_pos = path[0]  # Update next position


            if dqn_done:
                if config["simulation"]["auto_reset"]:
                    dqn_env.reset()
                    dqn_path = [start]

            if done:
                if config["simulation"]["auto_reset"]:
                    env.reset()

        # Update display
        win.fill(config["colors"]["background"])
        draw_grid(win, maze, path, dqn_path, agent_pos, config)
    while run:
        clock.tick(config["display"]["fps"])
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # Use env.step() to move the agent along the A* path
        if len(path) > 0:
            next_pos = path[0]
            current_x, current_y = agent_pos
            next_x, next_y = next_pos
            
            # Calculate the action based on the direction to the next position
            if next_x > current_x:  # Move down
                action = 2
            elif next_x < current_x:  # Move up
                action = 0
            elif next_y > current_y:  # Move right
                action = 1
            else:  # Move left
                action = 3
            
            # Take the step
            state_index, reward, done = env.step(action)
            agent_pos = (state_index // maze.shape[1], state_index % maze.shape[1])  # Convert state index to coordinates
            
            # Remove the current position from path if we reached it
            if agent_pos == next_pos:
                path.pop(0)
        
        agent_pos = agent_pos if not done else (start // maze.shape[1], start % maze.shape[1])

        if done:
            if config["simulation"]["auto_reset"]:
                env.reset()

        win.fill(config["colors"]["background"])
        draw_grid(win, maze, path, agent_pos, config)
        pygame.display.update()

    pygame.quit()

if __name__ == "__main__":
    main()