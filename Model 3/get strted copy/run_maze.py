import sys
import pygame
import numpy as np
import json
import torch
import torch.nn as nn
import random
import os
import time
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Simulation.envs.maze_env import MazeEnv
from Simulation.Utils.utils import load_maze_data, astar
from DuelingDQN import DuelingDQNAgent, preprocess_state

def load_config(config_file=None):
    try:
        if config_file is None:
            # Get the directory where the current script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_file = os.path.join(script_dir, "config.json")
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Config file {config_file} not found. Using default values.")
        return get_default_config()
    except json.JSONDecodeError:
        print(f"Error parsing {config_file}. Using default values.")
        return get_default_config()

def get_default_config():
    return {
        "display": {"width": 600, "height": 600, "caption": "Maze Simulation", "fps": 10},
        "files": {
            "maze_file": "D:\\maze_grid.npy",
            "source_destination_file": "D:\\source_destination.npy"
        },
        "colors": {
            "background": [255, 255, 255], "obstacle": [0, 0, 0], "free_space": [255, 255, 255],
            "grid_lines": [200, 200, 200], "path": [0, 0, 255], "start": [0, 255, 0],
            "goal": [255, 0, 255], "agent": [255, 100, 0]
        },
        "sizes": {
            "path_line_width": 3, "start_radius": 7, "goal_radius": 7,
            "agent_radius": 10, "grid_line_width": 1
        },
        "simulation": {"auto_reset": True, "loop_path": False},
        "training": {
            "epsilon": 1.0,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.9995
        }
    }

def initialize():
    config = load_config()
    pygame.init()
    
    maze, start, goal = load_maze_data(
        config["files"]["maze_file"], 
        config["files"]["source_destination_file"]
    )
    
    dqn_env = MazeEnv(maze, start, goal)
    path = astar(maze, start, goal)
    
    maze_shape = dqn_env.maze.shape
    state_size = (2, maze_shape[0], maze_shape[1])
    action_size = 4
    
    agent = DuelingDQNAgent(state_size, action_size, maze_shape)
    
    WIDTH = config["display"]["width"]
    HEIGHT = config["display"]["height"]
    ROWS, COLS = maze.shape
    CELL_SIZE = WIDTH // COLS
    
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(config["display"]["caption"])
    
    return config, maze, path, dqn_env, agent, win, CELL_SIZE

def draw_grid(win, maze, path, dqn_path, agent_pos, dqn_agent_pos, config, CELL_SIZE, start, goal):
    colors = config["colors"]
    sizes = config["sizes"]
    
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            rect = pygame.Rect(j*CELL_SIZE, i*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if maze[i][j] == 1:
                pygame.draw.rect(win, colors["obstacle"], rect)
            else:
                pygame.draw.rect(win, colors["free_space"], rect)

    if len(path) > 1:
        path_points = [(p[1]*CELL_SIZE + CELL_SIZE//2, p[0]*CELL_SIZE + CELL_SIZE//2) for p in path]
        pygame.draw.lines(win, colors["path"], False, path_points, sizes["path_line_width"])

    if len(dqn_path) > 1:
        dqn_path_points = [(p[1]*CELL_SIZE + CELL_SIZE//2, p[0]*CELL_SIZE + CELL_SIZE//2) for p in dqn_path]
        for i in range(len(dqn_path_points) - 1):
            if i % 2 == 0:
                pygame.draw.line(win, (255, 0, 0), dqn_path_points[i], dqn_path_points[i+1], sizes["path_line_width"] - 1)

    pygame.draw.circle(win, colors["start"], (start[1]*CELL_SIZE + CELL_SIZE//2, start[0]*CELL_SIZE + CELL_SIZE//2), sizes["start_radius"])
    pygame.draw.circle(win, colors["goal"], (goal[1]*CELL_SIZE + CELL_SIZE//2, goal[0]*CELL_SIZE + CELL_SIZE//2), sizes["goal_radius"])
    
    if len(dqn_path) > 0:
        pygame.draw.circle(win, colors["agent"], (dqn_agent_pos[1]*CELL_SIZE + CELL_SIZE//2, dqn_agent_pos[0]*CELL_SIZE + CELL_SIZE//2), sizes["agent_radius"] - 2)

def main():
    print("Starting initialization...", flush=True)
    config, maze, path, dqn_env, agent, win, CELL_SIZE = initialize()
    print("Initialization complete!", flush=True)
    
    # Initialize episode variables
    # Initialize data collection
    episode_count = 1
    episode_data = []
    total_steps = 0
    goal_reached_count = 0
    steps_to_goal = []
    first_goal_reached = False
    running_cumulative_reward = 0  # Track cumulative reward across episodes
    
    # Create results directory if it doesn't exist
    results_dir = 'training_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize epsilon parameters from config
    epsilon = config["training"]["epsilon"]  # Start with value from config
    initial_epsilon = epsilon  # Save initial value for logging
    epsilon_min = config["training"]["epsilon_min"]  # Minimum exploration rate
    epsilon_decay = config["training"]["epsilon_decay"]  # Decay rate
    
    # Log initial epsilon value
    print(f"Initial epsilon: {initial_epsilon:.6f}")
    
    # Initialize state with environment's state
    dqn_state = torch.FloatTensor(dqn_env.state).unsqueeze(0)
    
    clock = pygame.time.Clock()
    output_file = open("output.txt", "w", buffering=1)
    
    # Run continuous training for 5000 steps
    while episode_count <= 5000:
        # Reset environment and get initial state
        dqn_state = torch.FloatTensor(dqn_env.reset()).unsqueeze(0)
        
        # Reset episode variables
        current_episode_steps = 0
        done = False
        termination_reason = ""
        step_rewards = []  # Track rewards for return calculation
        cumulative_reward = 0  # Track cumulative reward
        
        while not done and current_episode_steps < dqn_env.max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            draw_grid(win, maze, path, [dqn_env.agent_pos], dqn_env.agent_pos, dqn_env.agent_pos, config, CELL_SIZE, dqn_env.start, dqn_env.goal)
            pygame.display.flip()

            action = agent.act(dqn_state)
            next_state, reward, done, termination_reason = dqn_env.step(action)
            
            # Track if goal was reached
            if termination_reason == "goal_reached":
                goal_reached_count += 1
                steps_to_goal.append(current_episode_steps + 1)
                first_goal_reached = True
            
            # Convert next state to tensor
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            
            # Store experience in replay memory - convert tensors to numpy arrays
            agent.remember(dqn_state.squeeze(0).numpy(), action, reward, next_state, done)
            
            # Train the agent
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()
                # Update target network periodically
                if current_episode_steps % 10 == 0:
                    agent.update_target_network()
            else:
                loss = 0.0
            
            # Update tracking variables
            step_rewards.append(reward)
            cumulative_reward += reward
            current_episode_steps += 1
            dqn_state = next_state_tensor
            
            # Control frame rate
            clock.tick(config["display"]["fps"])

        # Calculate episode return
        episode_return = sum([r * (agent.gamma ** t) for t, r in enumerate(step_rewards)])
        
        # Update epsilon with decay (don't decay on first episode to see initial value)
        if episode_count > 1:
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            agent.epsilon = epsilon
        
        # Calculate return
        # Format loss value safely
        loss_value = loss.item() if hasattr(loss, 'item') else loss
        
        # Calculate episode reward and update running cumulative reward
        episode_reward = sum(step_rewards)
        running_cumulative_reward += episode_reward
        
        # Collect episode data
        episode_info = {
            'Episode': episode_count,
            'Steps': current_episode_steps,
            'Return': episode_return,
            'Episode_Reward': episode_reward,
            'Cumulative_Reward': running_cumulative_reward,
            'Termination_Reason': termination_reason,
            'Epsilon': epsilon,
            'Loss': loss_value,
            'Goal_Reached': 1 if termination_reason == 'goal_reached' else 0
        }
        episode_data.append(episode_info)
        
        # Print episode information
        print(
            f"Episode {episode_count}/5000 - "
            f"Steps: {current_episode_steps} - "
            f"Return: {episode_return:.2f} - "
            f"Episode Reward: {episode_reward:.2f} - "
            f"Cumulative Reward: {running_cumulative_reward:.2f} - "
            f"Termination: {termination_reason}"
        )
        
        # Create DataFrame from episode data
        df = pd.DataFrame(episode_data)
        excel_path = os.path.join(results_dir, 'training_progress.xlsx')
        
        try:
            # Save to a temporary file first to prevent corruption
            temp_path = os.path.join(results_dir, 'temp_training_progress.xlsx')
            df.to_excel(temp_path, index=False)
            
            # Rename the temporary file to the final name (atomic operation)
            if os.path.exists(excel_path):
                os.remove(excel_path)
            os.rename(temp_path, excel_path)
            
            # Print progress every 10 episodes to avoid too much console output
            if episode_count % 10 == 0:
                print(f"Saved training progress to {excel_path}")
                
        except Exception as e:
            print(f"Error saving Excel file: {e}")
        
        if output_file:
            output_file.write(str(episode_info) + '\n')
            output_file.flush()
            
        # Update episode counter
        episode_count += 1
        
        # Reset environment for next episode
        if episode_count <= 5000:  # Only reset if we're continuing
            dqn_state = torch.FloatTensor(dqn_env.reset()).unsqueeze(0)
        else:
            # Save final results when done
            final_excel_path = os.path.join(results_dir, 'final_training_results.xlsx')
            try:
                df.to_excel(final_excel_path, index=False)
                total_episodes = len(df)
                goals_reached = df['Goal_Reached'].sum()
                avg_steps = df['Steps'].mean()
                
                print("\nTraining Summary:")
                print(f"Total episodes: {total_episodes}")
                print(f"Total goals reached: {goals_reached}")
                print(f"Goal reach rate: {goals_reached/total_episodes * 100:.2f}%")
                print(f"Average steps per episode: {avg_steps:.2f}")
                print(f"Final epsilon: {epsilon:.6f}")
                print(f"\nDetailed results saved to: {os.path.abspath(final_excel_path)}")
                print('-' * 80)
                
            except Exception as e:
                print(f"Error saving final results: {e}")
                
            break
            
            # Save summary to text file
            if output_file:
                output_file.write("\nTraining Summary:\n")
                output_file.write(f"Total episodes: {total_episodes}\n")
                output_file.write(f"Total goals reached: {goals_reached}\n")
                output_file.write(f"Goal reach rate: {goals_reached/total_episodes * 100:.2f}%\n")
                output_file.write(f"Average steps per episode: {avg_steps:.2f}\n")
                output_file.write(f"Final epsilon: {epsilon:.6f}\n")
                output_file.close()
            break

    pygame.quit()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining stopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Training stopped.")