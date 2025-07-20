import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import psutil
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from envs.maze_env import MazeEnv
from model import DuelingDQN
import os
import json

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)
    training_config = config.get('training', {})
    
    # Extract epsilon parameters
    epsilon = training_config.get('epsilon', 1.0)
    epsilon_min = training_config.get('epsilon_min', 0.010)
    epsilon_decay = training_config.get('epsilon_decay', 0.999)

def save_training_stats(stats, episode, output_dir="training_output"):
    """Save training statistics to text and Excel files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to text file
    with open(os.path.join(output_dir, f'training_stats_{episode}.txt'), 'w') as f:
        for key, value in stats.items():
            f.write(f'{key}: {value}\n')
    
    # Save to Excel file
    df = pd.DataFrame(stats.items(), columns=['Metric', 'Value'])
    df.to_excel(os.path.join(output_dir, f'training_stats_{episode}.xlsx'), index=False)
    
    # Also save cumulative stats
    if os.path.exists(os.path.join(output_dir, 'cumulative_stats.json')):
        with open(os.path.join(output_dir, 'cumulative_stats.json'), 'r') as f:
            cumulative_stats = json.load(f)
    else:
        cumulative_stats = {}
    
    cumulative_stats[episode] = stats
    with open(os.path.join(output_dir, 'cumulative_stats.json'), 'w') as f:
        json.dump(cumulative_stats, f, indent=4)

def print_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

def train_dueling_dqn():
    # Check MPS availability
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built with MPS: {torch.backends.mps.is_built()}")
    
    # Load maze from .npy file
    maze = np.load("/Users/pragya/Desktop/dueiling/maze_grid.npy")
    start = [0, 0]  # Define the starting position
    
    # Print initial memory usage
    print("Initial memory usage:")
    print_memory_usage()
    goal = [maze.shape[0] - 1, maze.shape[1] - 1]  # Define the goal position (bottom-right corner)

    # Initialize environment
    env = MazeEnv(maze, start, goal)

    # Initialize Dueling DQN agent
    state_size = env.maze_size
    action_size = 4  # up, right, down, left
    
    # Set device with MPS (Metal Performance Shaders) support for Apple Silicon
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) for GPU acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA for GPU acceleration")
    else:
        device = torch.device("cpu")
        print("Using CPU (no GPU acceleration available)")
    
    print(f"Using device: {device}")
    agent = DuelingDQNAgent(state_size, action_size, device)
    
    # Set epsilon parameters from config
    initial_epsilon = config['training']['epsilon']
    epsilon_min = config['training']['epsilon_min']
    epsilon_decay = config['training']['epsilon_decay']
    
    # Initialize epsilon tracking
    epsilon = initial_epsilon
    print(f'Initializing epsilon: {epsilon}, min: {epsilon_min}, decay: {epsilon_decay}')

    # Training parameters
    episodes = 5000  # Number of episodes to train
    max_steps = 200  # Maximum steps per episode
    batch_size = 64  # Size of batches for training
    update_target = 100  # Update target network every 100 episodes
    save_interval = 500  # Save model every 500 episodes
    
    # Experience replay buffer
    replay_buffer_size = 10000
    replay_buffer = []
    
    # Lists to store statistics
    goal_reached_steps = []  # Track steps when goals were reached
    goal_reached_rewards = []  # Track rewards when goals were reached
    cumulative_rewards = []  # Track cumulative rewards
    total_reward_sum = 0  # Track total cumulative reward
    
    # Training parameters
    gamma = 0.99  # Discount factor
    episodes = 5000  # Number of episodes to train
    
    # Track epsilon values for plotting
    epsilon_values = []
    
    # Initialize epsilon parameters from config
    initial_epsilon = config['training']['epsilon']  # Start value
    epsilon_min = config['training']['epsilon_min']  # Minimum value
    epsilon_decay = config['training']['epsilon_decay']  # Decay rate
    
    # Set initial epsilon in the model
    model.epsilon = initial_epsilon

    # Model saving parameters
    model_save_path = "models/"
    os.makedirs(model_save_path, exist_ok=True)
    
    # Best model tracking
    best_model_path = os.path.join(model_save_path, 'best_model.pth')
    best_reward = float('-inf')  # Track best reward achieved
    best_episode = 0  # Track which episode achieved the best reward

    # Save plots in training_plots directory
    plots_dir = 'training_plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create model save directory if it doesn't exist
    os.makedirs(model_save_path, exist_ok=True)

    # Initialize target network in agent
    agent.target_net.load_state_dict(agent.policy_net.state_dict())
    agent.target_net.eval()
    
    # Training loop
    print(f'\nStarting training for {episodes} episodes...')
    print(f'Max steps per episode: {max_steps}')
    print(f'-' * 50)
    
    for episode in range(episodes):
        # Update target network periodically
        if episode % update_target == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            print(f'\nEpisode {episode}: Updated target network')
            
        # Calculate epsilon for this episode
        new_epsilon = max(epsilon_min, initial_epsilon * (epsilon_decay ** episode))
        print(f'Episode {episode}: Epsilon changed from {epsilon:.6f} to {new_epsilon:.6f}')
        epsilon = new_epsilon
        agent.epsilon = epsilon  # Update agent's epsilon
            
        # Periodically save model
        if episode % save_interval == 0 and episode > 0:
            torch.save(model.state_dict(), os.path.join(model_save_path, f'dueling_dqn_maze_{episode}.pth'))
            print(f'\nEpisode {episode}: Model saved')
            
        print(f'\nEpisode {episode + 1}/{episodes}, Epsilon: {epsilon:.6f}')
        state = env.reset()
        done = False
        total_reward = 0
        
        for step in range(max_steps):
            # Get action from agent (handles epsilon-greedy internally)
            action = agent.act(state)
            
            # Take action
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            # Store transition in agent's memory and train
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            # Update state
            state = next_state
            
            # Print progress
            if (step + 1) % 100 == 0:
                print(f'Step {step + 1}/{max_steps}, Reward: {total_reward:.2f}')
            
            if done:
                break
            
        # Store statistics
        cumulative_rewards.append(total_reward)
        epsilon_values.append(epsilon)
        
        # Save statistics to files
        stats = {
            'Episode': episode + 1,
            'Total Reward': total_reward,
            'Average Reward (Last 100)': np.mean(cumulative_rewards[-100:]) if len(cumulative_rewards) >= 100 else np.mean(cumulative_rewards),
            'Epsilon': epsilon,
            'Steps Taken': step + 1,
            'Goal Reached': 1 if total_reward > 0 else 0
        }
        save_training_stats(stats, episode + 1)
        
        # Save best model if current reward is better
        if total_reward > best_reward:
            best_reward = total_reward
            best_episode = episode + 1
            torch.save(model.state_dict(), best_model_path)
            print(f'New best model saved with reward: {best_reward:.2f}')
            
        # Save model periodically
        if episode % save_interval == 0 and episode > 0:
            torch.save(model.state_dict(), os.path.join(model_save_path, f'dueling_dqn_maze_{episode}.pth'))
            print(f'Episode {episode + 1}: Model saved')
            
        # Print episode summary
        print(f'Episode {episode + 1} completed. Total reward: {total_reward:.2f}')
        print(f'Average reward (last 100 episodes): {np.mean(cumulative_rewards[-100:]):.2f}')
        print(f'Epsilon: {epsilon:.3f}')
        print(f'Best reward: {best_reward:.2f} (Episode {best_episode})')
        print('-' * 50)

    # Create multiple plots
    
    # Plot 1: Episode Lengths with Goal Reaching
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(episode_lengths)), episode_lengths, label='Episode Length', alpha=0.5)
    plt.stem(range(len(episode_lengths)), episode_lengths, linefmt='-', markerfmt='o', basefmt='-', label='Episode Length')
    plt.title('Episode Length Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    
    # Add vertical lines for episodes where goals were reached
    for episode_idx, goals in enumerate(goals_reached_per_episode):
        if goals > 0:
            plt.axvline(x=episode_idx, color='g', linestyle='--', alpha=0.3)
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('episode_length_plot.png')
    plt.show()

    # Plot 2: Goals Reached Per Episode
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(goals_reached_per_episode)), goals_reached_per_episode, label='Goals Reached')
    plt.stem(range(len(goals_reached_per_episode)), goals_reached_per_episode, linefmt='-', markerfmt='o', basefmt='-', label='Goals Reached')
    plt.title('Goals Reached Per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Number of Goals Reached')
    
    # Add trend line
    x = np.array(range(len(goals_reached_per_episode)))
    y = np.array(goals_reached_per_episode)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--", label='Trend Line')
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('goals_reached_plot.png')
    plt.show()

    # Plot epsilon decay
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(epsilon_values)), epsilon_values, 'r-', label='Epsilon Value')
    plt.title('Epsilon Decay Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon Value')
    
    # Add horizontal line for epsilon_min
    plt.axhline(y=epsilon_min, color='g', linestyle='--', label=f'Min Epsilon ({epsilon_min})')
    
    # Add trend line for epsilon decay
    x = np.array(range(len(epsilon_values)))
    y = np.array(epsilon_values)
    z = np.polyfit(x, y, 2)  # Using quadratic fit for the curve
    p = np.poly1d(z)
    plt.plot(x, p(x), "b--", label='Decay Trend')
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('epsilon_decay_plot.png')
    plt.show()
    
    # Plot cumulative rewards with epsilon overlay
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Cumulative Reward', color=color)
    ax1.plot(range(len(cumulative_rewards)), cumulative_rewards, color=color, label='Cumulative Reward')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Add trend line for rewards
    x = np.array(range(len(cumulative_rewards)))
    y = np.array(cumulative_rewards)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax1.plot(x, p(x), "b--", label='Reward Trend')
    
    # Create a second y-axis for epsilon values
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Epsilon', color=color)
    ax2.plot(range(len(epsilon_values)), epsilon_values, color=color, alpha=0.3, label='Epsilon')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('Cumulative Reward and Epsilon Decay')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('combined_reward_epsilon_plot.png')
    plt.show()

def test_value_agent(model, env, episodes=50, render=True):
    """
    Test the trained Dueling DQN model on the maze environment.
    
    Args:
        model: Trained Dueling DQN model
        env: Maze environment
        episodes: Number of test episodes
        render: Whether to render the environment during testing
    
    Returns:
        success_rate: Array containing success status for each episode
    """
    print(f'Info: Testing of the Agent has been started over the Maze Simulation...')
    print(f'Info: Source: {env.start} Destination: {env.goal}')
    print(f'-' * 147)

    success_rate = np.zeros(episodes)
    model.eval()

    for episode in range(episodes):
        # Memory management at the start of each episode
        if episode > 0 and episode % 10 == 0:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                torch.mps.synchronize()
            print(f"\nMemory usage after episode {episode}:")
            print_memory_usage()
            
        state = env.reset()
        state = preprocess_state(state, maze.shape)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # Clear any remaining computation graphs
        if hasattr(torch, 'mps'):
            torch.mps.empty_cache()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0)
            action = model(state_tensor).argmax().item()
            new_state, reward, done = env.step(action)

        state = new_state
        done = False
        returns = 0
        step = 0
        
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0)
                action = model(state_tensor).argmax().item()
                new_state, reward, done = env.step(action)

            state = new_state
            step += 1
            returns += reward
            
            if render:
                env.update_display()
                # Add a small delay for better visualization
                import time
                time.sleep(0.1)
            
            print(f'Episode {episode + 1}/{episodes} - Steps: {step}, Return: {returns:.2f}')
            success_rate[episode] = 1 if returns > 0 else 0  # Use cumulative reward as success metric

    print(f'-' * 147)
    print(f'Info: Testing has been completed...')
    return success_rate

    print(f'-' * 147)
    print(f'Info: Testing has been completed...')
    return success_rate

if __name__ == "__main__":
    # Train the model
    model = train_dueling_dqn()
    
    # Test the trained model
    maze = np.load("/Users/pragya/Desktop/dueiling/maze_grid.npy")
    start = [0, 0]
    goal = [maze.shape[0] - 1, maze.shape[1] - 1]
    env = MazeEnv(maze, start, goal)
    
    # Test the model
    success_rate = test_value_agent(model, env, episodes=100, render=True)
    
    # Print summary statistics
    print(f"\nTest Summary:")
    print(f"Success Rate: {success_rate.mean() * 100:.2f}%")
    print(f"Average Steps per Episode: {success_rate.sum() / success_rate.mean():.2f}")