def main():
    import torch
    import numpy as np
    from model import DuelingDQN
    from environment import MazeEnv
    from utils import preprocess_state, store_experience, update_target_network
    import random
    from collections import deque

    # Hyperparameters
    num_episodes = 1000
    max_steps = 100
    learning_rate = 0.001
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    batch_size = 32
    memory_size = 10000

    # Initialize environment and model
    maze_size = (5, 5)  # Example maze size
    destination = (4, 4)  # Example destination
    env = MazeEnv(maze_size, destination)
    state_size = env.get_state_size()
    action_size = env.get_action_size()
    model = DuelingDQN(state_size, action_size)
    target_model = DuelingDQN(state_size, action_size)
    target_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Experience replay memory
    memory = deque(maxlen=memory_size)

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))

        for step in range(max_steps):
            # Choose action
            if random.random() < epsilon:
                action = random.choice(range(action_size))
            else:
                with torch.no_grad():
                    action = model(torch.FloatTensor(preprocess_state(state))).argmax().item()

            # Take action
            next_state, reward, done = env.step(action)
            total_reward += reward

            # Store experience
            memory.append((state, action, reward, next_state, done))

            # Train model
            if len(memory) >= batch_size:
                experiences = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*experiences)

                states = torch.FloatTensor([preprocess_state(s) for s in states])
                next_states = torch.FloatTensor([preprocess_state(s) for s in next_states])
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                dones = torch.FloatTensor(dones)

                # Compute target
                with torch.no_grad():
                    target_q_values = target_model(next_states)
                    target = rewards + (1 - dones) * gamma * target_q_values.max(1)[0]

                # Compute loss
                predicted_q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze()
                loss = model.calculate_loss(predicted_q_values, target)

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update target network
                update_target_network(model, target_model)

            state = next_state
            if done:
                break

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

if __name__ == "__main__":
    main()