import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import datetime
import random
from collections import deque
import os

# Neural Network for Dueling DQ
class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()
        self.action_size = action_size
        
        channels, height, width = state_size
        
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # Calculate the size after convolutions
        conv_output_size = 32 * height * width
        self.fc1 = nn.Linear(conv_output_size, 128)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
    
    def forward(self, x):
        
        if x.dim() == 5:  
            x = x.squeeze(1)  
        
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


# Agent for Dueling DQN
class DuelingDQNAgent(object):
    def __init__(self, state_size, action_size, maze_shape):
        self.state_size = state_size
        self.action_size = action_size
        self.maze_shape = maze_shape
        
        self.policy_net = DuelingDQN(state_size, action_size)
        self.target_net = DuelingDQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.batch_size = 16  # Reduced from 64 to 16 for MPS memory efficiency
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        try:
            os.makedirs('models', exist_ok=True)
            file_path = 'models/best_model.pth'
            if os.path.isfile(file_path):
                checkpoint = torch.load(file_path)
                self.policy_net.load_state_dict(checkpoint['model_state_dict'])
                self.target_net.load_state_dict(checkpoint['model_state_dict'])
                self.epsilon = checkpoint.get('epsilon', self.epsilon)
        except:
            pass

    def remember(self, state, action, reward, next_state, done):
        # Ensure states are torch tensors with correct shape
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
            if state.dim() == 3:  # [2, 15, 15]
                state = state.unsqueeze(0)  # [1, 2, 15, 15]
        if isinstance(next_state, np.ndarray):
            next_state = torch.FloatTensor(next_state)
            if next_state.dim() == 3:  # [2, 15, 15]
                next_state = next_state.unsqueeze(0)  # [1, 2, 15, 15]

        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # Convert state to torch tensor and ensure correct shape [1, C, H, W]
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        
        if isinstance(state, torch.Tensor):
            if state.dim() == 3:  # [C, H, W]
                state = state.unsqueeze(0)  # [1, C, H, W]
            elif state.dim() == 4:
                if state.shape[0] != 1:  # [N, C, H, W] -> [1, C, H, W]
                    state = state[0].unsqueeze(0)
            else:
                raise ValueError(f"Unexpected state shape: {state.shape}")

        with torch.no_grad():
            self.policy_net.eval()
            q_values = self.policy_net(state)
            self.policy_net.train()
        return q_values.argmax().item()

    def save_model(self, episode_count, episode_reward, total_reward):
        try:
            torch.save({
                'model_state_dict': self.policy_net.state_dict(),
                'epsilon': self.epsilon
            }, 'models/dqn_model.pth')
        except:
            pass

    def save_best_model(self, episode_count, episode_reward, total_reward):
        try:
            torch.save({
                'model_state_dict': self.policy_net.state_dict(),
                'epsilon': self.epsilon,
                'episode_count': episode_count,
                'episode_reward': episode_reward,
                'total_reward': total_reward
            }, 'models/best_model.pth')
        except Exception as e:
            print(f"Error saving best model: {str(e)}")
            print(f"Detailed error: {traceback.format_exc()}")

    def __repr__(self):
        return ''

    def __str__(self):
        return ''

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0.0  # Return 0.0 when not enough samples
            
        # Sample minibatch
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Prepare tensors
        states = torch.stack([torch.FloatTensor(s[0]) for s in minibatch])
        actions = torch.LongTensor([int(s[1]) for s in minibatch])  # Convert action to int
        rewards = torch.FloatTensor([s[2] for s in minibatch])
        next_states = torch.stack([torch.FloatTensor(s[3]) for s in minibatch])
        dones = torch.FloatTensor([s[4] for s in minibatch])
        
        # Get Q-values from policy network
        q_values = self.policy_net(states)
        
        # Get next Q-values from target network
        next_q_values = self.target_net(next_states)
        
        # Calculate target values
        target = rewards + self.gamma * next_q_values.max(1)[0] * (1 - dones)
        
        # Create target tensor
        target_f = q_values.clone()
        
        # Update target values for the selected actions
        for i in range(self.batch_size):
            target_f[i, actions[i]] = target[i]
        
        # Calculate loss
        loss = nn.MSELoss()(q_values, target_f)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Preprocessing function
def preprocess_state(state, maze_shape):
    agent_pos = np.where(state[0] == 1)
    if len(agent_pos[0]) > 0:
        pos = (agent_pos[0][0], agent_pos[1][0])
        state_tensor = np.zeros((2, maze_shape[0], maze_shape[1]))
        state_tensor[0, pos[0], pos[1]] = 1
        state_tensor[1] = state[1]
        return torch.FloatTensor(state_tensor).unsqueeze(0)  # Add batch dimension
    return torch.FloatTensor(np.zeros((2, maze_shape[0], maze_shape[1]))).unsqueeze(0)  # Add batch dimension
