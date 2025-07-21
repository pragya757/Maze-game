import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DuelingDQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        # Feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
        # Move model to device
        self.device = torch.device("mps" if torch.backends.mps.is_available() else 
                                 "cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def forward(self, state):
        features = self.feature_layer(state)
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages using the Dueling DQN formula
        qvals = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals

class DuelingDQNAgent:
    def __init__(self, state_size, action_size, device):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Initialize networks
        self.policy_net = DuelingDQN(state_size, action_size)
        self.target_net = DuelingDQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Set up optimizer with weight decay for better generalization
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        # Enable memory pinning if using CUDA
        self.pin_memory = self.device.type == 'cuda'
        if self.pin_memory:
            print("CUDA memory pinning enabled")
        elif self.device.type == 'mps':
            print("MPS (Metal Performance Shaders) is being used")
        
    def encode_state(self, state):
        """Encode the state as a one-hot tensor on the correct device."""
        one_hot_tensor = torch.zeros(self.state_size, dtype=torch.float32, device=self.device)
        one_hot_tensor[state] = 1
        target_index = self.maze.destination[0] * self.maze.maze_size + self.maze.destination[1]
        one_hot_tensor[target_index] = 2
        return one_hot_tensor
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Select an action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = self.encode_state(state).unsqueeze(0)
            # Ensure the state tensor is on the same device as the model
            state_tensor = state_tensor.to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def replay(self):
        """Train the model on a batch of experiences from memory."""
        if len(self.memory) < self.batch_size:
            return 0.0  # Return zero loss if not enough samples
        
        # Sample a batch of experiences
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors and move to device
        states = torch.stack([self.encode_state(s) for s in states]).to(self.device)
        next_states = torch.stack([self.encode_state(s) for s in next_states]).to(self.device)
        actions = torch.tensor(actions, device=self.device, dtype=torch.long)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float32)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Clear CUDA cache if using CUDA
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())