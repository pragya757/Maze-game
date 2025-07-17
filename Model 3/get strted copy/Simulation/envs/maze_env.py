import numpy as np

class MazeEnv:
    def __init__(self, maze, start, goal, max_steps=200):
        """
        Initialize the Maze Environment.
        
        Args:
            maze: 2D numpy array representing the maze (0 = free space, 1 = wall)
            start: Tuple of (x, y) coordinates for the starting position
            goal: Tuple of (x, y) coordinates for the goal position
            max_steps: Maximum number of steps per episode
        """
        self.maze = maze
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.agent_pos = self.start
        self.prev_pos = self.start  # Track previous position for direction reward
        self.visited_states = set([self.start])  # Track visited states for exploration bonus
        self.maze_size = maze.shape
        self.state = self._get_state()
        self.steps_taken = 0
        self.max_steps = max_steps
        self.last_distance_to_goal = abs(self.goal[0] - self.start[0]) + abs(self.goal[1] - self.start[1])
        
    def reset(self):
        """Reset the environment to the starting state."""
        self.agent_pos = self.start
        self.visited_states = set([self.start])
        self.state = self._get_state()
        self.steps_taken = 0  # Reset step counter
        return self.state

    def _get_state(self):
        """
        Get the current state representation.
        
        Returns:
            3D numpy array with shape (2, height, width) where:
            - Channel 0: Binary mask with 1 at agent's position
            - Channel 1: Maze layout (0 = free, 1 = wall)
        """
        state = np.zeros((2, self.maze.shape[0], self.maze.shape[1]))
        state[0, self.agent_pos[0], self.agent_pos[1]] = 1  # Agent position
        state[1, :, :] = self.maze  # Maze layout
        return state

    def _is_valid_position(self, x, y):
        """Check if a position is valid (not a wall and within bounds)."""
        return (0 <= x < self.maze.shape[0] and 
                0 <= y < self.maze.shape[1] and 
                self.maze[x][y] == 0)

    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Integer representing the action (0=up, 1=right, 2=down, 3=left)
            
        Returns:
            tuple: (next_state, reward, done)
        """
        x, y = self.agent_pos
        
        # Calculate next position based on action
        if action == 0:  # up
            next_x, next_y = x - 1, y
        elif action == 1:  # right
            next_x, next_y = x, y + 1
        elif action == 2:  # down
            next_x, next_y = x + 1, y
        elif action == 3:  # left
            next_x, next_y = x, y - 1
        
        # Increment step counter
        self.steps_taken += 1
        
        # Check for wall collision or out of bounds
        if (next_x < 0 or next_x >= self.maze.shape[0] or 
            next_y < 0 or next_y >= self.maze.shape[1] or
            self.maze[next_x][next_y] == 1):
            # Wall hit or out of bounds - reduced penalty to encourage exploration
            reward = -0.75
            done = True
            termination_reason = "wall_collision"
        else:
            # Valid move
            self.prev_pos = self.agent_pos
            self.agent_pos = (next_x, next_y)
            self.visited_states.add(self.agent_pos)
            
            # Calculate reward
            reward = self._calculate_reward(x, y, next_x, next_y, action)
            
            # Check if goal is reached
            if self.agent_pos == self.goal:
                done = True
                termination_reason = "goal_reached"
            # Check if max steps reached
            elif self.steps_taken >= self.max_steps:
                done = True
                termination_reason = "max_steps_reached"
            else:
                done = False
                termination_reason = ""
        
        return self._get_state(), reward, done, termination_reason

    def _calculate_reward(self, x, y, next_x, next_y, action):
        """
        Calculate the reward for moving from (x,y) to (next_x, next_y).
        
        Returns:
            float: The calculated reward
        """
        # Initialize reward and flags
        reward = 0.0
        
        # Check if position is new (exploration bonus)
        new_position = (next_x, next_y)
        if new_position not in self.visited_states:
            reward += 0.05  # Small bonus for exploring new states
            self.visited_states.add(new_position)
        else:
            reward -= 0.07  # Small penalty for revisiting states
        
        # Calculate distance to goal
        current_dist = abs(self.goal[0] - next_x) + abs(self.goal[1] - next_y)
        prev_dist = abs(self.goal[0] - x) + abs(self.goal[1] - y)
        
        # Distance-based rewards
        if current_dist < prev_dist:
            reward += 0.07  # Reward for moving closer to the goal
        else:
            reward -= 0.10  # Penalty for moving away from the goal
        
        # Update last distance
        self.last_distance_to_goal = current_dist
        
        # Small step reward to encourage progress
        reward += 0.05
        
        # Check if goal is reached
        if (next_x, next_y) == self.goal:
            print(f'\nINFO: HURRAY!! Agent has reached its destination...')
            reward = 5.0  # Large reward for reaching the goal
        
        return reward
