class MazeEnv:
    def __init__(self, maze_size, destination):
        self.maze_size = maze_size
        self.destination = destination
        self.agent_position = (0, 0)  # Start at the top-left corner
        self.done = False
        self.steps = 0

    def reset(self):
        self.agent_position = (0, 0)
        self.done = False
        self.steps = 0
        return self.encode_state(self.agent_position)

    def step(self, action):
        if self.done:
            raise Exception("Environment is done. Please reset.")

        # Define possible actions: 0=up, 1=down, 2=left, 3=right
        if action == 0 and self.agent_position[0] > 0:  # Up
            self.agent_position = (self.agent_position[0] - 1, self.agent_position[1])
        elif action == 1 and self.agent_position[0] < self.maze_size[0] - 1:  # Down
            self.agent_position = (self.agent_position[0] + 1, self.agent_position[1])
        elif action == 2 and self.agent_position[1] > 0:  # Left
            self.agent_position = (self.agent_position[0], self.agent_position[1] - 1)
        elif action == 3 and self.agent_position[1] < self.maze_size[1] - 1:  # Right
            self.agent_position = (self.agent_position[0], self.agent_position[1] + 1)

        self.steps += 1
        reward = self._calculate_reward()
        state = self.encode_state(self.agent_position)

        return state, reward, self.done

    def _calculate_reward(self):
        if self.agent_position == self.destination:
            self.done = True
            return 10  # Positive reward for reaching the destination
        elif self.steps > 100:  # Arbitrary step limit
            self.done = True
            return -10  # Negative reward for exceeding step limit
        else:
            return -1  # Small negative reward for each step taken

    def encode_state(self, state):
        one_hot_state = np.zeros(self.maze_size)
        one_hot_state[state] = 1
        return one_hot_state.flatten()  # Return flattened one-hot encoded state

    def _get_state(self):
        return self.agent_position  # Return the current position of the agent