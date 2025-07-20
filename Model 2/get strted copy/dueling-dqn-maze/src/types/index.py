# File: /dueling-dqn-maze/dueling-dqn-maze/src/types/index.py

ACTION_SPACE = [0, 1, 2, 3]  # Example actions: 0=up, 1=down, 2=left, 3=right
STATE_SIZE = 16  # Example state size for a 4x4 maze
REWARD_REACH_DESTINATION = 10  # Reward for reaching the destination
REWARD_HIT_WALL = -5  # Penalty for hitting a wall
REWARD_STEP = -1  # Small penalty for each step taken

# Type definitions for experiences and transitions
Experience = tuple  # (state, action, reward, next_state, done)
Transition = tuple  # (state, action, reward, next_state)