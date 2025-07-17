# Dueling DQN Maze Game
This project implements a Dueling DQN algorithm to solve a maze game. It includes a maze environment, a Dueling DQN model, and training scripts.

## Setup
1. Clone the repository.
2. Install the required packages using `pip install -r requirements.txt`.
3. Run the training script with `python src/train_dueling_dqn.py`.

## Project Structure
```
dueling-dqn-maze
├── src
│   ├── train_dueling_dqn.py
│   ├── model.py
│   ├── environment.py
│   ├── utils.py
│   └── types
│       └── index.py
├── requirements.txt
└── README.md
```

## Dueling DQN Algorithm
Dueling DQN is an extension of the DQN algorithm that separates the representation of state values and advantages. This allows the model to learn which states are valuable and which actions are advantageous in those states, improving learning efficiency.

## Maze Environment
The maze environment is designed to simulate an agent navigating through a maze. The agent receives rewards based on its actions:
- Positive reward for reaching the destination.
- Negative reward for hitting walls or making invalid moves.
- Small negative reward for each step taken to encourage shorter paths.

## Usage
To train the Dueling DQN model, simply run the training script. The model will interact with the maze environment, learning to navigate through it over time.