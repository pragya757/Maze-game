# Reinforced Pathfinding in Maze using Dueling DQN and Neural Network Analysis

This project presents a Dueling Deep Q-Network (DQN) based approach for solving 15x15 maze pathfinding problems. Three distinct neural architectures (convolutional and dense layers) are evaluated to understand their impact on learning efficiency and performance.

# 📂 Project Structure
/environment/ – Maze environment code (15x15 grid)

/models/ – Dueling DQN agent implementations:

2× Convolutional Layers

2× Dense Layers

3× Dense Layers

/training/ – Training loop and logging system

/results/ – Output files:

output.txt

Training logs in .csv

Saved model weights

/plots/ – Performance graphs and analysis plots (.svg and .pdf formats)

# 🚀 Features
Dueling DQN Agent with shared representation layers.

Training over 5000 episodes for comparative evaluation.

Early termination conditions: wall hit, goal reached, max step count.

Performance logged and visualized across 6 key metrics.

Saved models and best weights.

Supports result reproduction via exported logs.

# 📊 Performance Metrics
Cumulative Reward

Temporal Difference (TD) Error

Episode Steps

Epsilon Decay

Average Return

Training Loss

Graphs saved as high-quality vector art suitable for publications.

# ⚙️ Usage
python train_agent.py

Modify config.py to select the model architecture and hyperparameters.
