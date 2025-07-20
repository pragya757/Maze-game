def preprocess_state(state):
    # Normalize or preprocess the state for input into the model
    return state / 255.0  # Example normalization

def store_experience(experience, memory):
    # Store experience in memory for replay
    memory.append(experience)

def update_target_network(model, target_model):
    # Update the target network weights
    target_model.load_state_dict(model.state_dict())