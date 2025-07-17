class DuelingDQN(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()
        self.action_size = action_size
        
        # Shared layers
        self.fc1 = torch.nn.Linear(state_size, 128)
        
        # Advantage stream
        self.adv_fc1 = torch.nn.Linear(128, 128)
        self.adv_fc2 = torch.nn.Linear(128, action_size)
        
        # Value stream
        self.val_fc1 = torch.nn.Linear(128, 128)
        self.val_fc2 = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        
        # Advantage stream
        adv = torch.relu(self.adv_fc1(x))
        adv = self.adv_fc2(adv)
        
        # Value stream
        val = torch.relu(self.val_fc1(x))
        val = self.val_fc2(val)
        
        # Combine value and advantage
        return val + (adv - adv.mean(dim=1, keepdim=True))

    def calculate_loss(self, predicted, target):
        return torch.nn.functional.mse_loss(predicted, target)