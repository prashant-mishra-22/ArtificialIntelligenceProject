import torch
from models.cnn_dqn import CNN_DQN

# Test the CNN model
state_shape = (15, 15, 3)  # height, width, channels
num_actions = 5

model = CNN_DQN(state_shape, num_actions)
print("CNN Model created successfully!")

# Test forward pass
dummy_input = torch.randn(1, 15, 15, 3)
output = model(dummy_input)
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output}")
print("Model forward pass works!")