import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CNN_DQN, self).__init__()
        
        # Input shape: (height, width, channels) -> we need (channels, height, width)
        self.conv_layers = nn.Sequential(
            # Input: (batch_size, 3, height, width)
            nn.Conv2d(input_shape[2], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # Calculate size after conv layers
        conv_out_size = self._get_conv_output(input_shape)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
    
    def _get_conv_output(self, shape):
        with torch.no_grad():
            # Convert from (H,W,C) to (C,H,W) for PyTorch
            input = torch.zeros(1, shape[2], shape[0], shape[1])
            output = self.conv_layers(input)
            return output.view(1, -1).size(1)
    
    def forward(self, x):
        # x comes as (batch_size, height, width, channels)
        # Convert to (batch_size, channels, height, width) for PyTorch
        x = x.permute(0, 3, 1, 2)
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        return self.fc_layers(x)