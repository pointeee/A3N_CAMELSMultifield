import torch
import torch.nn as nn
import torch.optim as optim

# Define the CNN model
class CNN_cosmo(nn.Module):
    def __init__(self, mode="cosmo"):
        super(CNN_cosmo, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv1_norm = nn.LayerNorm([16, 64, 64])
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2_norm = nn.LayerNorm([32, 32, 32])

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3_norm = nn.LayerNorm([64, 16, 16])
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.fc1 = nn.Linear(64 * 8 * 8, 1024)  # Assuming input images are 256x256; two dataset
        if mode=="cosmo":
            self.fc2 = nn.Linear(1024, 2)
        elif mode=="all":
            self.fc2 == nn.Linear(1024, 6)
        else:
            raise ValueError
        
        self.relu = nn.ReLU()

    def forward(self, x): 
        x = self.relu(self.conv1(x)) 
        x = self.conv1_norm(x) 
        x = self.pool(x) 
        x = self.relu(self.conv2(x)) 
        x = self.conv2_norm(x) 
        x = self.pool(x) 
        x = self.relu(self.conv3(x)) 
        x = self.conv3_norm(x) 
        x = self.pool(x) # [batch_size, channels=64, 8, 8]
        
        x = x.view(-1, 64 * 8 * 8) 
        
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

