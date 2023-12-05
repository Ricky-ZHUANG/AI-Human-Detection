'''
input dimension
600*600*3 for the one original image
'''
import torch
import torch.nn as nn

   
class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.fc = nn.Linear(4096, 1500)  # Fully connected layer to get a vector of size 1500x1
        self.dropout=0.2
   

        channels = [3, 16, 32, 64, 128, 256, 512, 1024,1024]

        for input_dim, output_dim in zip(channels, channels[1:]):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1), 
                    # nn.BatchNorm2d(output_dim), 
                    # nn.MaxPool2d(kernel_size=2, stride=2),
                    # nn.ReLU(),
                    )
            )

        for layer in self.conv_layers:
            nn.init.xavier_uniform_(layer[0].weight)
 
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = self.relu(self.mp(conv_layer(x)))
        
        x = self.flatten(x)
        # x = x.view(-1)
        x = self.fc(x)
        
        return x
    
