# import torch
# import torch.nn as nn

# # Define the basic block for ResNet
# class BasicBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channels)
        
#         # Shortcut connection for identity mapping
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
#                 nn.BatchNorm2d(out_channels)
#             )
#         else:
#             self.shortcut = nn.Identity()
        
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out += self.shortcut(x)
#         out = self.relu(out)
#         return out

# # Define the ResNet encoder
# class ResNetEncoder(nn.Module):
#     def __init__(self, block, num_blocks, num_channels=64):
#         super(ResNetEncoder, self).__init__()
#         self.in_channels = num_channels
#         self.conv = nn.Conv2d(3, num_channels, kernel_size=7, stride=2, padding=3)
#         self.bn = nn.BatchNorm2d(num_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
#         # Create the layers using a for loop
#         self.layers = []
#         for i, num_blocks in enumerate(num_blocks):
#             stride = 1 if i == 0 else 2
#             self.layers.append(self._make_layer(block, num_channels, num_blocks, stride))
#             num_channels *= 2
        
#         self.layers = nn.Sequential(*self.layers)
        
#     def _make_layer(self, block, num_channels, num_blocks, stride):
#         layers = []
#         layers.append(block(num_channels, num_channels, stride))
#         for _ in range(1, num_blocks):
#             layers.append(block(num_channels, num_channels))
#         return nn.Sequential(*layers)
        
#     def forward(self, x):
#         out = self.conv(x)
#         out = self.bn(out)
#         out = self.relu(out)
#         out = self.maxpool(out)
#         out = self.layers(out)
#         return out
# 
# # Create an instance of the ResNet encoder
# resnet_encoder = ResNetEncoder(BasicBlock, [2, 2, 2, 2])
# 
# # Test the ResNet encoder
# x = torch.randn(1, 3, 224, 224)  # Example input
# output = resnet_encoder(x)
# print(output.shape)  # Output shape: (1, 512, 7, 7)

import torch
import torch.nn as nn
import torchvision.models as models

class ResNetEncoder(nn.Module):
    def __init__(self, embedding_dim=1500):
        super(ResNetEncoder, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embedding_dim)

    def forward(self, x):
        x = self.resnet(x)
        return x

