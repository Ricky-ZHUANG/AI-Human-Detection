import torch
import torch.nn as nn
import torchvision.models as models

class VGGEncoder(nn.Module):
    def __init__(self, embedding_dim=1500):
        super(VGGEncoder, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        self.vgg.classifier = nn.Sequential(
            # nn.Linear(512*324, 4096),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, embedding_dim)
        )

    def forward(self, x):
        x = self.vgg.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.vgg.classifier(x)
        return x
