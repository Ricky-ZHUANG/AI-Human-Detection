import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

class VisionTransformerEncoder(nn.Module):
    def __init__(self, embedding_dim=1500, image_size=600, patch_size=16, num_classes=1000, num_layers=12, num_heads=12, hidden_dim=768, dropout=0.1):
        super(VisionTransformerEncoder, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        
        self.patch_embedding = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads), num_layers=num_layers)
        
        self.fc = nn.Linear(hidden_dim, embedding_dim)
        
    def forward(self, x):
        x = self.patch_embedding(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        b, n, _ = x.shape
        pos_emb = self.positional_embedding[:, :(n+1)]
        x = x + pos_emb
        x = self.dropout(x)
        
        x = self.transformer_encoder(x)
        x = x[:, 0]  # Only keep the first token (CLS token) representation
        
        x = self.fc(x)
        
        return x