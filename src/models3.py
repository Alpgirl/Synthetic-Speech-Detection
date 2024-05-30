import torch
import torch.nn as nn
import torchaudio.transforms as T
from torchvision import models

# Define Transformer Encoder based on provided scheme
class TransformerEncoder(nn.Module):
    def __init__(self, n_features, n_frames, hidden_size, num_heads, num_layers, batch_size):
        super(TransformerEncoder, self).__init__()
        self.linear = nn.Linear(n_features, hidden_size)
        self.positional_encoding = nn.Parameter(torch.randn(batch_size, n_frames, hidden_size))
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p=0.1)
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.encoder_layers = nn.ModuleList([self.make_encoder_layer(hidden_size, num_heads) for _ in range(num_layers)])

    def make_encoder_layer(self, hidden_size, num_heads):
        return nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, n_features, n_frames) -> (batch_size, n_frames, n_features)
        x = self.layer_norm1(self.linear(x)) + self.positional_encoding
        x = self.dropout(x)
        x = x.permute(1, 0, 2)  # (batch_size, n_frames, hidden_size) -> (n_frames, batch_size, hidden_size)
        x, _ = self.multihead_attn(x, x, x)
        x = self.feed_forward(x)
        for layer in self.encoder_layers:
            x = layer(x)
        x = x.permute(1, 0, 2)  # (n_frames, batch_size, hidden_size) -> (batch_size, n_frames, hidden_size)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

        # Downsample layer to match dimensions if necessary
        self.downsample = None
        if stride != 1 or input_channels != output_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.batch_norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



# Define TE_ResNet based on provided scheme
class TE_ResNet(nn.Module):
    def __init__(self, n_features, n_frames, num_layers, heads, hidden_size, output_channels, batch_size, device):
        super(TE_ResNet, self).__init__()
        self.transformer_encoder = TransformerEncoder(n_features=n_features, n_frames=n_frames, hidden_size=hidden_size, num_heads=heads, num_layers=num_layers, batch_size=batch_size)
        self.initial_conv = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.residual_blocks = nn.Sequential(
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 512, stride=2)
        )

        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.transformer_encoder(x)  # Expecting shape (batch_size, n_frames, hidden_size)
        x = x.unsqueeze(1)  # Adding a channel dimension: (batch_size, 1, n_frames, hidden_size)
        x = self.relu(self.batch_norm(self.initial_conv(x)))  # Initial Conv, BatchNorm, ReLU
        x = self.residual_blocks(x)  # Residual Blocks
        x = self.adaptive_avg_pool(x)  # Adaptive Average Pooling
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x
