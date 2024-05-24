import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the Transformer Encoder class
class TransformerEncoder(nn.Module):
    def __init__(self, n_features, n_frames, hidden_size, num_heads, num_layers, batch_size):
        super(TransformerEncoder, self).__init__()
        self.linear = nn.Linear(n_features, hidden_size)
        self.positional_encoding = nn.Parameter(torch.randn(batch_size, n_frames, hidden_size))
        self.dropout = nn.Dropout(p=0.1)
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.encoder_layers = nn.ModuleList([self.make_encoder_layer(hidden_size, num_heads) for _ in range(num_layers)])

    def make_encoder_layer(self, hidden_size, num_heads):
        return nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.linear(x)
        x += self.positional_encoding
        x = self.dropout(x)
        x = x.permute(1, 0, 2)  # Change the dimensions for Multihead Attention
        x = self.multihead_attn(x, x, x)[0]
        x = self.feed_forward(x)
        for layer in self.encoder_layers:
            x = layer(x)
        x = x.permute(1, 0, 2)
        return x

# Define the Residual Block class
class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.batch_norm(self.conv1(x)))
        out = self.batch_norm(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

# Complete TE-ResNet Model
class TE_ResNet(nn.Module):
    def __init__(self, n_features, n_frames, num_layers, heads, hidden_size, output_channels, batch_size, device):
        super(TE_ResNet, self).__init__()
        self.transformer_encoder = TransformerEncoder(n_features=n_features, n_frames=n_frames, hidden_size=hidden_size, num_heads=heads, num_layers=num_layers, batch_size=batch_size)
        self.resnet1 = ResidualBlock(input_channels=1, output_channels=64) 
        # self.resnet2 = ResidualBlock(input_channels=64, output_channels=128) 
        # self.resnet3 = ResidualBlock(input_channels=128, output_channels=256)
        # self.resnet4 = ResidualBlock(input_channels=256, output_channels=512)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64, 512)
        self.fc2 = nn.Linear(512, 2)  # 2 classes (fake and real)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = x.unsqueeze(1) 
        x = self.resnet1(x)
        # x = self.resnet2(x)
        # x = self.resnet3(x)
        # x = self.resnet4(x)
        x = self.adaptive_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x

