import torch
import torch.nn as nn
import torch.nn.functional as F

# Multi-Head Attention Component
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, embed_size, bias=False)
        self.keys = nn.Linear(self.head_dim, embed_size, bias=False)
        self.queries = nn.Linear(self.head_dim, embed_size, bias=False)
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.embed_size
        )

        out = self.fc_out(out)
        return out

# Feed-Forward Component
            mask = mask.unsqueeze(1).unsqueeze(2)
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
        out = out.reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_size, dropout):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden_size)
        self.fc2 = nn.Linear(ff_hidden_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(x)
        return x

# Transformer Encoder Layer
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_size, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, ff_hidden_size, dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.norm1(attention + query)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return out

# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self,
                 embed_size,
                 num_layers,
                 heads,
                 ff_hidden_size,
                 dropout,
                 device,
                 max_length):
class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, num_layers, heads, ff_hidden_size, dropout, device, max_length):
        super(TransformerEncoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, ff_hidden_size, dropout)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length, feature_dim = x.shape  # Adjusted to unpack three dimensions
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(x + self.position_embedding(positions))
        self.layers = nn.ModuleList([TransformerBlock(embed_size, heads, ff_hidden_size, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length, feature_dim = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        assert feature_dim == self.embed_size, f"Feature dimension {feature_dim} does not match embed size {self.embed_size}"
        x = x + self.position_embedding(positions)
        out = self.dropout(x)

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.identity = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        self.identity = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.identity = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.identity(x)
        out = self.relu(out)
        return out

# ResNet Component
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def _make_layer(self, block, out_channels, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x

# Complete TE-ResNet Model
class TE_ResNet(nn.Module):
    def __init__(self,
                 embed_size,
                 num_layers,
                 heads,
                 ff_hidden_size,
                 dropout,
                 device,
                 max_length):
        super(TE_ResNet, self).__init__()
class TE_ResNet(nn.Module):
    def __init__(self, embed_size, num_layers, heads, ff_hidden_size, dropout, device, max_length):
        super(TE_ResNet, self).__init__()
        self.linear_proj = nn.Linear(188, embed_size)  # Projecting from 188 to embed_size
        self.transformer_encoder = TransformerEncoder(embed_size, num_layers, heads, ff_hidden_size, dropout, device, max_length)
        self.resnet = ResNet(ResidualBlock, [3, 4, 6, 3])

    def forward(self, x, mask):
        x = self.transformer_encoder(x, mask)
        x = x.unsqueeze(1)  # Adding channel dimension for ResNet
        x = self.resnet(x)
        return x
        # Apply linear projection to the last dimension (feature_dim)
        x = self.linear_proj(x)  # Transform [batch_size, seq_length, 188] to [batch_size, seq_length, embed_size]
        x = self.transformer_encoder(x, mask)
        x = x.unsqueeze(1)  # Adding channel dimension for ResNet
        x = self.resnet(x)
        return x

