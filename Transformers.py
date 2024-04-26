import numpy as np
from einops import rearrange

from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor

np.random.seed(0)
torch.manual_seed(0)


def get_positional_embeddings(sequence_length, d):
    result = torch.zeros(sequence_length, d)
    position = torch.arange(0, sequence_length).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d, 2).float() * -(np.log(10000.0) / d))

    result[:, 0::2] = torch.sin(position * div_term)
    result[:, 1::2] = torch.cos(position * div_term)

    return result


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, emb_size, device):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, emb_size, kernel_size=patch_size, stride=patch_size)
        self.class_token = nn.Parameter(torch.randn(1, 1, emb_size)).to(device)
        self.positional_embeddings = nn.Parameter(get_positional_embeddings(self.num_patches + 1, emb_size)).to(device)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        b = x.shape[0]
        class_tokens = self.class_token.expand(b, -1, -1)
        x = torch.cat((class_tokens, x), dim=1)
        x += self.positional_embeddings
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, device):
        super(MultiheadAttention, self).__init__()
        assert emb_size % num_heads == 0

        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads

        self.linear_q = nn.Linear(emb_size, emb_size)
        self.linear_k = nn.Linear(emb_size, emb_size)
        self.linear_v = nn.Linear(emb_size, emb_size)

        self.out_proj = nn.Linear(emb_size, emb_size)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        # Transform Q, K, V
        query = self.linear_q(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.linear_k(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention scores (Q * K^T) / sqrt(d_k)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        attn_probs = nn.functional.softmax(attn_scores, dim=-1)

        context = torch.matmul(attn_probs, value)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.emb_size)
        output = self.out_proj(context)

        return output


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size, num_heads, dropout_rate=0.1, device=torch.device('cpu')):
        super(TransformerEncoderBlock, self).__init__()

        self.emb_size = emb_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.norm1 = nn.LayerNorm(self.emb_size)
        self.multihead = MultiheadAttention(self.emb_size, self.num_heads, device)
        self.norm2 = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, emb_size * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(emb_size * 4, emb_size),
            nn.Dropout(dropout_rate)
        )

    def forward(self, embedded_patches):
        norm1 = self.norm1(embedded_patches)
        attention = self.multihead(norm1, norm1, norm1)

        # Residual connection 1
        res_1 = attention + embedded_patches

        norm2 = self.norm2(res_1)
        mlp_out = self.mlp(norm2)

        # Residual connection 1
        return mlp_out + res_1


class ViT(nn.Module):
    def __init__(self, img_size=48, patch_size=4, emb_size=768, num_heads=8, num_encoder=4, num_classes=10,
                 dropout_rate=0.1, device=torch.device('cpu')):
        super(ViT, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.num_encoder = num_encoder
        self.device = device

        self.patch_embedding = PatchEmbedding(img_size, patch_size, emb_size, device)

        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(emb_size, num_heads, dropout_rate, device) for _ in range(num_encoder)
        ])

        self.MLP_Head = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, emb_size),
            nn.Linear(emb_size, num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        for block in self.encoder_blocks:
            x = block(x)
        x = self.MLP_Head(x[:, 0])
        return x


if __name__ == '__main__':
    test = torch.randn(4, 3, 224, 224)
    patch_size = 16
    emb_size = 768
    img_size = 224

    test_embedding = PatchEmbedding(img_size=img_size, patch_size=patch_size, emb_size=emb_size)
    test_embedding(test)
