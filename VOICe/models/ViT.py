from typing import Any
import torch
from torch import nn
from utils.torch_utils import compute_conv_kernel_size
from models.attention.CBAM import CBAMBlock
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from config import hparams
hp = hparams()

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads,
                        dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0., output_1d_embeddings: bool = hp.output_1d_embeddings):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * \
            (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout)

        self.output_1d_embeddings = output_1d_embeddings
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        if self.output_1d_embeddings:
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
            x = self.to_latent(x)
            x = self.mlp_head(x)

        return x


class VOICeViT(nn.Module):
    """ConvNeXt Model with output linear layer
    """

    def __init__(self,
                 num_classes: int = hp.num_classes,
                 input_height: int = hp.input_height, input_width: int = hp.input_width,
                 use_cbam: bool = hp.use_cbam, cbam_channels: int = hp.cbam_channels, cbam_reduction_factor: int = hp.cbam_reduction_factor, cbam_kernel_size: int = hp.cbam_kernel_size, output_1d_embeddings: bool = hp.output_1d_embeddings,
                 *args: Any, **kwargs: Any) -> None:

        super(VOICeViT, self).__init__(*args, **kwargs)
        self.use_cbam = use_cbam
        
        self.num_classes = num_classes
        self.input_height = input_height
        self.input_width = input_width
        self.output_1d_embeddings = output_1d_embeddings
        self.increase_channels_to_3_and_reduce_height = nn.Conv2d(
            1, 3, (2, 1))  # -> (batch_size, 3, 256, 40)
        self.ViT = ViT(
            image_size=max(self.input_height-1, self.input_width),
            patch_size=hp.patch_size,
            num_classes=3*self.num_classes,
            dim=hp.dim,
            depth=hp.depth,
            heads=hp.heads,
            mlp_dim=hp.mlp_dim,
            dropout=hp.vit_dropout,
            emb_dropout=hp.vit_emb_dropout,
            output_1d_embeddings=output_1d_embeddings
        )

        with torch.no_grad():
            if not self.output_1d_embeddings:
                # pann will then output a 2d image/tensor: (batch_size, 1, height, width)
                random_inp = torch.rand(
                    (1, 1, self.input_height, self.input_width))
                output = self.increase_channels_to_3_and_reduce_height(random_inp)
                output = self.ViT(output)
                self.vit_output_height = output.size(1)
                self.vit_output_width = output.size(2)

        if self.use_cbam:
            self.cbam = CBAMBlock(
                channel=self.vit_output_height, reduction=cbam_reduction_factor, kernel_size=cbam_kernel_size)

        if not self.output_1d_embeddings:
            self.head = nn.Sequential(
                nn.Conv1d(self.vit_output_height, hp.num_subwindows, compute_conv_kernel_size(self.vit_output_width, self.vit_output_width//2)),
                nn.Conv1d(hp.num_subwindows, hp.num_subwindows, compute_conv_kernel_size(self.vit_output_width//2, 3*self.num_classes))
            )
        else:
            self.head = nn.Sequential(nn.Identity())

    def forward(self, input):
        x = input.float()  # -> (batch_size, 1, num_frames, n_mels)
        x = self.increase_channels_to_3_and_reduce_height(x)
        x = self.ViT(x)
        x = torch.unsqueeze(x, 2)
        if self.use_cbam:
            x = self.cbam(x)
        x = torch.squeeze(x, 2)
        x = self.head(x)
        return x
