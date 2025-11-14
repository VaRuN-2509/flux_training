import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn

from flux.math import attention, rope


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x
    
class StrengthProjector(nn.Module):
    """
    Maps scalar strength s and pooled text embedding to modulation offsets.
    Implements the 4-layer MLP described in Appendix A.3 of the paper.

    Implementation notes:
    - We encode s with sinusoidal positional encoding to 128 dims, then
      linear -> 768, concat with pooled_text (768) => 1536 input.
    - MLP dims: 1536 -> 256 -> 128 -> D_out.
    - For simplicity and stability we output a per-token *shared* offset of
      shape (B, 1, 2*hidden), which is then broadcast to text token length.
      This matches the intended effect of the projector while keeping
      implementation robust across different text lengths.
    - If you need per-token unique offsets, you can grow the final layer
      to (2 * hidden * text_len) at init time (requires knowing text_len).
    """
    def __init__(self, hidden_size: int, proj_in_dim: int = 1536, posenc_dim: int = 128, mlp_dims=(256, 128)):
        super().__init__()
        self.hidden = hidden_size
        self.posenc_dim = posenc_dim

        # encode scalar s -> pos enc -> linear -> 768
        self.s_posenc_linear = nn.Linear(posenc_dim, 768)

        # projector MLP: 1536 -> 256 -> 128 -> 2*hidden
        h1, h2 = mlp_dims
        self.fc1 = nn.Linear(proj_in_dim, h1)
        self.act1 = nn.SiLU()
        self.fc2 = nn.Linear(h1, h2)
        self.act2 = nn.SiLU()
        # final outputs two vectors: delta_scale and delta_shift
        self.fc_out = nn.Linear(h2, 2 * hidden_size)

        self.init_weights()

    @staticmethod
    def sinusoidal_posenc(x: Tensor, dim: int = 128):
        # x: (B, 1) or (B,)
        # produce (B, dim)
        device = x.device
        x = x.float().view(-1, 1)
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / half)
        args = x * freqs[None, :]
        posenc = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2:
            posenc = torch.cat([posenc, torch.zeros(posenc.shape[0], 1, device=device)], dim=-1)
        return posenc  # (B, dim)

    def forward(self, s: Tensor, pooled_text: Tensor, txt_len: int | None = None) -> tuple[Tensor, Tensor]:
        """
        s: (B,) scalar value in [0,1]
        pooled_text: (B, hidden) (e.g. txt.mean(dim=1) or CLIP pooled embedding)
        returns: delta_shift, delta_scale each shaped (B, 1, hidden) to be broadcast over tokens
        """
        B = pooled_text.shape[0]
        posenc = self.sinusoidal_posenc(s, self.posenc_dim)            # (B, posenc_dim)
        s_emb = self.s_posenc_linear(posenc)                          # (B, 768)
        cat = torch.cat([s_emb, pooled_text], dim=-1)                 # (B, 1536)
        h = self.act1(self.fc1(cat))
        h = self.act2(self.fc2(h))
        out = self.fc_out(h)                                          # (B, 2*hidden)
        delta_scale, delta_shift = out.chunk(2, dim=-1)               # each (B, hidden)

        # reshape to (B, 1, hidden) to broadcast across token dim later
        delta_scale = delta_scale[:, None, :]
        delta_shift = delta_shift[:, None, :]

        return delta_shift, delta_scale
    
    

    def init_weights(self):

        nn.init.kaiming_uniform_(self.fc1.weight,nonlinearity="leaky_relu")
        nn.init.kaiming_uniform_(self.fc2.weight,nonlinearity="leaky_relu")
        nn.init.xavier_uniform_(self.fc_out.weight,gain=1e-2)
        

        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc_out.bias)



@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def debug(self,name, tensor):
        if not isinstance(tensor, torch.Tensor):
            print(f"[{name}] not a tensor")
            return
        t = tensor
        print(f"[{name}] min={t.min().item():.5f}, max={t.max().item():.5f}, "
            f"mean={t.mean().item():.5f}, any_nan={torch.isnan(t).any().item()}, "
            f"dtype={t.dtype}, shape={tuple(t.shape)}")

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor,txt_mod_deltas:tuple[Tensor,Tensor]) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        self.debug("txt_mod1",txt_mod1.shift)
        self.debug("txt_mod2",txt_mod2.shift)
        self.debug("txt_mod1",txt_mod1.scale)
        self.debug("txt_mod2",txt_mod2.scale)


        if txt_mod_deltas is not None:
            delta_shift, delta_scale = txt_mod_deltas  # (B, 1, hidden)
            # Add deltas to both modulation levels (text tokens). Broadcast along token axis
            txt_mod1.shift = txt_mod1.shift + delta_shift
            txt_mod1.scale = txt_mod1.scale + delta_scale
            txt_mod2.shift = txt_mod2.shift + delta_shift
            txt_mod2.scale = txt_mod2.scale + delta_scale



        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img blocks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # calculate the txt blocks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        return img, txt


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
