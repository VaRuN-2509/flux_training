# import torch
# from einops import rearrange
# from torch import Tensor


# def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
#     q, k = apply_rope(q, k, pe)

#     x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
#     x = rearrange(x, "B H L D -> B L (H D)")

#     return x


# def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
#     assert dim % 2 == 0
#     scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
#     omega = 1.0 / (theta**scale)
#     out = torch.einsum("...n,d->...nd", pos, omega)
#     out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
#     out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
#     return out.to(dtype=torch.float)


# def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
#     xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
#     xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
#     xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
#     xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
#     return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

import torch
from einops import rearrange
from torch import Tensor


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    # Apply RoPE in float32
    q, k = apply_rope(q, k, pe)

    # Save original dtype (bf16 or fp16)
    orig_dtype = q.dtype

    # Cast q, k, v to float32 for safe attention
    q = q.float()
    k = k.float()
    v = v.float()

    # Run SDPA in float32 to avoid overflow
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    # Cast result back to bf16
    x = x.to(orig_dtype)

    # Rearrange to final shape
    x = rearrange(x, "B H L D -> B L (H D)")
    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim
    omega = 1.0 / (theta**scale)

    out = torch.einsum("...n,d->...nd", pos.float(), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)

    # Return BF16-safe rope
    return out.to(torch.float32)   # keep rope in fp32


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    # Compute in float32
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)

    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]

    # Return in bf16
    return xq_out.reshape(*xq.shape).to(torch.bfloat16), \
           xk_out.reshape(*xk.shape).to(torch.bfloat16)
