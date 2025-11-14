from dataclasses import dataclass

import torch
from torch import Tensor, nn
import os

from flux.modules.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)
from flux.modules.lora import LinearLora, replace_linear_with_lora
from flux.modules.layers import StrengthProjector


def debug(name, tensor):
    if not isinstance(tensor, torch.Tensor):
        print(f"[{name}] not a tensor")
        return
    t = tensor
    print(f"[{name}] min={t.min().item():.5f}, max={t.max().item():.5f}, "
          f"mean={t.mean().item():.5f}, any_nan={torch.isnan(t).any().item()}, "
          f"dtype={t.dtype}, shape={tuple(t.shape)}")

@dataclass
class FluxParams:
    in_channels: int
    out_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.img_norm = nn.LayerNorm(self.hidden_size)
        self.txt_norm = nn.LayerNorm(self.hidden_size)
        self.vec_norm = nn.LayerNorm(self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)
        self.strength_projector = StrengthProjector(hidden_size=self.hidden_size)


        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        pooled_txt: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
        strengths: Tensor | None = None
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        print("==== Forward pass begin ====")

        debug("input img", img)
        debug("input txt", txt)
        debug("input pooled_txt", pooled_txt)
        debug("input y (clip)", y)

        # embeddings
        img = self.img_in(img)
        img = self.img_norm(img)
        debug("after img_in", img)

        vec = self.time_in(timestep_embedding(timesteps, 256))
        debug("after time_in", vec)

        if self.params.guidance_embed:
            g = self.guidance_in(timestep_embedding(guidance, 256))
            debug("after guidance_in", g)
            vec = vec + g
            debug("after vec + guidance", vec)

        v2 = self.vector_in(y)
        debug("after vector_in", v2)
        vec = vec + v2
        vec = self.vec_norm(vec)

        debug("after vec + vector_in", vec)

        txt = self.txt_in(txt)
        txt = self.txt_norm(txt)
        debug("after txt_in", txt)

        # positional encodings
        ids = torch.cat((txt_ids, img_ids), dim=1)
        debug("ids",ids)
        pe = self.pe_embedder(ids)
        debug("after pe_embedder", pe)

        # strength
        if strengths is None:
            strengths = timesteps.new_ones(txt.shape[0])

        delta_shift, delta_scale = self.strength_projector(strengths, pooled_txt)
        debug("delta_shift", delta_shift)
        debug("delta_scale", delta_scale)

        # double blocks
        for i, block in enumerate(self.double_blocks):
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe,
                            txt_mod_deltas=(delta_shift, delta_scale))
            debug(f"after DoubleBlock[{i}] img", img)
            debug(f"after DoubleBlock[{i}] txt", txt)

        # concat txt back
        img = torch.cat((txt, img), 1)
        debug("after concat txt,img", img)

        # single blocks
        for i, block in enumerate(self.single_blocks):
            img = block(img, vec=vec, pe=pe)
            debug(f"after SingleBlock[{i}]", img)

        # remove text tokens
        img = img[:, txt.shape[1]:, :]
        debug("after removing txt tokens", img)

        # final prediction
        img = self.final_layer(img, vec)
        debug("after final_layer", img)

        print("==== Forward pass end ====")

        return img



class FluxLoraWrapper(Flux):
    def __init__(
        self,
        lora_rank: int = 128,
        lora_scale: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.lora_rank = lora_rank

        replace_linear_with_lora(
            self,
            max_rank=lora_rank,
            scale=lora_scale,
        )

    def set_lora_scale(self, scale: float) -> None:
        for module in self.modules():
            if isinstance(module, LinearLora):
                module.set_scale(scale=scale)

if __name__ == 'main':
    model = Flux