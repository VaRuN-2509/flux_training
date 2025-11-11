import torch
from transformers import AutoModel
import torch.nn as nn
import argparse

from src.flux.util import load_flow_model
from src.flux.model import Flux, FluxParams, FluxLoraWrapper


device = "cuda" if torch.cuda.is_available() else "cpu"
offload = False

torch_device = torch.device(device)

# base_model = load_flow_model("flux-dev-kontext", device="cpu" if offload else torch_device)
# pretrained_state_dict = model.state_dict()

params=FluxParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        )

model = FluxLoraWrapper(lora_rank=4, lora_scale=1.0, params=params)
model,missing = load_flow_model(model)
model_state_dict = model.state_dict()

for name, param in model_state_dict.items():
    print(name)


for name, param in model.named_parameters():
    if name not in missing:
        param.requires_grad = False
    else:
        param.requires_grad = True

for name, param in model.named_parameters():
    print(f'{name} : {param.requires_grad}')






