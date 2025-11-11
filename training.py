import torch
from transformers import AutoModel
import torch.nn as nn
import argparse

from src.flux.util import load_flow_model
from src.flux.model import Flux, FluxParams

base_model = load_flow_model("flux-dev-kontext", device="cuda")

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

model = Flux(params=params)

# 3. Load pretrained weights partially
pretrained_state_dict = base_model.state_dict()
model_state_dict = model.state_dict()

# 4. Track which parameters matched
loaded_keys = []
unloaded_keys = []

for name, param in model_state_dict.items():
    if name in pretrained_state_dict and pretrained_state_dict[name].shape == param.shape:
        model_state_dict[name] = pretrained_state_dict[name]
        loaded_keys.append(name)
    else:
        unloaded_keys.append(name)

model.load_state_dict(model_state_dict)

# 5. Freeze only the parameters that came from the pretrained model
for name, param in model.named_parameters():
    if name in loaded_keys:
        param.requires_grad = False
    else:
        param.requires_grad = True

# 6. Optional: print summary
print("Loaded pretrained layers:")
for k in loaded_keys:
    print("  ", k)

print("\nNew trainable layers:")
for k in unloaded_keys:
    print("  ", k)

# 7. Optimizer for only new parameters
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
)