"""
Train Flux Kontext (LoRA + Strength Projector) on curated dataset in latent space.

This version uses:
  - Autoencoder to map RGB → latent patch tokens.
  - CLIP pooled text embeddings (1 token per prompt).
  - Flow-matching loss: || vθ(y_t) - (ε - x) ||²
  - Correct positional ID generation for image & text tokens.
"""

import os
import argparse
import json
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Import model utilities
from src.flux.model import FluxParams, FluxLoraWrapper
from src.flux.util import load_flow_model, load_ae, load_clip,load_t5
from src.flux.modules.layers import timestep_embedding


# ================================================================
# 1️⃣ Dataset: loads paired (x, y_s, s, prompt)
# ================================================================
class FluxLatentDataset(Dataset):
    """
    Dataset for curated edit dataset where:
      root/
        edit_type/
          0/
            0_s0.png, 0_s1.png, ... 0_s7.png
          1/
            1_s0.png, ...
        prompts.json : {"0": "replace sky with sunset", ...}
    """

    def __init__(self, root, json_dir, size=512):
        self.root = root
        self.size = size
        self.json_dir = json_dir
        self.samples = []

        # --- Load all experiment metadata into a dict keyed by exp_id ---
          # store full entry for later access

        # --- Now iterate through root folders (each folder = exp_id) ---
        for i,edit_type in enumerate(os.listdir(root)):
            et_path = os.path.join(root, edit_type)
            if not os.path.isdir(et_path):
                continue
            with open(os.path.join(self.json_dir, f"edit_{i+1}/data_loaded.json"), 'r') as f:
                self.exp_data = {}
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    exp_id = entry["exp_id"]
                    self.exp_data[exp_id] = entry

            for case in os.listdir(et_path):
                folder = os.path.join(et_path, case)
                if not os.path.isdir(folder):
                    continue

                files = sorted([f for f in os.listdir(folder) if f.endswith(".png")])

                if len(files) <= 1:
                    continue

                # Assume folder name (or edit_type) is exp_id
                try:
                    exp_id = int(case)  # or int(edit_type), depending on your structure
                except ValueError:
                    print(f"Skipping non-integer folder name: {case}")
                    continue

                # --- Fetch the corresponding prompts from JSON ---
                if exp_id not in self.exp_data:
                    print(f"⚠️ Warning: exp_id {exp_id} not found in JSON")
                    continue

                prompts = self.exp_data[exp_id]["edit_prompt"]

                src = os.path.join(folder, files[0])
                N = len(files) - 2

                for i, f in enumerate(files):
                    if i == 0:
                        continue
                    tgt = os.path.join(folder, f)
                    s_val = 0 if i == 1 else i / N
                    prompt = prompts[min(i-1, len(prompts)-1)]  # safe index
                    self.samples.append((src, tgt, s_val, prompt))

        self.preproc = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src, tgt, s_val, prompt = self.samples[idx]
        x = self.preproc(Image.open(src).convert("RGB"))
        y = self.preproc(Image.open(tgt).convert("RGB"))
        return x, y, torch.tensor(s_val, dtype=torch.float32), prompt


# ================================================================
# 2️⃣ AE + CLIP Preprocessor → latent tokens & text embeddings
# ================================================================
class FluxPreprocessor:
    def __init__(self, ae, clip, t5, device):
        self.ae = ae.eval().to(device)
        self.clip = clip.eval().to(device)
        self.t5 = t5.eval().to(device)
        self.device = device
        self.latent_proj = torch.nn.Linear(16, 64).to(device)

    @torch.no_grad()
    @torch.autocast("cuda", dtype=torch.bfloat16)
    def encode_images(self, imgs):
        lat = self.ae.encode(imgs.to(self.device))
        lat = lat.flatten(2).transpose(1, 2)
        lat = self.latent_proj(lat)
        return lat

    @torch.no_grad()
    def encode_texts(self, prompts):
        # CLIP pooled embedding (for strength projector)
        clip_emb = self.clip(list(prompts))  # [B, 768]

        # T5 contextual embeddings (for transformer conditioning)
        t5_emb = self.t5(list(prompts))      # [B, seq_len, 4096]
        txt_seq = t5_emb  
        B = txt_seq.shape[0]
           # assuming square latent grid
        txt_ids = self.make_txt_ids(B, txt_seq.shape[1], self.device)                  # rename for clarity
        
        return txt_seq, txt_ids, clip_emb
    
    def make_img_ids(self,H, W, B, device):
        y, x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )
        coords = torch.stack([x.flatten(), y.flatten(), torch.zeros_like(x.flatten())], dim=-1)
        coords = coords.unsqueeze(0).repeat(B, 1, 1)  # [B, H*W, 3]
        return coords

    def make_txt_ids(self,B, T, device):
        coords = torch.stack([
            torch.zeros(B, T, device=device),       # axis 1
            torch.arange(T, device=device)[None, :].repeat(B, 1),  # axis 2
            torch.zeros(B, T, device=device),       # axis 3
        ], dim=-1)
        return coords

    def prepare_batch(self, x_rgb, y_rgb, prompts):
        x_seq = self.encode_images(x_rgb)
        y_seq = self.encode_images(y_rgb)
        B, L, C = y_seq.shape
        H = W = int(y_seq.shape[1] ** 0.5)
        img_ids = self.make_img_ids(H, W, B, self.device)

        txt_seq, txt_ids, clip_emb = self.encode_texts(prompts)
        return x_seq, y_seq, img_ids, txt_seq, txt_ids, clip_emb



# ================================================================
# 3️⃣ Training Loop (flow matching loss)
# ================================================================
def train_flux_kontext(model, dataloader, preproc, device="cuda",
                       lr=2e-5, epochs=10, save_dir="checkpoints"):

    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for x_rgb, y_rgb, s, prompts in pbar:
            s = s.to(device)
            x_seq, y_seq, img_ids, txt_seq, txt_ids, pooled_txt = preproc.prepare_batch(x_rgb, y_rgb, prompts)

            # Sample random timestep t ∈ [0,1] and Gaussian noise ε
            eps = torch.randn_like(y_seq)
            print(eps.shape)
            t = torch.rand(y_seq.shape[0], 1, 1, device=y_seq.device)
            print(t.shape)
            print(y_seq.shape)
            y_t = (1 - t) * y_seq + t * eps

            # Drop slider conditioning with p=0.1 (regularization)
            if torch.rand(1) < 0.1:
                s = torch.zeros_like(s)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                print()
                print(f"input image shape : {y_t.shape}")
                print(f"text shape {txt_seq.shape}")
                print(f"conditioning image : {x_seq.shape}")
                print(f"img_ids.shape: {img_ids.shape}")
                print(f"txt_id shape : {txt_ids.shape}")
                v_pred =  model(
                    img=y_t,
                    img_ids=img_ids,
                    txt=txt_seq,           # 4096-dim T5 tokens
                    txt_ids=txt_ids,
                    pooled_txt = pooled_txt,
                    timesteps=t.squeeze(),
                    y=pooled_txt,
                    guidance=torch.ones_like(t.squeeze()),
                    strengths=s,
                )

                target = eps - x_seq
                loss = F.mse_loss(v_pred, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch+1}.pt"))
        print(f"✅ Saved checkpoint epoch {epoch+1} | Loss: {loss.item():.4f}")

    print("🎯 Training Complete")


# ================================================================
# 4️⃣ Main
# ================================================================
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("🔹 Loading Autoencoder and CLIP ...")
    ae = load_ae("flux-dev-kontext", device)
    clip = load_clip(device)
    t5 = load_t5(device)
    preproc = FluxPreprocessor(ae, clip,t5, device)

    print("🔹 Initializing Flux Kontext model ...")
    params = FluxParams(
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
    model, missing = load_flow_model(model)

    # Train only LoRA + Strength Projector
    for name, p in model.named_parameters():
        p.requires_grad = name in missing

    print(f"🔹 Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Dataset & loader
    dataset = FluxLatentDataset(args.data_root, args.json,size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Start training
    train_flux_kontext(model, dataloader, preproc, device, lr=args.lr,
                       epochs=args.epochs, save_dir=args.save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Flux Kontext in latent space (LoRA + projector)")
    parser.add_argument("--data_root", type=str, required=True, help="Path to curated dataset root")
    parser.add_argument("--json",type=str, required=True, help="Path to saved metadata root")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Checkpoint output dir")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--image_size", type=int, default=512)
    args = parser.parse_args()
    main(args)
