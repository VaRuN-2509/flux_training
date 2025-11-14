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
from src.flux.model import FluxParams, FluxLoraWrapper,Flux
from src.flux.util import load_flow_model, load_ae, load_clip,load_t5
from src.flux.modules.layers import timestep_embedding


class FluxLatentDataset(Dataset):
    """
    Dataset for curated edit dataset.

    Expected structure:
      images_dir/
        1/
          1_s0.png, 1_s1.png, ..., 1_s7.png
      prompt/
        edit_1/
          data_loaded.json : {"exp_id": 1, "edit_prompt": "..."}
    """

    def __init__(self, root, json_dir, size=512):
        self.root = root
        self.size = size
        self.json_dir = json_dir
        self.samples = []

        # --- Load JSON metadata ---
        json_path = os.path.join(json_dir, "edit_1", "data_loaded.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Missing {json_path}")

        with open(json_path, "r") as f:
            content = f.read().strip()
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    data = [data]
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                f.seek(0)
                data = [json.loads(line) for line in f if line.strip()]

        self.exp_data = {d["exp_id"]: d for d in data}
        print(f"📄 Loaded {len(self.exp_data)} experiment entries from {json_path}")

        # --- Collect samples ---
        for case in sorted(os.listdir(root)):
            folder = os.path.join(root, case)
            if not os.path.isdir(folder):
                continue

            files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".png")])
            if len(files) <= 1:
                print(f"⚠️ Skipping {folder} (not enough images)")
                continue

            try:
                exp_id = int(case)
            except ValueError:
                print(f"⚠️ Skipping invalid folder name: {case}")
                continue

            if exp_id not in self.exp_data:
                print(f"⚠️ exp_id {exp_id} not found in JSON. Using first available entry.")
                prompt = next(iter(self.exp_data.values()))["edit_prompt"]
            else:
                prompt = self.exp_data[exp_id]["edit_prompt"]

            src = os.path.join(folder, files[0])
            N = len(files) - 2 if len(files) > 2 else 1

            for i, f in enumerate(files[1:], 1):
                tgt = os.path.join(folder, f)
                s_val = 0 if i == 1 else i / N
                self.samples.append((src, tgt, s_val, prompt))

        self.preproc = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        print(f"✅ Loaded {len(self.samples)} samples from {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        out_dir = "debug_"
        src, tgt, s_val, prompt = self.samples[idx]
        float_s = float(s_val)
        src_img = Image.open(src).convert("RGB")
        tgt_img = Image.open(tgt).convert("RGB")

        os.makedirs(out_dir, exist_ok=True)

        src_img.save(f"{out_dir}/src_{idx}.png")
        tgt_img.save(f"{out_dir}/tgt_{idx}_s{float_s:.2f}.png")

        print(f"Saved debug samples for idx {idx}")
        x = self.preproc(Image.open(src).convert("RGB"))
        y = self.preproc(Image.open(tgt).convert("RGB"))
        return x, y, torch.tensor(s_val, dtype=torch.bfloat16), prompt


# ================================================================
# 2️⃣ AE + CLIP Preprocessor → latent tokens & text embeddings
# ================================================================
class FluxPreprocessor:
    def __init__(self, ae, clip, t5, device):
        self.device = device
        self.ae = ae.to(device)
        self.clip = clip.to(device)
        self.t5 = t5.to(device)
        self.latent_proj = torch.nn.Linear(16, 64).to(device)
        nn.init.xavier_normal_(self.latent_proj.weight, gain=0.1)
        nn.init.constant_(self.latent_proj.bias, 0)

    @torch.no_grad()
    def encode_images(self, imgs):
        imgs = imgs.to(self.device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lat = self.ae.encode(imgs)
            lat = lat.flatten(2).transpose(1, 2)
            lat = self.latent_proj(lat)
        # keep results on GPU (no .cpu())
        return lat


    @torch.no_grad()
    def encode_texts(self, prompts):
        clip_emb = self.clip(list(prompts))

        t5_emb = self.t5(list(prompts))

        txt_seq = t5_emb
        B = txt_seq.shape[0]
        txt_ids = self.make_txt_ids(B, txt_seq.shape[1], self.device)
        
        torch.cuda.empty_cache()
        return txt_seq.cpu(), txt_ids.cpu(), clip_emb.cpu()
    
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

# MEASURE USAGE
def print_gpu(prefix=""):
    """Print current GPU memory usage and peak since last reset."""
    if not torch.cuda.is_available():
        print("[CPU only]")
        return
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserv = torch.cuda.memory_reserved() / 1024**2
    print(f"[GPU] {prefix:<25} | Alloc: {alloc:8.2f} MB | Reserved: {reserv:8.2f} MB")

def report_tensor_size(name, t):
    """Print tensor name, shape, dtype, and memory size in MB."""
    if t is None:
        return
    if not isinstance(t, torch.Tensor):
        print(f"{name}: not a tensor")
        return
    numel = t.numel()
    mem_mb = numel * t.element_size() / 1024**2
    print(f"{name:<20}: shape={tuple(t.shape)}, dtype={t.dtype}, size={mem_mb:7.2f} MB")


# ================================================================
#  Training Loop (flow matching loss)
# ================================================================
def train_flux_kontext(model, dataloader, preproc, device="cuda",
                       lr=2e-5, epochs=10, save_dir="checkpoints"):

    os.makedirs(save_dir, exist_ok=True)
    model.to(device,dtype=torch.bfloat16)
    # model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for step, (x_rgb, y_rgb, s, prompts) in enumerate(pbar):
            
            print_gpu(f"Start of step {step}")

            s = s.to(device, non_blocking=True)
            x_seq, y_seq, img_ids, txt_seq, txt_ids, pooled_txt = preproc.prepare_batch(x_rgb, y_rgb, prompts)

            print_gpu("After preprocessing")
            for name, t in zip(
                ["x_seq", "y_seq", "img_ids", "txt_seq", "txt_ids", "pooled_txt"],
                [x_seq, y_seq, img_ids, txt_seq, txt_ids, pooled_txt]
            ):
                report_tensor_size(name, t)

            # Move to GPU (half precision)
            x_seq, y_seq, img_ids, txt_seq, txt_ids, clip_txt = [
                t.to(device, non_blocking=True) for t in (x_seq, y_seq, img_ids, txt_seq, txt_ids, pooled_txt)
            ]
            print_gpu("After moving tensors to GPU")

            x_seq = x_seq.to(device, non_blocking=True, dtype=torch.bfloat16)
            y_seq = y_seq.to(device, non_blocking=True, dtype=torch.bfloat16)
            txt_seq = txt_seq.to(device, non_blocking=True, dtype=torch.bfloat16)
            clip_txt = clip_txt.to(device, non_blocking=True, dtype=torch.bfloat16)
            
            # IDs can keep their original type (likely int or float)
            img_ids = img_ids.to(device, non_blocking=True,dtype=torch.bfloat16) 
            txt_ids = txt_ids.to(device, non_blocking=True,dtype=torch.bfloat16)
            report_tensor_size("img_ids",img_ids)
            report_tensor_size("txt_ids",txt_ids)
            print_gpu("After moving tensors to GPU")

            with torch.no_grad():
                eps = torch.randn_like(y_seq)
                t = torch.rand(y_seq.size(0), 1, 1, dtype=torch.bfloat16,device=device)
                y_seq = (1 - t) * y_seq + t * eps

            print_gpu("After noise mix")

            if torch.rand(1) < 0.1:
                s = torch.zeros_like(s)

            optimizer.zero_grad(set_to_none=True)
            model.train()
            v_pred32 = model(
                img=y_seq,
                img_ids=img_ids,
                txt=txt_seq,
                txt_ids=txt_ids,
                pooled_txt=clip_txt,
                timesteps=t.view(-1),
                y=clip_txt,
                guidance=torch.ones_like(t.view(-1)),
                strengths=s
            )
            print("v_pred32 min/max:", v_pred32.min(), v_pred32.max(), "anynan:", torch.isnan(v_pred32).any())

            print_gpu("After forward pass")
            report_tensor_size("v_pred", v_pred32)
            report_tensor_size("x_seq", x_seq)
            target = (eps - x_seq).to(dtype=torch.bfloat16)
            loss = F.mse_loss(v_pred32 , target)
            print(f"Target : {target}")
            print(f"loss: {loss}")

            if torch.isnan(loss):
                print("NaN detected!")
                print("v_pred min/max", v_pred32.min(), v_pred32.max())
                print("target min/max", target.min(), target.max())


            loss.backward()
            print_gpu("After backward pass")
            optimizer.step()
            print_gpu("After optimizer step")

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # cleanup
            del x_seq, y_seq, eps, v_pred32, pooled_txt, txt_seq
            torch.cuda.empty_cache()

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
    # model = Flux(params=params)
    model, missing = load_flow_model(model)

    if missing is None or len(missing) == 0:
        missing.append("final_layer.linear.weight")

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
    parser.add_argument("--data_root", type=str, help="Path to curated dataset root",default="images_dir")
    parser.add_argument("--json",type=str, help="Path to saved metadata root",default="prompt")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Checkpoint output dir")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--image_size", type=int, default=512)
    args = parser.parse_args()
    main(args)