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
from einops import rearrange, repeat

# Import model utilities
from src.flux.model import FluxParams, FluxLoraWrapper,Flux
from src.flux.util import load_flow_model, load_ae, load_clip,load_t5
from src.flux.modules.layers import timestep_embedding



import random

HF_TOKEN = os.environ["HF_TOKEN"]


class CleanedFluxDataset(Dataset):

    def __init__(self, json_path, size=512):
        with open(json_path) as f:
            self.samples = json.load(f)

        self.preproc = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        src = Image.open(item["src"]).convert("RGB")
        tgt = Image.open(item["tgt"]).convert("RGB")

        return (
            self.preproc(src),
            self.preproc(tgt),
            torch.tensor(item["s"], dtype=torch.float32),
            item["prompt"]
        )





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

        B, C, H, W = lat.shape
        assert H % 2 == 0 and W % 2 == 0, "Latent dims must be divisible by 2"

        # 2x2 patchify → token_dim = 16 * 4 = 64
        tokens = rearrange(
            lat,
            "b c (h ph) (w pw) -> b (h w) (c ph pw)",
            ph=2, pw=2
        )
        
        return tokens.to(torch.bfloat16)


    @torch.no_grad()
    def encode_texts(self, prompts):
        clip_emb = self.clip(list(prompts))

        t5_emb = self.t5(list(prompts))

        txt_seq = t5_emb
        B,T,_ = txt_seq.shape
        txt_ids = torch.zeros(B, T, 3, device=self.device, dtype=torch.bfloat16)
        txt_ids[..., 1] = torch.arange(T, device=self.device)
        torch.cuda.empty_cache()
        return txt_seq.cpu(), txt_ids.cpu(), clip_emb.cpu()
    
    def make_img_ids(self, lat_tokens):
        B, L, _ = lat_tokens.shape
        H = W = int(L**0.5)
        assert H * W == L

        coords = torch.zeros(B, L, 3, device=self.device, dtype=torch.bfloat16)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing="ij",
        )
        coords[..., 1] = grid_y.flatten()
        coords[..., 2] = grid_x.flatten()
        return coords

    def prepare_batch(self, x_rgb, y_rgb, prompts):
        x_seq = self.encode_images(x_rgb)
        y_seq = self.encode_images(y_rgb)
        B, L, C = y_seq.shape
        H = W = int(y_seq.shape[1] ** 0.5)
        img_ids = self.make_img_ids(y_seq)

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
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm


# --------------------------------------------------------
# Save BEST checkpoint (lowest loss)
# --------------------------------------------------------
def save_best_checkpoint(save_dir, epoch, loss, model, optimizer):
    ckpt_path = os.path.join(save_dir, f"best_epoch={epoch}_loss={loss:.6f}.pt")

    torch.save({
        "epoch": epoch,
        "loss": loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "type": "best"
    }, ckpt_path)

    print(f"💾 Saved BEST checkpoint: {ckpt_path}")


# --------------------------------------------------------
# Save LATEST checkpoint (every epoch)
# --------------------------------------------------------
def save_latest_checkpoint(save_dir, epoch, loss, model, optimizer):
    ckpt_path = os.path.join(save_dir, f"latest_epoch={epoch}.pt")

    torch.save({
        "epoch": epoch,
        "loss": loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "type": "latest"
    }, ckpt_path)

    print(f"📝 Saved LATEST checkpoint: {ckpt_path}")


# --------------------------------------------------------
# Load checkpoint safely (resume capability)
# --------------------------------------------------------
def load_checkpoint(checkpoint_path, model, optimizer=None, device="cuda"):
    ckpt = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    print(f"🔄 Loaded checkpoint epoch {ckpt['epoch']} (loss={ckpt.get('loss','N/A')})")
    return ckpt["epoch"], ckpt.get("loss", None)



# --------------------------------------------------------
# TRAINING LOOP (BEST + LATEST SAVING)
# --------------------------------------------------------
def train_flux_kontext(
    model,
    dataloader,
    preproc,
    device="cuda",
    lr=2e-5,
    epochs=10,
    save_dir="checkpoints",
    resume_from = None,
):

    os.makedirs(save_dir, exist_ok=True)
    model.to(device, dtype=torch.bfloat16)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    # --------------------------------------------------------
    # Resume from checkpoint (if provided)
    # --------------------------------------------------------
    start_epoch = 1
    best_loss = float("inf")

    if resume_from is not None:
        start_epoch, prev_loss = load_checkpoint(
            resume_from, model, optimizer, device
        )
        best_loss = prev_loss if prev_loss is not None else float("inf")
        start_epoch += 1
        print(f"▶ Resuming from epoch {start_epoch}, best_loss={best_loss}")

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------
    for epoch in range(start_epoch, epochs + 1):

        model.train()
        epoch_loss = 0
        count = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")

        for step, (x_rgb, y_rgb, s, prompts) in enumerate(pbar):

            s = s.to(device, non_blocking=True)

            # Data preprocessing
            x_seq, y_seq, img_ids, txt_seq, txt_ids, pooled_txt =preproc.prepare_batch(x_rgb, y_rgb, prompts)

            x_seq, y_seq, img_ids, txt_seq, txt_ids, clip_txt = [
                t.to(device, non_blocking=True) 
                for t in (x_seq, y_seq, img_ids, txt_seq, txt_ids, pooled_txt)
            ]

            x_seq = x_seq.to(dtype=torch.bfloat16)
            y_seq = y_seq.to(dtype=torch.bfloat16)
            txt_seq = txt_seq.to(dtype=torch.bfloat16)
            clip_txt = clip_txt.to(dtype=torch.bfloat16)
            img_ids = img_ids.to(dtype=torch.bfloat16)
            txt_ids = txt_ids.to(dtype=torch.bfloat16)

            # Add noise
            with torch.no_grad():
                eps = torch.randn_like(y_seq)
                t = torch.rand(y_seq.size(0), 1, 1, dtype=torch.bfloat16, device=device)
                y_seq = (1 - t) * y_seq + t * eps

            # Strength dropout
            if torch.rand(1).item() < 0.1:
                s = torch.zeros_like(s)

            optimizer.zero_grad(set_to_none=True)

            v_pred = model(
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

            target = (eps - x_seq).to(dtype=torch.bfloat16)
            loss = F.mse_loss(v_pred, target)

            if torch.isnan(loss):
                print("❌ NaN detected — skipping step")
                continue

            loss.backward()
            optimizer.step()

            # Track epoch loss
            epoch_loss += loss.item()
            count += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            del x_seq, y_seq, eps, v_pred

        epoch_loss /= max(count, 1)
        print(f"📉 Epoch {epoch} avg loss = {epoch_loss:.6f}")

        # --------------------------------------------------------
        # Save BEST checkpoint
        # --------------------------------------------------------
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_best_checkpoint(save_dir, epoch, best_loss, model, optimizer)

        # --------------------------------------------------------
        # Save LATEST checkpoint
        # --------------------------------------------------------
        save_latest_checkpoint(save_dir, epoch, epoch_loss, model, optimizer)

    print("🎉 Training Complete!")

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

    model = FluxLoraWrapper(lora_rank=4, lora_scale=1.0, params=params,strength_projector=True)
    # model = Flux(params=params)
    model,missing = load_flow_model(model)

    # Train only LoRA + Strength Projector
    for name, p in model.named_parameters():
        p.requires_grad = name in missing


    print(f"🔹 Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Dataset & loader
    dataset = CleanedFluxDataset("cleaned_samples_fixed.json")
    print(f"batch size : {args.batch_size}")
    num_workers = min(8, max(1, (os.cpu_count() or 4) - 1))
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    # Start training
    train_flux_kontext(model, dataloader, preproc, device, lr=args.lr,
                       epochs=args.epochs, save_dir=args.save_dir,resume_from="/root/data/checkpoints")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Flux Kontext in latent space (LoRA + projector)")
    parser.add_argument("--data_root", type=str, help="Path to curated dataset root",default="images_dir")
    parser.add_argument("--json",type=str, help="Path to saved metadata root",default="prompt")
    parser.add_argument("--save_dir", type=str, default="/root/data/flux-checkpoints", help="Checkpoint output dir")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--image_size", type=int, default=512)
    args = parser.parse_args()
    main(args)