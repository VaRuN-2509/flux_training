import os, json, random, argparse
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


# ------------------------------------------------------------
# Fast utility: convert an image to LPIPS tensor [-1,1]
# ------------------------------------------------------------
def to_lpips_tensor(path, resize_to=128):
    img = Image.open(path).convert("RGB")
    img = img.resize((resize_to, resize_to))
    t = transforms.ToTensor()(img) * 2 - 1  # [-1,1]
    return t.cuda()


# ------------------------------------------------------------
# Load JSONL prompt file
# ------------------------------------------------------------
def load_prompts(json_path):
    prompts = {}
    with open(json_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            exp = int(obj["exp_id"])
            prompts[exp] = obj["edit_prompt"]
    return prompts


# ------------------------------------------------------------
# FAST LPIPS filtering logic (torchmetrics squeeze)
# ------------------------------------------------------------
def passes_filter(lpips_model, traj_paths, strengths=8):

    # Load all s0..s7 images (downsampled for speed)
    try:
        imgs = [to_lpips_tensor(p) for p in traj_paths]
    except Exception as e:
        print(f"    ❌ Failed loading: {e}")
        return False

    # Stack into a single batch
    imgs = torch.stack(imgs)  # shape: (8, 3, H, W)

    # Compute LPIPS(s0, s7)
    dist = lpips_model(imgs[0].unsqueeze(0), imgs[-1].unsqueeze(0)).item()
    print(f"    ▶ LPIPS(s0, s7) = {dist:.4f}")

    if dist < 0.08:
        print("    ❌ Rejected: edit too weak")
        return False

    # Compute LPIPS for smoothness between steps
    for i in range(strengths - 1):
        d = lpips_model(imgs[i].unsqueeze(0), imgs[i+1].unsqueeze(0)).item()
        print(f"       step {i}->{i+1}: LPIPS = {d:.4f}")

        if d < 0.01:
            print("       ❌ Rejected: no change")
            return False
        if d > 0.35:
            print("       ❌ Rejected: abrupt jump")
            return False

    print("    ✔ Folder PASSED")
    return True


# ------------------------------------------------------------
# SCAN + FILTER A VOLUME
# ------------------------------------------------------------
def process_volume(volume_root, lpips_model, strengths=8):

    print(f"\n🔍 Scanning volume: {volume_root}")
    filtered_entries = []

    log_root = os.path.join(volume_root, "output_log")
    if not os.path.isdir(log_root):
        print("❌ Missing output_log → skipping")
        return filtered_entries

    categories = [
        c for c in sorted(os.listdir(volume_root))
        if os.path.isdir(os.path.join(volume_root, c))
        and c not in ["output_log", "lora", "output", "cleaned_samples"]
        and not c.startswith(".")
    ]

    print(f"📁 Categories found: {categories}")

    for category in categories:

        print(f"\n=== CATEGORY: {category} ===")
        cat_path = os.path.join(volume_root, category)
        log_cat = os.path.join(log_root, f"edit_{category}")

        if not os.path.isdir(log_cat):
            print(f"⚠ Missing log folder → {log_cat}")
            continue

        json_path = os.path.join(log_cat, "data_loaded.json")
        if not os.path.exists(json_path):
            print(f"⚠ Missing data_loaded.json in {log_cat}")
            continue

        prompts = load_prompts(json_path)

        exp_folders = [
            f for f in sorted(os.listdir(cat_path))
            if os.path.isdir(os.path.join(cat_path, f))
        ]

        print(f"   → Exp folders: {len(exp_folders)}")

        # scan experiment folders
        for exp_id in exp_folders:
            exp_path = os.path.join(cat_path, exp_id)
            print(f"  🔎 Checking folder: {exp_path}")

            # numeric ID
            try:
                exp_num = int(exp_id)
            except:
                print("    ⚠ Skipping non-numeric")
                continue

            if exp_num not in prompts:
                print("    ⚠ No prompt → skip")
                continue

            prompt = prompts[exp_num]

            # build full trajectory
            traj_paths = [
                os.path.join(exp_path, f"{exp_id}_s{s}.png")
                for s in range(strengths)
            ]

            if not all(os.path.exists(p) for p in traj_paths):
                print("    ⚠ Incomplete s0–s7 → skip")
                continue

            # run filtering
            if not passes_filter(lpips_model, traj_paths, strengths):
                print("    ❌ Folder rejected")
                continue

            print(f"    ✔ Accepted: {exp_path}")

            # save all src→tgt pairs
            src = traj_paths[0]
            for s_idx, tgt in enumerate(traj_paths):
                s_val = s_idx / (strengths - 1)

                filtered_entries.append({
                    "src": src,
                    "tgt": tgt,
                    "s": float(s_val),
                    "prompt": prompt
                })

    return filtered_entries


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--strengths", type=int, default=8)
    parser.add_argument("--output", type=str,
                        default="/root/data/cleaned_samples.json")
    args = parser.parse_args()

    # Manually set volumes (your use case)
    volumes = ["/root/data", "/root/data_2"]

    print("🔥 Loading fast LPIPS (torchmetrics, squeeze)...")
    lpips_model = LearnedPerceptualImagePatchSimilarity(
        net_type="squeeze"
    ).cuda()
    lpips_model.eval()

    all_entries = []

    for vol in volumes:
        entries = process_volume(vol, lpips_model, strengths=args.strengths)
        all_entries.extend(entries)

    random.shuffle(all_entries)

    print(f"\n💾 Saving {len(all_entries)} samples → {args.output}")
    with open(args.output, "w") as f:
        json.dump(all_entries, f, indent=2)

    print("\n🎉 DONE! Filtered dataset saved successfully.")
