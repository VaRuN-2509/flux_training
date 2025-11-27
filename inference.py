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
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file as load_sft


import os
import re
import time
from dataclasses import dataclass
from glob import iglob
from src.flux.model import Flux, FluxLoraWrapper

import torch
from src.flux.sampling import denoise, get_schedule, prepare_kontext, unpack
from src.flux.util import (
    aspect_ratio_to_height_width,
    check_onnx_access_for_trt,
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
    save_image,
)
HF_TOKEN = os.environ["HF_TOKEN"]


@dataclass
class SamplingOptions:
    prompt: str
    width: int | None
    height: int | None
    num_steps: int
    guidance: float
    seed: int | None
    img_cond_path: str

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


def parse_prompt(options: SamplingOptions) -> SamplingOptions | None:
    user_question = "Next prompt (write /h for help, /q to quit and leave empty to repeat):\n"
    usage = (
        "Usage: Either write your prompt directly, leave this field empty "
        "to repeat the prompt or write a command starting with a slash:\n"
        "- '/ar <width>:<height>' will set the aspect ratio of the generated image\n"
        "- '/s <seed>' sets the next seed\n"
        "- '/g <guidance>' sets the guidance (flux-dev only)\n"
        "- '/n <steps>' sets the number of steps\n"
        "- '/q' to quit"
    )

    while (prompt := input(user_question)).startswith("/"):
        if prompt.startswith("/ar"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, ratio_prompt = prompt.split()
            if ratio_prompt == "auto":
                options.width = None
                options.height = None
                print("Setting resolution to input image resolution.")
            else:
                options.width, options.height = aspect_ratio_to_height_width(ratio_prompt)
                print(f"Setting resolution to {options.width} x {options.height}.")
        elif prompt.startswith("/h"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, height = prompt.split()
            if height == "auto":
                options.height = None
            else:
                options.height = 16 * (int(height) // 16)
            if options.height is not None and options.width is not None:
                print(
                    f"Setting resolution to {options.width} x {options.height} "
                    f"({options.height * options.width / 1e6:.2f}MP)"
                )
            else:
                print(f"Setting resolution to {options.width} x {options.height}.")
        elif prompt.startswith("/g"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, guidance = prompt.split()
            options.guidance = float(guidance)
            print(f"Setting guidance to {options.guidance}")
        elif prompt.startswith("/s"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, seed = prompt.split()
            options.seed = int(seed)
            print(f"Setting seed to {options.seed}")
        elif prompt.startswith("/n"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, steps = prompt.split()
            options.num_steps = int(steps)
            print(f"Setting number of steps to {options.num_steps}")
        elif prompt.startswith("/q"):
            print("Quitting")
            return None
        else:
            if not prompt.startswith("/h"):
                print(f"Got invalid command '{prompt}'\n{usage}")
            print(usage)
    if prompt != "":
        options.prompt = prompt
    return options


def parse_img_cond_path(options: SamplingOptions | None) -> SamplingOptions | None:
    if options is None:
        return None

    user_question = "Next input image (write /h for help, /q to quit and leave empty to repeat):\n"
    usage = (
        "Usage: Either write a path to an image directly, leave this field empty "
        "to repeat the last input image or write a command starting with a slash:\n"
        "- '/q' to quit\n\n"
        "The input image will be edited by FLUX.1 Kontext creating a new image based"
        "on your instruction prompt."
    )

    while True:
        img_cond_path = input(user_question)

        if img_cond_path.startswith("/"):
            if img_cond_path.startswith("/q"):
                print("Quitting")
                return None
            else:
                if not img_cond_path.startswith("/h"):
                    print(f"Got invalid command '{img_cond_path}'\n{usage}")
                print(usage)
            continue

        if img_cond_path == "":
            break

        if not os.path.isfile(img_cond_path) or not img_cond_path.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            print(f"File '{img_cond_path}' does not exist or is not a valid image file")
            continue

        options.img_cond_path = img_cond_path
        break

    return options

def select_lora_from_prompt(prompt: str) :
    prompt = prompt.lower()

    # ---- Remove / Object removal LoRA ----
    if any(keyword in prompt for keyword in ["remove", "erase", "delete", "clean", "get rid of"]):
        return "Remove"

    # ---- Thumbnail LoRA ----
    if any(keyword in prompt for keyword in ["thumbnail", "youtube thumbnail", "small preview", "banner"]):
        return "thumbnail"

    # ---- Perspective / top-down / angle LoRA ----
    if any(keyword in prompt for keyword in ["top-down", "top down", "bird view", "perspective", "angle"]):
        return "perspective"

    # ---- 3D Chibi Style LoRA ----
    if any(keyword in prompt for keyword in ["chibi", "3d chibi", "cute chibi"]):
        return "3D_Chibi"   # or full path if needed
    
    if any(keyword in prompt for keyword in ["extract", "dress", "clothes"]):
        return "extract_clothes"
    
    if any(keyword in prompt for keyword in ["wear", "put on", "make"]):
        return "put_clothes"  
    





@torch.inference_mode()
def main(
    strength_projector,
    strengths,
    name: str = "flux-dev-kontext",
    aspect_ratio: str | None = None,
    seed: int | None = None,
    prompt: str = '',
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int = 30,
    loop: bool = False,
    guidance: float = 2.5,
    offload: bool = False,
    output_dir: str = "output",
    add_sampling_metadata: bool = True,
    img_cond_path: str = "sample.jpg",
    trt: bool = False,
    trt_transformer_precision: str = "bf16",
    track_usage: bool = False,
    
):
    """
    Sample the flux model. Either interactively (set `--loop`) or run for a
    single image.

    Args:
        height: height of the sample in pixels (should be a multiple of 16), None
            defaults to the size of the conditioning
        width: width of the sample in pixels (should be a multiple of 16), None
            defaults to the size of the conditioning
        seed: Set a seed for sampling
        output_name: where to save the output image, `{idx}` will be replaced
            by the index of the sample
        prompt: Prompt used for sampling
        device: Pytorch device
        num_steps: number of sampling steps (default 4 for schnell, 50 for guidance distilled)
        loop: start an interactive session and sample multiple times
        guidance: guidance value used for guidance distillation
        add_sampling_metadata: Add the prompt to the image Exif metadata
        img_cond_path: path to conditioning image (jpeg/png/webp)
        trt: use TensorRT backend for optimized inference
        track_usage: track usage of the model for licensing purposes
    """

    assert name == "flux-dev-kontext", f"Got unknown model name: {name}"

    torch_device = torch.device(device)

    output_name = os.path.join(output_dir, f"img_{prompt}_{strengths}.jpg")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        idx = 0
    else:
        fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
        if len(fns) > 0:
            idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
        else:
            idx = 0

    if aspect_ratio is None:
        width = None
        height = None
    else:
        width, height = aspect_ratio_to_height_width(aspect_ratio)

    t5 = load_t5(torch_device, max_length=512)
    clip = load_clip(torch_device)

    def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
        if len(missing) > 0 and len(unexpected) > 0:
            print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
            print("\n" + "-" * 79 + "\n")
            print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
        elif len(missing) > 0:
            print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        elif len(unexpected) > 0:
            print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


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

    model = FluxLoraWrapper(strength_projector=strength_projector,lora_rank=4, lora_scale=1.0, params=params)
    # model, missing = load_flow_model(model)
    if strength_projector:
        model = FluxLoraWrapper(strength_projector=strength_projector,lora_rank=4, lora_scale=1.0, params=params)
        CHECKPOINT_PATH = "/root/data/flux-checkpoints/latest_epoch=7.pt"
        print("Loading checkpoint:", CHECKPOINT_PATH)
        state = torch.load(CHECKPOINT_PATH, map_location="cpu")
        load_result = model.load_state_dict(state["model_state"], strict=False)
        if load_result.missing_keys:
            print("\n⚠️ Missing keys in checkpoint (not found in model):")
            for k in load_result.missing_keys:
                print("  -", k)

        if load_result.unexpected_keys:
            print("\n⚠️ Unexpected keys in checkpoint (not used by model):")
            for k in load_result.unexpected_keys:
                print("  -", k)

        if not load_result.missing_keys and not load_result.unexpected_keys:
            print("\n🎯 State dict loaded cleanly — no missing/unexpected keys!")

            # for name1, p in model.named_parameters():
            #     if p.dtype != torch.bfloat16:
            #         print("Not bf16:", name1, p.dtype)
    
    else:
        from huggingface_hub import hf_hub_download
        from diffusers import FluxKontextPipeline
        from diffusers.utils import load_image

        
        os.environ['HF_TOKEN'] = "***REMOVED***"

        style_type_lora_dict = {
            "3D_Chibi": "3D_Chibi_lora_weights.safetensors",
            "Remove" : "/root/data/lora/kontext_remove.safetensors",
            "thumbnail" : "/root/data/lora/thumbnails_lora_rank_32.safetensors",
            "perspective" : "/root/data/lora/Kontext-Top-Down-View.safetensors",
            "extract_clothes": "/root/data/lora/extract-clothes-kontext-dev-lora.safetensors",
            "put_clothes" : "/root/data/lora/virtual-tryon-kontext-lora.safetensors"
        }

        
        image = load_image(
            img_cond_path
        ).resize((1024, 1024))

        # Load Flux-Kontext pipeline
        pipeline = FluxKontextPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev",
            torch_dtype=torch.bfloat16
        ).to("cuda")

        file = select_lora_from_prompt(prompt)

        print(file)

        pipeline.load_lora_weights(f"/root/data/lora/Kontext-Top-Down-View.safetensors", adapter_name="lora_1")
        pipeline.load_lora_weights(f"/root/data/lora/LEGO_lora_weights.safetensors", adapter_name="lora")
        pipeline.set_adapters(["lora","lora_1"], adapter_weights=[0.5,0.1])

        print("Loaded adapters:", pipeline.get_active_adapters())



        out = pipeline(
            image=image,
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=10
        ).images[0]

        os.makedirs("/root/data/output", exist_ok=True)

        out.save(f"/root/data/output/{file}.png")

        return




    # Log everything clearly

    
    model.to(device="cuda", dtype=torch.bfloat16)

    ae = load_ae(name, device="cpu" if offload else torch_device)
    

    rng = torch.Generator(device="cpu")
    opts = SamplingOptions(
        prompt=prompt,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
        img_cond_path=img_cond_path,
    )

    if loop:
        opts = parse_prompt(opts)
        opts = parse_img_cond_path(opts)

    while opts is not None:
        if opts.seed is None:
            opts.seed = rng.seed()
        print(f"Generating with seed {opts.seed}:\n{opts.prompt}")
        t0 = time.perf_counter()

        

        if offload:
            t5, clip, ae = t5.to(torch_device), clip.to(torch_device), ae.to(torch_device)
        inp, height, width = prepare_kontext(
            t5=t5,
            clip=clip,
            prompt=opts.prompt,
            ae=ae,
            img_cond_path=opts.img_cond_path,
            target_width=opts.width,
            target_height=opts.height,
            bs=1,
            seed=opts.seed,
            device=torch_device,
        )
        from safetensors.torch import save_file

        save_file({k: v.cpu().contiguous() for k, v in inp.items()}, "output/noise.sft")
        inp.pop("img_cond_orig")
        opts.seed = None
        timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))

        # offload TEs and AE to CPU, load model to gpu
        if offload:
            t5, clip, ae = t5.cpu(), clip.cpu(), ae.cpu()
            torch.cuda.empty_cache()


        # denoise initial noise
        t00 = time.time()
        x = denoise(model, **inp, timesteps=timesteps, guidance=opts.guidance,strengths=strengths)
        torch.cuda.synchronize()
        t01 = time.time()
        print(f"Denoising took {t01 - t00:.3f}s")

        # offload model, load autoencoder to gpu
        if offload:
            model.cpu()
            torch.cuda.empty_cache()
            ae.decoder.to(x.device)

        # decode latents to pixel space
        x = unpack(x.float(), height, width)
        with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
            ae_dev_t0 = time.perf_counter()
            x = ae.decode(x)
            torch.cuda.synchronize()
            ae_dev_t1 = time.perf_counter()
            print(f"AE decode took {ae_dev_t1 - ae_dev_t0:.3f}s")


        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        print(f"Done in {t1 - t0:.1f}s")

        idx = save_image(
            None, name, output_name, idx, x, add_sampling_metadata, prompt, track_usage=track_usage
        )

        if loop:
            print("-" * 80)
            opts = parse_prompt(opts)
            opts = parse_img_cond_path(opts)
        else:
            opts = None


if __name__ == "__main__":

    
    MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"   # CPU or GPU auto
    )

    SYSTEM_PROMPT = """
                    You split complex image-editing instructions into several atomic edits.
                    Each atomic edit MUST:
                    - be a short natural-language command
                    - describe exactly ONE modification
                    - be independent
                    - not contain structured fields like action/target/value

                    Return ONLY a JSON list of natural-language edit commands.
                    """

    def decompose_edit_prompt(prompt: str):
        formatted = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{prompt}\n<|assistant|>"

        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        output = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.1,
            do_sample=False
        )

        text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract JSON list
        start = text.find("[")
        end = text.rfind("]") + 1
        json_str = text[start:end]

        try:
            return json.loads(json_str)
        except:
            # fallback – cleanup
            cleaned = json_str.replace("\n", " ").strip()
            return json.loads(cleaned)


    prompt = "make the background slightly darker"
    print(prompt)
    img_path = ["tridesh.jpeg"]
    # strengths=[torch.tensor(0.65, dtype=torch.bfloat16, device="cuda"),torch.tensor(0.1, dtype=torch.bfloat16, device="cuda")]
    strenght = [0.15]
    strengths = torch.tensor(strenght[0], dtype=torch.bfloat16, device="cuda")
    main(strength_projector=True,prompt=prompt,img_cond_path=img_path[0],strengths=strengths)

    # for i in range (len(prompt)):
    #     strengths = torch.tensor(strenght[i], dtype=torch.bfloat16, device="cuda") 
    #     main(strength_projector=False,prompt=prompt[i],img_cond_path=img_path[i],strengths=strengths)
