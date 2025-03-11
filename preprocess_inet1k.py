import os
cache_dir = "datasets"
os.environ["HF_HOME"] = cache_dir

import time
import datasets
from datasets import load_dataset
import numpy as np
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import io
from torchvision import transforms
from diffusers import AutoencoderDC
from multiprocess import set_start_method
import json
import lovely_tensors
lovely_tensors.monkey_patch()
TARGET_RES = 256


def safe_load_image(image_bytes):
    """Safely load an image while ignoring large metadata like ICC profiles"""
    try:
        # First try normal loading
        return Image.open(io.BytesIO(image_bytes))
    except ValueError as e:
        if "Decompressed Data Too Large" in str(e):
            # If that fails, try loading with LOAD_TRUNCATED_IMAGES flag
            Image.MAX_IMAGE_PIXELS = None  # Disable max image size limit
            return Image.open(io.BytesIO(image_bytes)).convert('RGB')
        raise e

def get_prng(seed):
    return np.random.RandomState(seed)

class BucketManager:
    def __init__(self, max_size=(256,256), divisible=16, min_dim=128, base_res=(256,256), bsz=64, world_size=1, global_rank=0, max_ar_error=4, seed=42, dim_limit=512, debug=False):
        self.max_size = max_size
        self.f = 8
        self.max_tokens = (max_size[0]/self.f) * (max_size[1]/self.f)
        self.div = divisible
        self.min_dim = min_dim
        self.dim_limit = dim_limit
        self.base_res = base_res
        self.bsz = bsz
        self.world_size = world_size
        self.global_rank = global_rank
        self.max_ar_error = max_ar_error
        self.prng = get_prng(seed)
        epoch_seed = self.prng.tomaxint() % (2**32-1)
        self.epoch_prng = get_prng(epoch_seed) # separate prng for sharding use for increased thread resilience
        self.epoch = None
        self.left_over = None
        self.batch_total = None
        self.batch_delivered = None

        self.debug = debug

        self.gen_buckets()

    def gen_buckets(self):
        if self.debug:
            timer = time.perf_counter()
        resolutions = []
        aspects = []
        w = self.min_dim
        while (w/self.f) * (self.min_dim/self.f) <= self.max_tokens and w <= self.dim_limit:
            h = self.min_dim
            got_base = False
            while (w/self.f) * ((h+self.div)/self.f) <= self.max_tokens and (h+self.div) <= self.dim_limit:
                if w == self.base_res[0] and h == self.base_res[1]:
                    got_base = True
                h += self.div
            if (w != self.base_res[0] or h != self.base_res[1]) and got_base:
                resolutions.append(self.base_res)
                aspects.append(1)
            resolutions.append((w, h))
            aspects.append(float(w)/float(h))
            w += self.div
        h = self.min_dim
        while (h/self.f) * (self.min_dim/self.f) <= self.max_tokens and h <= self.dim_limit:
            w = self.min_dim
            got_base = False
            while (h/self.f) * ((w+self.div)/self.f) <= self.max_tokens and (w+self.div) <= self.dim_limit:
                if w == self.base_res[0] and h == self.base_res[1]:
                    got_base = True
                w += self.div
            resolutions.append((w, h))
            aspects.append(float(w)/float(h))
            h += self.div
        res_map = {}
        for i, res in enumerate(resolutions):
            res_map[res] = aspects[i]
        self.resolutions = sorted(res_map.keys(), key=lambda x: x[0] * 4096 - x[1])
        self.aspects = np.array(list(map(lambda x: res_map[x], self.resolutions)))
        self.resolutions = np.array(self.resolutions)
        if self.debug:
            timer = time.perf_counter() - timer
            print(f"resolutions:\n{self.resolutions}")
            print(f"aspects:\n{self.aspects}")
            print(f"gen_buckets: {timer:.5f}s")

    def get_ideal_resolution(self, image_size) -> tuple[int, int]:
        w, h = image_size
        aspect = float(w)/float(h)
        bucket_id = np.abs(np.log(self.aspects) - np.log(aspect)).argmin()
        return self.resolutions[bucket_id]

def bytes_to_pil_image(image_bytes):
    return Image.open(io.BytesIO(image_bytes))

def preprocess_image(image, target_size, device='cuda'):
    """
    Merged 'resize_and_crop' + 'preprocess_image' into a single GPU-based pipeline:
    1) Ensure image is RGB.
    2) Convert to GPU tensor, normalized to [0,1].
    3) Resize to preserve aspect ratio (bicubic).
    4) Random-crop to target_size.
    5) Normalize channels to [-1, 1].
    """
    # Ensure image is in RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to GPU tensor, [C, H, W], in floating [0,1]
    img_tensor = torch.as_tensor(np.array(image), device=device).permute(2, 0, 1).float() / 255.0
    orig_c, orig_h, orig_w = img_tensor.shape

    # Unpack desired width/height
    target_w, target_h = target_size
    # Compute aspect ratios
    aspect_ratio = orig_w / orig_h
    target_aspect_ratio = target_w / target_h

    # Decide how to resize
    if abs(aspect_ratio - target_aspect_ratio) < 1e-6:
        # Aspect ratios match, just resize directly
        new_width, new_height = target_w, target_h
    else:
        if aspect_ratio > target_aspect_ratio:
            # Image is wider than target -> match target height
            new_height = target_h
            new_width = int(aspect_ratio * new_height)
        else:
            # Image is taller than target -> match target width
            new_width = target_w
            new_height = int(new_width / aspect_ratio)

    # Resize on GPU via interpolate
    # Need [N,C,H,W], so add batch dimension
    img_tensor_4d = img_tensor.unsqueeze(0)  # [1, C, H, W]
    resized_4d = F.interpolate(
        img_tensor_4d,
        size=(new_height, new_width),
        mode='bicubic',
        align_corners=False
    )
    # Remove batch dim => [C, H, W]
    resized = resized_4d.squeeze(0)

    # Random crop to target size
    # (Assumes new_width >= target_w and new_height >= target_h.)
    h_off = torch.randint(0, resized.shape[1] - target_h + 1, size=(1,), device=device).item()
    w_off = torch.randint(0, resized.shape[2] - target_w + 1, size=(1,), device=device).item()
    cropped = resized[:, h_off:h_off + target_h, w_off:w_off + target_w]

    # Normalize to [-1,1] the same way as:
    #   transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    # which is (x - 0.5)/0.5 => 2x - 1
    cropped = cropped * 2.0 - 1.0

    return cropped  # [C, target_h, target_w], in [-1,1], on GPU

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def generate_latents(model, images):
    """Assumes all images are the same shape"""
    latents = model.encode(images).latent

    return latents

# Set HF_HOME environment variable and cache_dir
bucket_manager = BucketManager(max_size=(256, 256), divisible=64, min_dim=128, base_res=(256, 256), bsz=64, world_size=1, global_rank=0, max_ar_error=4, seed=42, dim_limit=512)

ae = AutoencoderDC.from_pretrained("mit-han-lab/dc-ae-f64c128-mix-1.0-diffusers", cache_dir="ae", torch_dtype=torch.bfloat16)

ds = load_dataset("visual-layer/imagenet-1k-vl-enriched", split="validation", cache_dir=cache_dir)

ds = ds.remove_columns(["caption_enriched", "label_bbox_enriched", "issues"])

ds = ds.add_column("ae_latent", [None] * len(ds))
ds = ds.add_column("caption", [None] * len(ds))
ds = ds.add_column("ae_latent_shape", [None] * len(ds))


with open("combined_imagenet21k_captions.json", "r") as f:
    captions = json.load(f)

import torch
import numpy as np
from collections import defaultdict

def gpu_computation(row, rank):
    # Move the model on the right GPU if it's not there already
    device = f"cuda:{rank}"
    ae.to(device)

    res = bucket_manager.get_ideal_resolution(row["image"].size)
    image = preprocess_image(row["image"], res, device).to(device, torch.bfloat16).unsqueeze(0)
    latents = generate_latents(ae, image).squeeze(0).to(torch.float8_e5m2).view(torch.uint8)
    
    row["image"] = None    
    row["ae_latent"] = latents
    row["caption"] = captions[row["image_id"]]["caption"]
    row["ae_latent_shape"] = latents.shape
    return row


if __name__ == "__main__":
    set_start_method("spawn")

    print("Length before filtering:", len(ds))
    ds = ds.filter(lambda x: x["image"].width * x["image"].height >= TARGET_RES * TARGET_RES, num_proc=16)
    print("Length after filtering:", len(ds))

    ds = ds.map(
        gpu_computation,
        with_rank=True,
        num_proc=torch.cuda.device_count(),  # one process per GPU
    )

    ds = ds.remove_columns(["image"])

    ds.save_to_disk("imagenet1k_dcae-f64-latents_validation")
    ds.push_to_hub("SwayStar123/imagenet1k_dcae-f64-latents", split="validation")
