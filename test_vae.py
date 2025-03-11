from PIL import Image
import numpy as np
import torch
from diffusers import AutoencoderKL
import io
import time
from torchvision import transforms

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def generate_latents(model, images):
    """Assumes all images are the same shape"""
    latents = model.encode(images).latent_dist.sample()

    return latents

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

def resize_and_crop(image, target_size):
    # image: PIL Image
    # target_size: (width, height)
    target_w, target_h = target_size
    aspect_ratio = image.width / image.height
    target_aspect_ratio = target_w / target_h
    if abs(aspect_ratio - target_aspect_ratio) < 1e-6:
        # Aspect ratios match, resize directly
        image = image.resize((target_w, target_h), Image.BICUBIC)
    else:
        # Resize while preserving aspect ratio, then random crop
        if aspect_ratio > target_aspect_ratio:
            # Image is wider than target, resize height to target height
            new_height = target_h
            new_width = int(aspect_ratio * new_height)
        else:
            # Image is taller than target, resize width to target width
            new_width = target_w
            new_height = int(new_width / aspect_ratio)
        image = image.resize((new_width, new_height), Image.BICUBIC)
        # Random crop to target size
        left = np.random.randint(0, new_width - target_w + 1)
        upper = np.random.randint(0, new_height - target_h + 1)
        image = image.crop((left, upper, left + target_w, upper + target_h))
    return image

def preprocess_image(image):
    # Ensure image is in RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Convert to tensor, normalize
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0,1], shape (C,H,W)
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1,1]
    ])
    image_tensor = transform(image)
    return image_tensor

if __name__ == "__main__":
    vae = AutoencoderKL.from_pretrained("KBlueLeaf/EQ-SDXL-VAE", cache_dir="../../models/vae").cuda().to(torch.bfloat16)

    bucket_manager = BucketManager(max_size=(256, 256), divisible=16, min_dim=128, base_res=(256, 256), bsz=64, world_size=1, global_rank=0, max_ar_error=4, seed=42, dim_limit=512)

    image = Image.open("image.jpg") 
    res = bucket_manager.get_ideal_resolution(image.size)
    print(res)

    with torch.no_grad():
        image = resize_and_crop(image, res)
        image.save("resized.png")

        # preprocess image
        image = preprocess_image(image).to("cuda", torch.bfloat16)
        image = image.unsqueeze(0)
        latents = generate_latents(vae, image)

        # Quantize latents to 8 bit
        latents = latents.to(torch.float8_e5m2).view(torch.uint8)

        # back to bfloat16
        latents = latents.view(torch.float8_e5m2).to(torch.bfloat16)

        # decode and save
        decoded = vae.decode(latents).sample.to(torch.float32)
        decoded = decoded.clamp(-1, 1)
        decoded = decoded.cpu().numpy()
        decoded = (decoded * 0.5 + 0.5) * 255
        decoded = decoded.astype(np.uint8)
        decoded = decoded.squeeze(0)  # Remove batch dimension
        decoded = decoded.transpose(1, 2, 0)  # Change from CHW to HWC format
        
        decoded = Image.fromarray(decoded)
        decoded.save("decoded2.png")

