import os
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import AutoencoderKL
import torch
from open_clip import create_model_from_pretrained, get_tokenizer
import time
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.multiprocessing import Queue, Process
import torch.distributed as dist

DATASET = "commoncatalog-cc-by"
DATASET_DIR_BASE = "../../datasets"
MODELS_DIR_BASE = "../../models"
VAE_HF_NAME = "madebyollin/sdxl-vae-fp16-fix"
SIGLIP_HF_NAME = "hf-hub:timm/ViT-SO400M-14-SigLIP-384"
IMAGE_COLUMN_NAME = "jpg"
IMAGE_ID_COLUMN_NAME = "image_id"
BS = 64

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

def collect_and_move_parquets(base_dir):
    # aspect ratio : list of parquet filepaths
    aspect_ratio_parquets = {}

    base_dir = os.path.join(base_dir, DATASET)

    # Cycle through 0 to 9
    for i in range(10):
        # Construct the path for the i-th folder
        i_folder_path = os.path.join(base_dir, str(i))
        
        # Check if the folder exists
        if not os.path.exists(i_folder_path):
            continue

        # Cycle through all the resolution folders in the i-th folder
        for resolution_folder in os.listdir(i_folder_path):
            resolution_folder_path = os.path.join(i_folder_path, resolution_folder)
            
            # Check if it's a directory
            if not os.path.isdir(resolution_folder_path):
                continue

            # Cycle through all the aspect ratio folders in the resolution folder
            for aspect_ratio_folder in os.listdir(resolution_folder_path):
                aspect_ratio_folder_path = os.path.join(resolution_folder_path, aspect_ratio_folder)
                
                # Check if it's a directory
                if not os.path.isdir(aspect_ratio_folder_path):
                    continue

                # Collect all parquet files in the aspect ratio folder
                for parquet_file in os.listdir(aspect_ratio_folder_path):
                    if parquet_file.endswith('.parquet'):
                        parquet_file_path = os.path.join(aspect_ratio_folder_path, parquet_file)
                        
                        # Add the file path to the aspect ratio dictionary
                        if aspect_ratio_folder not in aspect_ratio_parquets:
                            aspect_ratio_parquets[aspect_ratio_folder] = []
                        aspect_ratio_parquets[aspect_ratio_folder].append(parquet_file_path)

    # filepath : aspect ratio
    new_parquet_filepaths = {}

    # Move files to new structure
    for aspect_ratio, file_paths in aspect_ratio_parquets.items():
        # Create new directory for the aspect ratio if it doesn't exist
        new_dir = os.path.join(base_dir, f"reorged_{DATASET}")
        new_dir = os.path.join(new_dir, aspect_ratio)

        os.makedirs(new_dir, exist_ok=True)

        # Move each file to the new directory
        for file_path in file_paths:
            shutil.move(file_path, new_dir)
            new_filepath = os.path.join(new_dir, os.path.basename(file_path))
            new_parquet_filepaths[new_filepath] = aspect_ratio

    return new_parquet_filepaths


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.float16)
def generate_captions(model, tokenizer, images):
    """Assumes all images are the same shape"""  
    captions = model.batch_answer(
        images=images,
        prompts=["Caption this image."] * len(images),
        tokenizer=tokenizer,
    )

    return captions

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.float16)
def generate_embeddings(model, tokenizer, captions):
    """Captions should be a list of strings"""
    texts = tokenizer(captions, context_length=model.context_length)
    text_embeddings = model.encode_text(texts)

    return text_embeddings

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.float16)
def generate_latents(model, images):
    """Assumes all images are the same shape"""
    latents = model.encode(images).latent_dist.sample()

    return latents

def process_parquets(rank, world_size, queue, moondream, moondream_tokenizer, vae, siglip_model, siglip_tokenizer, bucket_manager):
    while True:
        try:
            parquet_filepath = queue.get(timeout=10)
        except:
            break

        # Load parquet file
        df = pq.read_table(parquet_filepath)
        
        # Get ideal resolution
        original_resolution = batch[IMAGE_COLUMN_NAME][0].size
        new_resolution = bucket_manager.get_ideal_resolution(original_resolution)
        
        new_rows = []
        
        for batch_start in range(0, len(df), BS):
            batch = df.slice(batch_start, BS)
            
            # Resize images
            images = [resize_and_crop(img, new_resolution) for img in batch[IMAGE_COLUMN_NAME]]
            image_tensors = torch.stack([preprocess_image(img) for img in images])
            
            # Generate captions
            captions = generate_captions(moondream, moondream_tokenizer, image_tensors)
            
            # Generate VAE latents
            latents = generate_latents(vae, image_tensors)
            
            # Generate text embeddings
            text_embeddings = generate_embeddings(siglip_model, siglip_tokenizer, captions)
            
            # Add processed outputs to new rows
            for i in range(len(batch)):
                new_row = {
                    'image_id': batch[IMAGE_ID_COLUMN_NAME][i].as_py(),
                    'caption': captions[i],
                    'vae_latent': latents[i].cpu().numpy(),
                    'text_embedding': text_embeddings[i].cpu().numpy()
                }
                new_rows.append(new_row)
        
        # Create new parquet file
        new_df = pa.Table.from_pylist(new_rows)
        new_parquet_dir = os.path.join(DATASET_DIR_BASE, f"preprocessed_{DATASET}", f"{new_resolution[1]}x{new_resolution[0]}")
        os.makedirs(new_parquet_dir, exist_ok=True)
        new_parquet_path = os.path.join(new_parquet_dir, os.path.basename(parquet_filepath))
        pq.write_table(new_df, new_parquet_path)

def init_process(rank, world_size, queue, fn, backend='nccl'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    model_id = "vikhyatk/moondream2"
    revision = "2024-08-26"
    moondream = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=revision, torch_dtype=torch.float16, cache_dir=f"{MODELS_DIR_BASE}/moondream"
    ).to(rank)
    moondream_tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    vae = AutoencoderKL.from_pretrained(f"{VAE_HF_NAME}", torch_dtype=torch.float16, cache_dir=f"{MODELS_DIR_BASE}/vae").to(rank)

    siglip_model, _ = create_model_from_pretrained(SIGLIP_HF_NAME, precision="fp16", cache_dir=f"{MODELS_DIR_BASE}/siglip")
    siglip_model = siglip_model.to(rank)
    siglip_tokenizer = get_tokenizer(SIGLIP_HF_NAME)

    bucket_manager = BucketManager()

    fn(rank, world_size, queue, moondream, moondream_tokenizer, vae, siglip_model, siglip_tokenizer, bucket_manager)

def process_dataset():
    parquet_paths = collect_and_move_parquets(DATASET_DIR_BASE)
    
    world_size = torch.cuda.device_count()
    queue = Queue()
    
    for path in parquet_paths:
        queue.put(path)
    
    processes = []
    for rank in range(world_size):
        p = Process(target=init_process, args=(rank, world_size, queue, process_parquets))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

if __name__ == "__main__":
    process_dataset()

