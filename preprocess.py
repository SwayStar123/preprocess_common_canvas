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
from torch.multiprocessing import Queue, Process, Value, Event, set_start_method
import torch.distributed as dist
from tqdm import tqdm
from huggingface_hub import HfFileSystem, hf_hub_download, HfApi
import threading
import io

DATASET = "commoncatalog-cc-by-nd"
DATASET_DIR_BASE = "../datasets"
MODELS_DIR_BASE = "../models"
VAE_HF_NAME = "madebyollin/sdxl-vae-fp16-fix"
SIGLIP_HF_NAME = "hf-hub:timm/ViT-SO400M-14-SigLIP-384"
IMAGE_COLUMN_NAME = "jpg"
IMAGE_ID_COLUMN_NAME = "key"
BS = 32
# Deletes original parquet files after processing
DELETE_AFTER_PROCESSING = True
UPLOAD_TO_HUGGINGFACE = True
USERNAME = "SwayStar123"
PARTITIONS = [0,1,2,3,4,5,6,7,8,9]

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
def generate_embeddings(model, tokenizer, captions, device):
    """Captions should be a list of strings"""
    texts = tokenizer(captions, context_length=model.context_length).to(device)
    text_embeddings = model.encode_text(texts)

    return text_embeddings

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.float16)
def generate_latents(model, images):
    """Assumes all images are the same shape"""
    latents = model.encode(images).latent_dist.sample()

    return latents

def get_processed_files(tracking_file):
    processed_files = set()
    if os.path.exists(tracking_file):
        with open(tracking_file, 'r') as f:
            processed_files = set(line.strip() for line in f)
    return processed_files

def add_processed_file(tracking_file, filename):
    with open(tracking_file, 'a') as f:
        f.write(f"{filename}\n")

def download_and_queue_parquets(queue, download_progress, total_files, download_complete_event, tracking_file, pause_event):
    fs = HfFileSystem()
    parquet_files = []
    for partition in PARTITIONS:
        p_f = fs.glob(f"datasets/common-canvas/{DATASET}/{str(partition)}/**/*.parquet")
        parquet_files.extend(p_f)
    
    processed_files = get_processed_files(tracking_file)
    
    unprocessed_files = [file for file in parquet_files if os.path.basename(file) not in processed_files]
    total_files.value = len(unprocessed_files)
    
    for file in unprocessed_files:
        # Check if download should be paused
        pause_event.wait()
        
        local_file = hf_hub_download(repo_id=f"common-canvas/{DATASET}", filename=file.split(f"{DATASET}/")[-1], repo_type="dataset", local_dir=f"{DATASET_DIR_BASE}/{DATASET}")
        queue.put(local_file)
        with download_progress.get_lock():
            download_progress.value += 1
    
    download_complete_event.set()

def upload_to_huggingface(file_path, path_in_repo):
    api = HfApi()
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=path_in_repo,
        repo_id=f"{USERNAME}/preprocessed_{DATASET}",
        repo_type="dataset",
    )

def process_parquets(rank, world_size, queue, process_progress, total_files, total_images, download_complete_event, tracking_file):
    # Move CUDA initialization inside this function
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    
    model_id = "vikhyatk/moondream2"
    revision = "2024-08-26"
    moondream = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=revision, torch_dtype=torch.float16, cache_dir=f"{MODELS_DIR_BASE}/moondream"
    ).to(device)
    moondream_tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    vae = AutoencoderKL.from_pretrained(f"{VAE_HF_NAME}", torch_dtype=torch.float16, cache_dir=f"{MODELS_DIR_BASE}/vae").to(device)

    siglip_model, _ = create_model_from_pretrained(SIGLIP_HF_NAME, precision="fp16", cache_dir=f"{MODELS_DIR_BASE}/siglip")
    siglip_model = siglip_model.to(device)
    siglip_tokenizer = get_tokenizer(SIGLIP_HF_NAME)

    bucket_manager = BucketManager()
    
    while not (download_complete_event.is_set() and queue.empty()):
        try:
            parquet_filepath = queue.get(timeout=10)
        except:
            continue

        if os.path.basename(parquet_filepath) in get_processed_files(tracking_file):
            continue

        # Load parquet file
        df = pq.read_table(parquet_filepath)
        
        # Get ideal resolution
        sample_image_bytes = df[IMAGE_COLUMN_NAME][0].as_py()
        sample_image = bytes_to_pil_image(sample_image_bytes)
        original_resolution = sample_image.size
        new_resolution = bucket_manager.get_ideal_resolution(original_resolution)
        
        new_rows = []
        
        for batch_start in range(0, len(df), BS):
            batch = df.slice(batch_start, BS)
            
            # Resize images
            images = [bytes_to_pil_image(img.as_py()) for img in batch[IMAGE_COLUMN_NAME]]
            resized_images = [resize_and_crop(img, new_resolution) for img in images]
            image_tensors = torch.stack([preprocess_image(img) for img in resized_images]).to(device)
            
            # Generate captions
            captions = generate_captions(moondream, moondream_tokenizer, images)
            
            # Generate VAE latents
            latents = generate_latents(vae, image_tensors)
            
            # Generate text embeddings
            text_embeddings = generate_embeddings(siglip_model, siglip_tokenizer, captions, device)
            
            # Add only processed outputs to new rows
            for i in range(len(batch)):
                new_row = {
                    'image_id': batch[IMAGE_ID_COLUMN_NAME][i].as_py(),
                    'caption': captions[i],
                    'vae_latent': latents[i].cpu().numpy().tobytes(),  # Convert to bytes
                    'vae_latent_shape': latents[i].shape,  # Store shape separately
                    'text_embedding': text_embeddings[i].cpu().numpy().tobytes(),  # Convert to bytes
                    'text_embedding_shape': text_embeddings[i].shape,  # Store shape separately
                }
                new_rows.append(new_row)
            
            # Update total images processed
            with total_images.get_lock():
                total_images.value += len(batch)
        
        # Create new parquet file
        new_df = pa.Table.from_pylist(new_rows)
        new_parquet_dir = os.path.join(DATASET_DIR_BASE, f"preprocessed_{DATASET}", f"{new_resolution[1]}x{new_resolution[0]}")
        os.makedirs(new_parquet_dir, exist_ok=True)
        new_parquet_path = os.path.join(new_parquet_dir, os.path.basename(parquet_filepath))
        pq.write_table(new_df, new_parquet_path)

        if UPLOAD_TO_HUGGINGFACE:
            name_in_repo = f"{new_resolution[1]}x{new_resolution[0]}/{os.path.basename(parquet_filepath)}"
            upload_to_huggingface(new_parquet_path, name_in_repo)

        add_processed_file(tracking_file, os.path.basename(parquet_filepath))

        # Delete original parquet file if DELETE_AFTER_PROCESSING is True
        if DELETE_AFTER_PROCESSING:
            if UPLOAD_TO_HUGGINGFACE:
                os.remove(new_parquet_path) 
            os.remove(parquet_filepath)

        # Update process progress
        with process_progress.get_lock():
            process_progress.value += 1

def process_dataset():
    # Set start method to 'spawn'
    set_start_method('spawn', force=True)

    world_size = torch.cuda.device_count()
    queue = Queue()
    download_progress = Value('i', 0)
    process_progress = Value('i', 0)
    total_files = Value('i', 0)
    total_images = Value('i', 0)
    download_complete_event = Event()
    pause_event = Event()
    pause_event.set()  # Start in a non-paused state
    
    # Create tracking file path
    tracking_file = os.path.join(DATASET_DIR_BASE, f"{DATASET}_processed_files.txt")
    
    # Start the download thread
    download_thread = threading.Thread(target=download_and_queue_parquets, args=(queue, download_progress, total_files, download_complete_event, tracking_file, pause_event))
    download_thread.start()
    
    processes = []
    for rank in range(world_size):
        p = Process(target=process_parquets, args=(rank, world_size, queue, process_progress, total_files, total_images, download_complete_event, tracking_file))
        p.start()
        processes.append(p)
    
    # Progress bars with img/s estimate
    start_time = time.time()
    with tqdm(total=None, desc="Downloading files", position=1) as download_pbar, \
         tqdm(total=None, desc="Processing files", position=2) as process_pbar:
        last_images = 0
        while not (download_complete_event.is_set() and queue.empty()):
            current_download_progress = download_progress.value
            current_process_progress = process_progress.value
            current_images = total_images.value
            elapsed_time = time.time() - start_time
            
            # Update totals if changed
            if download_pbar.total != total_files.value:
                download_pbar.total = total_files.value
                process_pbar.total = total_files.value
            
            # Update progress bars
            download_pbar.n = current_download_progress
            process_pbar.n = current_process_progress
            
            # Calculate images per second
            if elapsed_time > 0:
                img_per_second = current_images / elapsed_time
                new_images = current_images - last_images
                if new_images > 0:
                    img_per_second = (img_per_second + new_images) / 2  # Average of overall and recent speed
                
                process_pbar.set_postfix({'img/s': f'{img_per_second:.2f}'})
            
            # Pause or resume download based on queue size
            if queue.qsize() > world_size * 2:
                pause_event.clear()  # Pause download
            else:
                pause_event.set()  # Resume download
            
            download_pbar.refresh()
            process_pbar.refresh()
            
            last_images = current_images
            time.sleep(1)
    
    download_thread.join()
    for p in processes:
        p.join()

if __name__ == "__main__":
    process_dataset()