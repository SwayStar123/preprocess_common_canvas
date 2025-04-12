import os
import shutil
import tarfile
import tempfile
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import AutoencoderDC
import torch
import time
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torch.multiprocessing import Queue, Process, Value, Event, set_start_method
import threading
import io
import json
from collections import defaultdict
from tqdm import tqdm
from huggingface_hub import HfFileSystem, hf_hub_download, HfApi

DATASET_OWNER = "drawthingsai"
DATASET = "megalith-10m"
DATASET_DIR_BASE = "./datasets"
MODELS_DIR_BASE = "../../models"
AE_HF_NAME = "mit-han-lab/dc-ae-f64c128-mix-1.0-diffusers"
IMAGE_COLUMN_NAME = "jpg"
IMAGE_ID_COLUMN_NAME = "key"
BS = 32
DELETE_AFTER_PROCESSING = True
UPLOAD_TO_HUGGINGFACE = True
UPLOAD_DS_PREFIX = "preprocessed_DCAE-f64_1024_"
USERNAME = "SwayStar123"
TARGET_RES = 1024

class BucketManager:
    def __init__(self, max_size=(TARGET_RES,TARGET_RES), divisible=64, min_dim=TARGET_RES//2, base_res=(TARGET_RES,TARGET_RES), dim_limit=TARGET_RES*2, debug=False):
        self.max_size = max_size
        self.f = 8
        self.max_tokens = (max_size[0]/self.f) * (max_size[1]/self.f)
        self.div = divisible
        self.min_dim = min_dim
        self.dim_limit = dim_limit
        self.base_res = base_res
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

def preprocess_images_batch(images, target_size, device='cuda'):
    """
    Process a batch of images on GPU in one operation:
    1) Convert PIL images to tensors
    2) Resize to preserve aspect ratio
    3) Center crop to target size
    4) Normalize to [-1, 1]
    
    Args:
        images (list): List of PIL images
        target_size (tuple): Target width and height as (w, h)
        device (str): Device to process on, default 'cuda'
        
    Returns:
        torch.Tensor: Batch of processed images [B, C, H, W]
    """
    # Target dimensions
    target_w, target_h = target_size
    
    # Convert all images to RGB first
    rgb_images = [img.convert('RGB') if img.mode != 'RGB' else img for img in images]
    
    # Create a batch of tensors [B, C, H, W]
    batch = []
    for img in rgb_images:
        # Convert to tensor [C, H, W]
        img_tensor = torch.as_tensor(np.array(img), device=device).permute(2, 0, 1).float() / 255.0
        batch.append(img_tensor)
    
    # Stack into batch
    batch_tensor = torch.stack(batch)  # [B, C, H, W]
    
    # Calculate resize dimensions for each image in batch
    b, c, h, w = batch_tensor.shape
    
    # Calculate aspect ratios
    aspect_ratios = w / h  # Assuming all images in batch have same shape due to bucketing
    target_aspect_ratio = target_w / target_h
    
    # Decide how to resize
    if abs(aspect_ratios - target_aspect_ratio) < 1e-6:
        # Aspect ratios match, just resize directly
        new_h, new_w = target_h, target_w
    else:
        if aspect_ratios > target_aspect_ratio:
            # Images are wider than target -> match target height
            new_h = target_h
            new_w = int(aspect_ratios * new_h)
        else:
            # Images are taller than target -> match target width
            new_w = target_w
            new_h = int(new_w / aspect_ratios)
    
    # Resize batch using interpolate
    resized_batch = F.interpolate(
        batch_tensor,
        size=(new_h, new_w),
        mode='bicubic',
        align_corners=False
    )
    
    # Center crop to target size
    if new_w > target_w:
        start_w = (new_w - target_w) // 2
        resized_batch = resized_batch[:, :, :, start_w:start_w + target_w]
    if new_h > target_h:
        start_h = (new_h - target_h) // 2
        resized_batch = resized_batch[:, :, start_h:start_h + target_h, :]
    
    # Normalize to [-1, 1]
    normalized_batch = resized_batch * 2.0 - 1.0
    
    return normalized_batch

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def generate_latents(model, images):
    """Assumes all images are the same shape"""
    latents = model.encode(images).latent
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

def download_with_retry(repo_id, filename, repo_type, local_dir, max_retries=500, retry_delay=60):
    for attempt in range(max_retries):
        try:
            return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type, local_dir=local_dir)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Download failed. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                print(f"Max retries reached. Skipping file: {filename}")
                return None

def download_and_queue_tars(queue, download_progress, total_files, download_complete_event, tracking_file, pause_event):
    fs = HfFileSystem()

    tar_files = fs.glob(f"datasets/{DATASET_OWNER}/{DATASET}/*.tar")

    processed_files = get_processed_files(tracking_file)
    processed_files_on_hub = fs.glob(f"datasets/{USERNAME}/{UPLOAD_DS_PREFIX}{DATASET}/data/*.parquet")
    processed_files_on_hub = set(os.path.basename(file).replace('.parquet', '.tar') for file in processed_files_on_hub)

    unprocessed_files = [file for file in tar_files if os.path.basename(file) not in processed_files and os.path.basename(file) not in processed_files_on_hub]

    total_files.value = len(unprocessed_files)
    
    for file in unprocessed_files:
        # Check if download should be paused
        pause_event.wait()
        
        local_file = download_with_retry(
            repo_id=f"{DATASET_OWNER}/{DATASET}",
            filename=file.split(f"{DATASET}/")[-1],
            repo_type="dataset",
            local_dir=f"{DATASET_DIR_BASE}/{DATASET}"
        )
        
        if local_file:
            queue.put(local_file)
            with download_progress.get_lock():
                download_progress.value += 1
        else:
            # If download failed after all retries, fail the program
            raise Exception(f"Failed to download {file} after maximum retries")
    
    download_complete_event.set()

def upload_with_retry(file_path, name_in_repo, max_retries=500, retry_delay=60):
    api = HfApi()
    for attempt in range(max_retries):
        try:
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=f"data/{name_in_repo}",
                repo_id=f"{USERNAME}/{UPLOAD_DS_PREFIX}{DATASET}",
                repo_type="dataset",
            )
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Upload failed: {str(e)}. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                print(f"Max retries reached. Failed to upload: {file_path}")
                return False

def upload_worker(upload_queue, upload_complete_event):
    while not (upload_complete_event.is_set() and upload_queue.empty()):
        try:
            file_path, name_in_repo = upload_queue.get(timeout=10)
        except:
            continue

        if upload_with_retry(file_path, name_in_repo):
            print(f"Successfully uploaded: {file_path}")
            if DELETE_AFTER_PROCESSING:
                os.remove(file_path)
        else:
            print(f"Failed to upload: {file_path}")
            # If upload failed after all retries, raise an exception
            raise Exception(f"Failed to upload {file_path} after maximum retries")

def get_short_captions(url, captions):
    """Extract short captions for an image URL"""
    if url in captions:
        cap_data = captions[url]
        return [
            cap_data.get("cap_internlm_short", ""),
            cap_data.get("cap_florence_short", ""),
            cap_data.get("cap_sharecap_short", "")
        ]
    return ["", "", ""]

def extract_and_process_batch(images, image_ids, image_urls, ae, device, bucket_manager, captions):
    """Process a batch of images with the same shape."""
    # Get ideal resolution for this batch
    original_resolution = images[0].size
    new_resolution = bucket_manager.get_ideal_resolution(original_resolution)
    
    # Batch process the images
    image_tensors = preprocess_images_batch(images, new_resolution, device)
    
    # Generate AE latents
    latents = generate_latents(ae, image_tensors).to(torch.float16)
    
    # Create rows
    new_rows = []
    for i in range(len(images)):
        image_id = image_ids[i]
        image_url = image_urls[i]
        
        # Get short captions from the external JSON file
        image_captions = get_short_captions(image_url, captions)
        
        new_row = {
            'image_id': image_id,
            'caption': image_captions,
            'ae_latent': latents[i].cpu().numpy().flatten(),
            'ae_latent_shape': latents[i].shape,
        }
        new_rows.append(new_row)
    
    return new_rows, len(images)

def process_tar_file(tar_filepath, ae, device, bucket_manager, upload_queue, tracking_file, total_images=None, captions=None):
    # Ensure the base directory exists
    os.makedirs(DATASET_DIR_BASE, exist_ok=True)
    
    # Use a temporary directory within DATASET_DIR_BASE
    with tempfile.TemporaryDirectory(dir=DATASET_DIR_BASE) as extract_dir:
        # Extract tar file contents
        with tarfile.open(tar_filepath, 'r') as tar:
            tar.extractall(path=extract_dir)
        
        # Find all JPG files and their corresponding JSON files
        image_files = []
        json_files = []
        
        for root, _, files in os.walk(extract_dir):
            for file in files:
                if file.endswith('.jpg'):
                    image_files.append(os.path.join(root, file))
                elif file.endswith('.json') and not file.startswith('.'):  # Exclude the special PaxHeader files
                    json_files.append(os.path.join(root, file))
        
        # Check for valid files
        if not image_files or not json_files:
            print(f"No valid files found in {tar_filepath}")
            if DELETE_AFTER_PROCESSING:
                os.remove(tar_filepath)
            return 0
        
        # Map JSON files to image files
        file_pairs = []
        
        for img_path in image_files:
            img_basename = os.path.basename(img_path).split('.')[0]
            matching_json = None
            
            for json_path in json_files:
                json_basename = os.path.basename(json_path).split('.')[0]
                if json_basename == img_basename:
                    matching_json = json_path
                    break
            
            if matching_json:
                # Verify the JSON has a valid key and url
                with open(matching_json, 'r') as f:
                    metadata = json.load(f)
                    if metadata.get('key') and metadata.get('url'):
                        file_pairs.append((img_path, matching_json))
        
        if not file_pairs:
            print(f"No valid file pairs found in {tar_filepath}")
            if DELETE_AFTER_PROCESSING:
                os.remove(tar_filepath)
            return 0
        
        # Process images using shape batching
        shape_batches = defaultdict(lambda: {'images': [], 'image_ids': [], 'image_urls': []})
        all_rows = []
        total_processed = 0
        
        # First pass: organize images by shape
        for img_path, json_path in file_pairs:
            try:
                # Get image ID and URL from JSON
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
                    image_id = metadata.get('key')
                    image_url = metadata.get('url')
                
                if not image_id or not image_url:
                    continue
                    
                # Try to fully load the image to verify it's not corrupted
                img = Image.open(img_path)
                img.load()  # This will force PIL to load the entire image and verify it's not truncated
                
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                # Get resolution bucket
                resolution = img.size
                if resolution[0] * resolution[1] >= TARGET_RES * TARGET_RES:
                    shape_batches[resolution]['images'].append(img)
                    shape_batches[resolution]['image_ids'].append(image_id)
                    shape_batches[resolution]['image_urls'].append(image_url)
                
                # If we have enough images of this shape, process them
                if len(shape_batches[resolution]['images']) >= BS:
                    batch = {
                        'images': shape_batches[resolution]['images'][:BS],
                        'image_ids': shape_batches[resolution]['image_ids'][:BS],
                        'image_urls': shape_batches[resolution]['image_urls'][:BS]
                    }
                    
                    try:
                        # Process batch
                        new_rows, batch_processed = extract_and_process_batch(
                            batch['images'], batch['image_ids'], batch['image_urls'], 
                            ae, device, bucket_manager, captions
                        )
                        
                        all_rows.extend(new_rows)
                        total_processed += batch_processed
                        
                        # Update total_images counter in real-time
                        if total_images is not None:
                            with total_images.get_lock():
                                total_images.value += batch_processed
                    except Exception as e:
                        print(f"Error processing batch: {str(e)}")
                    
                    # Remove processed images from the batch
                    shape_batches[resolution]['images'] = shape_batches[resolution]['images'][BS:]
                    shape_batches[resolution]['image_ids'] = shape_batches[resolution]['image_ids'][BS:]
                    shape_batches[resolution]['image_urls'] = shape_batches[resolution]['image_urls'][BS:]
            except Exception as e:
                print(f"Error processing file {img_path}: {str(e)}")
        
        # Process remaining items in shape batches
        for resolution, batch in shape_batches.items():
            if len(batch['images']) > 0:
                try:
                    # Process batch
                    new_rows, batch_processed = extract_and_process_batch(
                        batch['images'], batch['image_ids'], batch['image_urls'], 
                        ae, device, bucket_manager, captions
                    )
                    
                    all_rows.extend(new_rows)
                    total_processed += batch_processed
                    
                    # Update total_images counter in real-time
                    if total_images is not None:
                        with total_images.get_lock():
                            total_images.value += batch_processed
                except Exception as e:
                    print(f"Error processing final batch for resolution {resolution}: {str(e)}")
        
        # Create and save parquet if we have processed rows
        if all_rows:
            schema = pa.schema([
                ('image_id', pa.string()),
                ('caption', pa.list_(pa.string())),
                ('ae_latent', pa.list_(pa.float16())),
                ('ae_latent_shape', pa.list_(pa.int64())),
            ])
            
            new_df = pa.Table.from_pylist(all_rows, schema=schema)
            new_parquet_dir = os.path.join(DATASET_DIR_BASE, f"{UPLOAD_DS_PREFIX}{DATASET}")
            os.makedirs(new_parquet_dir, exist_ok=True)
            
            # Use tar filename to create unique parquet filename
            tar_base = os.path.basename(tar_filepath).replace('.tar', '')
            parquet_filename = f"{tar_base}.parquet"
            new_parquet_path = os.path.join(new_parquet_dir, parquet_filename)
            pq.write_table(new_df, new_parquet_path)
            
            if UPLOAD_TO_HUGGINGFACE:
                upload_queue.put((new_parquet_path, parquet_filename))
        
        # Delete the tar file (temp dir is auto-cleaned)
        if DELETE_AFTER_PROCESSING:
            os.remove(tar_filepath)
        
        add_processed_file(tracking_file, os.path.basename(tar_filepath))
        
        return total_processed

def process_tars(rank, world_size, queue, process_progress, total_files, total_images, download_complete_event, tracking_file, upload_queue):
    # Move CUDA initialization inside this function
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    ae = AutoencoderDC.from_pretrained(f"{AE_HF_NAME}", torch_dtype=torch.float16, cache_dir=f"{MODELS_DIR_BASE}/dc_ae").to(device)

    bucket_manager = BucketManager()
    
    with open('captions.json', 'r') as f:
        captions = json.load(f)
        
    while not (download_complete_event.is_set() and queue.empty()):
        try:
            tar_filepath = queue.get(timeout=10)
        except:
            continue

        if os.path.basename(tar_filepath) in get_processed_files(tracking_file):
            continue

        # Process tar file
        images_processed = process_tar_file(tar_filepath, ae, device, bucket_manager, upload_queue, tracking_file, total_images, captions)
        
        # Update process progress
        with process_progress.get_lock():
            process_progress.value += 1

def process_dataset():
    # Set start method to 'spawn'
    set_start_method('spawn', force=True)

    world_size = torch.cuda.device_count()
    queue = Queue()
    upload_queue = Queue()
    download_progress = Value('i', 0)
    process_progress = Value('i', 0)
    total_files = Value('i', 0)
    total_images = Value('i', 0)
    download_complete_event = Event()
    upload_complete_event = Event()
    pause_event = Event()
    pause_event.set()  # Start in a non-paused state
    
    # Create tracking file path
    tracking_file = os.path.join(DATASET_DIR_BASE, f"{DATASET}_processed_files.txt")
    
    # Start the download thread
    download_thread = threading.Thread(target=download_and_queue_tars, args=(queue, download_progress, total_files, download_complete_event, tracking_file, pause_event))
    download_thread.start()

    # Start the upload thread if enabled
    if UPLOAD_TO_HUGGINGFACE:
        upload_thread = threading.Thread(target=upload_worker, args=(upload_queue, upload_complete_event))
        upload_thread.start()
    
    processes = []
    for rank in range(world_size):
        p = Process(target=process_tars, args=(rank, world_size, queue, process_progress, total_files, total_images, download_complete_event, tracking_file, upload_queue))
        p.start()
        processes.append(p)
    
    # Progress bars with img/s estimate
    start_time = time.time()
    last_update_time = start_time
    with tqdm(total=None, desc="Downloading files", position=1) as download_pbar, \
         tqdm(total=None, desc="Processing files", position=2) as process_pbar:
        last_images = 0
        rates = []  # Keep track of recent rates for smoothing
        
        while not (download_complete_event.is_set() and queue.empty()):
            current_download_progress = download_progress.value
            current_process_progress = process_progress.value
            current_images = total_images.value
            current_time = time.time()
            elapsed_time = current_time - start_time
            time_since_last_update = current_time - last_update_time
            
            # Update totals if changed
            if download_pbar.total != total_files.value:
                download_pbar.total = total_files.value
                process_pbar.total = total_files.value
            
            # Update progress bars
            download_pbar.n = current_download_progress
            process_pbar.n = current_process_progress
            
            # Calculate images per second
            if time_since_last_update >= 0.5:
                images_since_last_update = current_images - last_images
                if images_since_last_update > 0:
                    current_rate = images_since_last_update / time_since_last_update
                    
                    # Add to rates list for smoothing (keep last 5)
                    rates.append(current_rate)
                    if len(rates) > 5:
                        rates.pop(0)
                    
                    # Calculate smoothed rate
                    smoothed_rate = sum(rates) / len(rates)
                    
                    # Calculate ETA
                    if current_process_progress > 0 and total_files.value > 0:
                        images_per_file = current_images / current_process_progress
                        remaining_files = total_files.value - current_process_progress
                        remaining_images = remaining_files * images_per_file
                        eta_seconds = remaining_images / smoothed_rate if smoothed_rate > 0 else 0
                        
                        # Format ETA
                        if eta_seconds < 60:
                            eta_str = f"{eta_seconds:.0f}s"
                        elif eta_seconds < 3600:
                            eta_str = f"{eta_seconds/60:.1f}m"
                        else:
                            eta_str = f"{eta_seconds/3600:.1f}h"
                        
                        process_pbar.set_postfix({
                            'img/s': f'{smoothed_rate:.2f}', 
                            'ETA': eta_str,
                            'images': f'{current_images}'
                        })
                    else:
                        process_pbar.set_postfix({
                            'img/s': f'{smoothed_rate:.2f}',
                            'images': f'{current_images}'
                        })
                
                last_images = current_images
                last_update_time = current_time
            
            # Pause or resume download based on queue size
            if queue.qsize() > world_size * 2:
                pause_event.clear()  # Pause download
            else:
                pause_event.set()  # Resume download
            
            download_pbar.refresh()
            process_pbar.refresh()
            
            time.sleep(0.2)
    
    download_thread.join()
    for p in processes:
        p.join()

    if UPLOAD_TO_HUGGINGFACE:
        upload_complete_event.set()
        upload_thread.join()

if __name__ == "__main__":
    process_dataset()