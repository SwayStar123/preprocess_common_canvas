import os
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch

def collect_and_move_parquets(base_dir):
    # aspect ratio to list of parquet filepaths
    aspect_ratio_parquets = {}

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

    # Move files to new structure
    for aspect_ratio, file_paths in aspect_ratio_parquets.items():
        # Create new directory for the aspect ratio if it doesn't exist
        new_dir = os.path.join(base_dir, aspect_ratio)
        os.makedirs(new_dir, exist_ok=True)

        # Move each file to the new directory
        for file_path in file_paths:
            shutil.move(file_path, new_dir)

def generate_captions(model, batch):
    """Assumes all images are the same shape"""
    images = batch["jpg"]
    
    captions = model.batch_answer(
        images=images,
        prompts=["Caption this image."] * len(images),
        tokenizer=tokenizer,
    )

    return captions

if __name__ == "__main__":
    base_directory = "path/to/your/base/directory"
    collect_and_move_parquets(base_directory)

    model_id = "vikhyatk/moondream2"
    revision = "2024-08-26"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=revision, torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)