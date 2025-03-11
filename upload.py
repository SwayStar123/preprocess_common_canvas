from datasets import load_from_disk
import json
import tqdm

if __name__ == "__main__":
    ds = load_from_disk("imagenet1k_eqsdxlvae_latents_train")

    ds = ds.add_column("caption", [""] * len(ds))

    # load combined_imagenet21k_captions.json
    with open("combined_imagenet21k_captions.json", "r") as f:
        captions = json.load(f)

    def attach_caption(row):
        row["caption"] = captions[row["image_id"]]["caption"]
        return row

    ds = ds.map(attach_caption)

    ds.push_to_hub("SwayStar123/imagenet1k_eqsdxlvae_latents", split="train")
