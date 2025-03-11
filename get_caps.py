from datasets import load_dataset
import json
from collections import defaultdict

if __name__ == "__main__":
    ds = load_dataset("CaptionEmporium/flickr-megalith-10m-internvl2-multi-caption", cache_dir="datasets", split="train")

    print(len(ds))

    caps = defaultdict(dict)

    def add_caps(row):
        url = row["url_highres"]
        cap_internlm = row["caption_internlm2"]
        cap_florence = row["caption_florence2"]
        cap_sharecap = row["caption_sharecap"]
        cap_internlm_short = row["caption_internlm2_short"]
        cap_florence_short = row["caption_florence2_short"]
        cap_sharecap_short = row["caption_sharecap_short"]

        caps[url]["cap_internlm"] = cap_internlm
        caps[url]["cap_florence"] = cap_florence
        caps[url]["cap_sharecap"] = cap_sharecap
        caps[url]["cap_internlm_short"] = cap_internlm_short
        caps[url]["cap_florence_short"] = cap_florence_short
        caps[url]["cap_sharecap_short"] = cap_sharecap_short

        return row

    ds.map(add_caps)

    output_file = "captions.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(caps, f)

 