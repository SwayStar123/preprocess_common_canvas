Preprocesses a common canvas dataset (you can easily edit this to change to the CC-BY, CC-BY-SA, CC-BY-NC, CC-BY-ND, CC-BY-NC-ND, or CC-BY-NC-SA datasets, CC-BY by default).

1. Generates new captions using moondream2.
2. Generates latents using SDXL VAE.

Requires the whole dataset to be downloaded

First it reorganises to remove the 10 top layer folders, then since we resize all images to the same pixel count, we remove the resolutions folders, only keeping the aspect ratio folders.

Folder structure goes from:
```
0
    least_dim_range=256-512
        aspect_ratio_bucket=1-1
            00001.parquet
            00002.parquet
            ...
            12345.parquet
        aspect_ratio_bucket=1-2
        ...
        aspect_ratio_bucket=19-13
    least_dim_range=512-768
    ...
    least_dim_range=2048-4096
1
    ...
...
9
    ...
```

to

```
aspect_ratio_bucket=1-1
    00001.parquet
    00002.parquet
    ...
    12345.parquet
aspect_ratio_bucket=1-2
...
aspect_ratio_bucket=19-13
```