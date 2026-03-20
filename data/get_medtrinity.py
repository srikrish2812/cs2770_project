from datasets import load_dataset
import os

ds = load_dataset(
    "UCSC-VLAA/MedTrinity-25M",
    "25M_demo",
    token=os.environ['HF_TOKEN'],
    cache_dir='/ix/cs2770_2026s/abn80/cs2770_project/data/medtrinity-25m'
)
print(ds)