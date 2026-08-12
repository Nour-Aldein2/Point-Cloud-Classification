import os
import random
import shutil
from pathlib import Path

random.seed(42)
src = Path("/Users/nour/Downloads/ModelNet10/")
dst = Path("/Users/nour/Downloads/ModelNet10_3_splits/")

splits = [0.8, 0.05, 0.15]
all_files = {c.name: [] for c in src.iterdir() if c.is_dir()}

## Collect the paths for the data files
for c in all_files.keys():
    category_dir = src/c
    data = list((category_dir/"train").glob("*.off")) + list((category_dir/"test").glob("*.off"))
    random.shuffle(data)
    all_files[c] = data

# Create new train / val / test splits
for c in all_files:
    category_dir = dst / c

    for split_name in ["train", "val", "test"]:
        (category_dir / split_name).mkdir(parents=True, exist_ok=True)

    files = all_files[c]
    n = len(files)

    train_end = int(splits[0] * n)
    val_end = train_end + int(splits[1] * n)

    train_split = files[:train_end]
    val_split = files[train_end:val_end]
    test_split = files[val_end:]

    # Copy files
    for f in train_split:
        shutil.copy2(f, category_dir / "train" / f.name)

    for f in val_split:
        shutil.copy2(f, category_dir / "val" / f.name)

    for f in test_split:
        shutil.copy2(f, category_dir / "test" / f.name)

    # Sanity check
    print(
        f"{c:15s} | "
        f"total={n:4d} | "
        f"train={len(train_split):4d} | "
        f"val={len(val_split):4d} | "
        f"test={len(test_split):4d}"
    )