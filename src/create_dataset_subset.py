import os
import shutil
from pathlib import Path
from collections import defaultdict
import random

def subset_dataset(source_dir, target_dir, images_per_age=3):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    age_count = defaultdict(int)
    age_files = defaultdict(list)

    print(f"Scanning source directory: {source_dir}")
    for img_path in source_dir.glob('*.jpg'):
        try:
            age = int(img_path.stem.split('_')[0])
            age_files[age].append(img_path)
        except ValueError:
            print(f"Failed to extract age from filename: {img_path}")
            continue

    total_copied = 0
    for age, files in age_files.items():
        selected_files = random.sample(files, min(images_per_age, len(files)))
        for file in selected_files:
            new_path = target_dir / file.name
            shutil.copy2(file, new_path)
            total_copied += 1

        print(f"Copied {len(selected_files)} images for age {age}")

    print(f"Total images copied: {total_copied}")
    print(f"Images copied to: {target_dir}")

if __name__ == "__main__":
    source_dir = "../AgeDetection/data/UTKFace"  # Update this to your source UTKFace directory
    target_dir = "../AgeDetection/data/UTKFace_subset"  # This will be the new directory with the subset
    subset_dataset(source_dir, target_dir, images_per_age=3)