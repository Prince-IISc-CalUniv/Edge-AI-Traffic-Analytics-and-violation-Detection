import os
import shutil
import random
import yaml
from pathlib import Path

random.seed(42)

BASE_DIR = Path("edge-ai-monitoring")
RAW_POTHOLE = BASE_DIR / "data" / "raw" / "pothole"
RAW_TS_IMAGES = BASE_DIR / "data" / "raw" / "traffic_sign" / "Dataset" / "images"
RAW_TS_LABELS = BASE_DIR / "data" / "raw" / "traffic_sign" / "Dataset" / "labels"
MERGED_DIR = BASE_DIR / "data" / "processed" / "merged"
SPLITS = ["train", "val", "test"]

for split in SPLITS:
    (MERGED_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (MERGED_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

CLASSES = [
    "Pothole",
    "TrafficSign_SpeedLimit",
    "TrafficSign_Other1",
    "TrafficSign_Other2",
    "TrafficSign_Other3",
    "TrafficSign_Other4"
]

dataset_items = []

pothole_files = list(RAW_POTHOLE.glob("*.*"))
pothole_images = [f for f in pothole_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
for img_path in pothole_images:
    label_path = img_path.with_suffix('.txt')
    if label_path.exists():
        dataset_items.append({
            "img": img_path,
            "lbl": label_path,
            "prefix": "pothole_",
            "type": "pothole"
        })

ts_images = list(RAW_TS_IMAGES.glob("*.*"))
ts_images = [f for f in ts_images if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
for img_path in ts_images:
    label_path = RAW_TS_LABELS / (img_path.stem + ".txt")
    if label_path.exists():
        dataset_items.append({
            "img": img_path,
            "lbl": label_path,
            "prefix": "ts_",
            "type": "ts"
        })

random.shuffle(dataset_items)
total = len(dataset_items)
train_end = int(total * 0.8)
val_end = int(total * 0.95)

splits_map = {
    "train": dataset_items[:train_end],
    "val": dataset_items[train_end:val_end],
    "test": dataset_items[val_end:]
}

print(f"Total items: {total}")
print(f"Train: {len(splits_map['train'])}, Val: {len(splits_map['val'])}, Test: {len(splits_map['test'])}")

for split, items in splits_map.items():
    print(f"Processing {split} split...")
    for item in items:
        img_src = item["img"]
        lbl_src = item["lbl"]
        prefix = item["prefix"]

        new_basename = prefix + img_src.name
        img_dst = MERGED_DIR / "images" / split / new_basename
        lbl_dst = MERGED_DIR / "labels" / split / (prefix + lbl_src.name)

        shutil.copy(img_src, img_dst)

        with open(lbl_src, "r") as f_in, open(lbl_dst, "w") as f_out:
            for line in f_in:
                parts = line.strip().split()
                if parts:
                    try:
                        old_class = int(parts[0])
                        if item["type"] == "pothole":
                            new_class = 0
                        else:
                            new_class = old_class + 1
                        new_line = f"{new_class} " + " ".join(parts[1:]) + "\n"
                        f_out.write(new_line)
                    except ValueError:
                        continue

yaml_content = {
    "path": str(MERGED_DIR.absolute()),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": len(CLASSES),
    "names": CLASSES
}

with open(MERGED_DIR / "data.yaml", "w") as f:
    yaml.dump(yaml_content, f, sort_keys=False)

print(f"Dataset preparation complete. YAML saved at {MERGED_DIR / 'data.yaml'}")