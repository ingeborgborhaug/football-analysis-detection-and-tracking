import os
import random
from pathlib import Path


DATASET_PATH = "/datasets/tdt4265/other/rbk"
LOCAL_PATH = "/work/imborhau/football-analysis-detection-and-tracking"

SOURCE_PATHS = [
    Path("/datasets/tdt4265/other/rbk/1_train-val_1min_aalesund_from_start"),
    Path("/datasets/tdt4265/other/rbk/2_train-val_1min_after_goal"),
    Path("/datasets/tdt4265/other/rbk/3_test_1min_hamkam_from_start")
]

OUTPUT_PATHS = [
    Path("/work/imborhau/football-analysis-detection-and-tracking/dataset_1"),
    Path("/work/imborhau/football-analysis-detection-and-tracking/dataset_2"),
    Path("/work/imborhau/football-analysis-detection-and-tracking/dataset_3")
]

def convert_gt_2_YOLO(out_path, gt_file):
    print(f'Importing YOLO files for labels...')

    image_width = 1920
    image_height = 1080

    with open(gt_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        frame_id, track_id, x, y, w, h, class_id, vis, conf = line.strip().split(',')

        x_center = (float(x) + float(w)/2) / image_width
        y_center = (float(y) + float(h)/2) / image_height
        width = float(w) / image_width
        height = float(h) / image_height

        yolo_line = f"{class_id} {x_center} {y_center} {width} {height}\n"

        yolo_file = out_path / "labels" / f"{int(frame_id):06}.txt"
        with open(yolo_file, "a") as out_f:
            out_f.write(yolo_line)

# Create symlinks from SSH for images in train and val folders
for src_path, out_path in zip(SOURCE_PATHS, OUTPUT_PATHS):

    # Import YOLO files to labels folder
    if not (out_path / "labels").exists():
        (out_path / "labels").mkdir(parents=True, exist_ok=True)
        convert_gt_2_YOLO(out_path, src_path / "gt/gt.txt")

    train_img_dir = out_path / "images/train"
    val_img_dir = out_path / "images/val"
    train_label_dir = out_path / "labels/train"
    val_label_dir = out_path / "labels/val"

    src_img_dir = src_path / "img1"
    src_label_dir =  out_path / "labels"

    # Create output directories
    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)
    train_label_dir.mkdir(parents=True, exist_ok=True)
    val_label_dir.mkdir(parents=True, exist_ok=True)

    # Get and shuffle image list
    image_files = sorted(src_path.glob("*.jpg"))
    random.seed(42)
    random.shuffle(image_files)

    # 80/20 split
    split_index = int(len(image_files) * 0.8)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    # Create symlinks and move labes to respective folders
    for img in train_files:
        target = train_img_dir / img.name
        if not target.exists():
            target.symlink_to(img)

        label_file = src_label_dir / (img.stem + ".txt")
        label_dir = train_label_dir / label_file.name
        if label_file.exists():
            shutil.move(str(label_file), str(label_dir))

    for img in val_files:
        target = val_img_dir / img.name
        if not target.exists():
            target.symlink_to(img)

        label_file = src_label_dir / (img.stem + ".txt")
        label_dir = val_label_dir / label_file.name
        if label_file.exists():
            shutil.move(str(label_file), str(label_dir))

    print(f"Preparation of data done.")



