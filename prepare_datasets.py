import os
import random
from pathlib import Path
import shutil

DATASET_PATH = "/datasets/tdt4265/other/rbk"
LOCAL_PATH = "/work/imborhau/football-analysis-detection-and-tracking"

SOURCE_DATASETS_STR = [
    "/1_train-val_1min_aalesund_from_start",
    "/2_train-val_1min_after_goal",
    "/3_test_1min_hamkam_from_start"
]

OUTPUT_PATHS = [
    Path(LOCAL_PATH + "/datasets/dataset_train_val"),
    Path(LOCAL_PATH + "/datasets/dataset_test"),
]

def save_as_yolo_format(gt_file, ds_number, out_path):
    print(f'Importing YOLO files for labels...')

    image_width = 1920
    image_height = 1080

    with open(gt_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        frame_id, track_id, x, y, w, h, conf, class_id, vis = line.strip().split(',')

        if conf == "0":
            continue

        class_id_yolo = int(class_id) - 1
        x_center = (float(x) + float(w)/2) / image_width
        y_center = (float(y) + float(h)/2) / image_height
        width = float(w) / image_width
        height = float(h) / image_height

        yolo_line = f"{class_id_yolo} {x_center} {y_center} {width} {height}\n"

        yolo_file = out_path / f"ds{ds_number}_{int(frame_id):06}.txt"
        with open(yolo_file, "a") as out_f:
            out_f.write(yolo_line)

## Prepare training and validation data
for src_dataset_str in SOURCE_DATASETS_STR[:2]:

    ds_number = src_dataset_str[1]

    train_img_dir = OUTPUT_PATHS[0] / "images/train"
    val_img_dir = OUTPUT_PATHS[0] / "images/val"
    train_label_dst = OUTPUT_PATHS[0] / "labels/train"
    val_label_dst = OUTPUT_PATHS[0] / "labels/val"

    src_img_dir = Path(DATASET_PATH + src_dataset_str + "/img1")
    src_gt_dir = Path(DATASET_PATH + src_dataset_str + "/gt/gt.txt")

    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)
    train_label_dst.mkdir(parents=True, exist_ok=True)
    val_label_dst.mkdir(parents=True, exist_ok=True)

    image_files = sorted(src_img_dir.glob("*.jpg"))
    random.seed(42)
    random.shuffle(image_files)

    split_index = int(len(image_files) * 0.8)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    for img in train_files:
        new_name = f"ds{ds_number}_{img.name}"
        target_img = train_img_dir / new_name
        target_label = train_label_dst / new_name.replace(".jpg", ".txt")

        if not target_img.exists():
            target_img.symlink_to(img)

        if not target_label.exists():
            save_as_yolo_format(src_gt_dir, ds_number, train_label_dst)

    for img in val_files:
        new_name = f"ds{ds_number}_{img.name}"
        target_img = val_img_dir / new_name
        target_label = val_label_dst / new_name.replace(".jpg", ".txt")

        if not target_img.exists():
            target_img.symlink_to(img)

        if not target_label.exists():
            save_as_yolo_format(src_gt_dir, ds_number, val_label_dst)


## Prepare test data
test_img_dir = OUTPUT_PATHS[1] / "images"
test_label_dst = OUTPUT_PATHS[1] / "labels"

test_src_img_dir = Path(DATASET_PATH + SOURCE_DATASETS_STR[2] + "img1")
test_src_gt_dir = Path(DATASET_PATH + SOURCE_DATASETS_STR[2] + "/gt/gt.txt")

test_img_dir.mkdir(parents=True, exist_ok=True)
test_label_dst.mkdir(parents=True, exist_ok=True)

test_image_files = sorted(src_img_dir.glob("*.jpg"))

for img in test_image_files:
        new_name = f"ds3_{img.name}"
        target_img = test_img_dir / new_name
        target_label = test_label_dst / new_name.replace(".jpg", ".txt")

        if not target_img.exists():
            target_img.symlink_to(img)

        if not target_label.exists():
            save_as_yolo_format(test_src_gt_dir, 3, test_label_dst)


print("Preparation of all datasets done.")
