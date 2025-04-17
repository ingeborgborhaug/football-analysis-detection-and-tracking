import os
import random
from pathlib import Path
import shutil

DATASET_PATH = "/datasets/tdt4265/other/rbk"
LOCAL_PATH = "/work/imborhau/football-analysis-detection-and-tracking"

SOURCE_PATHS = [
    Path(DATASET_PATH + "/1_train-val_1min_aalesund_from_start"),
    Path(DATASET_PATH + "/2_train-val_1min_after_goal"),
    Path(DATASET_PATH + "/3_test_1min_hamkam_from_start")
]

SOURCE_DATASETS_STR = [
    "1_train-val_1min_aalesund_from_start",
    "2_train-val_1min_after_goal",
    "3_test_1min_hamkam_from_start"
]

OUTPUT_PATHS = [
    Path(LOCAL_PATH + "/datasets/dataset_train_val"),
    Path(LOCAL_PATH + "/datasets/dataset_test"),
]

def convert_gt_2_YOLO(out_path, dataset_name):
    print(f'Importing YOLO files for labels...')

    image_width = 1920
    image_height = 1080

    gt_file = Path(DATASET_PATH / gt_path + "gt/gt.txt")
    ds_number = dataset_name[0]

    with open(gt_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        frame_id, track_id, x, y, w, h, conf, class_id, vis = line.strip().split(',')

        class_id_yolo = int(class_id) - 1
        x_center = (float(x) + float(w)/2) / image_width
        y_center = (float(y) + float(h)/2) / image_height
        width = float(w) / image_width
        height = float(h) / image_height

        yolo_line = f"{class_id_yolo} {x_center} {y_center} {width} {height}\n"


        yolo_file = out_path / "labels" / f"{int(frame_id):06}.txt"
        with open(yolo_file, "a") as out_f:
            out_f.write(yolo_line)

## Prepare training and validation data
for src_dataset_str in SOURCE_DATASETS_STR[:2]:

    train_img_dir = OUTPUT_PATHS[0] / "images/train"
    val_img_dir = OUTPUT_PATHS[0] / "images/val"
    train_label_dst = OUTPUT_PATHS[0] / "labels/train"
    val_label_dst = OUTPUT_PATHS[0] / "labels/val"

    src_img_dir = DATASET_PATH / src_dataset_str / "img1"
    src_gt_dir = DATASET_PATH / src_dataset_str / "gt/gt.tx"

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
        target = train_img_dir / img.name
        if not target.exists():
            target.symlink_to(img)

        label_file = src_label_dst / (img.stem + ".txt")
        label_dst = train_label_dst / label_file.name
        if label_file.exists():
            shutil.move(str(label_file), str(label_dst))

    for img in val_files:
        target = val_img_dir / img.name
        if not target.exists():
            target.symlink_to(img)

        label_file = src_label_dst / (img.stem + ".txt")
        label_dst = val_label_dst / label_file.name
        if label_file.exists():
            shutil.move(str(label_file), str(label_dst))
            
    if not (OUTPUT_PATHS[0] / "labels").exists():
        (OUTPUT_PATHS[0] / "labels").mkdir(parents=True, exist_ok=True)
        convert_gt_2_YOLO(OUTPUT_PATHS[0], src_dataset_str / "gt/gt.txt")

## Prepare test data
test_labels_dir = OUTPUT_PATHS[2] / "labels"
test_img_dir = SOURCE_PATHS[2] / "img1"
test_img_dst = OUTPUT_PATHS[2] / "images"

if not (test_labels_dir).exists():
    test_labels_dir.mkdir(parents=True, exist_ok=True)
    convert_gt_2_YOLO(OUTPUT_PATHS[2], SOURCE_PATHS[2] / "gt/gt.txt")

test_img_dst.mkdir(parents=True, exist_ok=True)

for img in sorted(test_img_dir.glob("*.jpg")):
    target = test_img_dst / img.name
    if not target.exists():
        target.symlink_to(img)

def merge_datasets(source_datasets, target_path):
    target_images_train = target_path / "images/train"
    target_labels_train = target_path / "labels/train"
    target_images_val = target_path / "images/val"
    target_labels_val = target_path / "labels/val"

    for p in [target_images_train, target_images_val, target_labels_train, target_labels_val]:
        p.mkdir(parents=True, exist_ok=True)

    for idx, dataset in enumerate(source_datasets, start=1):
        prefix = f"ds{idx}_"
        for split in ["train", "val"]:
            img_dir = dataset / f"images/{split}"
            lbl_dir = dataset / f"labels/{split}"
            out_img_dir = target_path / f"images/{split}"
            out_lbl_dir = target_path / f"labels/{split}"

            for img_file in sorted(img_dir.glob("*.jpg")):
                new_img_name = prefix + img_file.name
                new_lbl_name = prefix + img_file.stem + ".txt"

                target_img = out_img_dir / new_img_name
                if not target_img.exists():
                    target_img.symlink_to(img_file.resolve())

                original_label = lbl_dir / img_file.with_suffix(".txt").name
                target_label = out_lbl_dir / new_lbl_name
                if original_label.exists():
                    shutil.copy(original_label, target_label)

    print(f"Merged datasets into: {target_path}")

# Merge first and second dataset
merge_datasets(
    [OUTPUT_PATHS[0], OUTPUT_PATHS[1]],
    Path(LOCAL_PATH + "/datasets/dataset_combined")
)

print("Preparation of all datasets done.")
