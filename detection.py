from ultralytics import YOLO
from PIL import Image
import os


def convert_gt_2_YOLO(dataset_path):
    print(f'Converting gt.txt in {dataset_path} to YOLO format...')

    gt_file = dataset_path + '/gt/gt.txt'
    yolo_file_path = dataset_path + '/labels/'

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

        yolo_file = yolo_file_path + f"{int(frame_id):06}.txt"
        with open(yolo_file, "a") as out_f:
            out_f.write(yolo_line)


model = YOLO("yolov8s.pt")

# results = model.predict("rbk/1_train-val_1min_aalesund_from_start/img1/000001.jpg")
# Image.fromarray(results[0].plot()).show()

if len(os.listdir('train_folder/labels')) == 0:
    convert_gt_2_YOLO('train_folder')

if len(os.listdir('val_folder/labels')) == 0:
    convert_gt_2_YOLO('test_folder')

results = model.train(data="mini_dataset/data.yaml", epochs=30)
    