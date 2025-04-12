from ultralytics import YOLO
from PIL import Image
import os

DATASET_PATH = "/datasets/tdt4265/other/rbk"
LOCAL_PATH = "/work/imborhau/rbk"

def convert_gt_2_YOLO(video_folder):
    print(f'Converting gt.txt in {DATASET_PATH} to YOLO format in {LOCAL_PATH}...')

    gt_file = DATASET_PATH + '/gt/gt.txt'
    yolo_file_path = LOCAL_PATH + video_folder + '/labels/'

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

# Check if (YOLO) labels folder exist, if not create them
DIR_1 = ("/work/imborhau/rbk/1_train-val_1min_aalesund_from_start/labels")
DIR_2 = ("/work/imborhau/rbk/2_train-val_1min_after_goal/labels")
CHECK_LABELS_IN_FOLDER_1 = os.path.isdir(DIR_1)
CHECK_LABELS_IN_FOLDER_2 = os.path.isdir(DIR_2)

if not CHECK_LABELS_IN_FOLDER_1:
    os.makedirs(DIR_1)
    print("created folder : ", DIR_1)
else:
    print(DIR_1, "folder already exists.")

if not CHECK_LABELS_IN_FOLDER_2:
    os.makedirs(DIR_2)
    print("created folder : ", DIR_2)
else:
    print(DIR_2, "folder already exists.")

# Check if (YOLO) labels folder is empty, if so convert gt.txt to YOLO format
if len(os.listdir(LOCAL_PATH + '1_train-val_1min_aalesund_from_start/labels')) == 0:
    convert_gt_2_YOLO('1_train-val_1min_aalesund_from_start')

if len(os.listdir(LOCAL_PATH + '2_train-val_1min_after_goal/labels')) == 0:
    convert_gt_2_YOLO('2_train-val_1min_after_goal')



# results = model.predict("rbk/1_train-val_1min_aalesund_from_start/img1/000001.jpg")
# Image.fromarray(results[0].plot()).show()

# results = model.train(data="data.yaml", epochs=30)




    