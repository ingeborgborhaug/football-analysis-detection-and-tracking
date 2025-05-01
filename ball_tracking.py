from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import os
import cv2
from datetime import datetime
import pandas as pd
import motmetrics as mm
import itertools
import torch

LOCAL_PATH = "/work/imborhau/football-analysis-detection-and-tracking"
DATASET_PATH = "/datasets/tdt4265/other/rbk"
TEST_IMAGES_PATH = DATASET_PATH + "/3_test_1min_hamkam_from_start/img1" #LOCAL_PATH + "/datasets/dataset_test/images"

gt_path = DATASET_PATH + "/3_test_1min_hamkam_from_start/gt/gt.txt"
gt_file = pd.read_csv(gt_path, header=None)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

IMG_SIZE = 1088
CONFIDENCE_THRESHOLD = 0.0

NAME_2_COLOR = {
    "Ball": (0,200,200),
    "Player": (255,0,0),
}

ID_2_COLOR = {
    "0": (0,200,200),
    "1": (255,0,0),
}

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)

model = YOLO(LOCAL_PATH + '/runs/detect/train66/weights/best.pt')

# Initialize video writer
example_image = cv2.imread(LOCAL_PATH + "/datasets/dataset_test/images/ds3_000001.jpg")
VIDEO_HEIGHT, VIDEO_WIDTH, _ = example_image.shape

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# output_dir = Path('/work/imborhau/video_outputs')
# output_dir.mkdir(parents=True, exist_ok=True)
# video_path = output_dir / f"output_video_{timestamp}.mp4"
# video = cv2.VideoWriter(str(video_path), fourcc, 20, (VIDEO_WIDTH, VIDEO_HEIGHT))

precentage_done = 0
results = []
start = datetime.now()

acc_ball = mm.MOTAccumulator(auto_id=True)

frame_paths = sorted(Path(TEST_IMAGES_PATH).glob("*.jpg"))
results = model.track(
    source=frame_paths, 
    persist=True, 
    tracker="bytetrack.yaml", 
    stream=True,
    conf=0.05,
    imgsz=640
)

for frame_idx, (image_path, result) in enumerate(zip(frame_paths, results), start=1):

    torch.cuda.memory_summary(device=None, abbreviated=False)

    # frame_detections = results.boxes[results.boxes.frame == frame_idx]
    frame = cv2.imread(str(image_path))

    gt_frames = gt_file[gt_file[0] == frame_idx]
    gt_frames = gt_frames[gt_frames[7] == 1] # Only keep ball
    pred_boxes_ball = []
    pred_ids_ball = []
        
    for box in result.boxes:
        cls_id = int(box.cls[0])

        if not (cls_id == 0): # Skip if not ball
            continue

        conf = float(box.conf[0])
        track_id = int(box.id[0]) if box.id is not None else -1
        # print(f"track_id: {track_id}, conf: {conf}")
        pred_ids_ball.append(track_id)

        label = f"{model.names[cls_id]} {conf:.2f} ID:{track_id}"
        xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
        width = xmax - xmin
        height = ymax - ymin
        pred_boxes_ball.append([xmin, ymin, width, height])

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
        cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)
        cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)


    # -----------------

    # Update metrics
    gt_boxes_ball = gt_frames[[2, 3, 4, 5]].values
    gt_ids_ball = gt_frames[1].values

    distances_ball = mm.distances.iou_matrix(gt_boxes_ball, pred_boxes_ball, max_iou=0.5)

    # print(f'leng(gt(gt_ids_ball) {len(gt_ids_ball)}, len(pred_ids_ball) {len(pred_ids_ball)}, len(distances_ball.shape) {len(distances_ball)}')
    acc_ball.update(gt_ids_ball, pred_ids_ball, distances_ball)

    torch.cuda.empty_cache()

    # print(f"Frame {frame_number}: {len(gt_ids_ball)} GT ball, {len(pred_ids_ball)} predicted")

    # video.write(frame)


mh = mm.metrics.create()
summary_ball = mh.compute(acc_ball, metrics=['mota', 'idf1', 'precision', 'recall', 'num_switches'], name='Tracking')
mota_ball = summary_ball.loc['Tracking']['mota']

print(f"\n{summary_ball}")
        
end = datetime.now()   

# video.release()