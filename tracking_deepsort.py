from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import cv2
from datetime import datetime
import pandas as pd
import motmetrics as mm
import itertools

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

model = YOLO(LOCAL_PATH + '/runs/detect/train17/weights/best.pt')

# Initialize video writer
example_image = cv2.imread(LOCAL_PATH + "/datasets/dataset_test/images/ds3_000001.jpg")
VIDEO_HEIGHT, VIDEO_WIDTH, _ = example_image.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_dir = Path('/work/imborhau/video_outputs')
output_dir.mkdir(parents=True, exist_ok=True)
video_path = output_dir / f"output_video_{timestamp}.mp4"
video = cv2.VideoWriter(str(video_path), fourcc, 20, (VIDEO_WIDTH, VIDEO_HEIGHT))

param_grid = {
    "max_age": [4],
    "n_init": [15], 
    "max_iou_distance": [0.9],
}

param_combinations = list(itertools.product(
    param_grid["max_age"],
    param_grid["n_init"],
    param_grid["max_iou_distance"]
))

number_of_combinations = len(param_combinations)
precentage_done = 0
best_mota = -9999
best_params = None
results = []
start = datetime.now()

for max_age, n_init, max_iou_distance in param_combinations:
    print(f"Trying: max_age={max_age}, n_init={n_init}, max_iou_distance={max_iou_distance}")

    tracker = DeepSort(
        max_age=max_age, # Maximum number of frames to keep a track alive without detection
        n_init=n_init, # Number of frames to initialize a track
        max_iou_distance=max_iou_distance, # Maximum distance between a detection and a track to consider it a match
        nms_max_overlap=1.0, # NMS overlap threshold
        # persist=True, # Keep the tracker alive even if no detections are present
    )
    acc = mm.MOTAccumulator(auto_id=True)
    frame_number = 1

    for image_path in sorted(Path(TEST_IMAGES_PATH).glob("*.jpg")):

        frame = cv2.imread(str(image_path))

        gt_frames = gt_file[gt_file[0] == frame_number]
        pred_boxes = []
        pred_ids = []

        ## DeepSort
        detections = model(frame, verbose=False, conf=0.1)[0]

        results_yolo = []

        for data in detections.boxes.data.tolist():

            confidence = data[4]

            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            class_id = int(data[5])

            # if class_id == 0: # Skip ball
            #     continue
            
            results_yolo.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

        tracks = tracker.update_tracks(results_yolo, frame=frame)

        for track in tracks:

            if not track.is_confirmed():
                continue

            # get the track id and the bounding box
            track_id = track.track_id
            ltrb = track.to_ltrb()
            pred_ids.append(track_id)

            xmin, ymin, xmax, ymax = int(ltrb[0]), int(
                ltrb[1]), int(ltrb[2]), int(ltrb[3])
            width = xmax - xmin
            height = ymax - ymin
            pred_boxes.append([xmin, ymin, width, height])


            # draw the bounding box and the track id
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
            cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)

            cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
        # -----------------

        # Update metrics
        gt_boxes = gt_frames[[2, 3, 4, 5]].values
        gt_ids = gt_frames[1].values

        distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        acc.update(gt_ids, pred_ids, distances)

        video.write(frame)

        frame_number += 1

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['mota', 'idf1', 'precision', 'recall', 'num_switches'], name='Tracking')
    print(summary)
    mota = summary.loc['Tracking']['mota']

    results.append({
        "max_age": max_age,
        "n_init": n_init,
        "max_iou_distance": max_iou_distance,
        "mota": mota
    })

    precentage_done += 1/number_of_combinations
    print(f"Progress: {precentage_done:.2%}")

    if mota > best_mota:
        best_mota = mota
        best_params = (max_age, n_init, max_iou_distance)
        
end = datetime.now()   
print(f"Time to run hyperparameter search: {(end - start).total_seconds()/ 60:.2f} minutes")
print(f"Best MOTA: {best_mota:.4f} with params: max_age={best_params[0]}, n_init={best_params[1]}, max_iou_distance={best_params[2]}")

video.release()

# Save results to CSV
df = pd.DataFrame(results)
df.to_csv('hyperparameter_search_results.csv', index=False)

# Plotting results
# fig, ax = plt.subplots(figsize=(10,6))
# scatter = ax.scatter(
#     df["max_age"], df["n_init"],
#     c=df["mota"], cmap="viridis", s=200, edgecolors='k'
# )
# plt.colorbar(scatter, label='MOTA Score')
# ax.set_xlabel('Max Age')
# ax.set_ylabel('N Init')
# ax.set_title('Hyperparameter Search: MOTA Scores (color)')
# plt.grid(True)
# plt.show()
# plt.savefig('plot.png')


## Bytetrack
        # results_yolo = model.track(frame, persist=True, show=False, tracker="botsort.yaml")
        # annotated_frame = results_yolo[0].plot()

        # for box in results_yolo[0].boxes:
        #     cls_id = int(box.cls[0])
        #     conf = float(box.conf[0])
        #     track_id = int(box.id[0]) if box.id is not None else -1

        #     label = f"{model.names[cls_id]} {conf:.2f} ID:{track_id}"
        #     xmin, ymin, xmax, ymax = map(int, box.xyxy[0])

        #     cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
        #     cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)
        #     cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
        # -----------------------------