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

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# output_dir = Path('/work/imborhau/video_outputs')
# output_dir.mkdir(parents=True, exist_ok=True)
# video_path = output_dir / f"output_video_{timestamp}.mp4"
# video = cv2.VideoWriter(str(video_path), fourcc, 20, (VIDEO_WIDTH, VIDEO_HEIGHT))

param_grid = {
    "max_age": [6],
    "n_init": [2], 
    "max_iou_distance": [0.9],
}

#  "max_age": [6, 10, 20, 20],
#     "n_init": [1, 2, 3, 4, 5], 
#     "max_iou_distance": [0.7, 0.8, 0.9],

param_combinations = list(itertools.product(
    param_grid["max_age"],
    param_grid["n_init"],
    param_grid["max_iou_distance"]
))

number_of_combinations = len(param_combinations)
precentage_done = 0
best_mota = -9999
best_num_switches = 9999
best_params = None
results = []
start = datetime.now()

for max_age, n_init, max_iou_distance in param_combinations:
    print(f"Trying: max_age={max_age}, n_init={n_init}, max_iou_distance={max_iou_distance}")

    tracker_players = DeepSort(
        max_age=6, # Maximum number of frames to keep a track alive without detection
        n_init=50, # Number of frames to initialize a track
        max_iou_distance=0.9, # Maximum distance between a detection and a track to consider it a match
        nms_max_overlap=1.0, # NMS overlap threshold
        # persist=True, # Keep the tracker_players alive even if no detections are present
    )

    # tracker_ball = DeepSort(
    #     max_age=max_age, # Maximum number of frames to keep a track alive without detection
    #     n_init=n_init, # Number of frames to initialize a track
    #     max_iou_distance=max_iou_distance, # Maximum distance between a detection and a track to consider it a match
    #     nms_max_overlap=1.0, # NMS overlap threshold
    #     # persist=True, # Keep the tracker_players alive even if no detections are present
    # )

    acc_players = mm.MOTAccumulator(auto_id=True)
    acc_ball = mm.MOTAccumulator(auto_id=True)
    frame_number = 1

    for image_path in sorted(Path(TEST_IMAGES_PATH).glob("*.jpg")):

        frame = cv2.imread(str(image_path))

        gt_frames = gt_file[gt_file[0] == frame_number]
        pred_boxes_players = []
        pred_ids_players = []

        pred_boxes_ball = []
        pred_ids_ball = []

        ## DeepSort
        detections = model(frame, verbose=False, conf=0.25)[0]

        results_yolo_players = []
        results_yolo_ball = []

        for data in detections.boxes.data.tolist():

            print(f"data: {data}")

            confidence = data[5]

            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            class_id = int(data[6])

            # if class_id == 0:
            #     results_yolo_ball.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])
            if class_id == 1:
                results_yolo_players.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])
            
        # ball_tracks = tracker_ball.update_tracks(results_yolo_ball, frame=frame)
        result_ball = model.track(source=frame, persist=True, conf=0.05, tracker="bytetrack.yaml")[0]
        # print(f'Number of player detections: {len(results_yolo_players)}')
        player_tracks = tracker_players.update_tracks(results_yolo_players, frame=frame)

        for box in result_ball.boxes:
            cls_id = int(box.cls[0])

            if not (cls_id == 0): # Skip if not ball
                continue

            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else -1
            pred_ids_ball.append(track_id)

            label = f"{model.names[cls_id]} {conf:.2f} ID:{track_id}"
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
            width = xmax - xmin
            height = ymax - ymin
            pred_boxes_ball.append([xmin, ymin, width, height])

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
            cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)
            cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

        # print(f'Number of player tracks: {len(player_tracks)}')
        for player_track in player_tracks:

            if not player_track.is_confirmed():
                continue

            # Get player track id and the bounding box
            player_track_id = player_track.track_id
            ltrb_player = player_track.to_ltrb()
            pred_ids_players.append(player_track_id)

            # Draw player bounding box
            xmin, ymin, xmax, ymax = int(ltrb_player[0]), int(
                ltrb_player[1]), int(ltrb_player[2]), int(ltrb_player[3])
            width = xmax - xmin
            height = ymax - ymin
            pred_boxes_players.append([xmin, ymin, width, height])

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
            cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)

            cv2.putText(frame, str(player_track_id), (xmin + 5, ymin - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

        # for ball_track in ball_tracks:

        #     if not ball_track.is_confirmed():
        #         continue
        
        #     # Get ball track id and the bounding box
        #     ball_track_id = ball_track.track_id
        #     ltrb_ball = ball_track.to_ltrb()
        #     pred_ids_ball.append(ball_track_id)

        #     # Draw ball bounding box
        #     xmin, ymin, xmax, ymax = int(ltrb_ball[0]), int(
        #         ltrb_ball[1]), int(ltrb_ball[2]), int(ltrb_ball[3])
        #     width = xmax - xmin
        #     height = ymax - ymin
        #     pred_boxes_ball.append([xmin, ymin, width, height])

        #     cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
        #     cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)

        #     cv2.putText(frame, str(ball_track_id), (xmin + 5, ymin - 8),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

        # -----------------

        # Update metrics
        gt_boxes_ball = gt_frames[gt_frames[7] == 1][[2, 3, 4, 5]].values
        gt_ids_ball = gt_frames[gt_frames[7] == 1][1].values

        gt_boxes_players = gt_frames[gt_frames[7] == 2][[2, 3, 4, 5]].values
        gt_ids_players = gt_frames[gt_frames[7] == 2][1].values

        distances_ball = mm.distances.iou_matrix(gt_boxes_ball, pred_boxes_ball, max_iou=0.5)
        distances_players = mm.distances.iou_matrix(gt_boxes_players, pred_boxes_players, max_iou=0.5)

        # print(f'leng(gt(gt_ids_ball) {len(gt_ids_ball)}, len(pred_ids_ball) {len(pred_ids_ball)}, len(distances_ball.shape) {len(distances_ball)}')
        acc_ball.update(gt_ids_ball, pred_ids_ball, distances_ball)
        print(f'leng(gt(gt_ids_players) {len(gt_ids_players)}, len(pred_ids_players) {len(pred_ids_players)}, len(distances_players.shape) {len(distances_players)}')
        acc_players.update(gt_ids_players, pred_ids_players, distances_players)

        # print(f"Frame {frame_number}: {len(gt_ids_players)} GT players, {len(pred_ids_players)} predicted")
        # print(f"Frame {frame_number}: {len(gt_ids_ball)} GT ball, {len(pred_ids_ball)} predicted")

        # video.write(frame)

        frame_number += 1

    mh = mm.metrics.create()
    summary_players = mh.compute(acc_players, metrics=['mota', 'idf1', 'precision', 'recall', 'num_switches'], name='Tracking')
    summary_ball = mh.compute(acc_ball, metrics=['mota', 'idf1', 'precision', 'recall', 'num_switches'], name='Tracking')
    
    mota_ball = summary_ball.loc['Tracking']['mota']
    mota_players = summary_players.loc['Tracking']['mota']
    num_switches_players = summary_players.loc['Tracking']['num_switches']

    results.append({
        "max_age_ball": max_age,
        "n_init_ball": n_init,
        "max_iou_distance_ball": max_iou_distance,
        "mota_ball": mota_ball,
        "mota_players": mota_players,
        "num_switches_players": num_switches_players
    })

    print(f"summary_ball: \n{summary_ball}")
    print(f"summary_players: \n{summary_players}")

    precentage_done += 1/number_of_combinations
    print(f"Progress: {precentage_done:.2%}")

    if mota_ball > best_mota:
        best_mota = mota_ball
        best_params = (max_age, n_init, max_iou_distance)
        
end = datetime.now()   
print(f"Time to run hyperparameter search: {(end - start).total_seconds()/ 60:.2f} minutes")
print(f"Best best_mota for ball: {best_mota:.4f} with params for : max_age={best_params[0]}, n_init={best_params[1]}, max_iou_distance={best_params[2]}")

# video.release()

# Save results to CSV
df = pd.DataFrame(results)
df.to_csv(f"hyperparameter_search_results_{timestamp}.csv", index=False)

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
        # results_yolo = model.track(frame, persist=True, show=False, tracker_players="botsort.yaml")
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