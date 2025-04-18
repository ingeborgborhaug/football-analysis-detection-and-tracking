from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import cv2
from datetime import datetime

LOCAL_PATH = "/work/imborhau/football-analysis-detection-and-tracking"
TEST_IMAGES_PATH = LOCAL_PATH + "/datasets/dataset_test/images"

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

model = YOLO('runs/detect/train17/weights/best.pt')
tracker = DeepSort(max_age=300, max_iou_distance=0.9, nms_max_overlap=0.3)#, n_init=3, nn_budget=100, override_track_class=True, embed)

# Initialize video writer
example_image = cv2.imread(LOCAL_PATH + "/datasets/dataset_test/images/ds3_000001.jpg")
VIDEO_HEIGHT, VIDEO_WIDTH, _ = example_image.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_dir = Path('/work/imborhau/video_outputs')
output_dir.mkdir(parents=True, exist_ok=True)
video_path = output_dir / f"output_video_{timestamp}.mp4"
video = cv2.VideoWriter(str(video_path), fourcc, 20, (VIDEO_WIDTH, VIDEO_HEIGHT))

frame_number = 1

for image_path in sorted(Path(TEST_IMAGES_PATH).glob("*.jpg")):
    start = datetime.now()

    frame = cv2.imread(str(image_path))

    ## Bytetrack
    # results = model.track(frame, persist=True, show=False, tracker="botsort.yaml")
    # annotated_frame = results[0].plot()

    # for box in results[0].boxes:
    #     cls_id = int(box.cls[0])
    #     conf = float(box.conf[0])
    #     track_id = int(box.id[0]) if box.id is not None else -1

    #     label = f"{model.names[cls_id]} {conf:.2f} ID:{track_id}"
    #     xmin, ymin, xmax, ymax = map(int, box.xyxy[0])

    #     cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
    #     cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)
    #     cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
    # -----------------------------
    ## DeepSort
    detections = model(frame, verbose=False, conf=0.25)[0]

    results = []

    for data in detections.boxes.data.tolist():

        confidence = data[4]

        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        class_id = int(data[5])
        
        results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

    tracks = tracker.update_tracks(results, frame=frame)


    for track in tracks:

        if not track.is_confirmed():
            continue

        # get the track id and the bounding box
        track_id = track.track_id
        ltrb = track.to_ltrb()

        xmin, ymin, xmax, ymax = int(ltrb[0]), int(
            ltrb[1]), int(ltrb[2]), int(ltrb[3])

        # draw the bounding box and the track id
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
        cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)
        cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
    # -----------------

    video.write(frame)

    end = datetime.now()
    print(f"Time to process frame {frame_number}: {(end - start).total_seconds() * 1000:.0f} milliseconds")
    frame_number += 1

video.release()

