from ultralytics import YOLO
from PIL import Image
from pathlib import Path
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import cv2

LOCAL_PATH = "/work/imborhau/football-analysis-detection-and-tracking"
TEST_IMAGES_PATH = LOCAL_PATH + "/datasets/dataset_3/images"

CONFIDENCE_THRESHOLD = 0.5
COLORS = {
    "ball": (0,200,200),
    "player": (255,0,0),
}

example_image = cv2.imread(LOCAL_PATH + "/datasets/dataset_3/images/000001.jpg")
VIDEO_HEIGHT, VIDEO_WIDTH, _ = example_image.shape

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('output_video.mp4', fourcc, 20, (VIDEO_WIDTH, VIDEO_HEIGHT))

# model = YOLO("yolov8s.pt")
model = YOLO('runs/detect/train6/weights/best.pt')
tracker = DeepSort(max_age=50)

results = model.train(data= LOCAL_PATH + "/data_combined.yaml", epochs=300, dropout=0.3, patience=30, plots=True)

def draw_detections(frame, detections):
    frame = frame.copy()

    for data in detections.boxes.data.tolist():
        confidence = data[4]
        class_id = data[5]
        class_name = model.names[class_id]

        # if float(confidence) < CONFIDENCE_THRESHOLD:
        #     continue

        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        label = f"{class_name} {confidence:.2f}"

        cv2.rectangle(frame, (xmin, ymin) , (xmax, ymax), COLORS[class_name], 2)
        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[class_name], 1)

    return frame


for image_path in sorted(Path(TEST_IMAGES_PATH).glob("*.jpg")):
    frame = cv2.imread(str(image_path))
    detections = model(frame)[0]

    frame = draw_detections(frame, detections)
    
    video.write(frame)


video.release()

