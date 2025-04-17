from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import cv2
from datetime import datetime

LOCAL_PATH = "/work/imborhau/football-analysis-detection-and-tracking"
TEST_IMAGES_PATH = LOCAL_PATH + "/datasets/dataset_3/images"

IMG_SIZE = 1088
CONFIDENCE_THRESHOLD = 0.5
COLORS = {
    "Ball": (0,200,200),
    "Player": (255,0,0),
}

# model = YOLO("yolov8s.pt")
model = YOLO('runs/detect/train16/weights/best.pt')
# tracker = DeepSort(max_age=50)

# results = model.train(data= LOCAL_PATH + "/data.yaml", epochs=300, imgsz=IMG_SIZE, dropout=0.3, patience=100, plots=True)

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

# Initialize video writer
example_image = cv2.imread(LOCAL_PATH + "/datasets/dataset_3/images/000001.jpg")
VIDEO_HEIGHT, VIDEO_WIDTH, _ = example_image.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_dir = Path('/work/imborhau/video_outputs')
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
video_path = output_dir / f"timestamp" / f"output_video_{timestamp}.mp4"
video = cv2.VideoWriter(str(video_path), fourcc, 20, (VIDEO_WIDTH, VIDEO_HEIGHT))

# Debug
print(model.names)


for image_path in sorted(Path(TEST_IMAGES_PATH).glob("*.jpg")):
    frame = cv2.imread(str(image_path))
    frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    detections = model(frame_resized, verbose=False, conf=0.25)[0]
    print(f'Number of detections in frame: {len(detections.boxes.data.tolist())}')

    frame_resized_back = cv2.resize(frame_resized, (frame.shape[1], frame.shape[0]))
    frame_with_detections = draw_detections(frame_resized_back, detections)
    
    video.write(frame_with_detections)


video.release()

