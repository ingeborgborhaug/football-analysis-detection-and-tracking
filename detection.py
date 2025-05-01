from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import cv2
from datetime import datetime

DATASET_PATH = "/datasets/tdt4265/other/rbk"
LOCAL_PATH = "/work/imborhau/football-analysis-detection-and-tracking"
TEST_IMAGES_PATH = DATASET_PATH + "/3_test_1min_hamkam_from_start/img1"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

IMG_SIZE = 1088
CONFIDENCE_THRESHOLD_PLAYER = 0.4
CONFIDENCE_THRESHOLD_BALL = 0.2
NAME_2_COLOR = {
    "Ball": (0,200,200),
    "Player": (255,0,0),
}
model = YOLO("yolov8s.pt")

## Already trained model

model = YOLO('football-analysis-detection-and-tracking/runs/detect/train66/weights/best.pt')

## Find the best hyperparameters

# results_tuning = model.tune(
#     data= LOCAL_PATH + "/data.yaml",
#     batch=-1,
#     epochs=50,            # Smaller epochs per experiment
#     imgsz=IMG_SIZE,
#     iterations=30,        # Number of hyperparameter candidates to test
#     plots=True,           
#     patience=20,          # Optional, early stopping patience
#     val=True              # Validate performance
# )
# end = datetime.now()
# print(f"Tuning time: {end - start}")

## Train the model with the best hyperparameters

# start = datetime.now()
# results_training = model.train(
#     data= LOCAL_PATH + "/data.yaml", 
#     epochs=300, 
#     imgsz=IMG_SIZE, 
#     warmup_epochs=2.77509,
#     weight_decay=0.00051,
#     box=7.41325,
#     cls=0.4922,
#     dfl=1.43177,
#     hsv_s=0.72628,
#     scale=0.48445,
#     dropout=0.3, 
#     patience=100, 
#     val=True,
#     plots=True
# )
# end = datetime.now()
# print(f"Training complete. Best model at: {results_training.save_dir}/weights/best.pt")
# print(f"Training time: {end - start}")

def draw_detections(frame, detections):
    frame = frame.copy()

    for data in detections.boxes.data.tolist():
        confidence = data[4]
        class_id = data[5]
        class_name = model.names[class_id]

        if class_name == "Player":
            if float(confidence) < CONFIDENCE_THRESHOLD_PLAYER:
                continue
        elif class_name == "Ball":
            if float(confidence) < CONFIDENCE_THRESHOLD_BALL:
                continue

        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        label = f"{class_name} {confidence:.2f}"

        cv2.rectangle(frame, (xmin, ymin) , (xmax, ymax), NAME_2_COLOR[class_name], 2)
        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, NAME_2_COLOR[class_name], 1)

    return frame

# Initialize video writer
example_image = cv2.imread(LOCAL_PATH + "/datasets/dataset_test/images/ds3_000001.jpg")
VIDEO_HEIGHT, VIDEO_WIDTH, _ = example_image.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_dir = Path('/work/imborhau/video_outputs')
output_dir.mkdir(parents=True, exist_ok=True)
video_path = output_dir / f"output_video_{timestamp}.mp4"
video = cv2.VideoWriter(str(video_path), fourcc, 20, (VIDEO_WIDTH, VIDEO_HEIGHT))


for image_path in sorted(Path(TEST_IMAGES_PATH).glob("*.jpg")):
    frame = cv2.imread(str(image_path))
    # frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    detections = model(frame, verbose=False, conf=0.25)[0]
    # print(f'Number of detections in frame: {len(detections.boxes.data.tolist())}')

    # frame_resized_back = cv2.resize(frame_resized, (frame.shape[1], frame.shape[0]))
    frame_with_detections = draw_detections(frame, detections)
    
    video.write(frame_with_detections)


video.release()

