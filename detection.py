from ultralytics import YOLO
from PIL import Image
import os

LOCAL_PATH = "/work/imborhau/football-analysis-detection-and-tracking"

model = YOLO("yolov8s.pt")

# results = model.predict("rbk/1_train-val_1min_aalesund_from_start/img1/000001.jpg")
# Image.fromarray(results[0].plot()).show()

results = model.train(data= LOCAL_PATH + "/data.yaml", epochs=30)

for image in LOCAL_PATH + "/dataset_3/images/train":
    


    
