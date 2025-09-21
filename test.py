# from ultralytics import YOLO
import torch

from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.pt")  # load an official model
model = YOLO("checkpoint/ssmnet_usod.pt")  # load a custom model

# Validate the model
metrics = model.val(split="test", data="ultralytics/cfg/datasets/USOD.yaml", batch=1)  # no arguments needed, dataset and settings remembered
