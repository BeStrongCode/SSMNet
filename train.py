from ultralytics import YOLO
import os

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.yaml")  # build a new model from scratch
# model = YOLO("yolov8m.pt")
# Use the model
model.train(data="ultralytics/cfg/datasets/USOD.yaml", epochs=200, batch=8)  # train the model
#
# import torch
# print(torch.version.cuda)