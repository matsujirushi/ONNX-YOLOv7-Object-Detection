import argparse
import time
import cv2
from yolov7 import YOLOv7
from yolov7.utils import class_names
import numpy as np

DEFAULT_MODEL = 'models/yolov7-tiny_384x640.onnx'

parser = argparse.ArgumentParser()
parser.add_argument('--disable-trt', action='store_true')
parser.add_argument('--disable-cuda', action='store_true')
parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
parser.add_argument('--interval-ms', type=int, default=0)
parser.add_argument('--headless', action='store_true')
args = parser.parse_args()

print(f' - TensorRT: {"Disable" if args.disable_trt else "Enable"}')
print(f' - CUDA: {"Disable" if args.disable_cuda else "Enable"}')
print(f' - model: {args.model}')
print(f' - interval-ms: {args.interval_ms}')
print(f' - Headless mode: {args.headless}')

detector = YOLOv7(args.model, conf_thres=0.5, iou_thres=0.5, trt=not args.disable_trt, cuda=not args.disable_cuda)
person_class_id = class_names.index('person')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not args.headless:
    cv2.namedWindow('YOLOv7', cv2.WINDOW_NORMAL)

capture_time = 0
while cap.isOpened():
    wait_time = (capture_time + args.interval_ms / 1000) - time.perf_counter()
    if cv2.waitKey(max(int(wait_time * 1000), 1)) == ord('q'):
        break

    capture_time = time.perf_counter()
    if args.interval_ms != 0:
        cap.grab()
    _, frame = cap.read()

    _, _, class_ids = detector(frame)

    if not args.headless:
        combined_img = detector.draw_detections(frame)
        cv2.imshow('YOLOv7', combined_img)
    else:
        if person_class_id in class_ids:
            print(f'Person count: {np.bincount(class_ids)[person_class_id]}')
