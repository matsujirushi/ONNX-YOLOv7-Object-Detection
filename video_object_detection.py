import cv2
from yolov7 import YOLOv7
import argparse

default_model = 'models/yolov7-tiny_384x640.onnx'

parser = argparse.ArgumentParser()
parser.add_argument('--disable-trt', action='store_true')
parser.add_argument('--disable-cuda', action='store_true')
parser.add_argument('--headless', action='store_true')
parser.add_argument('--model', type=str, default=default_model)
parser.add_argument('--video', type=str)
args = parser.parse_args()

trt = not args.disable_trt
cuda = not args.disable_cuda
headless = args.headless
model_path = args.model
video_path = args.video
print(' - TensorRT: {}'.format('Disable' if args.disable_trt else 'Enable'))
print(' - CUDA: {}'.format('Disable' if args.disable_cuda else 'Enable'))
print(' - Headless mode: {}'.format(headless))
print(' - model: {}'.format(model_path))
print(' - video: {}'.format('USB Camera' if video_path is None else video_path))

# # Initialize video
if video_path is None:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(video_path)

start_time = 0  # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * 30)

# Initialize YOLOv7 model
yolov7_detector = YOLOv7(model_path, conf_thres=0.5, iou_thres=0.5, trt=trt, cuda=cuda)

if not headless:
    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

while cap.isOpened():

    # Press key q to stop
    if cv2.waitKey(1) == ord('q'):
        break

    try:
        # Read frame from the video
        ret, frame = cap.read()
        if not ret:
            break
    except Exception as e:
        print(e)
        continue

    # Update object localizer
    boxes, scores, class_ids = yolov7_detector(frame)

    if not headless:
        combined_img = yolov7_detector.draw_detections(frame)
        cv2.imshow("Detected Objects", combined_img)
    # out.write(combined_img)

# out.release()
