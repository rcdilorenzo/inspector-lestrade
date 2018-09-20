import sys
sys.path.append('keras-yolo3')

from yolo import YOLO, detect_video

model = YOLO(model_path='./models/v4-12k-adam1e3-train10/ep039-loss16.249-val_loss15.892.h5',
             anchors_path='./keras-yolo3/model_data/yolo_anchors.txt',
             classes_path='./classes-yolo-format.txt',
             score=0.01)

# Pretrained YOLO model
# yolo_model = YOLO(model_path='../../../data/yolov3/yolov3-320.h5',
#                   anchors_path='./keras-yolo3/model_data/yolo_anchors.txt',
#                   classes_path='./keras-yolo3/model_data/coco_classes.txt')

detect_video(model, 0)
