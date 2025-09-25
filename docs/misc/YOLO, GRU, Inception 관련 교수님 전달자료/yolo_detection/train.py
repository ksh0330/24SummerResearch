import os
from ultralytics import YOLO

#환경변수 관련 오류 해결 코드
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# Load a model
model = YOLO('yolov8m.yaml')  # build a new model from YAML
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

def main():
    results = model.train(save_period=1, data='C:/Users/user/Desktop/Lab/VITAS/Project/yolov8/data/data.yaml', project='C:/', name='0406', epochs=250, imgsz=640, device=0, batch=32)

if __name__ == '__main__':
    main()