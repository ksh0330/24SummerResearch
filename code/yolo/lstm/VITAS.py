from ultralytics import YOLO
import socket
from _thread import *
import datetime #이미지의 FPS를 측정하기 위해서 필요.

# 모델 불러오기
model = YOLO("E:/캡디/weight/0406_weight/weights/epoch249.pt")

# 소켓 통신

model.predict(source='D:/Straight_Data_0607/Straight_Data_0623/Original/Caution/ACBC_FF1.mp4', imgsz=640, show=True, device=0)