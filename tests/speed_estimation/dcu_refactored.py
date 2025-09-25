import sys
import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from ultralytics import YOLO
from collections import defaultdict, deque
import torch
import time

# =========================
# 상수 및 설정값
# =========================
WINDOW_TITLE = "CCTV Monitor"
WINDOW_SIZE = (1800, 720)
VIDEO_LABEL_SIZE = (1280, 720)
FPS_DEFAULT = 30

# 색상
RED = (0, 0, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)

# 영역 좌표
AREA_A = np.array([[176, 290], [689, 515], [807, 464], [242, 274]])

# =========================
# 유틸리티 함수
# =========================
def cvimg_to_qpixmap(img):
    rgb_image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

def create_table_widget(parent, x, y, width, height, headers):
    table = QTableWidget(parent)
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setGeometry(x, y, width, height)
    return table

def create_label(parent, x, y, width, height, text, style=""):
    label = QLabel(parent)
    label.setGeometry(x, y, width, height)
    label.setText(text)
    if style:
        label.setStyleSheet(style)
    return label

# =========================
# ViewTransformer 클래스
# =========================
class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

# =========================
# VideoMonitor 클래스
# =========================
class VideoMonitor(QWidget):
    def __init__(self, video_source):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.setGeometry(0, 0, *WINDOW_SIZE)

        # YOLO 모델 로드
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO("yolov8s.pt").to(self.device)

        # 영역 및 변환기
        self.areas = {"A": AREA_A}
        self.T_W, self.T_H = 5, 30
        target = np.array([[0, 0], [self.T_W, 0], [0, self.T_H], [self.T_W, self.T_H]])
        self.transformerA = ViewTransformer(source=AREA_A, target=target)

        # 비디오 캡처
        self.cap = cv.VideoCapture(video_source)
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            sys.exit()
        self.fps = self.cap.get(cv.CAP_PROP_FPS) or FPS_DEFAULT

        # =========================
        # GUI 위젯 생성
        # =========================
        self.video_label = create_label(self, 0, 0, *VIDEO_LABEL_SIZE, "", "")
        self.is_paused = False
        self.play_pause_button = QPushButton("일시정지", self)
        self.play_pause_button.setGeometry(1300, 650, 200, 60)
        self.play_pause_button.clicked.connect(self.toggle_play_pause)

        self.show_bbox = True
        self.toggle_button = QPushButton("객체 BBOX\nOFF", self)
        self.toggle_button.setGeometry(1550, 650, 200, 60)
        self.toggle_button.clicked.connect(self.toggle_display)

        self.text_table_widget = create_label(self, 1300, 0, 300, 20, "모든 객체 정보")
        self.table_widget = create_table_widget(self, 1300, 20, 420, 100, ["Track ID", "Class", "Confidence", "위치(X,Y)"])
        self.text_a_area_table_widget = create_label(self, 1300, 140, 300, 20, "A 영역 내 객체 정보")
        self.a_area_table_widget = create_table_widget(self, 1300, 160, 420, 100, ["Track ID", "Class", "위치(x,y)", "속도(km/h)"])
        self.text_b_area_table_widget = create_label(self, 1300, 280, 300, 20, "B 영역 내 객체 정보")
        self.b_area_table_widget = create_table_widget(self, 1300, 300, 420, 100, ["Track ID", "Class", "위치(x,y)", "속도(km/h)"])
        self.collision_label = create_label(self, 1300, 420, 500, 50, "위험도: 안전", "font-size: 30px; color: red;")
        self.text_label = create_label(self, 1300, 500, 500, 50, "객체 탐지 중...", "font-size: 20px;")

        # =========================
        # 기타 변수 초기화
        # =========================
        self.FPS = self.fps
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(int(self.FPS))

        self.last_frame = None
        self.last_results = None

        self.T_FPS = 10
        self.PREDICT = 15
        self.track_history = defaultdict(lambda: deque(maxlen=self.T_FPS))
        self.trackmA_history = defaultdict(lambda: deque(maxlen=self.T_FPS))
        self.trackmB_history = defaultdict(lambda: deque(maxlen=self.T_FPS))
        self.predictA_history = defaultdict(lambda: deque(maxlen=self.PREDICT))
        self.predictB_history = defaultdict(lambda: deque(maxlen=self.PREDICT))

        self.A_slope, self.B_slope = 0, 0
        self.av, self.bv = 1, 1
        self.A_v_list, self.B_v_list = [], []

        self.current_a_position = None
        self.current_b_position = None
        self.current_a_speed = None
        self.current_b_speed = None

        self.exit_times = {}
        self.active_tracks = {'A': set(), 'B': set()}
        self.out_tracks = {'A': set(), 'B': set()}
        self.out_time_a = None
        self.out_time_b = None
        self.trakcA_len = None
        self.trakcB_len = None

    def Mouse(self, event, x, y, flags, param):
        if event == cv.EVENT_MOUSEMOVE:
            point = [x, y]
            print(point)

    def toggle_play_pause(self):
        if self.is_paused:
            self.timer.start(0)
            self.play_pause_button.setText("일시정지")
        else:
            self.timer.stop()
            self.play_pause_button.setText("재생")
        self.is_paused = not self.is_paused

    def toggle_display(self):
        self.show_bbox = not self.show_bbox
        self.toggle_button.setText("객체 BBOX\nOFF" if self.show_bbox else "객체 BBOX\nON")
        if self.last_frame is not None and self.last_results is not None:
            self.update_display(self.last_frame, self.last_results)

    def update_frame(self):
        start_time = time.time()
        ret, frame = self.cap.read()
        if ret:
            results = self.model.track(frame, conf=0.6, imgsz=(736, 1280), persist=True, verbose=False)
            self.last_frame = frame.copy()
            self.last_results = results
            #cv.imshow("T_TEST", frame)
            self.update_display(frame, results)
            elapsed_time = time.time() - start_time
            wait_time = max(1.0 / self.FPS - elapsed_time, 0)
            time.sleep(wait_time)
            print(f"FPS: {self.FPS:.2f}")
        else:
            self.cap.release()

    def calculate_speed(self, slope):
        v = round((slope * 3.6 * int(self.FPS)), 0)
        v1 = round((slope * 0.0025 * 30), 2)
        v2 = round((v1 * 3.6), 2)
        v3 = round((v2 * 7.5), 2)
        speed = v
        return speed

    def update_display(self, frame, results):
        self.table_widget.setRowCount(0)
        self.a_area_table_widget.setRowCount(0)
        self.b_area_table_widget.setRowCount(0)
        cv.polylines(frame, [np.array(self.areas['A'], np.int32)], True, RED, 1)
        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            boxes_C = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            names = results[0].boxes.cls.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            for box, box_C, track_id, name, conf in zip(boxes, boxes_C, track_ids, names, confs):
                x1, y1, x2, y2 = map(int, box)
                x, y, w, h = map(int, box_C)
                cls_name = self.model.names[int(name)]
                track = self.track_history[track_id]
                track.append((x, y))
                transformed_pointA = self.transformerA.transform_points(np.array([[x, y]]))
                transformed_textA = f'({int(transformed_pointA[0][0])}, {int(transformed_pointA[0][1])})'
                if self.show_bbox:
                    in_A = cv.pointPolygonTest(np.array(self.areas['A'], np.int32), (x, y), False) >= 0
                    if in_A and cls_name == 'car':
                        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv.putText(frame, f'ID: {int(track_id)} {cls_name} {self.av}',
                                   (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 1.5, GREEN, 5)
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv.polylines(frame, [points], isClosed=False, color=GREEN, thickness=3)
                        self.active_tracks['A'].add(track_id)
                        self.trackmA_history[track_id].append(transformed_pointA)
                        self.A_v_list.append(int(transformed_pointA[0][-1]))
                        self.A_slope = round(abs(self.A_v_list[-1] - self.A_v_list[0]) / len(self.A_v_list), 2)
                        self.av = self.calculate_speed(self.A_slope)
                        self.current_a_position = transformed_pointA[0][-1]
                        self.current_a_speed = self.av
                        row_position = self.a_area_table_widget.rowCount()
                        self.a_area_table_widget.insertRow(row_position)
                        self.a_area_table_widget.setItem(row_position, 0, QTableWidgetItem(str(int(track_id))))
                        self.a_area_table_widget.setItem(row_position, 1, QTableWidgetItem(cls_name))
                        self.a_area_table_widget.setItem(row_position, 2, QTableWidgetItem(transformed_textA))
                        self.a_area_table_widget.setItem(row_position, 3, QTableWidgetItem(str(self.av)))
                self.collision_label.setText(f"속도: {self.av}")
                row_position = self.table_widget.rowCount()
                self.table_widget.insertRow(row_position)
                self.table_widget.setItem(row_position, 0, QTableWidgetItem(str(int(track_id))))
                self.table_widget.setItem(row_position, 1, QTableWidgetItem(cls_name))
                self.table_widget.setItem(row_position, 2, QTableWidgetItem(f'{conf:.2f}'))
                self.table_widget.setItem(row_position, 3, QTableWidgetItem(f'({x}, {y})'))
        frame = cv.resize(frame, (1280, 720))
        rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        self.cap.release()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    video_source = r"D:\REAL\DCU_30.mp4"  # 또는 비디오 파일 경로
    window = VideoMonitor(video_source)
    window.show()
    sys.exit(app.exec_()) 