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

class VideoMonitor(QWidget):
    def __init__(self, video_source):
        super().__init__()
        self.setWindowTitle("CCTV Monitor")
        self.setGeometry(0, 0, 1800, 720)

        # YOLO 모델 로드
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO("yolov8s.pt").to(self.device)

        # Areas and colors
        self.areas = {
            "A": [(562, 134), (63, 458), (344, 542), (694, 154)]
        }
        self.RED, self.BLACK, self.WHITE, self.BLUE, self.GREEN = (0, 0, 255), (0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0)
        self.T_W, self.T_H = 5, 20

        cv.namedWindow('T_TEST')
        cv.setMouseCallback('T_TEST', self.Mouse)

        # Video capture setup
        self.cap = cv.VideoCapture(video_source)
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            sys.exit()

        self.fps = self.cap.get(cv.CAP_PROP_FPS)
        print(f"Video FPS: {self.fps}")

        # Video display widget
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(1280, 720)

        # Play/Pause button
        self.is_paused = False
        self.play_pause_button = QPushButton("일시정지", self)
        self.play_pause_button.setGeometry(1300, 650, 200, 60)
        self.play_pause_button.clicked.connect(self.toggle_play_pause)

        # BBox와 객체 정보 표시 토글 버튼
        self.show_bbox = True
        self.toggle_button = QPushButton("객체 BBOX\nOFF", self)
        self.toggle_button.setGeometry(1550, 650, 200, 60)
        self.toggle_button.clicked.connect(self.toggle_display)

        self.text_table_widget = QLabel(self)
        self.text_table_widget.setGeometry(1300, 0, 300, 20)
        self.text_table_widget.setText("모든 객체 정보")

        # Table widget to display object information
        self.table_widget = QTableWidget(self)
        self.table_widget.setColumnCount(4)
        self.table_widget.setHorizontalHeaderLabels(["Track ID", "Class", "Confidence", "위치(X,Y)"])
        self.table_widget.setGeometry(1300, 20, 420, 100)

        self.text_a_area_table_widget = QLabel(self)
        self.text_a_area_table_widget.setGeometry(1300, 140, 300, 20)
        self.text_a_area_table_widget.setText("A 영역 내 객체 정보")

        # A영역에 있는 객체 정보 테이블 위젯 (두 번째 표)
        self.a_area_table_widget = QTableWidget(self)
        self.a_area_table_widget.setColumnCount(4)
        self.a_area_table_widget.setHorizontalHeaderLabels(["Track ID", "Class", "위치(x,y)", "속도(km/h)"])
        self.a_area_table_widget.setGeometry(1300, 160, 420, 100)

        self.text_b_area_table_widget = QLabel(self)
        self.text_b_area_table_widget.setGeometry(1300, 280, 300, 20)
        self.text_b_area_table_widget.setText("B 영역 내 객체 정보")

        # B영역에 있는 객체 정보 테이블 위젯 (세 번째 표)
        self.b_area_table_widget = QTableWidget(self)
        self.b_area_table_widget.setColumnCount(4)
        self.b_area_table_widget.setHorizontalHeaderLabels(["Track ID", "Class", "위치(x,y)", "속도(km/h)"])
        self.b_area_table_widget.setGeometry(1300, 300, 420, 100)

        # 충돌 예측 라벨
        self.collision_label = QLabel(self)
        self.collision_label.setGeometry(1300, 420, 500, 50)
        self.collision_label.setStyleSheet("font-size: 30px; color: red;")
        self.collision_label.setText("위험도: 안전")

        # 텍스트 디스플레이 라벨 -> 통신 상태 표시로 전환
        self.text_label = QLabel(self)
        self.text_label.setGeometry(1300, 500, 500, 50)
        self.text_label.setStyleSheet("font-size: 20px;")
        self.text_label.setText("객체 탐지 중...")

        self.FPS = self.fps

        # Timer setup for frame updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(int(self.FPS))

        # 마지막 프레임 저장
        self.last_frame = None
        self.last_results = None

        # Initialize the ViewTransformers for areas A and B
        sourceA = np.array([[562, 134], [694, 154], [63, 458], [344, 542]])


        target = np.array([[0, 0], [self.T_W, 0], [0, self.T_H], [self.T_W, self.T_H]])
        self.transformerA = ViewTransformer(source=sourceA, target=target)


        # Initialize variables for drawing
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

        # 객체의 출구 시간을 기록하기 위한 딕셔너리
        self.exit_times = {}

        # 각 영역의 현재 트랙킹 중인 객체 ID를 저장하는 딕셔너리
        self.active_tracks = {'A': set(), 'B': set()}

        # 각 영역에서 나온 객체의 ID를 저장하는 딕셔너리
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
        if self.show_bbox:
            self.toggle_button.setText("객체 BBOX\nOFF")
        else:
            self.toggle_button.setText("객체 BBOX\nON")

        if self.last_frame is not None and self.last_results is not None:
            self.update_display(self.last_frame, self.last_results)

    def update_frame(self):
        start_time = time.time()  # 프레임 시작 시간 기록
        ret, frame = self.cap.read()
        if ret:
            # YOLO 모델을 통해 객체 추적
            results = self.model.track(frame, conf=0.6, imgsz=(736, 1280), persist=True, verbose=False)
            self.last_frame = frame.copy()
            self.last_results = results
            cv.imshow("T_TEST", frame)

            # 화면 업데이트
            self.update_display(frame, results)
            # 프레임 처리 속도를 계산하여 FPS를 출력
            #end_time = time.time()
            #self.FPS = 1 / (end_time - start_time)


            # 프레임 처리 속도를 조정하여 원래 FPS(예: 24FPS)를 유지
            elapsed_time = time.time() - start_time
            wait_time = max(1.0 / self.FPS - elapsed_time, 0)  # 원래 fps(예: 24)에 맞춘 처리 대기 시간 계산
            time.sleep(wait_time)
            print(f"FPS: {self.FPS:.2f}")
        else:
            self.cap.release()

    def calculate_speed(self, slope):
        v = round((slope * 3.6 * int(self.FPS)), 0)
        print(v)

        v1 = round((slope * 0.0025 * 30), 2)
        v2 = round((v1 * 3.6), 2)
        v3 = round((v2 * 7.5), 2)
        speed = v
        return speed


    def update_display(self, frame, results):
        self.table_widget.setRowCount(0)
        self.a_area_table_widget.setRowCount(0)
        self.b_area_table_widget.setRowCount(0)


        cv.polylines(frame, [np.array(self.areas['A'], np.int32)], True, self.RED, 1)
        #cv.polylines(frame, [np.array(self.areas['B'], np.int32)], True, self.RED, 1)

        # 각 객체에 대해 BBox 그리기
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
                    # A와 B 영역 내에 있는지 확인
                    in_A = cv.pointPolygonTest(np.array(self.areas['A'], np.int32), (x, y), False) >= 0

                    if in_A and cls_name == 'car':
                        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv.putText(frame, f'ID: {int(track_id)} {cls_name} {self.av}',
                                   (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 1.5, self.GREEN, 5)

                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv.polylines(frame, [points], isClosed=False, color=self.GREEN, thickness=3)

                        self.active_tracks['A'].add(track_id)

                        # A 영역에서의 속도 계산
                        self.trackmA_history[track_id].append(transformed_pointA)
                        self.A_v_list.append(int(transformed_pointA[0][-1]))
                        self.A_slope = round(abs(self.A_v_list[-1] - self.A_v_list[0]) / len(self.A_v_list), 2)
                        #print(self.A_slope)

                        self.av = self.calculate_speed(self.A_slope)
                        self.current_a_position = transformed_pointA[0][-1]
                        self.current_a_speed = self.av

                        # A 영역 테이블 업데이트
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
    video_source = r"D:\REAL\HRC_30.mp4" # 웹캠은 주소 대신 그냥 0으로 바꿔주면 됨
    window = VideoMonitor(video_source)
    window.show()
    sys.exit(app.exec_())