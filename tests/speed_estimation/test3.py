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
        self.setWindowTitle("Video Monitor")
        self.setGeometry(100, 100, 1800, 720)

        # YOLO 모델 로드
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO("epoch52.pt").to(self.device)

        # Areas and colors
        self.areas = {
            "B": [(1116, 52), (775, 251), (966, 337), (1244, 87)],
            "A": [(74, 85), (361, 337), (548, 251), (200, 51)]
        }
        self.RED, self.BLACK, self.WHITE, self.BLUE, self.GREEN = (0, 0, 255), (0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0)
        self.T_W, self.T_H = 160, 720

        # Video capture setup
        self.cap = cv.VideoCapture(video_source)
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            sys.exit()

        # Video display widget
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(1280, 720)

        # Play/Pause button
        self.is_paused = False
        self.play_pause_button = QPushButton("Pause", self)
        self.play_pause_button.setGeometry(1300, 600, 100, 30)
        self.play_pause_button.clicked.connect(self.toggle_play_pause)

        # BBox와 객체 정보 표시 토글 버튼
        self.show_bbox = True
        self.toggle_button = QPushButton("Hide BBox", self)
        self.toggle_button.setGeometry(1300, 640, 100, 30)
        self.toggle_button.clicked.connect(self.toggle_display)

        self.text_table_widget = QLabel(self)
        self.text_table_widget.setGeometry(1300, 0, 300, 20)
        self.text_table_widget.setText("모든 객체 정보")

        # Table widget to display object information
        self.table_widget = QTableWidget(self)
        self.table_widget.setColumnCount(4)
        self.table_widget.setHorizontalHeaderLabels(["Track ID", "Class", "Confidence", "Center (X, Y)"])
        self.table_widget.setGeometry(1300, 20, 420, 100)

        self.text_a_area_table_widget = QLabel(self)
        self.text_a_area_table_widget.setGeometry(1300, 140, 300, 20)
        self.text_a_area_table_widget.setText("A 영역 내 객체 정보")

        # A영역에 있는 객체 정보 테이블 위젯 (두 번째 표)
        self.a_area_table_widget = QTableWidget(self)
        self.a_area_table_widget.setColumnCount(4)
        self.a_area_table_widget.setHorizontalHeaderLabels(["Track ID", "Class", "Transform (X, Y)", "Speed"])
        self.a_area_table_widget.setGeometry(1300, 160, 420, 100)

        self.text_b_area_table_widget = QLabel(self)
        self.text_b_area_table_widget.setGeometry(1300, 280, 300, 20)
        self.text_b_area_table_widget.setText("B 영역 내 객체 정보")

        # B영역에 있는 객체 정보 테이블 위젯 (세 번째 표)
        self.b_area_table_widget = QTableWidget(self)
        self.b_area_table_widget.setColumnCount(4)
        self.b_area_table_widget.setHorizontalHeaderLabels(["Track ID", "Class", "Transform (X, Y)", "Speed"])
        self.b_area_table_widget.setGeometry(1300, 300, 420, 100)

        # 충돌 예측 라벨
        self.collision_label = QLabel(self)
        self.collision_label.setGeometry(1300, 420, 500, 50)
        self.collision_label.setStyleSheet("font-size: 20px; color: red;")
        self.collision_label.setText("위험도: 안전")

        # 텍스트 디스플레이 라벨
        self.text_label = QLabel(self)
        self.text_label.setGeometry(1300, 500, 500, 50)
        self.text_label.setText("객체 탐지 중...")

        # Timer setup for frame updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(0)

        # 마지막 프레임 저장
        self.last_frame = None
        self.last_results = None

        # Initialize the ViewTransformers for areas A and B
        sourceA = np.array([[74, 85], [200, 51], [361, 337], [548, 251]])
        sourceB = np.array([[1116, 52], [1244, 87], [775, 251], [966, 337]])
        target = np.array([[0, 0], [self.T_W, 0], [0, self.T_H], [self.T_W, self.T_H]])
        self.transformerA = ViewTransformer(source=sourceA, target=target)
        self.transformerB = ViewTransformer(source=sourceB, target=target)

        # Initialize variables for drawing
        self.T_FPS = 5
        self.PREDICT = 10
        self.track_history = defaultdict(lambda: deque(maxlen=self.T_FPS))
        self.trackmA_history = defaultdict(lambda: deque(maxlen=self.T_FPS))
        self.trackmB_history = defaultdict(lambda: deque(maxlen=self.T_FPS))
        self.predictA_history = defaultdict(lambda: deque(maxlen=self.PREDICT))
        self.predictB_history = defaultdict(lambda: deque(maxlen=self.PREDICT))

        self.A_slope, self.B_slope = 0, 0
        self.A_v_list, self.B_v_list = [], []


        # 충돌 예측 정보 저장
        self.last_a_position = None
        self.last_b_position = None
        self.last_a_speed = None
        self.last_b_speed = None

        self.current_a_position = None
        self.current_b_position = None
        self.current_a_speed = None
        self.current_b_speed = None

        self.is_in_A = False
        self.is_in_B = False

        # 객체의 출구 시간을 기록하기 위한 딕셔너리
        self.exit_times = {}

        # 각 영역의 현재 트랙킹 중인 객체 ID를 저장하는 딕셔너리
        self.active_tracks = {'A': set(), 'B': set()}


    def calculate_time_to_safe(self, speed):
        max_speed = 25  # 가장 빠른 속도
        min_speed = 5   # 가장 느린 속도
        min_time = 2    # 가장 빠를 때 추적 시간 (1초)
        max_time = 6    # 가장 느릴 때 추적 시간 (5초)

        if speed >= max_speed:
            return min_time
        elif speed <= min_speed:
            return max_time
        else:
            ratio = (max_speed - speed) / (max_speed - min_speed)
            return round(min_time + (max_time - min_time) * ratio, 2)

    def toggle_play_pause(self):
        if self.is_paused:
            self.timer.start(0)
            self.play_pause_button.setText("Pause")
        else:
            self.timer.stop()
            self.play_pause_button.setText("Play")
        self.is_paused = not self.is_paused

    def toggle_display(self):
        self.show_bbox = not self.show_bbox
        if self.show_bbox:
            self.toggle_button.setText("Hide BBox")
        else:
            self.toggle_button.setText("Show BBox")

        if self.last_frame is not None and self.last_results is not None:
            self.update_display(self.last_frame, self.last_results)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # YOLO 모델을 통해 객체 추적
            results = self.model.track(frame, conf=0.6, imgsz=(736, 1280), persist=True, verbose=False)
            self.last_frame = frame.copy()
            self.last_results = results

            # 화면 업데이트
            self.update_display(frame, results)
        else:
            self.cap.release()

    def calculate_speed(self, slope):
        v1 = round((slope * 0.0025 * 30), 2)
        v2 = round((v1 * 3.6), 2)
        v3 = round((v2 * 7.5), 2)
        speed = v3
        return speed

    def predict_future_position(self, track_id, area, position, speed):
        time_to_safe = self.calculate_time_to_safe(speed)
        future_position = position + speed * time_to_safe


        if area == 'A':
            self.current_a_position = future_position

        elif area == 'B':
            self.current_b_position = future_position

    def evaluate_collision_risk(self):
        if self.current_a_position is not None and self.current_b_position is not None:
            distance = abs(self.current_a_position - self.current_b_position)
            #print(self.current_a_position, self.current_b_position)
            #print(distance)
            if distance < 150:
                return "위험"
            elif distance < 300:
                return "주의"
            else:
                return "안전"

    def draw_tracking_path(self, frame, track_history, color=(0, 255, 0)):
        for track_id, points in track_history.items():
            if len(points) > 1:
                points_array = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                cv.polylines(frame, [points_array], isClosed=False, color=color, thickness=2)

    def update_display(self, frame, results):
        self.table_widget.setRowCount(0)
        self.a_area_table_widget.setRowCount(0)
        self.b_area_table_widget.setRowCount(0)

        text_to_display = ""
        collision_status = "안전"
        current_time = time.time()

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
                transformed_pointB = self.transformerB.transform_points(np.array([[x, y]]))

                transformed_textA = f'({int(transformed_pointA[0][0])}, {int(transformed_pointA[0][1])})'
                transformed_textB = f'({int(transformed_pointB[0][0])}, {int(transformed_pointB[0][1])})'

                if self.show_bbox:
                    # A와 B 영역 내에 있는지 확인
                    in_A = cv.pointPolygonTest(np.array(self.areas['A'], np.int32), (x, y), False) >= 0
                    in_B = cv.pointPolygonTest(np.array(self.areas['B'], np.int32), (x, y), False) >= 0

                    if in_A:
                        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv.putText(frame, f'ID: {int(track_id)} {cls_name}',
                                   (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, self.GREEN, 2)

                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv.polylines(frame, [points], isClosed=False, color=self.GREEN, thickness=3)

                        self.active_tracks['A'].add(track_id)

                        # A 영역에서의 속도 계산
                        self.trackmA_history[track_id].append(transformed_pointA)
                        self.A_v_list.append(int(transformed_pointA[0][1]))
                        self.A_slope = round(abs(self.A_v_list[-1] - self.A_v_list[0]) / len(self.A_v_list), 2)
                        av3 = self.calculate_speed(self.A_slope)

                        self.current_a_position = transformed_pointA[0][1]
                        self.current_a_speed = av3

                        # A 영역 테이블 업데이트
                        row_position = self.a_area_table_widget.rowCount()
                        self.a_area_table_widget.insertRow(row_position)
                        self.a_area_table_widget.setItem(row_position, 0, QTableWidgetItem(str(int(track_id))))
                        self.a_area_table_widget.setItem(row_position, 1, QTableWidgetItem(cls_name))
                        self.a_area_table_widget.setItem(row_position, 2, QTableWidgetItem(transformed_textA))
                        self.a_area_table_widget.setItem(row_position, 3, QTableWidgetItem(str(av3)))
                        #text_to_display += f"A 영역: ID: {int(track_id)}, Name: {cls_name}, 변환된 좌표: {transformed_textA}, 속도: {av3}\n"

                        time_to_track_a = self.calculate_time_to_safe(av3)
                        self.exit_times[track_id] = current_time + time_to_track_a
                    elif track_id in self.active_tracks['A']:
                        # 객체가 A 영역을 벗어났을 때
                        if (track_id, 'A') not in self.exit_times:
                            # 벗어난 순간의 시간을 기록
                            self.exit_times[(track_id, 'A')] = current_time

                        time_to_safe_a = self.calculate_time_to_safe(self.current_a_speed)
                        #print(f"A: {time_to_safe_a}, {current_time - self.exit_times[(track_id, 'A')]}")
                        if current_time - self.exit_times[(track_id, 'A')] > time_to_safe_a:
                            self.active_tracks['A'].discard(track_id)
                            del self.exit_times[(track_id, 'A')]  # 안전 판단 후 시간 삭제
                        else:
                            self.predict_future_position(track_id, 'A', self.current_a_position, self.current_a_speed)
                            #text_to_display += f"A 영역을 벗어난 객체 {int(track_id)} 교차로 진행중\n"


                    if in_B:
                        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv.putText(frame, f'ID: {int(track_id)} {cls_name}',
                                   (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, self.GREEN, 2)

                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv.polylines(frame, [points], isClosed=False, color=self.GREEN, thickness=3)

                        self.active_tracks['B'].add(track_id)

                        # B 영역에서의 속도 계산
                        self.trackmB_history[track_id].append(transformed_pointB)
                        self.B_v_list.append(int(transformed_pointB[0][1]))
                        self.B_slope = round(abs(self.B_v_list[-1] - self.B_v_list[0]) / len(self.B_v_list), 2)

                        bv3 = self.calculate_speed(self.B_slope)

                        self.current_b_position = transformed_pointB[0][1]
                        self.current_b_speed = bv3

                        # B 영역 테이블 업데이트
                        row_position = self.b_area_table_widget.rowCount()
                        self.b_area_table_widget.insertRow(row_position)
                        self.b_area_table_widget.setItem(row_position, 0, QTableWidgetItem(str(int(track_id))))
                        self.b_area_table_widget.setItem(row_position, 1, QTableWidgetItem(cls_name))
                        self.b_area_table_widget.setItem(row_position, 2, QTableWidgetItem(transformed_textB))
                        self.b_area_table_widget.setItem(row_position, 3, QTableWidgetItem(str(bv3)))
                        #text_to_display += f"B 영역: ID: {int(track_id)}, Name: {cls_name}, 변환된 좌표: {transformed_textB}, 속도: {bv3}\n"
                    elif track_id in self.active_tracks['B']:
                        # 객체가 B 영역을 벗어났을 때
                        if (track_id, 'B') not in self.exit_times:
                            # 벗어난 순간의 시간을 기록
                            self.exit_times[(track_id, 'B')] = current_time

                        time_to_safe_b = self.calculate_time_to_safe(self.current_b_speed)
                        #print(f"B: {time_to_safe_b}, {current_time - self.exit_times[(track_id, 'B')]}")
                        if current_time - self.exit_times[(track_id, 'B')] > time_to_safe_b:
                            self.active_tracks['B'].discard(track_id)
                            del self.exit_times[(track_id, 'B')]  # 안전 판단 후 시간 삭제
                        else:
                            self.predict_future_position(track_id, 'B', self.current_b_position, self.current_b_speed)
                            #text_to_display += f"B 영역을 벗어난 객체 {int(track_id)} 교차로 진행중\n"


                    # 충돌 예측
                    if self.active_tracks['A'] and self.active_tracks['B']:
                        #collision_status = self.evaluate_collision_risk()
                        if (track_id, 'A') in self.exit_times and (track_id, 'B') in self.exit_times:
                            collision_status = "위험"
                            print("위험")
                        elif ((track_id, 'A') in self.exit_times and (track_id, 'B') not in self.exit_times) or ((track_id, 'B') in self.exit_times and (track_id, 'A') not in self.exit_times):
                            # A의 time_to_safe_a는 있고 B의 time_to_safe_b가 없는 경우
                            if self.current_a_position >= 720 or self.current_b_position >= 720:
                                continue
                            collision_status = self.evaluate_collision_risk()
                            if collision_status == "위험":
                                print("통신으로 차량  제어1")#위치 값이 좀 더 작은 놈을 정지 혹은 속도가 느린 놈을 정지
                        else:
                            if self.current_a_position >= 720 or self.current_b_position >= 720:
                                continue
                            collision_status = self.evaluate_collision_risk()
                            if collision_status == "위험":
                                print("통신으로 차량  제어2")#위치 값이 좀 더 작은 놈을 정지 혹은 속도가 느린 놈을 정지




                self.collision_label.setText(f"위험도: {collision_status}")
                #self.text_label.setText(text_to_display or "객체 탐지 중...")



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
    video_source = r"D:\Straight_Data_0623\Safe\ACBC_F3F.mp4" # 웹캠은 주소 대신 그냥 0으로 바꿔주면 됨
    window = VideoMonitor(video_source)
    window.show()
    sys.exit(app.exec_())