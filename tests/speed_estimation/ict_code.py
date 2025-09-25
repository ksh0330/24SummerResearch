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
import socket
import threading

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

        # 서버 소켓 설정
        self.server_host = '192.168.2.13'
        self.server_port = 8878
        self.clients = []
        self.start_server()

        # Areas and colors
        self.areas = {
            "B": [(1116, 52), (775, 251), (966, 337), (1244, 87)],
            "A": [(74, 85), (361, 337), (548, 251), (200, 51)]
        }
        self.RED, self.BLACK, self.WHITE, self.BLUE, self.GREEN = (0, 0, 255), (0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0)
        self.T_W, self.T_H = 160, 720

        # Video capture setup
        self.cap = cv.VideoCapture(video_source)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 736)
        self.cap.set(cv.CAP_PROP_FPS, 30)
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            sys.exit()

        # Video display widget
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(1280, 720)

        # Play/Pause button
        self.is_paused = False
        self.play_pause_button = QPushButton("긴급상황접수", self)
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

        # Timer setup for frame updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

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
        self.a_list, self.b_list = [], []

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

    def calculate_time_to_safe(self, speed):
        max_speed = 40  # 가장 빠른 속도
        min_speed = 10  # 가장 느린 속도
        min_time = 0.5  # 가장 빠를 때 추적 시간 (1초)
        max_time = 5  # 가장 느릴 때 추적 시간 (5초)

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
            self.play_pause_button.setText("긴급상황접수")

        else:
            self.timer.stop()
            self.play_pause_button.setText("긴급상황\n접수 취소")
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

    def evaluate_collision_risk(self):
        if self.current_a_position is not None and self.current_b_position is not None:
            distance = abs(self.current_a_position - self.current_b_position)

            if distance < 120:
                return "위험"
            elif distance < 350:
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

        collision_status = "안전"
        current_time = time.time()

        cv.polylines(frame, [np.array(self.areas['A'], np.int32)], True, self.RED, 1)
        cv.polylines(frame, [np.array(self.areas['B'], np.int32)], True, self.RED, 1)

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
                        self.A_v_list.append(int(transformed_pointA[0][-1]))
                        self.A_slope = round(abs(self.A_v_list[-1] - self.A_v_list[0]) / len(self.A_v_list), 2)
                        self.av = self.calculate_speed(self.A_slope)
                        self.current_a_position = transformed_pointA[0][-1]
                        self.current_a_speed = self.av

                        trackA = self.predictA_history[track_id]
                        trackA.append(((transformed_pointA[0][0]), (transformed_pointA[0][1])))
                        pointsA = np.hstack(trackA).astype(np.int32).reshape((-1, 1, 2))

                        # A 영역 테이블 업데이트
                        row_position = self.a_area_table_widget.rowCount()
                        self.a_area_table_widget.insertRow(row_position)
                        self.a_area_table_widget.setItem(row_position, 0, QTableWidgetItem(str(int(track_id))))
                        self.a_area_table_widget.setItem(row_position, 1, QTableWidgetItem(cls_name))
                        self.a_area_table_widget.setItem(row_position, 2, QTableWidgetItem(transformed_textA))
                        self.a_area_table_widget.setItem(row_position, 3, QTableWidgetItem(str(self.av)))

                        self.time_to_track_a = self.calculate_time_to_safe(self.av)

                    elif track_id in self.active_tracks['A']:
                        # 객체가 A 영역을 벗어났을 때
                        if self.out_time_a is None:
                            self.out_time_a = current_time
                            self.out_tracks['A'].add(track_id)

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
                        self.bv = self.calculate_speed(self.B_slope)
                        self.current_b_position = transformed_pointB[0][1]
                        self.current_b_speed = self.bv

                        trackB = self.predictB_history[track_id]
                        trackB.append(((transformed_pointB[0][0]), (transformed_pointB[0][1])))
                        pointsB = np.hstack(trackB).astype(np.int32).reshape((-1, 1, 2))

                        # B 영역 테이블 업데이트
                        row_position = self.b_area_table_widget.rowCount()
                        self.b_area_table_widget.insertRow(row_position)
                        self.b_area_table_widget.setItem(row_position, 0, QTableWidgetItem(str(int(track_id))))
                        self.b_area_table_widget.setItem(row_position, 1, QTableWidgetItem(cls_name))
                        self.b_area_table_widget.setItem(row_position, 2, QTableWidgetItem(transformed_textB))
                        self.b_area_table_widget.setItem(row_position, 3, QTableWidgetItem(str(self.bv)))

                        self.time_to_track_b = self.calculate_time_to_safe(self.bv)

                    elif track_id in self.active_tracks['B']:
                        # 객체가 B 영역을 벗어났을 때
                        if self.out_time_b is None:
                            self.out_time_b = current_time
                            self.out_tracks['B'].add(track_id)

                    # 충돌 예측
                    if self.current_a_position and self.current_b_position is not None:
                        if self.current_a_position >= 500 or self.current_b_position >= 500:
                            collision_status = self.evaluate_collision_risk()
                            self.collision_label.setText(f"위험도: {collision_status}")

                            if collision_status == "위험":
                                self.broadcast_message(collision_status)
                            elif collision_status == "주의":
                                self.broadcast_message(collision_status)
                            else:
                                self.broadcast_message("안전")

                        elif (self.current_a_position is None) != (self.current_b_position is None):
                            collision_status = "주의"
                        else:
                            collision_status = "안전"

                    if collision_status == "None":
                        collision_status = "안전"

                row_position = self.table_widget.rowCount()
                self.table_widget.insertRow(row_position)
                self.table_widget.setItem(row_position, 0, QTableWidgetItem(str(int(track_id))))
                self.table_widget.setItem(row_position, 1, QTableWidgetItem(cls_name))
                self.table_widget.setItem(row_position, 2, QTableWidgetItem(f'{conf:.2f}'))
                self.table_widget.setItem(row_position, 3, QTableWidgetItem(f'({x}, {y})'))

        if self.out_time_a is not None:
            # 영역 벗어난 후 측정중(위험 상황 or 아닌 상황)
            # 벗어난 후 시간 계산

            if results and results[0].boxes.id is not None:
                self.exit_times[(track_id, 'A')] = self.out_time_a + self.time_to_track_a

                if track_id not in self.a_list:
                    self.a_list.append(track_id)
                    at = track_id

                # 벗어난 이후
                if current_time - self.exit_times[(at, 'A')] > 0:
                    if collision_status == "안전" or collision_status == "주의":
                        self.active_tracks['A'].discard(at)
                        del self.exit_times[(at, 'A')]  # 안전 판단 후 시간 삭제
                        self.out_time_a = None
                        self.current_a_position = None
                        self.av = 1
                        print("A OUT => SAFE")

        if self.out_time_b is not None:
            # 영역 벗어난 후 측정중(위험 상황 or 아닌 상황)
            # 벗어난 후 시간 계산
            if results and results[0].boxes.id is not None:
                self.exit_times[(track_id, 'B')] = self.out_time_b + self.time_to_track_b

                if track_id not in self.a_list:
                    self.a_list.append(track_id)
                    bt = track_id

                # 벗어난 이후
                if current_time - self.exit_times[(bt, 'B')] > 0:
                    if collision_status == "안전" or collision_status == "주의":
                        self.active_tracks['B'].discard(bt)
                        del self.exit_times[(bt, 'B')]  # 안전 판단 후 시간 삭제
                        self.out_time_b = None
                        self.current_b_position = None
                        self.av = 1
                        print("B OUT => SAFE")

        # 객체 둘 다 인식이 안되는 경우(둘 다 안전해서 벗어난 경우, 충돌해서 인식 안되는 경우)
        if results and results[0].boxes.id is None:
            if self.out_time_a and self.out_time_b is not None:
                if collision_status == "위험":
                    print("CRASH")
            else:
                print("NO CAR")

        frame = cv.resize(frame, (1280, 720))
        rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def start_server(self):
        server_thread = threading.Thread(target=self._start_server_thread)
        server_thread.start()

    def _start_server_thread(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.server_host, self.server_port))
        server.listen(5)
        print(f"[LISTENING] Server is listening on {self.server_host}:{self.server_port}")

        while True:
            client_socket, addr = server.accept()
            client_handler = threading.Thread(target=self.handle_client, args=(client_socket, addr))
            client_handler.start()

    def broadcast_message(self, message):
        for client in self.clients:
            try:
                client.sendall(message.encode('utf-8'))
            except:
                pass

    def handle_client(self, client_socket, addr):
        print(f"[NEW CONNECTION] {addr} connected.")
        self.clients.append(client_socket)
        try:
            while True:
                message = client_socket.recv(1024).decode('utf-8')
                if not message:
                    break
                print(f"[{addr}] {message}")
        finally:
            print(f"[DISCONNECTED] {addr} disconnected.")
            self.clients.remove(client_socket)
            client_socket.close()

    def closeEvent(self, event):
        self.cap.release()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    video_source = 0
    window = VideoMonitor(video_source)
    window.show()
    sys.exit(app.exec_())