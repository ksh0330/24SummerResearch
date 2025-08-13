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

        # Load YOLO Model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO("yolov8s.pt").to(self.device)

        # Areas & colors
        #Hard coding :(
        self.areas = {
            "A": [(176, 290), (689, 515), (807, 464), (242, 274)]
        }
        self.RED, self.BLACK, self.WHITE, self.BLUE, self.GREEN = (0, 0, 255), (0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0)
        self.T_W, self.T_H = 5, 30

        cv.namedWindow('Estimation_S&V')
        cv.setMouseCallback('Estimation_S&V', self.Mouse)

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
        self.play_pause_button = QPushButton("Pause", self)
        self.play_pause_button.setGeometry(1300, 650, 200, 60)
        self.play_pause_button.clicked.connect(self.toggle_play_pause)

        # ToggleButten_BBox_ObjectInform
        self.show_bbox = True
        self.toggle_button = QPushButton("Object BBOX\nOFF", self)
        self.toggle_button.setGeometry(1550, 650, 200, 60)
        self.toggle_button.clicked.connect(self.toggle_display)

        # Table widget to display object information(First Table)
        self.text_table_widget = QLabel(self)
        self.text_table_widget.setGeometry(1300, 0, 300, 20)
        self.text_table_widget.setText("Information of all")

        self.table_widget = QTableWidget(self)
        self.table_widget.setColumnCount(4)
        self.table_widget.setHorizontalHeaderLabels(["Track ID", "Class", "Confidence", "Posi(X,Y)"])
        self.table_widget.setGeometry(1300, 20, 420, 100)

        # Table widget to display object in area A(Second Table)
        self.text_a_area_table_widget = QLabel(self)
        self.text_a_area_table_widget.setGeometry(1300, 140, 300, 20)
        self.text_a_area_table_widget.setText("Inform in Area A")

        self.a_area_table_widget = QTableWidget(self)
        self.a_area_table_widget.setColumnCount(4)
        self.a_area_table_widget.setHorizontalHeaderLabels(["Track ID", "Class", "Posi(x,y)", "Speed(km/h)"])
        self.a_area_table_widget.setGeometry(1300, 160, 420, 100)

        # Predict Situation Label
        self.collision_label = QLabel(self)
        self.collision_label.setGeometry(1300, 420, 500, 50)
        self.collision_label.setStyleSheet("font-size: 30px; color: red;")
        self.collision_label.setText("Situation: Safe")

        # It Maybe Change to ComStatus
        self.text_label = QLabel(self)
        self.text_label.setGeometry(1300, 500, 500, 50)
        self.text_label.setStyleSheet("font-size: 20px;")
        self.text_label.setText("Object Detecting...")

        self.FPS = self.fps

        # Timer setup for frame updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(int(self.FPS))

        # Save Last Frame
        self.last_frame = None
        self.last_results = None

        # Initialize the ViewTransformers for areas A and B
        sourceA = np.array([[176, 290], [242, 274], [689, 515], [807, 464]])
        target = np.array([[0, 0], [self.T_W, 0], [0, self.T_H], [self.T_W, self.T_H]])
        self.transformerA = ViewTransformer(source=sourceA, target=target)

        # Initialize variables for drawing
        self.T_FPS = 10
        self.PREDICT = 15
        self.track_history = defaultdict(lambda: deque(maxlen=self.T_FPS))
        self.trackmA_history = defaultdict(lambda: deque(maxlen=self.T_FPS))
        self.predictA_history = defaultdict(lambda: deque(maxlen=self.PREDICT))
        self.A_slope = 0
        self.av = 1
        self.A_v_list = []
        self.current_a_position = None
        self.current_a_speed = None

        # Dic for record of object exit times
        self.exit_times = {}

        # Dic for object ID
        self.active_tracks = {'A': set(), 'B': set()}

        # Dic for out object ID
        self.out_tracks = {'A': set()}
        self.out_time_a = None

        self.trakcA_len = None

    # Print Mouse Point Pixel Position
    def Mouse(self, event, x, y, flags, param):
        if event == cv.EVENT_MOUSEMOVE:
            point = [x, y]
            print(point)

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
            self.toggle_button.setText("Object BBOX\nOFF")
        else:
            self.toggle_button.setText("Object BBOX\nON")

        if self.last_frame is not None and self.last_results is not None:
            self.update_display(self.last_frame, self.last_results)

    def update_frame(self):
        start_time = time.time()  #Record Frame Start Time
        ret, frame = self.cap.read()
        if ret:
            # ObjectTracking use YOLO
            results = self.model.track(frame, conf=0.6, imgsz=(736, 1280), persist=True, verbose=False)
            self.last_frame = frame.copy()
            self.last_results = results
            cv.imshow("Estimation_S&V", frame)

            # Update Display
            self.update_display(frame, results)
            # Print FPS Calculate FrameRate
            #end_time = time.time()
            #self.FPS = 1 / (end_time - start_time)

            # Tuning FrameRate, Keep Origin FPS(Ex FPS=24)
            elapsed_time = time.time() - start_time
            wait_time = max(1.0 / self.FPS - elapsed_time, 0) 
            time.sleep(wait_time)
            print(f"FPS: {self.FPS:.2f}")
        else:
            self.cap.release()

    def calculate_speed(self, slope):
        v = round((slope * 3.6 * int(self.FPS)), 0)
        return v


    def update_display(self, frame, results):
        self.table_widget.setRowCount(0)
        self.a_area_table_widget.setRowCount(0)

        cv.polylines(frame, [np.array(self.areas['A'], np.int32)], True, self.RED, 1)

        # Drawing BBox each object
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
                    # Check object in area A
                    in_A = cv.pointPolygonTest(np.array(self.areas['A'], np.int32), (x, y), False) >= 0

                    if in_A and cls_name == 'car':
                        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv.putText(frame, f'ID: {int(track_id)} {cls_name} {self.av}',
                                   (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 1.5, self.GREEN, 5)

                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv.polylines(frame, [points], isClosed=False, color=self.GREEN, thickness=3)

                        self.active_tracks['A'].add(track_id)

                        # Calculate speed In area A
                        self.trackmA_history[track_id].append(transformed_pointA)
                        self.A_v_list.append(int(transformed_pointA[0][-1]))
                        self.A_slope = round(abs(self.A_v_list[-1] - self.A_v_list[0]) / len(self.A_v_list), 2)

                        self.av = self.calculate_speed(self.A_slope)
                        self.current_a_position = transformed_pointA[0][-1]
                        self.current_a_speed = self.av

                        # Update area A table
                        row_position = self.a_area_table_widget.rowCount()
                        self.a_area_table_widget.insertRow(row_position)
                        self.a_area_table_widget.setItem(row_position, 0, QTableWidgetItem(str(int(track_id))))
                        self.a_area_table_widget.setItem(row_position, 1, QTableWidgetItem(cls_name))
                        self.a_area_table_widget.setItem(row_position, 2, QTableWidgetItem(transformed_textA))
                        self.a_area_table_widget.setItem(row_position, 3, QTableWidgetItem(str(self.av)))


                self.collision_label.setText(f"Speed: {self.av}")

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
    video_source = r"D:\REAL\DCU_30.mp4" #Change dir to '0' When use wecam
    window = VideoMonitor(video_source)
    window.show()
    sys.exit(app.exec_())
