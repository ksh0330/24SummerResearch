import cv2 as cv
import numpy as np
import math
import torch
from ultralytics import YOLO
from collections import defaultdict, deque
import time
import datetime
import os
import joblib
import pandas as pd


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

BLUE, GREEN, RED = (255, 0, 0), (0, 255, 0), (0, 0, 255)
BLACK, GRAY, WHITE = (0, 0, 0), (127, 127, 127), (255, 255, 255)
SKY, MAGENTA, YELLOW = (255, 255, 0), (255, 0, 255), (0, 255, 255)

#area
Aarea = [(74,85), (361,337), (548,251), (200,51)]#2사분면, 3사분면, 4사분면, 1사분면
Barea = [(1116,52), (775,251), (966,337), (1244,87)]#2사분면, 3사분면, 4사분면, 1사분면

T_W, T_H = 160, 720

list_id_inA, list_id_inB = [], []
list_id_A, list_id_B = [], []

l_i_o_A, l_i_o_B = [], []
l_i_p_A, l_i_p_B = [], []
Passed_A, Passed_B = [], []
list_pass_predictA1, list_pass_predictA2, list_pass_predictA3, list_pass_predictB1, list_pass_predictB2, list_pass_predictB3  = [], [], [], [], [], []

csv_A_py, csv_B_py = [], []

#fps_list = []
offset = 0.3
#persent = 7.5 #실제랑 테스트베드 비율

F_V = 160
M_V = 120
S_V = 80
A_V_offset = 0

# class명 이름으로 출력하기 위해 txt파일 읽어옴
my_file = open("313.txt", 'r')
data = my_file.read()
class_list = data.split('\n')
my_file.close()


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

def Mouse(event, x, y, flags, param):
    if event == cv.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

def Distance_dot_line(a, b, c, x1, y1):  # 점의 좌표(x1, y1)와 직선의 기울기, 절편,  점과 직선의 거리 리턴
    numerator = abs(a * x1 + b * y1 + c)
    denominator = math.sqrt(a ** 2 + b ** 2)
    distance = numerator / denominator
    return int(distance)

def Make_line_2dot(dot1, dot2):  # 2 점의 좌표 입력, 기울기와 절편 리턴
    x1, y1 = dot1[0], dot1[1]
    x2, y2 = dot2[0], dot2[1]
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    return a, b, c


T_FPS = 5
PREDICT = 12
track_history = defaultdict(lambda: deque(maxlen=T_FPS))# A, B 원본 창 둘 다 저장

trackmA_history = defaultdict(lambda: deque(maxlen=T_FPS))# A, B 변환 창 둘 다 저장
trackmB_history = defaultdict(lambda: deque(maxlen=T_FPS))# A, B 변환 창 둘 다 저장

predictA_history = defaultdict(lambda: deque(maxlen=PREDICT))
predictB_history = defaultdict(lambda: deque(maxlen=PREDICT))

spla_history = defaultdict(lambda: deque(maxlen=T_FPS))# 좌표 거리값의 차이를 저장하는 리스트
splb_history = defaultdict(lambda: deque(maxlen=T_FPS))# 좌표 거리값의 차이를 저장하는 리스트

sourceA = np.array([[74,85], [200,51], [361,337], [548,251]]) # 2, 1, 3 ,4
sourceB = np.array([[1116,52], [1244,87], [775,251], [966,337]]) # 2, 1, 3 ,4
target = np.array([[0, 0], [T_W, 0], [0, T_H], [T_W, T_H]])

transformerA = ViewTransformer(source=sourceA, target=target)
transformerB = ViewTransformer(source=sourceB, target=target)

imgA = np.array([[74,85], [200,51], [361,337], [548,251]], dtype=np.float32) # 2, 1, 3 ,4
imgB = np.array([[1116,52], [1244,87], [775,251], [966,337]], dtype=np.float32) # 2, 1, 3, 4
trn = np.array([[0, 0], [T_W, 0], [0, T_H], [T_W, T_H]], dtype=np.float32)

matrixA = cv.getPerspectiveTransform(imgA, trn)  # 원본창, 변환창
matrixB = cv.getPerspectiveTransform(imgB, trn)  # 원본창, 변환창

# 모델과 스케일러 불러오기
mlp_model_filename = '(88,76)0610_avgVMLP_best.joblib'
svm_model_filename = '(84,72)0610_avgVSVM_best.joblib'
scaler_filename = '(88,76)0610_avgVscaler.joblib'

mlp = joblib.load(mlp_model_filename)
svm = joblib.load(svm_model_filename)
scaler = joblib.load(scaler_filename)

def main():
    cap = cv.VideoCapture(r"D:\Straight_Data_0607\Warn\ALBC_S3S.mp4")
    if not cap.isOpened():
        print("Can't open video. check your spelling")
        return
    w, h, fps = (int(cap.get(x)) for x in (cv.CAP_PROP_FRAME_WIDTH, cv.CAP_PROP_FRAME_HEIGHT, cv.CAP_PROP_FPS))
    print(cap.get(cv.CAP_PROP_FPS))

    video = cv.VideoWriter(r"D:\KHB_Videos\0610\ALBC_S3S_1.mp4",
                           cv.VideoWriter_fourcc(*'mp4v'),
                           fps, (1600, 720))

    model = YOLO("epoch52.pt").to(device)

    global distCA, distCB
    distCA, distCB = T_H, T_H

    global CC, CT, FPS, FPS15, pt
    CC, CT, FPS, FPS15, pt = 0, 0, 1, 1, 0

    global pvA, pvB, pva, pvb
    pvA, pvB, pva, pvb = 0, 0, 0, 0
    global ETA_A, ETA_B
    ETA_A, ETA_B = 15.0, 15.0

    global situation, mlp_situation, svm_situation
    situation, mlp_situation, svm_situation = -1, -1, -1

    global CTVa, CTVb
    CTVa, CTVb = 0, 0

    #0602
    global PSTART_A, PSTART_B, PEND_A, PEND_B, Ams, Akh, Bms, Bkh, TA, TB
    PSTART_A, PSTART_B, PEND_A, PEND_B, Ams, Akh, Bms, Bkh, TA, TB = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    global out_ETA_A, out_ETA_B
    out_ETA_A, out_ETA_B = 0, 0

    #0604
    global Ayp1, Ayp2, Ayp3, Byp1, Byp2, Byp3
    Ayp1, Ayp2, Ayp3, Byp1, Byp2, Byp3 = 0, 0, 0, 0, 0, 0
    global Ay, By, Ax, Bx
    Ay, By, Ax, Bx = 0, 0, 0, 0
    global AyBy1, AyBy2, AyBy3, ByAy1, ByAy2, ByAy3
    AyBy1, AyBy2, AyBy3, ByAy1, ByAy2, ByAy3 = 0, 0, 0, 0, 0, 0
    global Sresult
    Sresult = 0

    #0606
    global A_slope, B_slope
    A_slope, B_slope = 0, 0

    global colum_passA, colum_passB
    colum_passA, colum_passB = 0, 1


    while True:
        start = datetime.datetime.now()
        ret, frame = cap.read()

        CC += 1
        if CC % T_FPS == 0:
            CTa = CT
            CT = datetime.datetime.now()
            if CTa != 0:
                FPS = (CT - CTa).total_seconds()
                if list_id_A:
                    CTVa += 1
                if list_id_B:
                    CTVb += 1
        if CC % PREDICT == 0:
            pta = pt
            pt = datetime.datetime.now()
            if pta != 0:
                FPS15 = (pt - pta).total_seconds()

        if ret:
            results = model.track(frame, conf=0.6, imgsz=(736, 1280), persist=True, verbose=False)  # verbose=False: 화면 기본 출력값 없애기

            cv.polylines(frame, [np.array(Aarea, np.int32)], True, RED, 1)
            cv.polylines(frame, [np.array(Barea, np.int32)], True, RED, 1)

            cv.putText(frame, ('A'), (490, 330), cv.FONT_HERSHEY_COMPLEX, 0.8, BLACK, 2)
            cv.putText(frame, ('B'), (830, 330), cv.FONT_HERSHEY_COMPLEX, 0.8, BLACK, 2)

            cv.putText(frame, ('A T: ' + str(TA) + ' vs ' + str(out_ETA_A) + ' s'), (400, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
            cv.putText(frame, ('A S: ' + str(int(distCA)) + ' S'), (400, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       WHITE, 2)
            cv.putText(frame, ('A LV: ' + str(Ams) + 'm/s ' + str(Akh) + 'k/h'), (400, 70), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       WHITE, 2)
            cv.putText(frame, ('A slop: ' + str(A_slope)), (400, 90),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       WHITE, 2)
            cv.putText(frame, ('ETA_A: ' + str(ETA_A) + 's'), (400, 110), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       WHITE, 2)

            cv.putText(frame, ('B T: ' + str(TB) + ' vs ' + str(out_ETA_B) + ' s'), (800, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
            cv.putText(frame, ('B S: ' + str(int(distCB)) + ' S'), (800, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       WHITE, 2)
            cv.putText(frame, ('B LV: ' + str(Bms) + ' m/s ' + str(Bkh) + 'k/h'), (800, 70), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       WHITE, 2)
            cv.putText(frame, ('B slop: ' + str(B_slope)), (800, 90),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       WHITE, 2)
            cv.putText(frame, ('ETA_B: ' + str(ETA_B) + 's'), (800, 110), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       WHITE, 2)

            cv.putText(frame, ('5FPS: ' + str(round(FPS, 2)) + 's'), (600, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       RED, 2)
            cv.putText(frame, ('12FPS: ' + str(round(FPS15, 2)) + 's'), (600, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       RED, 2)


            cv.putText(frame, ('mlp_Situation: ' + str(mlp_situation)), (600, 230), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       BLUE, 2)
            cv.putText(frame, ('svm_Situation: ' + str(svm_situation)), (600, 250), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       BLUE, 2)


            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu()  # box 1,2 좌표값 텐서 반환
                boxes_C = results[0].boxes.xywh.cpu()  # box 중앙, 높이, 너비값 텐서 반환
                track_ids = results[0].boxes.id.int().cpu().tolist()  # box id값(숫자로) 반환
                names = results[0].boxes.cls.cpu().tolist()  # box 클래스 숫자로 반환
                con = results[0].boxes.conf.cpu().tolist()  # box 정확도(리스트 숫자로) 반환
            else:
                continue

            resultA = cv.warpPerspective(frame, matrixA, (T_W, T_H))  # 이미지, 행렬, (너비, 높이)
            resultB = cv.warpPerspective(frame, matrixB, (T_W, T_H))  # 이미지, 행렬, (너비, 높이)

            for box, box_C, track_id, name, conF in zip(boxes, boxes_C, track_ids, names, con):
                x1, y1, x2, y2 = map(int,box.tolist())
                x, y, w, h = map(int, box_C.tolist())  # x,y,w,h는 tensor값으로 받음
                CLS = int(name)
                CON = conF  # 정확도 값

                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point

                # 차량 객체 클래스에 해당하는 객체가 인식되었을 때
                if CLS == 2:
                    A = cv.pointPolygonTest(np.array(Aarea, np.int32), (x, y),
                                            False)
                    if A < 0:
                        if str(track_id) not in l_i_o_A and str(track_id) in list_id_inA:
                            l_i_o_A.append(str(track_id))
                            PEND_A = round(time.time(), 2)
                            TA = round(PEND_A - PSTART_A, 2) # 100cm=1M 이동하는데 걸리는 시간
                            print("\nTA: ", TA)
                            print("ETA_B: ", ETA_B)
                            Ams = round((1 / TA),2) #* persent #차량 비율 1:10 도로 비율은 1:7.5
                            Akh = round((Ams * 3.6),2)
                            print('LV_A: '+ str((Ams)) + 'm/s' + ' -> ' + str(Akh) + 'km/h')

                            ETA_A = 0.0
                            pvA = 0
                            out_ETA_A = ETA_B

                            # 여기서 csv 특징 추출 및 예측
                            features = [colum_passA, A_slope, B_slope, distCA, distCB, Ax, Bx]
                            features = scaler.transform([features])

                            # MLP 예측
                            mlp_prediction = mlp.predict(features)
                            mlp_situation = mlp_prediction[0]
                            print("MLP Predicted situation A: ", mlp_situation)

                            # SVM 예측
                            svm_prediction = svm.predict(features)
                            svm_situation = svm_prediction[0]
                            print("SVM Predicted situation A: ", svm_situation)


                        #차량이 영역에서 나오고, 이미 지나간 차량이 아닐 때
                        if PEND_A > 0 and str(track_id) not in Passed_A and str(track_id) in list_id_A:
                            pointsTA = np.array([[x], [y]])
                            pointsTA = transformerA.transform_points(points=pointsTA).astype(int)
                            a, b, c = Make_line_2dot((0, T_H), (T_W, T_H))
                            distCA = Distance_dot_line(a, b, c, pointsTA[0][0], pointsTA[0][1])
                            if distCA >= 0:
                                distCA = distCA * -1
                            if A_slope >= 0:
                                A_slope = A_slope * -1
                            if Akh >= 0:
                                Akh = Akh * -1
                            out_A_time = round(time.time(), 2)  # 나온 시간 계속 업데이트
                            oa = round(abs(out_A_time - PEND_A), 2)
                            # 사고가 발생하지 않았을 때 차량 초기화
                            if oa + FPS15 + offset >= TA and situation != 2:
                                Passed_A.append(str(track_id))
                                ETA_A = 15.0
                                print("Passed A")
                                csv_A_py.clear()

                    if A >= 0:
                        track_mA = trackmA_history[track_id]
                        spla = spla_history[track_id]
                        predictA = predictA_history[track_id]

                        if str(track_id) not in list_id_inA and str(track_id) not in list_id_A:
                            list_id_inA.append(str(track_id))
                            list_id_A.append(str(track_id))

                        cv.circle(frame, (x, y), 2, WHITE, -1)
                        cv.rectangle(frame, (x1, y1), (x2, y2), GREEN, 1)
                        cv.rectangle(frame, (x1, y1 - 14), (x2, y1), GREEN, -1)
                        cv.putText(frame,
                                   'id:' + str(track_id) + ' ' + class_list[0] + ' ' + str(round(CON, 2)) + '%',
                                   (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv.polylines(frame, [points], isClosed=False, color=WHITE, thickness=3)

                        pointsTA = np.array([[x], [y]])
                        pointsTA = transformerA.transform_points(points=pointsTA).astype(int)

                        a, b, c = Make_line_2dot((0, T_H), (T_W, T_H))
                        distCA = Distance_dot_line(a, b, c, pointsTA[0][0], pointsTA[0][1])

                        track_mA.append([pointsTA[0][0], pointsTA[0][1]])
                        predictA.append([pointsTA[0][0], pointsTA[0][1]])
                        Ay = pointsTA[0][1]


                        if pointsTA[0][1] >= 160:
                            if str(track_id) not in l_i_p_A:
                                l_i_p_A.append(str(track_id))
                                PSTART_A = round(time.time(), 2)

                            cv.putText(frame, str(A_slope), (x, y+15), cv.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

                            # 변환 창에 tracking 선분 표시
                            pointsTA = np.hstack(track_mA).astype(np.int32).reshape((-1, 1, 2))
                            cv.polylines(resultA, [pointsTA], isClosed=False, color=WHITE, thickness=3)

                            pointsPA = np.hstack(predictA).astype(np.int32).reshape((-1, 1, 2))
                            pointsPA = np.array([[80, y] for y in pointsPA[:, 0, 1]], dtype=np.int32)
                            pointsPA = pointsPA.reshape((-1, 1, 2))
                            cv.polylines(resultA, [pointsPA], isClosed=False, color=GREEN, thickness=3)

                            csv_A_py.append(pointsPA[-1][0][1])
                            a_slope = round(abs(csv_A_py[-1] - csv_A_py[0]) / len(csv_A_py), 2)
                            A_slope = a_slope
                            Ax = pointsTA[0][0][0]

                            dis = abs(pointsPA[0][0][1] - pointsPA[-1][0][1])
                            if Ayp1 != -1:
                                Ayp1 = pointsPA[-1][0][1] + dis
                            if Ayp2 != -1:
                                Ayp2 = pointsPA[-1][0][1] + dis * 2
                            if Ayp3 != -1:
                                Ayp3 = pointsPA[-1][0][1] + dis * 3

                            cv.circle(resultA, (80, Ayp1), 5, RED, -1)
                            cv.circle(resultA, (80, Ayp2), 5, YELLOW, -1)
                            cv.circle(resultA, (80, Ayp3), 5, GREEN, -1)

                            if Ayp1 >= T_H and not list_pass_predictA1:
                                list_pass_predictA1.append(str(track_id))
                                Ayp1 = -1

                            if Ayp2 >= T_H and not list_pass_predictA2:
                                list_pass_predictA2.append(str(track_id))
                                Ayp2 = -1

                            if Ayp3 >= T_H and not list_pass_predictA3:
                                list_pass_predictA3.append(str(track_id))
                                Ayp3 = -1

                            #매프레임동안 변한 좌표량
                            if len(track_mA) == T_FPS:
                                m_start = track_mA[-1][1]
                                m_end = track_mA[0][1]
                                diss = abs(m_start - m_end)
                                spla.append(diss)
                                # 매프레임마다 측정하고, 변한 좌표량의 평균을 구함
                                summ = sum(spla) / len(spla)

                                if CTVa >= 2:
                                    CTVa = 1

                                if CTVa == 1:
                                    pva = round(summ/FPS, 2) #5프레임동안 변한 좌표량
                                    ETA_A = round(distCA/pva, 2)
                                    pvA = int(pva)# 이거 이유가 필요한데...
                                    CTVa = 0
                                    #print(pva, pvA)


                    B = cv.pointPolygonTest(np.array(Barea, np.int32), (x, y),
                                            False)

                    if B < 0:
                        if str(track_id) not in l_i_o_B and str(track_id) in list_id_inB:
                            l_i_o_B.append(str(track_id))
                            PEND_B = round(time.time(), 2) # 교차로 나온 시간
                            TB = round(PEND_B - PSTART_B, 2)
                            print("\nTB: ", TB)
                            print("ETA_A:", ETA_A)
                            Bms = round((1 / TB),2) #* persent
                            Bkh = round((Bms * 3.6),2)
                            print('LV_B: '+str(Bms) + 'm/s' + ' -> ' + str(Bkh) + 'km/h')


                            ETA_B = 0.0
                            pvB = 0

                            out_ETA_B = ETA_A

                            # 여기서 csv 특징 추출 및 예측
                            features = [colum_passB, A_slope, B_slope, distCA, distCB, Ax, Bx]
                            features = scaler.transform([features])

                            # MLP 예측
                            mlp_prediction = mlp.predict(features)
                            mlp_situation = mlp_prediction[0]
                            print("MLP Predicted situation B: ", mlp_situation)

                            # SVM 예측
                            svm_prediction = svm.predict(features)
                            svm_situation = svm_prediction[0]
                            print("SVM Predicted situation B: ", svm_situation)

                        if PEND_B > 0 and str(track_id) not in Passed_B and str(track_id) in list_id_B:
                            pointsTB = np.array([[x], [y]])
                            pointsTB = transformerA.transform_points(points=pointsTB).astype(int)

                            a, b, c = Make_line_2dot((0, T_H), (T_W, T_H))
                            distCB = Distance_dot_line(a, b, c, pointsTB[0][0], pointsTB[0][1])
                            if distCB >= 0:
                                distCB = distCB * -1
                            if B_slope >= 0:
                                B_slope = B_slope * -1
                            if Bkh >= 0:
                                Bkh = Bkh * -1
                            # print('B', distCB)
                            out_B_time = round(time.time(), 2)  # 나온 시간 계속 업데이트
                            ob = round(abs(out_B_time - PEND_B), 2)
                            if ob + FPS15 + offset >= TB and situation != 2:
                                Passed_B.append(str(track_id))
                                ETA_B = 15.0
                                print("Passed B")
                                csv_B_py.clear()

                    if B >= 0:
                        track_mB = trackmB_history[track_id]
                        splb = splb_history[track_id]
                        predictB = predictB_history[track_id]
                        if str(track_id) not in list_id_inB and str(track_id) not in list_id_B:
                            list_id_inB.append(str(track_id))
                            list_id_B.append(str(track_id))

                        cv.circle(frame, (x, y), 2, WHITE, -1)
                        cv.rectangle(frame, (x1, y1), (x2, y2), GREEN, 1)
                        cv.rectangle(frame, (x1, y1 - 14), (x2, y1), GREEN, -1)
                        cv.putText(frame,
                                   'id:' + str(track_id) + ' ' + class_list[0] + ' ' + str(round(CON, 2)) + '%',
                                   (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv.polylines(frame, [points], isClosed=False, color=WHITE, thickness=3)

                        pointsTB = np.array([[x], [y]])
                        pointsTB = transformerB.transform_points(points=pointsTB).astype(int)

                        a, b, c = Make_line_2dot((0, T_H), (T_W, T_H))
                        distCB = Distance_dot_line(a, b, c, pointsTB[0][0], pointsTB[0][1])

                        track_mB.append([pointsTB[0][0], pointsTB[0][1]])
                        predictB.append([pointsTB[0][0], pointsTB[0][1]])
                        By = pointsTB[0][1]


                        if pointsTB[0][1] >= 160:
                            if str(track_id) not in l_i_p_B:
                                l_i_p_B.append(str(track_id))
                                PSTART_B = round(time.time(), 2)

                            cv.putText(frame, str(B_slope), (x, y + 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

                            pointsTB = np.hstack(track_mB).astype(np.int32).reshape((-1, 1, 2))
                            cv.polylines(resultB, [pointsTB], isClosed=False, color=WHITE, thickness=3)

                            pointsPB = np.hstack(predictB).astype(np.int32).reshape((-1, 1, 2))
                            pointsPB = np.array([[80, y] for y in pointsPB[:, 0, 1]], dtype=np.int32)
                            pointsPB = pointsPB.reshape((-1, 1, 2))
                            cv.polylines(resultB, [pointsPB], isClosed=False, color=GREEN, thickness=3)


                            csv_B_py.append(pointsPB[-1][0][1])
                            b_slope = round(abs(csv_B_py[-1] - csv_B_py[0]) / len(csv_B_py),2)
                            B_slope = b_slope
                            Bx = pointsTB[0][0][0]

                            dis = abs(pointsPB[0][0][1] - pointsPB[-1][0][1])
                            if Byp1 != -1:
                                Byp1 = pointsPB[-1][0][1] + dis
                            if Byp2 != -1:
                                Byp2 = pointsPB[-1][0][1] + dis * 2
                            if Byp3 != -1:
                                Byp3 = pointsPB[-1][0][1] + dis * 3


                            cv.circle(resultB, (80, Byp1), 5, RED, -1)
                            cv.circle(resultB, (80, Byp2), 5, YELLOW, -1)
                            cv.circle(resultB, (80, Byp3), 5, GREEN, -1)

                            if Byp1 >= T_H and not list_pass_predictB1:
                                list_pass_predictB1.append(str(track_id))
                                Byp1 = -1

                            if Byp2 >= T_H and not list_pass_predictB2:
                                list_pass_predictB2.append(str(track_id))
                                Byp2 = -1

                            if Byp3 >= T_H and not list_pass_predictB3:
                                list_pass_predictB3.append(str(track_id))
                                Byp3 = -1

                            if len(track_mB) >= T_FPS:
                                m_start = track_mB[-1][1]
                                m_end = track_mB[0][1]
                                diss = abs(m_start - m_end)
                                splb.append(round(diss, 0))
                                summ = sum(splb) / len(splb)

                                if CTVb >= 2:
                                    CTVb = 1

                                if CTVb == 1:
                                    pvb = round(summ / FPS, 2)
                                    ETA_B = round(distCB / pvb, 2)
                                    pvB = pvb
                                    CTVb = 0

            #time.sleep(0.004)
            end = datetime.datetime.now()
            total = (end - start).total_seconds()

            fps_S = f'FPS: {1/total:.2f}'

            cv.putText(frame, fps_S, (600,680), cv.FONT_HERSHEY_SIMPLEX, 0.5,GREEN, thickness=2)

            combined_image = cv.hconcat([resultA, resultB, frame])
            cv.imshow("Combined Images", combined_image)

            video.write(combined_image)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv.destroyAllWindows()
    video.release()

if __name__ == "__main__":
    main()
