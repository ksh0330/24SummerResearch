import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import time


# InceptionV3 모델을 사용하여 특징을 추출하는 함수
def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input
    inputs = keras.Input((224, 224, 3))
    preprocessed = preprocess_input(inputs)
    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


# 비디오 파일에서 예측 함수
def video_file_prediction(gru_model_path, video_path):
    feature_extractor = build_feature_extractor()

    # 모델 로드 시 custom_objects에 AdamW를 포함하여 옵티마이저를 인식하게 함
    seq_model = keras.models.load_model(gru_model_path, custom_objects={'AdamW': tfa.optimizers.AdamW})

    # 비디오 파일 초기화
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    MAX_SEQ_LENGTH = 839  # 모델과 일치하도록 수정
    NUM_FEATURES = 2048
    IMG_SIZE = 224

    frame_features = np.zeros((1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
    frame_mask = np.zeros((1, MAX_SEQ_LENGTH), dtype="bool")
    frame_count = 0
    last_prediction_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 영상 전처리
        frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame_resized = frame_resized[:, :, [2, 1, 0]]  # BGR에서 RGB로 변환
        frame_resized = np.expand_dims(frame_resized, axis=0)
        frame_resized = keras.applications.inception_v3.preprocess_input(frame_resized)

        # 특징 추출
        features = feature_extractor.predict(frame_resized)
        frame_features = np.roll(frame_features, -1, axis=1)
        frame_features[0, -1, :] = features
        frame_mask = np.roll(frame_mask, -1, axis=1)
        frame_mask[0, -1] = 1

        frame_count += 1

        # 일정 시간마다 예측 수행 (예: 1초마다)
        if time.time() - last_prediction_time >= 1:  # 1초에 한 번 예측
            predictions = seq_model.predict([frame_features, frame_mask])
            predicted_class = np.argmax(predictions, axis=1)[0]
            last_prediction_time = time.time()  # 예측 시간 갱신

            # 분류 결과 출력
            if predicted_class == 0:
                label = "Danger"
                color = (0, 0, 255)  # Red
                print("Danger")
            elif predicted_class == 1:
                label = "Caution"
                color = (0, 255, 255)  # Yellow
                print("Caution")
            else:
                label = "Safe"
                color = (0, 255, 0)  # Green
                print("Safe")

            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # 영상 출력
        cv2.imshow('Video Classification', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# 학습된 GRU 모델 경로 설정 및 비디오 파일 경로 설정
gru_model_path = 'C:/Users/user/Desktop/VideoClassification/tmp/video_classifier_epoch_24.h5'
video_path = 'C:/Users/user/Desktop/VideoClassification/test/danger/ACBC_F1MD.mp4'

# 비디오 파일 예측 실행
video_file_prediction(gru_model_path, video_path)
