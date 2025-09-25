import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

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

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y: start_y + min_dim, start_x: start_x + min_dim]

def preprocess_frame(frame, img_size=224):
    frame = crop_center_square(frame)
    frame = cv2.resize(frame, (img_size, img_size))
    frame = frame[:, :, [2, 1, 0]]  # BGR에서 RGB로 변환
    frame = np.expand_dims(frame, axis=0)
    frame = keras.applications.inception_v3.preprocess_input(frame)
    return frame

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
    frame_features = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 매 프레임마다 특징 추출
        preprocessed_frame = preprocess_frame(frame, IMG_SIZE)
        features = feature_extractor.predict(preprocessed_frame)
        frame_features.append(features[0])
        frame_count += 1

    cap.release()

    # 프레임 특징을 numpy 배열로 변환
    frame_features = np.array(frame_features)
    num_frames = len(frame_features)

    # 시퀀스 길이가 MAX_SEQ_LENGTH보다 길 경우 자르기, 짧을 경우 패딩
    if num_frames > MAX_SEQ_LENGTH:
        frame_features = frame_features[:MAX_SEQ_LENGTH]
        frame_mask = np.ones((MAX_SEQ_LENGTH,), dtype="bool")
    else:
        padding_length = MAX_SEQ_LENGTH - num_frames
        frame_features = np.pad(frame_features, ((0, padding_length), (0, 0)), mode='constant')
        frame_mask = np.concatenate([np.ones((num_frames,), dtype="bool"), np.zeros((padding_length,), dtype="bool")])

    frame_features = np.expand_dims(frame_features, axis=0)
    frame_mask = np.expand_dims(frame_mask, axis=0)

    # 예측 수행
    predictions = seq_model.predict([frame_features, frame_mask])
    predicted_class = np.argmax(predictions, axis=1)[0]

    if predicted_class == 0:
        label = "Danger"
        color = (0, 0, 255)  # Red
    elif predicted_class == 1:
        label = "Caution"
        color = (0, 255, 255)  # Yellow
    else:
        label = "Safe"
        color = (0, 255, 0)  # Green

    # 결과 출력
    print(f"Predicted class: {label}")

    # 영상에 결과를 표시
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        cv2.imshow('Video Classification', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 학습된 GRU 모델 경로 설정 및 비디오 파일 경로 설정
gru_model_path = ('C:/Users/user/Desktop/VideoClassification/tmp/video_classifier_epoch_24.h5')
video_path = 'C:/Users/user/Desktop/VideoClassification/train/danger/ACBR_FFD.mp4'

# 비디오 파일 예측 실행
video_file_prediction(gru_model_path, video_path)
