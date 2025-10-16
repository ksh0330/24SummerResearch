from concurrent.futures import ThreadPoolExecutor
import traceback
import os
import numpy as np
import pandas as pd
import cv2
from tensorflow import keras

# 데이터셋 경로
train_dataset_path = 'C:/Users/user/Desktop/VideoClassification/train'
test_dataset_path = 'C:/Users/user/Desktop/VideoClassification/test'
val_dataset_path = 'C:/Users/user/Desktop/VideoClassification/validation'

# 데이터셋 내의 레이블(디렉토리 목록) 가져오기
label_types = os.listdir(train_dataset_path)
print(label_types)

# 학습 데이터 준비
rooms = []
for label in label_types:
    files = os.listdir(os.path.join(train_dataset_path, label))
    for file in files:
        file_path = os.path.join(train_dataset_path, label, file)
        rooms.append((label, file_path))

train_df = pd.DataFrame(data=rooms, columns=['tag', 'video_name'])
train_df.to_csv('train.csv', index=False)

# 테스트 데이터 준비
room_types = os.listdir(test_dataset_path)
rooms = []
for item in room_types:
    all_rooms = os.listdir(os.path.join(test_dataset_path, item))
    for room in all_rooms:
        rooms.append((item, os.path.join(test_dataset_path, item, room)))

test_df = pd.DataFrame(data=rooms, columns=['tag', 'video_name'])
test_df.to_csv('test.csv', index=False)

# 검증 데이터 준비
room_types = os.listdir(val_dataset_path)
rooms = []
for item in room_types:
    all_rooms = os.listdir(os.path.join(val_dataset_path, item))
    for room in all_rooms:
        rooms.append((item, os.path.join(val_dataset_path, item, room)))

val_df = pd.DataFrame(data=rooms, columns=['tag', 'video_name'])
val_df.to_csv('val.csv', index=False)

from tensorflow_docs.vis import embed
import tensorflow as tf
import imageio

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=12000)])
    except RuntimeError as e:
        print(e)

# 데이터 준비
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
val_df = pd.read_csv("val.csv")

print(f"Total videos for training: {len(train_df)}")
print(f"Total videos for testing: {len(test_df)}")
print(f"Total videos for validation: {len(val_df)}")

# 이미지 크기 및 프레임 수 설정
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
NUM_FEATURES = 2048

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y: start_y + min_dim, start_x: start_x + min_dim]

def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)
            if max_frames > 0 and len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

def load_video_with_error_handling(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    try:
        return load_video(path, max_frames, resize)
    except Exception as e:
        print(f"Error loading video {path}: {e}")
        traceback.print_exc()
        return np.array([])

def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)
    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()

label_processor = keras.layers.StringLookup(num_oov_indices=0, vocabulary=np.unique(train_df["tag"]))
print(label_processor.get_vocabulary())

labels = train_df["tag"].values
labels = label_processor(labels[..., None]).numpy()

def find_max_seq_length(df, root_dir):
    max_length = 0
    video_paths = df["video_name"].values.tolist()
    for path in video_paths:
        frames = load_video(os.path.join(root_dir, path))
        max_length = max(max_length, frames.shape[0])
    return max_length

max_seq_length_train = find_max_seq_length(train_df, train_dataset_path)
max_seq_length_test = find_max_seq_length(test_df, test_dataset_path)
max_seq_length_val = find_max_seq_length(val_df, val_dataset_path)
MAX_SEQ_LENGTH = max(max_seq_length_train, max_seq_length_test, max_seq_length_val)

print(f"MAX_SEQ_LENGTH: {MAX_SEQ_LENGTH}")

def prepare_all_videos_parallel(df, root_dir, max_seq_length):
    num_samples = len(df)
    video_paths = df["video_name"].values.tolist()
    labels = df["tag"].values
    labels = label_processor(labels[..., None]).numpy()
    frame_masks = np.zeros(shape=(num_samples, max_seq_length), dtype="bool")
    frame_features = np.zeros(shape=(num_samples, max_seq_length, NUM_FEATURES), dtype="float32")

    def process_video(idx, path):
        print(f"Processing video {idx+1}/{len(video_paths)}: {path}")  # 현재 처리 중인 비디오 경로 출력
        frames = load_video_with_error_handling(os.path.join(root_dir, path), max_frames=max_seq_length)
        if frames.size == 0:
            return np.zeros((max_seq_length, NUM_FEATURES), dtype="float32"), np.zeros(max_seq_length, dtype="bool")

        frames = frames[None, ...]
        temp_frame_features = np.zeros((max_seq_length, NUM_FEATURES), dtype="float32")
        temp_frame_mask = np.zeros(max_seq_length, dtype="bool")

        video_length = frames.shape[1]
        length = min(max_seq_length, video_length)
        for j in range(length):
            temp_frame_features[j, :] = feature_extractor.predict(frames[0, j, :][None, :])
        temp_frame_mask[:length] = 1

        return temp_frame_features, temp_frame_mask

    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda p: process_video(*p), enumerate(video_paths))

    for idx, (features, mask) in enumerate(results):
        frame_features[idx] = features
        frame_masks[idx] = mask

    return (frame_features, frame_masks), labels

# 학습, 테스트, 검증 데이터 준비
train_data, train_labels = prepare_all_videos_parallel(train_df, train_dataset_path, MAX_SEQ_LENGTH)
test_data, test_labels = prepare_all_videos_parallel(test_df, test_dataset_path, MAX_SEQ_LENGTH)
val_data, val_labels = prepare_all_videos_parallel(val_df, val_dataset_path, MAX_SEQ_LENGTH)

# 특징값 및 마스크 저장
os.makedirs('./data', exist_ok=True)
np.save('./data/train_frame_features.npy', train_data[0])
np.save('./data/train_frame_masks.npy', train_data[1])
np.save('./data/train_labels.npy', train_labels)

np.save('./data/test_frame_features.npy', test_data[0])
np.save('./data/test_frame_masks.npy', test_data[1])
np.save('./data/test_labels.npy', test_labels)

np.save('./data/val_frame_features.npy', val_data[0])
np.save('./data/val_frame_masks.npy', val_data[1])
np.save('./data/val_labels.npy', val_labels)
