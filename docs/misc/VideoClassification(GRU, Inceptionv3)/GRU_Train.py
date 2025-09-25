import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler, EarlyStopping
import datetime

# 데이터 불러오기
train_frame_features = np.load('C:/Users/user/Desktop/VideoClassification/data/train_frame_features.npy')
train_frame_masks = np.load('C:/Users/user/Desktop/VideoClassification/data/train_frame_masks.npy')
train_labels = np.load('C:/Users/user/Desktop/VideoClassification/data/train_labels.npy').astype(str)

test_frame_features = np.load('C:/Users/user/Desktop/VideoClassification/data/test_frame_features.npy')
test_frame_masks = np.load('C:/Users/user/Desktop/VideoClassification/data/test_frame_masks.npy')
test_labels = np.load('C:/Users/user/Desktop/VideoClassification/data/test_labels.npy').astype(str)

val_frame_features = np.load('C:/Users/user/Desktop/VideoClassification/data/val_frame_features.npy')
val_frame_masks = np.load('C:/Users/user/Desktop/VideoClassification/data/val_frame_masks.npy')
val_labels = np.load('C:/Users/user/Desktop/VideoClassification/data/val_labels.npy').astype(str)


def split_data(features, masks, labels, num_frames=100):
    new_features = []
    new_masks = []
    new_labels = []

    for i in range(features.shape[0]):
        num_splits = features.shape[1] // num_frames
        for j in range(num_splits):
            new_features.append(features[i, j * num_frames:(j + 1) * num_frames, :])
            new_masks.append(masks[i, j * num_frames:(j + 1) * num_frames])
            new_labels.append(labels[i])

    new_features = np.array(new_features)
    new_masks = np.array(new_masks)
    new_labels = np.array(new_labels)

    return new_features, new_masks, new_labels


# 데이터를 100 프레임씩 분할
train_frame_features, train_frame_masks, train_labels = split_data(train_frame_features, train_frame_masks,
                                                                   train_labels)
test_frame_features, test_frame_masks, test_labels = split_data(test_frame_features, test_frame_masks, test_labels)
val_frame_features, val_frame_masks, val_labels = split_data(val_frame_features, val_frame_masks, val_labels)

print(f"Train frame features shape: {train_frame_features.shape}")
print(f"Train frame masks shape: {train_frame_masks.shape}")
print(f"Train labels shape: {train_labels.shape}")
print(f"Test frame features shape: {test_frame_features.shape}")
print(f"Test frame masks shape: {test_frame_masks.shape}")
print(f"Test labels shape: {test_labels.shape}")
print(f"Validation frame features shape: {val_frame_features.shape}")
print(f"Validation frame masks shape: {val_frame_masks.shape}")
print(f"Validation labels shape: {val_labels.shape}")

# 레이블 처리기
label_processor = keras.layers.StringLookup(num_oov_indices=0, vocabulary=np.unique(train_labels))
print(label_processor.get_vocabulary())

# 하이퍼파라미터 정의
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 200
NUM_FEATURES = 2048
MAX_SEQ_LENGTH = 100  # 각 시퀀스의 최대 길이를 100 프레임으로 설정


# 시퀀스 모델 정의
def get_sequence_model():
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    x = keras.layers.GRU(112, return_sequences=True)(frame_features_input, mask=mask_input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.GRU(16)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"]
    )
    return rnn_model


# 학습률 스케줄러
def lr_scheduler(epoch, lr):
    if epoch < 50:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


# 실험 실행 함수
def run_experiment():
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # 디렉토리가 존재하지 않으면 생성
    os.makedirs('./tmp', exist_ok=True)
    filepath = "./tmp/video_classifier_epoch_{epoch:02d}.h5"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=False, save_best_only=True, verbose=1
    )

    lr_scheduler_callback = LearningRateScheduler(lr_scheduler)
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',  # 모니터링할 성능 지표
        patience=30,  # 몇 epoch 동안 개선되지 않으면 멈출지 설정
        min_delta=0.0,  # 최소 변화량
        restore_best_weights=True,  # 가장 좋은 모델의 가중치를 복원할지 여부
        verbose=1  # 콜백의 진행 상황을 출력할지 여부
    )

    seq_model = get_sequence_model()
    history = seq_model.fit(
        [train_frame_features, train_frame_masks],
        label_processor(train_labels).numpy(),
        validation_data=([val_frame_features, val_frame_masks], label_processor(val_labels).numpy()),
        epochs=EPOCHS,
        callbacks=[checkpoint, tensorboard_callback, lr_scheduler_callback, early_stopping_callback],
    )

    # 가장 좋은 모델 가중치를 로드
    best_epoch = np.argmin(history.history['val_loss']) + 1
    best_model_path = filepath.format(epoch=best_epoch)
    seq_model.load_weights(best_model_path)
    _, accuracy = seq_model.evaluate([test_frame_features, test_frame_masks], label_processor(test_labels).numpy())
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history, seq_model


# 실험 실행
_, sequence_model = run_experiment()
