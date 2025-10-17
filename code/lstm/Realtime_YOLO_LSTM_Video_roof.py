import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, precision_recall_fscore_support, confusion_matrix
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, TensorDataset
import logging
import optuna

# LSTM 모델 정의
class RiskPredictionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_prob=0.5):
        super(RiskPredictionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        lengths = lengths.cpu().long()  # lengths를 CPU로 이동시키고 int64 타입으로 변환
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        out = self.dropout(output[range(len(output)), lengths - 1])
        out = self.fc(out)
        return out

# 하이퍼파라미터 튜닝을 위한 Objective 함수 정의
def objective(trial):
    # 하이퍼파라미터 설정
    input_size = 6  # 특징 수: 차량 A 위치(y), 차량 B 위치(y), 차량 A 속도, 차량 B 속도
    hidden_size = trial.suggest_int('hidden_size', 128, 512)
    num_classes = 3  # 위험도: 안전, 주의, 위험
    num_epochs = 50  # 튜닝 시 에폭 수를 줄입니다.
    batch_size = trial.suggest_int('batch_size', 32, 128)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    dropout_prob = trial.suggest_uniform('dropout_prob', 0.1, 0.5)
    patience = 10  # 조기 종료를 위한 인내 횟수

    # 데이터 로드 및 전처리 함수
    def load_data(data_path, classes):
        data = []
        labels = []
        lengths = []
        max_length = 0
        file_names = []

        # 데이터의 최대 길이를 찾습니다.
        for class_idx, class_name in enumerate(classes):
            class_folder = os.path.join(data_path, class_name)
            for file_name in os.listdir(class_folder):
                file_path = os.path.join(class_folder, file_name)
                if not os.path.isfile(file_path):
                    print(f"파일을 찾을 수 없습니다: {file_path}")
                    continue
                df = pd.read_csv(file_path).iloc[:, 1:]
                lengths.append(len(df))
                max_length = max(max_length, len(df))
                file_names.append(file_name)

        # 데이터를 최대 길이에 맞추어 패딩합니다.
        for class_idx, class_name in enumerate(classes):
            class_folder = os.path.join(data_path, class_name)
            for file_name in os.listdir(class_folder):
                file_path = os.path.join(class_folder, file_name)
                if not os.path.isfile(file_path):
                    print(f"파일을 찾을 수 없습니다: {file_path}")
                    continue
                df = pd.read_csv(file_path).iloc[:, 1:]
                padding = np.zeros((max_length - len(df), df.shape[1]))
                padded_data = np.vstack((df.values, padding))
                data.append(padded_data)
                labels.append(class_idx)

        data = np.array(data, dtype=np.float32)  # dtype을 float32로 설정
        labels = np.array(labels, dtype=np.int64)
        lengths = np.array(lengths, dtype=np.int64)

        return data, labels, lengths, max_length, file_names

    train_data_path = 'D:/Straight_Data_0607/Straight_Data_0623/CSV_masking/train'
    val_data_path = 'D:/Straight_Data_0607/Straight_Data_0623/CSV_masking/validation'
    test_data_path = 'D:/Straight_Data_0607/Straight_Data_0623/CSV_masking/test'
    classes = ['caution', 'safe', 'danger']

    # 훈련 데이터 로드
    x_train, y_train, lengths_train, train_max_length, train_file_names = load_data(train_data_path, classes)
    # 검증 데이터 로드
    x_val, y_val, lengths_val, val_max_length, val_file_names = load_data(val_data_path, classes)
    # 테스트 데이터 로드
    x_test, y_test, lengths_test, test_max_length, test_file_names = load_data(test_data_path, classes)

    # 가장 긴 시퀀스 길이 설정
    max_length = max(train_max_length, val_max_length, test_max_length)

    # 데이터 로더 설정
    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.long),
                                  torch.tensor(lengths_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(x_val, dtype=torch.float32),
                                torch.tensor(y_val, dtype=torch.long),
                                torch.tensor(lengths_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32),
                                 torch.tensor(y_test, dtype=torch.long),
                                 torch.tensor(lengths_test, dtype=torch.long))

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    # 모델 초기화 및 학습 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RiskPredictionLSTM(input_size, hidden_size, num_classes, dropout_prob).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 조기 종료 및 체크포인팅 설정
    best_val_loss = float('inf')
    early_stop_counter = 0

    # 모델 학습 및 검증
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0
        all_train_labels = []
        all_train_predictions = []

        for inputs, labels, lengths in train_loader:
            inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)

            # 순방향 전파
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)

            # 역방향 전파 및 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # 학습 정확도 계산
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            all_train_labels.extend(labels.cpu().numpy())
            all_train_predictions.extend(predicted.cpu().numpy())

        train_loss /= len(train_loader)
        train_accuracy = 100 * correct_train / total_train

        # 검증 과정
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        all_val_labels = []
        all_val_predictions = []
        with torch.no_grad():
            for inputs, labels, lengths in val_loader:
                inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)

                outputs = model(inputs, lengths)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                all_val_labels.extend(labels.cpu().numpy())
                all_val_predictions.extend(predicted.cpu().numpy())

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct_val / total_val

        # 조기 종료 및 체크포인팅 로직
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            break

    return best_val_loss

# Optuna를 사용하여 하이퍼파라미터 튜닝
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print('Best trial:')
trial = study.best_trial

print(f'  Loss: {trial.value}')
print('  Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')
