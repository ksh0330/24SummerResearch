import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, TensorDataset
import logging
import seaborn as sns
import matplotlib.pyplot as plt

# LSTM 모델 정의
class RiskPredictionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_prob=0.5):
        super(RiskPredictionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        lengths = lengths.cpu().long()
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        out = self.dropout(output[range(len(output)), lengths - 1])
        out = self.fc(out)
        return out

# 하이퍼파라미터 설정
input_size = 6
hidden_size = 419
num_classes = 3
num_epochs = 500
batch_size = 86
learning_rate = 0.0007
dropout_prob = 0.2
patience = 20

# 데이터 로드 및 전처리 함수
def load_data(data_path, classes):
    data = []
    labels = []
    lengths = []
    max_length = 0

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

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    lengths = np.array(lengths, dtype=np.int64)

    return data, labels, lengths, max_length

train_data_path = 'D:/Straight_Data_0607/Straight_Data_0623/CSV_masking/train'
val_data_path = 'D:/Straight_Data_0607/Straight_Data_0623/CSV_masking/validation'
test_data_path = 'D:/Straight_Data_0607/Straight_Data_0623/CSV_masking/test'
classes = ['caution', 'safe', 'danger']

# 훈련 데이터 로드
x_train, y_train, lengths_train, train_max_length = load_data(train_data_path, classes)
# 검증 데이터 로드
x_val, y_val, lengths_val, val_max_length = load_data(val_data_path, classes)
# 테스트 데이터 로드
x_test, y_test, lengths_test, test_max_length = load_data(test_data_path, classes)

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

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 모델 저장을 위한 경로 설정
model_save_path = 'D:/Straight_Data_0607/Straight_Data_0623/CSV_masking/model'
os.makedirs(model_save_path, exist_ok=True)
best_model_path = os.path.join(model_save_path, 'best_model.pth')

# CSV 파일로 학습 과정 저장
csv_file = 'training_log.csv'
csv_columns = ['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy']

# CSV 파일 생성
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = pd.DataFrame(columns=csv_columns)
        writer.to_csv(file, index=False)

# 조기 종료 및 체크포인팅 설정
best_val_loss = float('inf')
best_epoch = 0
early_stop_counter = 0

# 모델 학습 및 검증
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    correct_train = 0
    total_train = 0
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

    train_loss /= len(train_loader)
    train_accuracy = 100 * correct_train / total_train

    # 검증 과정
    model.eval()
    val_loss = 0
    correct_val = 0
    total_val = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels, lengths in val_loader:
            inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)

            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    val_loss /= len(val_loader)
    val_accuracy = 100 * correct_val / total_val

    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    logging.info(
        f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}')

    # CSV 파일에 결과 저장
    with open(csv_file, mode='a', newline='') as file:
        writer = pd.DataFrame(
            [[epoch + 1, train_loss, train_accuracy, val_loss, val_accuracy, precision, recall, f1]],
            columns=csv_columns + ['precision', 'recall', 'f1'])
        writer.to_csv(file, header=False, index=False)

    # 조기 종료 및 체크포인팅 로직
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), best_model_path)
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    # 현재 epoch 모델 저장
    epoch_model_path = os.path.join(model_save_path, f'epoch_{epoch + 1}.pth')
    torch.save(model.state_dict(), epoch_model_path)

    if early_stop_counter >= patience:
        logging.info("조기 종료 조건 충족. 학습을 종료합니다.")
        break

logging.info(f"모델 학습 완료. Best Epoch: {best_epoch}")

# 테스트 데이터셋에 대한 평가
model.eval()
test_loss = 0
correct_test = 0
total_test = 0
all_test_labels = []
all_test_predictions = []

with torch.no_grad():
    for inputs, labels, lengths in test_loader:
        inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)

        outputs = model(inputs, lengths)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

        all_test_labels.extend(labels.cpu().numpy())
        all_test_predictions.extend(predicted.cpu().numpy())

test_loss /= len(test_loader)
test_accuracy = 100 * correct_test / total_test

precision_test = precision_score(all_test_labels, all_test_predictions, average='weighted')
recall_test = recall_score(all_test_labels, all_test_predictions, average='weighted')
f1_test = f1_score(all_test_labels, all_test_predictions, average='weighted')

logging.info(
    f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, Test Precision: {precision_test:.2f}, Test Recall: {recall_test:.2f}, Test F1-Score: {f1_test:.2f}')

# 결과를 CSV 파일에 추가 저장
with open(csv_file, mode='a', newline='') as file:
    writer = pd.DataFrame(
        [[num_epochs, test_loss, test_accuracy, None, None, precision_test, recall_test, f1_test]],
        columns=csv_columns + ['precision', 'recall', 'f1'])
    writer.to_csv(file, header=False, index=False)

logging.info("테스트 데이터셋 평가 완료.")

# Classification report 생성 및 출력
train_report = classification_report(y_train, model(torch.tensor(x_train, dtype=torch.float32).to(device), torch.tensor(lengths_train, dtype=torch.long).to(device)).argmax(dim=1).cpu().numpy(), target_names=classes)
val_report = classification_report(y_val, model(torch.tensor(x_val, dtype=torch.float32).to(device), torch.tensor(lengths_val, dtype=torch.long).to(device)).argmax(dim=1).cpu().numpy(), target_names=classes)
test_report = classification_report(y_test, model(torch.tensor(x_test, dtype=torch.float32).to(device), torch.tensor(lengths_test, dtype=torch.long).to(device)).argmax(dim=1).cpu().numpy(), target_names=classes)

print("Training Classification Report:\n", train_report)
print("Validation Classification Report:\n", val_report)
print("Testing Classification Report:\n", test_report)

# 혼동 행렬 계산 및 출력
conf_matrix = confusion_matrix(all_test_labels, all_test_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
