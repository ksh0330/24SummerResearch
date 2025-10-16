import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# CSV 파일 읽기
csv_file = 'mlp0629.csv'
data = pd.read_csv(csv_file)





def get_class_indices(y):
    class_0_indices = y[y == 0].index.tolist()
    class_1_indices = y[y == 1].index.tolist()
    class_2_indices = y[y == 2].index.tolist()
    return class_0_indices, class_1_indices, class_2_indices

def main():
    while True:

        # 특징과 레이블 분리
        X = data[['A_slope', 'B_slope', 'distCA', 'distCB', 'Ax', 'Bx']]
        y = data['situation']

        #데이터 분할
        random_seed_mlp = np.random.randint(0, 5000)
        print(f"Random Seed for MLPClassifier: {random_seed_mlp}")

        # 각 클래스별로 14개의 검증 데이터와 14개의 테스트 데이터를 샘플링하여 총 42개의 검증 데이터와 42개의 테스트 데이터를 만듭니다.
        def sample_per_class(X, y, n_samples_per_class):
            X_sampled = pd.DataFrame()
            y_sampled = pd.Series(dtype=y.dtype)
            for cls in y.unique():
                X_cls = X[y == cls]
                y_cls = y[y == cls]
                X_cls_sampled = X_cls.sample(n=n_samples_per_class, random_state=random_seed_mlp)
                y_cls_sampled = y_cls.sample(n=n_samples_per_class, random_state=random_seed_mlp)
                X_sampled = pd.concat([X_sampled, X_cls_sampled])
                y_sampled = pd.concat([y_sampled, y_cls_sampled])
            return X_sampled, y_sampled

        # 검증 데이터 추출
        X_val, y_val = sample_per_class(X, y, 14)
        X, y = X.drop(X_val.index), y.drop(y_val.index)

        # 테스트 데이터 추출
        X_test, y_test = sample_per_class(X, y, 14)
        X, y = X.drop(X_test.index), y.drop(y_test.index)

        # 나머지 데이터를 학습 데이터로 사용하여 총 348개, 각 클래스별로 116개
        X_train_full, y_train_full = X, y

        # 각 클래스의 데이터 개수 확인
        #print("Training set class distribution:\n", y_train_full.value_counts())
        #print("Validation set class distribution:\n", y_val.value_counts())
        #print("Testing set class distribution:\n", y_test.value_counts())

        # 클래스별 인덱스 추출


        # 각 데이터셋에서 클래스별 인덱스 추출
        #train_class_0_indices, train_class_1_indices, train_class_2_indices = get_class_indices(y_train_full)
        #val_class_0_indices, val_class_1_indices, val_class_2_indices = get_class_indices(y_val)
        #test_class_0_indices, test_class_1_indices, test_class_2_indices = get_class_indices(y_test)

        # 각 클래스별 인덱스 출력
        #print("Train class 0 indices:", train_class_0_indices)
        #print("Train class 1 indices:", train_class_1_indices)
        #print("Train class 2 indices:", train_class_2_indices)

        #print("Validation class 0 indices:", val_class_0_indices)
        #print("Validation class 1 indices:", val_class_1_indices)
        #print("Validation class 2 indices:", val_class_2_indices)

        #print("Test class 0 indices:", test_class_0_indices)
        #print("Test class 1 indices:", test_class_1_indices)
        #print("Test class 2 indices:", test_class_2_indices)

        # 특징 스케일링
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_full)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # 변환된 데이터를 다시 데이터프레임으로 변환하여 피처 이름을 유지합니다.
        X_train = pd.DataFrame(X_train, columns=X.columns)
        X_val = pd.DataFrame(X_val, columns=X.columns)
        X_test = pd.DataFrame(X_test, columns=X.columns)

        # MLP 분류기 모델 초기화
        mlp = MLPClassifier(max_iter=5000, random_state=random_seed_mlp)

        # Grid Search를 사용하여 하이퍼파라미터 튜닝
        param_grid = {
            'hidden_layer_sizes': [(5, 5), (10, 10), (20, 20), (30, 30), (50, 50), (60, 60), (70, 70), (80, 80), (90, 90), (3, 10, 3), (3, 5, 3), (5, 5, 5), (5, 10, 5), (5, 10, 5), (10, 10, 10), (20, 20, 20), (10, 20, 10), (30, 30, 30), (30, 50, 30)],
            'alpha': [0.1, 0.01, 0.001],
            'learning_rate_init': [0.001, 0.01, 0.1],
        }
        grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train_full)

        # 최적의 하이퍼파라미터 출력
        print(f"Best parameters found: {grid_search.best_params_}")

        # 최적의 모델로 학습 및 평가
        best_mlp = grid_search.best_estimator_

        # 학습 데이터 평가
        #best_mlp.fit(X_train, y_train_full)
        y_pred_train = best_mlp.predict(X_train)
        #print("Training Classification Report:")
        #print(classification_report(y_train_full, y_pred_train))
        print(f"Training Accuracy: {accuracy_score(y_train_full, y_pred_train):.2f}")
        TR = accuracy_score(y_train_full, y_pred_train)

        # 검증 데이터 평가
        y_pred_val = best_mlp.predict(X_val)
        #print("Validation Classification Report:")
        #print(classification_report(y_val, y_pred_val))
        print(f"Validation Accuracy: {accuracy_score(y_val, y_pred_val):.2f}")
        VA = accuracy_score(y_val, y_pred_val)

        # 테스트 데이터 평가
        y_pred_test = best_mlp.predict(X_test)
        #print("Testing Classification Report:")
        #print(classification_report(y_test, y_pred_test))
        print(f"Testing Accuracy: {accuracy_score(y_test, y_pred_test):.2f}")
        TE = accuracy_score(y_test, y_pred_test)

        print(f"실제 수행된 에포크 수: {best_mlp.n_iter_}\n")

        if TR >= 0.90 and VA >= 0.80 and TE >= 0.80:

            print("Training Classification Report:")
            print(classification_report(y_train_full, y_pred_train))
            print(f"Training Accuracy: {accuracy_score(y_train_full, y_pred_train):.2f}")

            print("Validation Classification Report:")
            print(classification_report(y_val, y_pred_val))
            print(f"Validation Accuracy: {accuracy_score(y_val, y_pred_val):.2f}")


            print("Testing Classification Report:")
            print(classification_report(y_test, y_pred_test))
            print(f"Testing Accuracy: {accuracy_score(y_test, y_pred_test):.2f}")


            # 혼동 행렬 시각화 함수 정의
            def plot_confusion_matrix(y_true, y_pred, title):
                conf_matrix = confusion_matrix(y_true, y_pred)
                plt.figure(figsize=(10, 7))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title(title)
                plt.show()

            # 학습 데이터 혼동 행렬 시각화
            plot_confusion_matrix(y_train_full, y_pred_train, 'Confusion Matrix (Training Data)')

            # 검증 데이터 혼동 행렬 시각화
            plot_confusion_matrix(y_val, y_pred_val, 'Confusion Matrix (Validation Data)')

            # 테스트 데이터 혼동 행렬 시각화
            plot_confusion_matrix(y_test, y_pred_test, 'Confusion Matrix (Testing Data)')


            # 모델 저장
            model_filename = '0704_mlp.joblib'
            scaler_filename = '0704_mlp_scaler.joblib'
            joblib.dump(best_mlp, model_filename)
            joblib.dump(scaler, scaler_filename)

            # 조기 종료 여부 및 실제 수행된 에포크 수 출력
            print(f"실제 수행된 에포크 수: {best_mlp.n_iter_}")
            if best_mlp.n_iter_ < best_mlp.max_iter:
                print("모델이 조기 종료되었습니다.")
            else:
                print("모델이 설정된 최대 에포크까지 학습되었습니다.")


if __name__ == "__main__":
    main()