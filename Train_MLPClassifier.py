import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

#Read CSV
csv_file = 'mlp0624.csv'
data = pd.read_csv(csv_file)

#Spe_Feture_Lable
X = data[['A_slope', 'B_slope', 'distCA', 'distCB', 'Ax', 'Bx']]
y = data['situation']

#Spe_Dataset(Train:Val:Test = 80:10:10)
X_train_full, X_temp, y_train_full, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
#random_state=0 92/70/59
#0  84/56/50 {'alpha': 0.1, 'early_stopping': False, 'hidden_layer_sizes': (20, 20), 'learning_rate_init': 0.1}
#42 96/77/73 {'alpha': 0.1, 'early_stopping': False, 'hidden_layer_sizes': (30, 30), 'learning_rate_init': 0.001}
#42 97/60/73 {'alpha': 0.001, 'early_stopping': False, 'hidden_layer_sizes': (80, 80), 'learning_rate_init': 0.001}
#42 92/67/68 {'alpha': 0.1, 'early_stopping': False, 'hidden_layer_sizes': (30, 30, 30), 'learning_rate_init': 0.01}
#42 94/72/70 {'alpha': 0.1, 'early_stopping': False, 'hidden_layer_sizes': (70, 70), 'learning_rate_init': 0.001}
#42 92/72/75 {'alpha': 0.1, 'early_stopping': False, 'hidden_layer_sizes': (70, 70), 'learning_rate_init': 0.001}
#100 79/72/68 {'alpha': 0.1, 'early_stopping': False, 'hidden_layer_sizes': (30, 30, 30), 'learning_rate_init': 0.1}

#Feature_Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_full)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

#ReTransformData
X_train = pd.DataFrame(X_train, columns=X.columns)
X_val = pd.DataFrame(X_val, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

#Init_MLPClassifier
mlp = MLPClassifier(max_iter=5000)

#HyperParamsTuning use Grid Search
param_grid = {
    'hidden_layer_sizes': [(20, 20), (30, 30), (50, 50), (60, 60), (70, 70), (80, 80), (90, 90), (100, 100), (10, 20, 10), (30, 30, 30), (30, 50, 30)],
    'alpha': [0.1],
    'learning_rate_init': [0.001],
    'early_stopping': [True, False]
}
grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train_full)

#Best_HyperParams
print(f"Best parameters found: {grid_search.best_params_}")

#BestModel_Train_Test
best_mlp = grid_search.best_estimator_

#Eval_Train
best_mlp.fit(X_train, y_train_full)
y_pred_train = best_mlp.predict(X_train)
print("Training Classification Report:")
print(classification_report(y_train_full, y_pred_train))
print(f"Training Accuracy: {accuracy_score(y_train_full, y_pred_train):.2f}")

#Eval_Val
y_pred_val = best_mlp.predict(X_val)
print("Validation Classification Report:")
print(classification_report(y_val, y_pred_val))
print(f"Validation Accuracy: {accuracy_score(y_val, y_pred_val):.2f}")

#Eval_Test
y_pred_test = best_mlp.predict(X_test)
print("Testing Classification Report:")
print(classification_report(y_test, y_pred_test))
print(f"Testing Accuracy: {accuracy_score(y_test, y_pred_test):.2f}")

#Save_Model
model_filename = '0624_mlp.joblib'
scaler_filename = '0624_mlp_scaler.joblib'
joblib.dump(best_mlp, model_filename)
joblib.dump(scaler, scaler_filename)

#Epoch_Check
print(f"Real Epoch_Num: {best_mlp.n_iter_}")
if best_mlp.n_iter_ < best_mlp.max_iter:
    print("Model Early Stop.")
else:
    print("Model Train Max Epoch")
