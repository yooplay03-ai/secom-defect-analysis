import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.metrics

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/"
X = pd.read_csv(URL + "secom.data", sep=" ", header=None)
y = pd.read_csv(URL + "secom_labels.data", sep=" ", header=None)
y = (y[0] == 1).astype(int)

X = X.loc[:, X.nunique() > 1]
X = X.loc[:, X.isnull().mean() < 0.5]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
median = X_train.median()
X_train = X_train.fillna(median)
X_test = X_test.fillna(median)

# 정상 데이터만 골라서 크기 맞추기
X_normal = X_train[y_train == 0]
scaler = StandardScaler()
scaler.fit(X_normal)
normal_t = torch.tensor(scaler.transform(X_normal), dtype=torch.float32)
test_t = torch.tensor(scaler.transform(X_test), dtype=torch.float32)

# 오토인코더 만들기
torch.manual_seed(42)
n = X.shape[1]
model = nn.Sequential(
    nn.Linear(n, 64), nn.ReLU(),
    nn.Linear(64, 16), nn.ReLU(),
    nn.Linear(16, 64), nn.ReLU(),
    nn.Linear(64, n),
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# 학습: 정상 데이터를 압축했다 복원하는 연습
for epoch in range(200):
    optimizer.zero_grad()
    loss = loss_fn(model(normal_t), normal_t)
    loss.backward()
    optimizer.step()
    if epoch % 40 == 0:
        print(f"epoch {epoch}: 복원 오차 {loss.item():.4f}")

# 복원 오차 계산
with torch.no_grad():
    normal_err = ((model(normal_t) - normal_t) ** 2).mean(dim=1).numpy()
    test_err = ((model(test_t) - test_t) ** 2).mean(dim=1).numpy()

# 기준 정하고 평가, MLflow에 기록
mlflow.set_experiment("secom-defect")
for q in [95, 90, 80, 70, 50]:
    th = np.percentile(normal_err, q)
    pred = (test_err >= th).astype(int)
    false_alarm = ((pred == 1) & (y_test.values == 0)).sum()
    r = sklearn.metrics.recall_score(y_test, pred)
    p = sklearn.metrics.precision_score(y_test, pred, zero_division=0)

    with mlflow.start_run(run_name=f"ae_q{q}"):
        mlflow.log_param("model", "AutoEncoder")
        mlflow.log_param("threshold_percentile", q)
        mlflow.log_metric("recall", r)
        mlflow.log_metric("precision", p)
        mlflow.log_metric("false_alarm", false_alarm)

    print(f"정상 상위 {100 - q}% 기준: recall {r:.2f}, precision {p:.2f}, 헛경보 {false_alarm}")