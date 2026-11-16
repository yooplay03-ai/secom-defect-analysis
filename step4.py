import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/"
X = pd.read_csv(URL + "secom.data", sep=" ", header=None)
y = pd.read_csv(URL + "secom_labels.data", sep=" ", header=None)
y = (y[0] == 1).astype(int)

X = X.loc[:, X.nunique() > 1]
X = X.loc[:, X.isnull().mean() < 0.5]
print("남은 센서 수:", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

median = X_train.median()
X_train = X_train.fillna(median)
X_test = X_test.fillna(median)

mlflow.set_experiment("secom-defect")

model = RandomForestClassifier(class_weight="balanced", random_state=42)
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]

for th in [0.5, 0.3, 0.2, 0.1, 0.05]:
    pred = (proba >= th).astype(int)
    false_alarm = ((pred == 1) & (y_test.values == 0)).sum()

    with mlflow.start_run(run_name=f"rf_th{th}"):
        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("threshold", th)
        mlflow.log_metric("recall", sklearn.metrics.recall_score(y_test, pred))
        mlflow.log_metric("precision", sklearn.metrics.precision_score(y_test, pred, zero_division=0))
        mlflow.log_metric("false_alarm", false_alarm)

    print(f"임계값 {th} 기록 완료")