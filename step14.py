import numpy as np
import pandas as pd
import mlflow
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/"
X = pd.read_csv(URL + "secom.data", sep=" ", header=None)
labels = pd.read_csv(URL + "secom_labels.data", sep=" ", header=None)
y = (labels[0] == 1).astype(int)
time = pd.to_datetime(labels[1], format="%d/%m/%Y %H:%M:%S")

X = X.loc[:, X.nunique() > 1]
X = X.loc[:, X.isnull().mean() < 0.5]

order = time.sort_values().index
X = X.loc[order].reset_index(drop=True)
y = y.loc[order].reset_index(drop=True)

n = len(X)
size = n // 6

def walk_forward(y_used):
    gains = {"all_past": [], "recent_400": []}
    for i in range(2, 6):
        start = i * size
        end = (i + 1) * size if i < 5 else n
        y_te = y_used.iloc[start:end]
        base = y_te.mean()
        for name, train_from in [("all_past", 0), ("recent_400", max(0, start - 400))]:
            model = HistGradientBoostingClassifier(class_weight="balanced", random_state=42)
            model.fit(X.iloc[train_from:start], y_used.iloc[train_from:start])
            ap = average_precision_score(y_te, model.predict_proba(X.iloc[start:end])[:, 1])
            gains[name].append(ap - base)
    return {k: np.mean(v) for k, v in gains.items()}

real = walk_forward(y)
print("진짜 정답 향상:", {k: round(float(v), 3) for k, v in real.items()})

rng = np.random.default_rng(42)
null = {"all_past": [], "recent_400": []}
for k in range(50):
    y_fake = pd.Series(rng.permutation(y.values))
    result = walk_forward(y_fake)
    for name in null:
        null[name].append(result[name])
    if (k + 1) % 10 == 0:
        print(f"가짜 실험 {k + 1}/50 완료")

mlflow.set_experiment("secom-defect")
with mlflow.start_run(run_name="permutation_test"):
    for name in null:
        arr = np.array(null[name])
        p = (np.sum(arr >= real[name]) + 1) / (len(arr) + 1)
        print(f"{name}: 진짜 {real[name]:+.3f} | 우연 상위 5% 경계 {np.percentile(arr, 95):+.3f} | p값 {p:.3f}")
        mlflow.log_metric(f"{name}_real_gain", real[name])
        mlflow.log_metric(f"{name}_null_p95", np.percentile(arr, 95))
        mlflow.log_metric(f"{name}_p_value", p)