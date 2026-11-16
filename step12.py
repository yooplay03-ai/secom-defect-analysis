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
mlflow.set_experiment("secom-defect")
summary = {"baseline": [], "all_past": [], "recent_400": []}

for i in range(2, 6):
    start = i * size
    end = (i + 1) * size if i < 5 else n
    X_te, y_te = X.iloc[start:end], y.iloc[start:end]

    scores = {}
    for name, train_from in [("all_past", 0), ("recent_400", max(0, start - 400))]:
        model = HistGradientBoostingClassifier(class_weight="balanced", random_state=42)
        model.fit(X.iloc[train_from:start], y.iloc[train_from:start])
        scores[name] = average_precision_score(y_te, model.predict_proba(X_te)[:, 1])

    base = y_te.mean()
    summary["baseline"].append(base)
    summary["all_past"].append(scores["all_past"])
    summary["recent_400"].append(scores["recent_400"])
    print(f"구간 {i}: 불량 {y_te.sum()}개 | 기준선 {base:.3f} | "
          f"과거 전체 {scores['all_past']:.3f} | 최근 400개 {scores['recent_400']:.3f}")

    with mlflow.start_run(run_name=f"walkforward_{i}"):
        mlflow.log_param("test_chunk", i)
        mlflow.log_metric("baseline", base)
        mlflow.log_metric("pr_auc_all_past", scores["all_past"])
        mlflow.log_metric("pr_auc_recent_400", scores["recent_400"])

print("\n평균:", {k: round(float(np.mean(v)), 3) for k, v in summary.items()})