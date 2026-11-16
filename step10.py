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

# 시간 순서로 과거 75% / 미래 25%
order = time.sort_values().index
cut = int(len(order) * 0.75)
past, future = order[:cut], order[cut:]
print("과거:", time[past].min(), "~", time[past].max(), "/ 불량", y[past].sum())
print("미래:", time[future].min(), "~", time[future].max(), "/ 불량", y[future].sum())

X_p, y_p = X.loc[past], y.loc[past]
X_f, y_f = X.loc[future], y.loc[future]

# 1. 모델
model = HistGradientBoostingClassifier(class_weight="balanced", random_state=42)
model.fit(X_p, y_p)
score = model.predict_proba(X_f)[:, 1]
ap = average_precision_score(y_f, score)
print(f"미래 데이터 PR-AUC: {ap:.3f} (기준선 {y_f.mean():.3f})")

def report(name, flag):
    flag = np.asarray(flag)
    caught = (flag & (y_f.values == 1)).sum()
    total = y_f.sum()
    print(f"{name}: 검사 {flag.mean():.0%} → 불량 {caught}/{total}개 (recall {caught / total:.2f})")
    return caught / total, flag.mean()

r_model, f_model = report("모델 상위 20%", score >= np.quantile(score, 0.8))

# 2. 59번 규칙 (기준선은 과거로만 정함)
th59 = X_p[59].quantile(0.8)
r_rule, f_rule = report(f"59번 > {th59:.2f}", (X_f[59] > th59).values)

mlflow.set_experiment("secom-defect")
with mlflow.start_run(run_name="time_split_check"):
    mlflow.log_param("split", "time order, past 75% / future 25%")
    mlflow.log_metric("future_pr_auc", ap)
    mlflow.log_metric("model_recall_at20", r_model)
    mlflow.log_metric("rule59_recall", r_rule)
    mlflow.log_metric("rule59_inspect_rate", f_rule)