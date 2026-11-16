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

def walk_forward(X_used):
    gains = {"all_past": [], "recent_400": []}
    for i in range(2, 6):
        start = i * size
        end = (i + 1) * size if i < 5 else n
        y_te = y.iloc[start:end]
        base = y_te.mean()
        for name, train_from in [("all_past", 0), ("recent_400", max(0, start - 400))]:
            model = HistGradientBoostingClassifier(class_weight="balanced", random_state=42)
            model.fit(X_used.iloc[train_from:start], y.iloc[train_from:start])
            ap = average_precision_score(y_te, model.predict_proba(X_used.iloc[start:end])[:, 1])
            gains[name].append(ap - base)
    return {k: np.mean(v) for k, v in gains.items()}

drift = pd.read_csv("drift_sensors.csv", index_col=0)
top20_adv = drift["adversarial"].sort_values(ascending=False).head(20).index
both8 = [34, 45, 58, 116, 132, 543, 544, 545]

feature_sets = {
    "all_446": X,
    "drop_both8": X.drop(columns=both8),
    "drop_adv20": X.drop(columns=top20_adv),
}

mlflow.set_experiment("secom-defect")
for name, X_used in feature_sets.items():
    result = walk_forward(X_used)
    print(f"{name} (센서 {X_used.shape[1]}개): 과거 전체 {result['all_past']:+.3f} | 최근 400개 {result['recent_400']:+.3f}")
    with mlflow.start_run(run_name=f"drift_drop_{name}"):
        mlflow.log_param("feature_set", name)
        mlflow.log_param("note", "drift list used future data (upper bound)")
        mlflow.log_metric("gain_all_past", result["all_past"])
        mlflow.log_metric("gain_recent_400", result["recent_400"])