import numpy as np
import pandas as pd
import mlflow
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/"
X = pd.read_csv(URL + "secom.data", sep=" ", header=None)
y = pd.read_csv(URL + "secom_labels.data", sep=" ", header=None)
y = (y[0] == 1).astype(int)
X = X.loc[:, X.nunique() > 1]
X = X.loc[:, X.isnull().mean() < 0.5]

# 1. 비슷하게 움직이는 센서 묶기
corr = X.corr(method="spearman").fillna(0).abs().values
dist = 1 - corr
dist = (dist + dist.T) / 2
np.fill_diagonal(dist, 0)
Z = linkage(squareform(dist, checks=False), method="average")
group_id = fcluster(Z, t=0.2, criterion="distance")

groups = {}
for col, g in zip(X.columns, group_id):
    groups.setdefault(g, []).append(col)
print(f"센서 {X.shape[1]}개 → 묶음 {len(groups)}개")

# 2. 교차 검증 5번, 묶음째로 섞어서 중요도 재기
rng = np.random.default_rng(42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
records = []

for fold, (tr, te) in enumerate(cv.split(X, y)):
    model = HistGradientBoostingClassifier(class_weight="balanced", random_state=42)
    model.fit(X.iloc[tr], y.iloc[tr])
    X_te, y_te = X.iloc[te], y.iloc[te]
    base = average_precision_score(y_te, model.predict_proba(X_te)[:, 1])

    for g, cols in groups.items():
        drops = []
        for _ in range(3):
            X_shuf = X_te.copy()
            X_shuf[cols] = X_te[cols].values[rng.permutation(len(X_te))]
            score = average_precision_score(y_te, model.predict_proba(X_shuf)[:, 1])
            drops.append(base - score)
        records.append({"fold": fold, "group": g, "drop": np.mean(drops)})

    print(f"fold {fold} 완료 (기본 PR-AUC {base:.3f})")

# 3. 다섯 번 결과 합치기
df = pd.DataFrame(records)
df["rank"] = df.groupby("fold")["drop"].rank(ascending=False)
summary = df.groupby("group").agg(
    mean_drop=("drop", "mean"),
    std_drop=("drop", "std"),
    top10_count=("rank", lambda r: (r <= 10).sum()),
)
summary["sensors"] = [groups[g] for g in summary.index]
summary = summary.sort_values("mean_drop", ascending=False)
print(summary.head(10).to_string())
summary.to_csv("group_importance_cv.csv")

mlflow.set_experiment("secom-defect")
with mlflow.start_run(run_name="group_importance_cv"):
    mlflow.log_param("model", "HistGradientBoosting")
    mlflow.log_param("method", "grouped permutation, 5-fold CV")
    mlflow.log_param("group_threshold", "spearman |corr| about 0.8")
    mlflow.log_metric("n_groups", len(groups))
    mlflow.log_artifact("group_importance_cv.csv")