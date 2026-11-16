import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
import sklearn.metrics

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/"
X = pd.read_csv(URL + "secom.data", sep=" ", header=None)
y = pd.read_csv(URL + "secom_labels.data", sep=" ", header=None)
y = (y[0] == 1).astype(int)

X = X.loc[:, X.nunique() > 1]
X = X.loc[:, X.isnull().mean() < 0.5]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)

model = HistGradientBoostingClassifier(class_weight="balanced", random_state=42)
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]
base = sklearn.metrics.average_precision_score(y_test, proba)
print(f"시험용 PR-AUC: {base:.3f}")

result = permutation_importance(
    model, X_test, y_test, scoring="average_precision",
    n_repeats=10, random_state=42)

imp = pd.DataFrame({
    "sensor": X.columns,
    "mean": result.importances_mean,
    "std": result.importances_std,
})
imp = imp.sort_values("mean", ascending=False)
print(imp.head(10))

top = imp.head(10)
plt.barh(top["sensor"].astype(str), top["mean"], xerr=top["std"])
plt.gca().invert_yaxis()
plt.xlabel("PR-AUC drop when shuffled")
plt.title("Top 10 sensors")
plt.tight_layout()
plt.savefig("top_sensors.png")
imp.to_csv("sensor_importance.csv", index=False)

mlflow.set_experiment("secom-defect")
with mlflow.start_run(run_name="importance_HGB"):
    mlflow.log_param("model", "HistGradientBoosting")
    mlflow.log_param("method", "permutation_importance")
    mlflow.log_metric("pr_auc_test", base)
    mlflow.log_artifact("top_sensors.png")
    mlflow.log_artifact("sensor_importance.csv")