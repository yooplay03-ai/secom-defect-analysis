import mlflow
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/"
X = pd.read_csv(URL + "secom.data", sep=" ", header=None)
y = pd.read_csv(URL + "secom_labels.data", sep=" ", header=None)
y = (y[0] == 1).astype(int)

X = X.loc[:, X.nunique() > 1]
X = X.loc[:, X.isnull().mean() < 0.5]

models = {
    "LogisticRegression": make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000)),
    "RandomForest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(class_weight="balanced", random_state=42)),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        class_weight="balanced", random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
mlflow.set_experiment("secom-defect")

for name, model in models.items():
    pr = cross_val_score(model, X, y, cv=cv, scoring="average_precision")
    roc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    with mlflow.start_run(run_name=f"cv_{name}"):
        mlflow.log_param("model", name)
        mlflow.log_param("evaluation", "5-fold CV")
        mlflow.log_metric("pr_auc_mean", pr.mean())
        mlflow.log_metric("pr_auc_std", pr.std())
        mlflow.log_metric("roc_auc_mean", roc.mean())

    print(f"{name}: PR-AUC {pr.mean():.3f} ± {pr.std():.3f}, ROC-AUC {roc.mean():.3f}")

print("아무거나 찍었을 때 PR-AUC 기준선:", round(y.mean(), 3))