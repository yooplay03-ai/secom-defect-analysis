import pandas as pd
import mlflow
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/"
X = pd.read_csv(URL + "secom.data", sep= " ", header=None)
labels = pd.read_csv(URL + "secom_labels.data", sep=" ", header=None)
y = (labels[0] == 1).astype(int)
time = pd.to_datetime(labels[1], format="%d/%m/%Y %H:%M:%S")

X = X.loc[:, X.nunique() >1]
X = X.loc[:, X.isnull().mean() < 0.5]

order= pd.DataFrame({"order": time.rank(method="first")})

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
mlflow.set_experiment("secom-defect")

def cv_pr_auc(features, name):
    model = HistGradientBoostingClassifier(class_weight="balanced", random_state=42)
    scores = cross_val_score(model, features, y, cv=cv, scoring="average_precision")
    print(f"{name}: PR-AUC {scores.mean():.3f} ± {scores.std():.3f}")
    with mlflow.start_run(run_name=f"confound_{name}"):
        mlflow.log_param("features", name)
        mlflow.log_metric("pr_auc_mean", scores.mean())
        mlflow.log_metric("pr_auc_std", scores.std())


cv_pr_auc(X, "sensors_446")
cv_pr_auc(order, "order_only")
cv_pr_auc(X[[59]], "sensor59_only")
print("기준선 (아무거나 찍기):", round(y.mean(), 3))