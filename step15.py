import numpy as np
import pandas as pd
import mlflow
from scipy.stats import ks_2samp
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/"
X = pd.read_csv(URL + "secom.data", sep=" ", header=None)
labels = pd.read_csv(URL + "secom_labels.data", sep=" ", header=None)
time = pd.to_datetime(labels[1], format="%d/%m/%Y %H:%M:%S")

X = X.loc[:, X.nunique() > 1]
X = X.loc[:, X.isnull().mean() < 0.5]
X = X.loc[time.sort_values().index].reset_index(drop=True)

n = len(X)
cut = int(n * 0.75)
is_future = pd.Series((np.arange(n) >= cut).astype(int))

# 1. 과거/미래를 얼마나 잘 구분하나
model = HistGradientBoostingClassifier(random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc = cross_val_score(model, X, is_future, cv=cv, scoring="roc_auc")
print(f"과거/미래 구분 AUC: {auc.mean():.3f} ± {auc.std():.3f}")

# 2. 구분할 때 어떤 센서를 봤나
X_tr, X_te, f_tr, f_te = train_test_split(
    X, is_future, test_size=0.3, stratify=is_future, random_state=42)
model.fit(X_tr, f_tr)
result = permutation_importance(model, X_te, f_te, scoring="roc_auc",
                                n_repeats=5, random_state=42)
adv = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)

# 3. 센서 하나씩 분포 비교 (KS)
past, future = X.iloc[:cut], X.iloc[cut:]

def ks_stat(col):
    a, b = past[col].dropna(), future[col].dropna()
    if len(a) < 10 or len(b) < 10:
        return np.nan
    return ks_2samp(a, b).statistic

ks = pd.Series({c: ks_stat(c) for c in X.columns}).sort_values(ascending=False)

print("적대적 검증 상위 10:", list(adv.head(10).index))
print("KS 상위 10:", list(ks.head(10).index))
both = sorted(set(adv.head(20).index) & set(ks.head(20).index))
print("둘 다 상위 20에 든 센서:", both)
print("59번 순위  적대적:", list(adv.index).index(59) + 1, " KS:", list(ks.index).index(59) + 1)

pd.DataFrame({"adversarial": adv, "ks": ks}).to_csv("drift_sensors.csv")
mlflow.set_experiment("secom-defect")
with mlflow.start_run(run_name="adversarial_validation"):
    mlflow.log_metric("past_vs_future_auc", auc.mean())
    mlflow.log_metric("n_sensors_in_both_top20", len(both))
    mlflow.log_artifact("drift_sensors.csv")