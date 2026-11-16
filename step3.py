import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, precision_score

model = RandomForestClassifier(class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:,1]

for th in [0.5, 0.3, 0.2, 0.1]:
    pred = (proba >= th).astype(int)
    r = recall_score(y_test, pred)
    p = precision_score(y_test, pred, zero_division=0)
    print(f"임계값 {th}: 잡은 불량 비율 {r:.2f}, 불량이라 한 것 중 진짜 {p:.2f}")
    print(confusion_matrix(y_test, pred))