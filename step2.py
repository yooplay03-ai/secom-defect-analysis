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

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("정확도:", accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))