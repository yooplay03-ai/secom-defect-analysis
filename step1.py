import pandas as pd

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/"
X = pd.read_csv(URL + "secom.data", sep=" ", header=None)
y = pd.read_csv(URL + "secom_labels.data", sep=" ", header= None)

print(X.shape)
print(y.shape)
print(y[0].value_counts())

print("전체 빈 칸: ", X.isnull().sum().sum())

missing_rate = X.isnull().mean()*100
print(missing_rate.sort_values(ascending=False).head(10))

print("값이 하나뿐인 센서 수: ", (X.nunique()==1).sum())