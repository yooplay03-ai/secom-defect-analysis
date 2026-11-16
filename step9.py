import pandas as pd
import matplotlib.pyplot as plt

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/"
X = pd.read_csv(URL + "secom.data", sep=" ", header=None)
y = pd.read_csv(URL + "secom_labels.data", sep=" ", header=None)
y = (y[0] == 1).astype(int)

s = X[59]
print("59번 빈칸 비율:", round(s.isnull().mean(), 3))
print(s.groupby(y).describe())

bins = pd.qcut(s, q=5, duplicates="drop")
rate = y.groupby(bins, observed=True).agg(["mean", "count"])
print(rate)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].boxplot([s[y == 0].dropna(), s[y == 1].dropna()])
axes[0].set_xticks([1, 2], ["pass", "fail"])
axes[0].set_title("Sensor 59: pass vs fail")

axes[1].bar(range(len(rate)), rate["mean"])
axes[1].set_xticks(range(len(rate)), [str(i) for i in rate.index], rotation=30, fontsize=7)
axes[1].axhline(y.mean(), color="red", linestyle="--", label="overall defect rate")
axes[1].set_title("Defect rate by sensor 59 range")
axes[1].legend()

plt.tight_layout()
plt.savefig("sensor59.png")
axes[0].set_ylim(-30, 60)