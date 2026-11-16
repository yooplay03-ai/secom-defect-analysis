import pandas as pd
import matplotlib.pyplot as plt

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/"
X = pd.read_csv(URL + "secom.data", sep=" ", header=None)
labels = pd.read_csv(URL + "secom_labels.data", sep=" ", header=None)

df = pd.DataFrame({
    "time": pd.to_datetime(labels[1], format="%d/%m/%Y %H:%M:%S"),
    "s59": X[59],
    "y": (labels[0] == 1).astype(int),
}).sort_values("time").reset_index(drop=True)

# 최근 100개 제품의 평소 값 (현재 제품은 빼고)
df["base"] = df["s59"].shift(1).rolling(100, min_periods=30).median()
df["rel"] = df["s59"] - df["base"]

cut = int(len(df) * 0.75)
past, future = df.iloc[:cut], df.iloc[cut:]
print("59번 중앙값  과거:", round(past["s59"].median(), 2), " 미래:", round(future["s59"].median(), 2))

for name, col in [("절대값 규칙", "s59"), ("평소 대비 규칙", "rel")]:
    th = past[col].quantile(0.8)
    flag = future[col] > th
    caught = (flag & (future["y"] == 1)).sum()
    total = future["y"].sum()
    print(f"{name} (기준 {th:.2f}): 검사 {flag.mean():.0%} → 불량 {caught}/{total}개 (recall {caught / total:.2f})")

# 시간에 따른 59번 그래프
plt.figure(figsize=(11, 4))
plt.plot(df.index, df["s59"], ".", markersize=2, alpha=0.4, label="sensor 59")
plt.plot(df.index, df["base"], color="orange", label="recent median (100)")
fails = df[df["y"] == 1]
plt.plot(fails.index, fails["s59"], "rx", markersize=5, label="fail")
plt.axvline(cut, color="black", linestyle="--", label="past | future")
plt.ylim(-30, 60)
plt.xlabel("product order (by time)")
plt.legend()
plt.tight_layout()
plt.savefig("sensor59_time.png")