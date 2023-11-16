import pandas as pd
import matplotlib.pyplot as plt

# データの読み込み
df = pd.read_pickle("/mnt/data/LSWMD_25519.pkl")

# failureTypeの値ごとのカウントを計算
failure_counts = df['failureType'].value_counts()

# 棒グラフの描画
plt.figure(figsize=(10, 6))
failure_counts.plot(kind='bar')
plt.title('Counts of Each Failure Type')
plt.xlabel('Failure Type')
plt.ylabel('Counts')
plt.xticks(rotation=45)  # X軸のラベルを45度回転
plt.show()
