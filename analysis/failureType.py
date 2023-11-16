import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# データの読み込み
df = pd.read_pickle("../work/input/LSWMD_25519.pkl")

# # failureTypeの値ごとのカウントを計算
# failure_counts = df['failureType'].value_counts()

# # 棒グラフの描画
# plt.figure(figsize=(10, 6))
# failure_counts.plot(kind='bar')
# plt.title('Counts of Each Failure Type')
# plt.xlabel('Failure Type')
# plt.ylabel('Counts')
# plt.xticks(rotation=45)  # X軸のラベルを45度回転
# plt.show()

# train_test_splitを使用してデータを分割
train_df, test_df = train_test_split(df, stratify=df['failureType'], test_size=0.10, random_state=42)

# train_df の failureType の値ごとのカウントを計算
failure_counts = train_df['failureType'].value_counts()

# 棒グラフを描画
plt.figure(figsize=(10, 6))
failure_counts.plot(kind='bar')
plt.title('Counts of Each Failure Type in Training Data')
plt.xlabel('Failure Type')
plt.ylabel('Counts')
plt.xticks(rotation=45)
plt.show()
