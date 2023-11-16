import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# データの読み込み
df = pd.read_pickle("../work/input/LSWMD_25519.pkl")

# lotNameから数字部分（lotNumber）を抽出
df['lotNumber'] = df['lotName'].str.extract('(\d+)').astype(int)

# 密度プロットの作成
# plt.figure(figsize=(12, 8))
# # bw_adjust=0.5で帯域幅を調整
# sns.kdeplot(df['lotNumber'], bw_adjust=0.5)
# plt.xlabel('lot 1 to 47542')
# plt.ylabel('Density')
# plt.title('Density of lotNumber')
# plt.show()

# ヒストグラムの作成
# plt.figure(figsize=(12, 8))
# # bin=100でヒストグラムの棒の数を100に指定 alpha=0.5で透明度を指定
# plt.hist(df['lotNumber'], bins=100, alpha=0.5)
# plt.xlabel('lot 1 to 47542')
# plt.ylabel('Density')
# plt.title('Density of lotNumber')
# plt.show()