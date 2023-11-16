import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# データの読み込み
df = pd.read_pickle("../work/input/LSWMD_25519.pkl")

# lotNameから数字部分を抽出し整数型に変換
df['lotNumber'] = df['lotName'].str.extract('(\d+)').astype(int)

# 指定された範囲に基づいてlotNameをグループ化
conditions = [
    (df['lotNumber'] <= 10000),
    (df['lotNumber'] > 10000) & (df['lotNumber'] <= 30000),
    (df['lotNumber'] > 30000)
]
choices = ['lot1-10000', 'lot10001-30000', 'lot30001-47542']
df['lotGroup'] = np.select(conditions, choices, default=np.nan)

# クロスタブ（分割表）を作成
cross_tab = pd.crosstab(df['lotGroup'], df['failureType'])

# カイ二乗検定を実行
chi2, p, dof, expected = chi2_contingency(cross_tab)

# 結果の出力
print(f"Chi-squared Test statistic: {chi2}")
print(f"p-value: {p}")
print(f"Degrees of freedom: {dof}")
print("Expected frequencies:")
print(expected)

# 分割表の表示
cross_tab
