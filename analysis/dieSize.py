import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# データの読み込み
def load_data(pickle_path):
    return pd.read_pickle(pickle_path)

# dieSizeとwaferMapのピクセル数の相関係数を計算
def calculate_pixel_count(wafer_map):
    return wafer_map.size

# 相関係数の計算とプロット
def plot_correlation(df):
    df['pixelCount'] = df['waferMap'].apply(calculate_pixel_count)
    
    # プロット
    plt.figure(figsize=(10, 5))
    plt.scatter(df['dieSize'], df['pixelCount'], alpha=0.5)
    plt.title('dieSize vs pixelCount')
    plt.xlabel('dieSize')
    plt.ylabel('pixelCount')
    plt.grid(True)
    plt.show()

    # 相関係数の計算
    correlation_coefficient = df['dieSize'].corr(df['pixelCount'])
    print(f"correlation coefficient: {correlation_coefficient}")

# データの読み込み
df = load_data("../work/input/LSWMD_25519.pkl")
# 相関係数の計算とプロット
plot_correlation(df)
