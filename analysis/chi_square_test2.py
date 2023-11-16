import pandas as pd
import numpy as np
import scipy.stats as stats

df = pd.read_pickle("../work/input/LSWMD_25519.pkl")

def is_sequential(wafer_indices):
    wafer_indices = wafer_indices.astype(int)
    return list(wafer_indices) == list(range(min(wafer_indices), max(wafer_indices) + 1))

df['sequential'] = df.groupby('lotName')['waferIndex'].transform(lambda x: 1 if is_sequential(x.values) else 0)

contingency_table = pd.crosstab(df['sequential'], df['failureType'])

chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print(f"Chi-squared: {chi2}")
print(f"P-value: {p}")
