import pandas as pd
import scipy.stats as stats

df=pd.read_pickle("../work/input/LSWMD_25519.pkl")

contingency_table = pd.crosstab(df['lotName'], df['failureType'])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print(f"Chi-squared: {chi2}")
print(f"P-value: {p}")
