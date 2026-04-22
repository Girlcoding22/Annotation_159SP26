import pandas as pd
from sklearn.metrics import cohen_kappa_score

df = pd.read_csv("merged.csv")
df = df.dropna(subset=["label_A", "label_B"])
df["label_A"] = df["label_A"].astype(int)
df["label_B"] = df["label_B"].astype(int)

kappa = cohen_kappa_score(df["label_A"], df["label_B"])
kappa_weighted = cohen_kappa_score(df["label_A"], df["label_B"], weights="linear")
kappa_weighted_q = cohen_kappa_score(df["label_A"], df["label_B"], weights="quadratic")

print(f"Pairs evaluated : {len(df)}")
print(f"Cohen's Kappa (unweighted) : {kappa:.4f}")
print(f"Cohen's Kappa (linear weighted) : {kappa_weighted:.4f}")
print(f"Cohen's Kappa (quadratic weighted): {kappa_weighted_q:.4f}")

# Interpretation guide
def interpret(k):
    if k < 0:      return "Poor (worse than chance)"
    elif k < 0.20: return "Slight"
    elif k < 0.40: return "Fair"
    elif k < 0.60: return "Moderate"
    elif k < 0.80: return "Substantial"
    else:          return "Almost perfect"

print(f"\nInterpretation (Landis & Koch):")
print(f"  Unweighted : {interpret(kappa)}")
print(f"  Linear     : {interpret(kappa_weighted)}")
print(f"  Quadratic  : {interpret(kappa_weighted_q)}")
