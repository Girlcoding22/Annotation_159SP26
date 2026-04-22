import pandas as pd

df = pd.read_csv("merged.csv")
df["label_A"] = df["label_A"].astype("Int64")
df["label_B"] = df["label_B"].astype("Int64")

df["label"] = df[["label_A", "label_B"]].min(axis=1)

out = df[["id", "label", "text"]]
out.to_csv("adjudicated.csv", index=False)
print(f"Adjudicated {len(out)} rows")
print(f"Agreement (A==B): {(df['label_A'] == df['label_B']).sum()} rows")
print(f"Disagreement    : {(df['label_A'] != df['label_B']).sum()} rows")
