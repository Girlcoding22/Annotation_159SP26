import pandas as pd

annotators = ["A", "B", "C", "D"]
parts = []

for i, ann in enumerate(annotators):
    df = pd.read_csv(f"output_annotation_file_{i}.txt", sep="\t", header=0,
                     names=["id", "label", "text"], skiprows=1)
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.dropna(subset=["id"])
    df["id"] = df["id"].astype(int)
    df["text"] = df["text"].str.strip()
    df["annotator"] = ann
    parts.append(df)

stacked = pd.concat(parts, ignore_index=True)

# For each (id, text), assign the two annotator labels as label_A and label_B
def pivot_labels(group):
    labels = group["label"].tolist()
    return pd.Series({
        "label_A": labels[0] if len(labels) > 0 else None,
        "label_B": labels[1] if len(labels) > 1 else None,
    })

merged = (
    stacked.groupby(["id", "text"], sort=False)
    .apply(pivot_labels)
    .reset_index()
    .sort_values("id")
    .reset_index(drop=True)
)

merged = merged[["id", "label_A", "label_B", "text"]]
merged.to_csv("merged.csv", index=False)
print(f"Merged {len(merged)} rows")
