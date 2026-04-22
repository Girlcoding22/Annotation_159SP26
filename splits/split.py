import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("adjudicated.txt", sep="\t")

train, temp = train_test_split(df, test_size=0.4, random_state=42)
dev, test = train_test_split(temp, test_size=0.5, random_state=42)

train.to_csv("train.txt", sep="\t", index=False)
dev.to_csv("dev.txt", sep="\t", index=False)
test.to_csv("test.txt", sep="\t", index=False)

print(f"Total : {len(df)}")
print(f"Train : {len(train)} ({len(train)/len(df):.0%})")
print(f"Dev   : {len(dev)} ({len(dev)/len(df):.0%})")
print(f"Test  : {len(test)} ({len(test)/len(df):.0%})")
