import pandas as pd

df = pd.read_csv("vectors_full.csv", header=None)
print(df[4].values[0].shape)