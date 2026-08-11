

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n, p = 34, 20

X = rng.normal(size=(n, p))
signal = X[:, 0] * 1.2 - X[:, 1] * 0.8 + X[:, 2] * 0.6 + rng.normal(scale=0.5, size=n)
group = pd.cut(signal, bins=[-np.inf, -0.4, 0.4, np.inf],
               labels=["low", "the", "strong"]).astype(str)

df = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
df.insert(0, "id", [f"c{i:02d}" for i in range(n)])
df["group"] = group
df.loc[df.sample(4, random_state=1).index, "f5"] = np.nan   # a few missing values

df.to_csv("examples/demo_features.csv", index=False)
print("wrote examples/demo_features.csv", df.shape,
      "| groups:", df.group.value_counts().to_dict())
