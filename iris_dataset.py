import pandas as pd
import numpy as np


def get_target_features():
    ds = pd.read_csv("iris.csv")
    features = ds[list(ds.columns[:4])]
    target = np.array(range(150)).reshape(150, 1)
    for i in range(1, 151):
        if i <= 50:
            target[i - 1][0] = 0
        elif 50 < i <= 100:
            target[i - 1][0] = 1
        else:
            target[i - 1][0] = 2
    return target, features
