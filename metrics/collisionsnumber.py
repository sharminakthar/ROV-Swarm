from tokenize import group
import pandas as pd
import numpy as np
from pathlib import Path
from BaseMetric import BaseMetric
from pandas import DataFrame
from pandas import *
from matplotlib import pyplot as plt

class CollisionsNumber(BaseMetric):

    def __init__(self):
        super().__init__()

    def myfunction(self, df: pd.DataFrame, index1, index2) -> pd.DataFrame:
        arr1 = df.to_numpy()[index1]
        arr2 = df.to_numpy()[index2]
        distances = np.linalg.norm(arr1[:,1:] - arr2[:,1:], axis=1)
        return (distances < 7.5).sum()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data[["Timestep", "X Position", "Y Position"]]     
        
        i = len(df.loc[df["Timestep"] == 0])
        ia2, ib2 = np.broadcast_arrays(*np.ogrid[:i,:i])
        indices = np.triu_indices(i,1)
        index1 = ia2[indices]
        index2 = ib2[indices]

        df1= df.groupby('Timestep').apply(self.myfunction, index1=index1, index2=index2).reset_index()

        return df1

if __name__ == "__main__":
    metric = CollisionsNumber()

    # Replace path name with absolute path if not running from inside the metrics folder
    path_name = "../out/FLOCK_SIZE"
    p = Path(path_name)
    print("running")
    data = metric.run_metric(p)
    # "dictionary comprehension"
    heights = {k:0 for k in data.keys()}
    for k,d in data.items():
        # Get mean number of collisions over each run
        val = d.loc[:, d.columns != "Timestep"].to_numpy().sum() / ( d.shape[1] - 1 )
        heights[k] = val
    plt.bar(heights.keys(), heights.values())
    plt.legend()
    plt.xlabel("Flock size")
    plt.ylabel("Total number of collisions")
    plt.show()