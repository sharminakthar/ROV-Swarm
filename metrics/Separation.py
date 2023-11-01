from tokenize import group
import pandas as pd
import numpy as np
from pathlib import Path
from BaseMetric import BaseMetric
from pandas import DataFrame
from pandas import *
from matplotlib import pyplot as plt

class Separation(BaseMetric):

    def __init__(self, reduction="min"):
        try:
            self.reduction_func = {"min": np.min, "max": np.max, "mean": np.mean}[reduction]
        except KeyError:
            print("Error: expected reduction to be one of 'min', 'max', or 'mean'")
            quit()
        super().__init__()

    def myfunction(self, df: pd.DataFrame, indices) -> pd.DataFrame:
        la, lb = len(df), len(df)
        ia2, ib2 = np.broadcast_arrays(*np.ogrid[:la,:lb])
        x = np.column_stack([df.values[ia2[indices],1:], df.values[ib2[indices],1:]])
        distances = np.linalg.norm(x[:,:2] - x[:,2:], axis=1)
        return self.reduction_func(distances)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data[["Timestep", "X Position", "Y Position"]]
        i = len(df.loc[df["Timestep"] == 0])
        indices = np.triu_indices(5,1)
        df1= df.groupby('Timestep').apply(self.myfunction, indices=indices).reset_index()

        return(df1)

if __name__ == "__main__":
    metric = Separation(reduction="min")

    # Replace path name with absolute path if not running from inside the metrics folder
    path_name = "../out/BANDWIDTH"
    p = Path(path_name)
    print("running")
    data = metric.run_metric(p)
    for k,d in data.items():
        plt.plot(d["Timestep"], d.loc[:, d.columns != "Timestep"].mean(axis=1), label=k)
    plt.legend()
    plt.show()
    