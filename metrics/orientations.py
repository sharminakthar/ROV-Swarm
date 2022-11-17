from pathlib import Path
from .BaseMetric import BaseMetric
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
from .Helper_Functions import getOrientations

class OrientationMetric(BaseMetric):

    def __init__(self):
        super().__init__()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data[["Timestep", "X Velocity", "Y Velocity"]]
        bearings = getOrientations(df)
        df = df[["Timestep"]]
        new_df = df.assign(Orientations = bearings)
        groups = new_df.groupby("Timestep").std()
        df = df.merge(groups, on="Timestep")
        return df

   

if __name__ == "__main__":
    metric = OrientationMetric()

    # Replace path name with absolute path if not running from inside the metrics folder
    path_name = "/Users/sharmin/Desktop/GDP/swarm-simulator/out/RANGE_ERROR"
    p = Path(path_name)

    data = metric.run_metric(p)
    for k,d in data.items():
        plt.plot(d["Timestep"], d.loc[:, d.columns != "Timestep"].mean(axis=1), label=k)
    plt.legend()
    plt.title("Standard deviation of orientations with varying Range error")
    plt.xlabel("Timestep")
    plt.ylabel("Standard deviation of orientations")
    plt.show()