from pathlib import Path
from BaseMetric import BaseMetric
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class CentreDistMetric(BaseMetric):

    def __init__(self):
        super().__init__()
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data[["Timestep", "X Position", "Y Position"]]
        groups = df.groupby("Timestep")
        centres = groups.mean().reset_index()
        centres = centres.rename(columns={"X Position": "X Centre", "Y Position": "Y Centre"})
        df = df.merge(centres, on="Timestep")
        distances = np.sqrt(((df["X Position"] - df["X Centre"]).pow(2) + (df["Y Position"] - df["Y Centre"]).pow(2)))
        df["Distances"] = distances
        df = df[["Timestep", "Distances"]].groupby("Timestep").mean().reset_index()
        print(df)
        return df

if __name__ == "__main__":
    metric = CentreDistMetric()

    # Replace path name with absolute path if not running from inside the metrics folder
    path_name = "../out/FLOCK_SIZE"
    p = Path(path_name)

    data = metric.run_metric(p)
    for k,d in data.items():
        plt.plot(d["Timestep"], d.loc[:, d.columns != "Timestep"].mean(axis=1), label=k)
    plt.legend()
    plt.show()
