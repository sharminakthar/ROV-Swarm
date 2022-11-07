import os
import sys
from BaseMetric import BaseMetric
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

class Speed(BaseMetric):

    def __init__(self):
        super().__init__()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:

        df = data[["Timestep", "X Velocity", "Y Velocity"]]
        groups = df.groupby("Timestep")
        absdata = df.abs()
        absdata = absdata.rename(columns = {"X Velocity": "X Speed", "Y Velocity": "Y Speed"})
        df = df.merge( absdata, on="Timestep" )
        speed = np.sqrt ( (df["X Speed"]).pow(2) + (df["Y Speed"]).pow(2) )
        df["Speed"] = speed
        df = df[["Timestep", "Speed"]].groupby("Timestep").mean().reset_index()
        
        return df


if __name__ == "__main__":
    metric = Speed()

    # Replace path name with absolute path if not running from inside the metrics folder
    path_name = "/Users/sharmin/Desktop/GDP/swarm-simulator/out/SPEED_ERROR"
    p = Path(path_name)

    data = metric.run_metric(p)
    for k,d in data.items():
        plt.plot(d["Timestep"], d.loc[:, d.columns != "Timestep"].mean(axis=1), label=k)
    plt.legend()
    plt.xlabel("Timestep")
    plt.ylabel("Speed of flock")
    plt.title("The Average speed of the flock with varying Speed error")
    plt.show()
