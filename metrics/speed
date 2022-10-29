import os
import sys
from BaseMetric import BaseMetric
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

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
    metric = ExampleMetric()
    data = metric.run_metric(r"\Users\shinee\Desktop\EEE YEAR 4\GDP\clone\swarm-simulator\out\FLOCK_SIZE")
    print(metric.std(data[0]))
