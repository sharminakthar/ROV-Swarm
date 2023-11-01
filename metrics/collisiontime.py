import os
import sys
from BaseMetric import BaseMetric
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class CollisionTime(BaseMetric):

    def __init__(self):
        super().__init__()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.merge(data, how="cross")
        df = df.loc[df["Drone ID_x"] < df["Drone ID_y"]]

        distances = np.sqrt( ((df["X Position_x"] - df["X Position_y"]).pow(2)) + ((df["Y Position_x"] - df["Y Position_y"]).pow(2)))
        
        df["Distances"] = distances

        time = [df["Distances"] < 7.5].index.to_numpy()
        first_time = time[0]

        return first_time

if __name__ == "__main__":
    metric = CollisionTime()
    data = metric.run_metric("..\\out\\FLOCK_SIZE")
    print(metric.std(data[0]))