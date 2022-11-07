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

    def myfunction(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.merge(data, how="cross")
        df = df.loc[df["Drone ID_x"] < df["Drone ID_y"]]

        distances = np.sqrt( ((df["X Position_x"] - df["X Position_y"]).pow(2)) + ((df["Y Position_x"] - df["Y Position_y"]).pow(2)))

        df["Distances"] = distances

        df.loc[df["Distances"] < 75, 'Less_than_separation_distance'] = True
        df.loc[df["Distances"] >= 75, 'Less_than_separation_distance'] = False

        collisions = df.Less_than_separation_distance[df.Less_than_separation_distance==True].count()
        return(collisions)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data[["Timestep" ,"Drone ID", "X Position", "Y Position"]]     
        print("grouping and applying")

        df= df.groupby('Timestep').apply(self.myfunction).reset_index()
        df = df.rename(columns={0: "Collisions"})

        print(df)
        return(df)

if __name__ == "__main__":
    metric = CollisionsNumber()

    # Replace path name with absolute path if not running from inside the metrics folder
    path_name = "/Users/sharmin/Desktop/GDP/swarm-simulator/out/FLOCK_SIZE"
    p = Path(path_name)
    print("running")
    data = metric.run_metric(p)
    # "list comprehension"
    heights = {k:0 for k in data.keys()}
    for k,d in data.items():
        val = d.loc[:, d.columns != "Timestep"].to_numpy().sum() /( d.shape[1] -1 )
        heights[k] = val
    plt.bar(heights.keys(), heights.values())
    plt.legend()
    plt.xlabel("Flock size")
    plt.ylabel("Total number of collisions")
    plt.show()