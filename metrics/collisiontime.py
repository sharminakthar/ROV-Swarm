import os
import sys
from BaseMetric import BaseMetric
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

class CollisionTime(BaseMetric):

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
        df2 = df.loc[df['Collisions']>0]
        col = df2['Timestep'].min()
        print("Timestep of first collision is")
        print(col)

        return (col)

if __name__ == "__main__":
    metric = CollisionTime()
    path_name = "../out/FLOCK_SIZE"
    p = Path(path_name) 
    data = metric.run_metric(p)
    print(metric.std(data[0]))