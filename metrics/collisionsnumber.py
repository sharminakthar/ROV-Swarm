from tokenize import group
import pandas as pd
import numpy as np
from pathlib import Path
from BaseMetric import BaseMetric
from pandas import DataFrame
from pandas import *

class CollisionsNumber(BaseMetric):

    def __init__(self):
        super().__init__()

    def myfunction(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.merge(data, how='cross')

        distances = np.sqrt( ((df["X Position_x"] - df["X Position_y"]).pow(2)) + ((df["Y Position_x"] - df["Y Position_y"]).pow(2)))
        distances = distances[distances > 0]

        df["Distances"] = distances

        df.loc[df["Distances"] < 7.5, 'Less_than_separation_distance'] = 'True'
        df.loc[df["Distances"] >= 7.5, 'Less_than_separation_distance'] = 'False'

        collisions = df.Less_than_separation_distance[df.Less_than_separation_distance==True].count()
        return(collisions)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data[["Timestep" ,"Drone ID", "X Position", "Y Position"]]     
        print("grouping and applying")

        df= df.groupby('Timestep').apply(self.myfunction).reset_index()
        df = df.rename(index={0: "Timestep", 1: "Collisions"})

        print(df)
        return(df)

if __name__ == "__main__":
    metric = CollisionsNumber()

    # Replace path name with absolute path if not running from inside the metrics folder
    path_name = "/Users/sharmin/Desktop/GDP/swarm-simulator/out/BANDWIDTH"
    p = Path(path_name)
    print("running")
    data = metric.run_metric(p)
    print(metric.std(data[0]))