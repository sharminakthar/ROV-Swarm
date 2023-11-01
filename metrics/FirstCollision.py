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

        df = df.loc[df['Less_than_separation_distance'] == True ]

        return df.head(1)

        

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data[["Timestep" ,"Drone ID", "X Position", "Y Position"]]     

        df= df.groupby('Timestep').apply(self.myfunction).reset_index()
        df = df.rename(columns={0: "Collisions"})

        print(df)
        return(df)

if __name__ == "__main__":
    metric = CollisionsNumber()

    # Replace path name with absolute path if not running from inside the metrics folder
    path_name = "../out/FLOCK_SIZE"
    p = Path(path_name)
    print("running")
    data = metric.run_metric(p)

    for k,d in data.items():
        plt.plot(d["Timestep"], d.loc[:, d.columns != "Timestep"].mean(axis=1), label=k)
    plt.legend()
    plt.show()
