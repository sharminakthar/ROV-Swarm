from tokenize import group
import pandas as pd
import numpy as np
from pathlib import Path
from BaseMetric import BaseMetric
from pandas import DataFrame
from pandas import *

class SeparationMean(BaseMetric):

    def __init__(self):
        super().__init__()

    def myfunction(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.merge(data, how='cross')

        distances = np.sqrt( ((df["X Position_x"] - df["X Position_y"]).pow(2)) + ((df["Y Position_x"] - df["Y Position_y"]).pow(2)))
        distances = distances[distances > 0]
        # df["Distances"] = distances
        #df = df[["Timestep_x", "Drone ID_x", "Drone ID_y", "Distances"]]

        mean_distance = distances.mean()
        return(pd.Series([df["Timestep_x"].iloc[0], mean_distance]))

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data[["Timestep" ,"Drone ID", "X Position", "Y Position"]]     
        print("grouping and applying")
        df1= df.groupby('Timestep').apply(self.myfunction)

        #print(df1)
        return(df1)

if __name__ == "__main__":
    metric = SeparationMean()

    # Replace path name with absolute path if not running from inside the metrics folder
    path_name = "../out/BANDWIDTH"
    p = Path(path_name)
    print("running")
    data = metric.run_metric(p)
    print(metric.std(data[0]))