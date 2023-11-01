import pandas as pd
import numpy as np
from pathlib import Path
from BaseMetric import BaseMetric
from pandas import DataFrame

class SeparationMin(BaseMetric):

    def __init__(self):
        super().__init__()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data[["Timestep" ,"Drone ID", "X Position", "Y Position"]]     
        df = df.loc[df['Timestep'] == 0]
        df = df.merge(df, how='cross')

        distances = np.sqrt( ((df["X Position_x"] - df["X Position_y"]).pow(2)) + ((df["Y Position_x"] - df["Y Position_y"]).pow(2)))
        df["Distances"] = distances
        df = df[["Timestep_x", "Drone ID_x", "Drone ID_y", "Distances"]]

        min_distance = df.loc[df['Distances']>0, 'Distances'].min()
        df["Minimum Separation"] = min_distance
        df = df[["Timestep_x", "Minimum Separation"]]
        print(df.iloc[0])


if __name__ == "__main__":
    metric = SeparationMin()

    # Replace path name with absolute path if not running from inside the metrics folder
    path_name = "/Users/sharmin/Desktop/GDP/swarm-simulator/out/FLOCK_SIZE"
    p = Path(path_name)

    data = metric.run_metric(p)
    print(metric.std(data[0]))
    