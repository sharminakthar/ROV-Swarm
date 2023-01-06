import pandas as pd
from pathlib import Path
from .BaseMetric import BaseMetric
from .orientations import OrientationMetric
from .Helper_Functions import getOrientations
from matplotlib import pyplot as plt
import numpy as np

class FHTrajectoryMetric(BaseMetric):
    def __init__(self):
        super().__init__()

    def optimaltrajectories(self, df: pd.DataFrame) -> pd.DataFrame:
        optimalBearings = 269
        
        return optimalBearings

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data[["Timestep", "X Velocity", "Y Velocity"]]
        
        bearings = getOrientations(df)
        optimal_df = data[["Timestep", "X Position", "Y Position"]]

        optimalBearings = self.optimaltrajectories(optimal_df)
        bearing_diff = abs(np.subtract(bearings, optimalBearings))

        new_df = df.assign(Mean_Abs_Orientation_Error = bearing_diff)
        new_df.drop(labels=['X Velocity', 'Y Velocity'], axis=1, inplace=True)
        groups = new_df.groupby("Timestep").mean()
        print(groups)
        df = df[["Timestep"]]
        df = df.merge(groups, on="Timestep")

        return df
        

if __name__ == "__main__":
    metric = FHTrajectoryMetric()
    
    # Replace path name with absolute path if not running from inside the metrics folder
    path_name = "../out/FLOCK_SIZE"
    p = Path(path_name)

    data = metric.run_metric(p)
    for k,d in data.items():
        plt.plot(d["Timestep"], d.loc[:, d.columns != "Timestep"].mean(axis=1), label=k)
    plt.legend()
    plt.show()



