import pandas as pd
from pathlib import Path
from BaseMetric import BaseMetric
import math
from orientations import OrientationMetric
from matplotlib import pyplot as plt


class ExampleMetric(BaseMetric):
    #TARGET_X = 2500
    #TARGET_Y = 2500

    def __init__(self):
        super().__init__()
    
    def optimaltrajectories(self, data: pd.DataFrame) -> pd.DataFrame:
        #STRAIGHT LINE HEADING:
        targX = 2500
        targY = 2500
        df = data.sub
        df["X Position"] = df["X Position"].sub(targX)
        df["Y Position"] = df["Y Position"].sub(targY)
        df['X Position'] = df['X Position'].apply(lambda x: x*-1)
        df['Y Position'] = df['Y Position'].apply(lambda x: x*-1)

        optimalBearings = OrientationMetric.getOrientations(df)

        return optimalBearings

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data[["Timestep", "X Velocity", "Y Velocity"]]
        bearings = OrientationMetric.getOrientations(df)
        optimal_df = data[["Timestep", "X Position", "Y Position"]]

        optimalBearings = self.optimaltrajectories(optimal_df)
        bearing_diff = math.abs(bearings - optimalBearings)

        df = data["Timestep"]
        new_df = df.assign(Mean_Abs_Orientation_Error = bearing_diff)
        groups = new_df.groupby("Timestep").mean()
        df = df.merge(groups, on="Timestep")

        return df
        

if __name__ == "__main__":
    metric = OrientationMetric()

    # Replace path name with absolute path if not running from inside the metrics folder
    path_name = "../out/FLOCK_SIZE"
    p = Path(path_name)

    data = metric.run_metric(p)
    for k,d in data.items():
        plt.plot(d["Timestep"], d.loc[:, d.columns != "Timestep"].mean(axis=1), label=k)
    plt.legend()
    plt.show()



