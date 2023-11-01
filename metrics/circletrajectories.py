import pandas as pd
from pathlib import Path
from .BaseMetric import BaseMetric
from .orientations import OrientationMetric
from .Helper_Functions import getOrientations
from matplotlib import pyplot as plt
import numpy as np
import math

class CircleTrajectoryMetric(BaseMetric):
    def __init__(self):
        super().__init__()
    
    def distance(self, df: pd.DataFrame) -> pd.DataFrame:
      
        targx = 2500
        targy = 2500

        r = 1000
        df1 = abs(df["X Position"] - targx)
        df2 = abs(df["Y Position"] - targy)
        return (df1.multiply(df1, fill_value=0) + df2.multiply(df2, fill_value=0))**0.5 - r



    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data[["Timestep", "X Position", "Y Position"]]
        df1 = self.distance(df).rename("Distance_From_Circle")
        new_df = df.assign(DistanceFromCircle =  df1)
    

        final_df = new_df[["Timestep", "DistanceFromCircle"]].groupby("Timestep").mean().reset_index()
   
        return final_df

if __name__ == "__main__":
    metric = CircleTrajectoryMetric()

    # Replace path name with absolute path if not running from inside the metrics folder
    path_name = "../FOLLOW_CIRCLE_EXTENDED/SPEED_ERROR"
    p = Path(path_name)

    data = metric.run_metric(p)
    for k,d in data.items():
        plt.plot(d["Timestep"], d.loc[:, d.columns != "Timestep"].mean(axis=1), label=k)
    plt.legend()
    plt.show()



class Circle():
    def __init__(self, r):
        self.radius = r

    def area(self):
        return self.radius**2*3.14
    
    def perimeter(self):
        return 2*self.radius*3.14