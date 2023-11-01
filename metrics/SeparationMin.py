from tokenize import group
import pandas as pd
import numpy as np
from pathlib import Path
from BaseMetric import BaseMetric
from pandas import DataFrame
from pandas import *
import time

class SeparationMin(BaseMetric):

    def __init__(self):
        super().__init__()

    def myfunction(self, df: pd.DataFrame) -> pd.DataFrame:
        la, lb = len(df), len(df)
        ia2, ib2 = np.broadcast_arrays(*np.ogrid[:la,:lb])
        x = np.column_stack([df.values[ia2.ravel(),1:], df.values[ib2.ravel(),1:]])
        distances = np.linalg.norm(x[:,:2] - x[:,2:])

        return distances.min()

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data[["Timestep", "X Position", "Y Position"]]     
        print("grouping and applying")
        df1= df.groupby('Timestep').apply(self.myfunction).reset_index()

        #print(df1)
        return(df1)

if __name__ == "__main__":
    metric = SeparationMin()

    # Replace path name with absolute path if not running from inside the metrics folder
    path_name = "../out/BANDWIDTH"
    p = Path(path_name)
    print("running")
    data = metric.run_metric(p)
    print(metric.std(data[0]))
    