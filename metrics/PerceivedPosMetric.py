from pathlib import Path
from BaseMetric import BaseMetric
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class PerceivedPosMetric(BaseMetric):

    def __init__(self):
        super().__init__()
    
    def calc_error(self, df: pd.DataFrame) -> pd.DataFrame:
        x_error = df["X Position"] - df["X Approx Position"]
        y_error = df["Y Position"] - df["Y Approx Position"]
        error = np.sqrt(x_error.pow(2) + y_error.pow(2))
        return error.mean()
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data[["Timestep", "X Position", "Y Position", "X Approx Position", "Y Approx Position"]]
        return df.groupby("Timestep").apply(self.calc_error).reset_index()

if __name__ == "__main__":
    metric = PerceivedPosMetric()

    # Replace path name with absolute path if not running from inside the metrics folder
    path_name = "../out/SPEED_ERROR"
    p = Path(path_name)

    data = metric.run_metric(p)
    for k,d in data.items():
        plt.plot(d["Timestep"], d.loc[:, d.columns != "Timestep"].mean(axis=1), label=k)
    plt.title("Error Between Actual Position and Approximated Position \n With Varying Speed Error")
    plt.xlabel("Timestep (s)")
    plt.ylabel("Position Error (m)")
    plt.legend()
    plt.savefig("..\\..\\..\\Figures\\speed_per_pos.png")
    plt.show()
