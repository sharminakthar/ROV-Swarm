import os
import sys
from BaseMetric import BaseMetric
import pandas as pd

class ExampleMetric(BaseMetric):

    def __init__(self):
        super().__init__()
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        return data[["Timestep", "X Position"]]
        

if __name__ == "__main__":
    metric = ExampleMetric()
    data = metric.run_metric(r"\Users\shinee\Desktop\EEE YEAR 4\GDP\clone\swarm-simulator\out\FLOCK_SIZE")
    print(metric.std(data[0]))

