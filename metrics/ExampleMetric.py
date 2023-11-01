import os
import sys

'''sys.path.append(os.getcwd() + "\metrics")
print(os.getcwd())'''

print(os.getcwd())
from BaseMetric import BaseMetric
import pandas as pd

class ExampleMetric(BaseMetric):

    def __init__(self):
        super().__init__()
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        return data[["Timestep", "X Position"]]

if __name__ == "__main__":
    metric = ExampleMetric()
    print(metric.run_metric("..\\out\\FLOCK_SIZE"))
