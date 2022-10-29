import pandas as pd
from pathlib import Path
from BaseMetric import BaseMetric

class ExampleMetric(BaseMetric):

    def __init__(self):
        super().__init__()
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        return data[["Timestep", "X Position"]]
        

if __name__ == "__main__":
    metric = ExampleMetric()
    path_name = "../out/FLOCK_SIZE"
    p = Path(path_name)
    data = metric.run_metric(p)
    print(metric.std(data[0]))

