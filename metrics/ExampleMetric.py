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

    # Replace path name with absolute path if not running from inside the metrics folder
    path_name = "/Users/sharmin/Desktop/GDP/swarm-simulator/out/FLOCK_SIZE"
    p = Path(path_name)

    data = metric.run_metric(p)
    print(metric.std(data[0]))
