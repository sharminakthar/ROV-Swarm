import pandas as pd
from abc import ABC, abstractmethod
import os

class BaseMetric(ABC):
    """
    Abstract class defining key methods for every metric
    """

    def __init__(self):
        pass

    def read_csv(self, csv_file: str) -> pd.DataFrame:
        pass

    def run_metric(self, folder):
        results = []
        for run in os.listdir(folder):
            data = pd.read_csv(run)
            result = self.calculate(data)
            results.append(result)
        
        return results

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the metric on the raw data of a single simulation

        Parameters:
            data: pd.Dataframe
                The raw data to run the metric on
        
        Returns:
            A pandas dataframe with one column for the timestep and another column for the metric.
        """
        pass
