import numpy as np
from numpy.random import Generator
from .abstract_sensor import AbstractSensor


class RangeSensor(AbstractSensor):
    error_m = 0
    rng = None

    def __init__(self, rng : Generator, error_m : float, calibration_error_m : float):
        self.error_m = error_m
        self.rng = rng
        self.calibration_error = self.rng.normal(0, calibration_error_m)

    def get_reading(self, self_position, god_other_position) -> float:
        range_reading = np.linalg.norm(god_other_position - self_position)
        
        # get percentage of error
        relative_range_error = range_reading * (self.error_m/100)
        # use gaussian to get a reading with error
        range_reading += self.rng.normal(0, relative_range_error)
        range_reading += self.calibration_error

        return range_reading
