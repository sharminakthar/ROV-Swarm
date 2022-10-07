from numpy.random import Generator
import numpy as np
from .abstract_sensor import AbstractSensor
from util.vector_helpers import vector_to_bearing


class BearingSensor(AbstractSensor):
    error_deg = 0
    rng = None

    def __init__(self, rng : Generator, error_deg : float, calibration_error : float):
        self.error_deg = error_deg
        self.rng = rng
        self.calibration_error = self.rng.normal(0, calibration_error)

    def get_reading(self, self_position : np.ndarray, god_other_position : np.ndarray) -> float:
        bearing_reading = vector_to_bearing(god_other_position - self_position)

        # use gaussian to get a reading with error
        bearing_reading += self.rng.normal(0, self.error_deg)
        bearing_reading += self.calibration_error

        return bearing_reading
