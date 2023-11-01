import numpy as np
from numpy.random import Generator
from .abstract_sensor import AbstractSensor
from util.vector_helpers import vector_to_bearing


class HeadingSensor(AbstractSensor):
    error_deg = 0
    rng = None

    def __init__(self, rng : Generator, error_deg : float, calibration_error : float):
        self.error_deg = error_deg
        self.rng = rng
        self.calibration_error = self.rng.normal(0, calibration_error)

    def get_reading(self, velocity):
        heading_reading = vector_to_bearing(velocity)

        heading_reading += self.rng.normal(0, self.error_deg)
        heading_reading += self.calibration_error

        return heading_reading
