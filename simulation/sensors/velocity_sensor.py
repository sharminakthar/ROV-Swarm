from numpy.random import Generator
from .abstract_sensor import AbstractSensor
from util.vector_helpers import bearing_to_vector, get_random_vector_normal


class VelocitySensor(AbstractSensor):
    error_speed = 0
    rng = None

    def __init__(self, rng : Generator, error_speed : float, calibration_error : float):
        self.error_speed = error_speed
        self.rng = rng
        self.calibration_error = get_random_vector_normal(rng, calibration_error)

    def get_reading(self, exact_speed : float, heading_reading : float):
        
        error_size = exact_speed * (self.error_speed/100)

        random_error = self.rng.normal(0, error_size)

        speed_with_error = exact_speed + random_error

        return (speed_with_error * bearing_to_vector(heading_reading)) + self.calibration_error