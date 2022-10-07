import numpy as np
from numpy.random import Generator
from util.vector_helpers import get_random_vector_normal
from .abstract_sensor import AbstractSensor


class AccelerationSensor(AbstractSensor):
    error_acc = 0
    rng = None

    def __init__(self, rng : Generator, error_acc : float, calibration_error : float):
        self.error_acc = error_acc
        self.rng = rng
        self.calibration_error = get_random_vector_normal(rng, calibration_error)

    def get_reading(self, crt_acc):
        acc_reading = crt_acc
        
        error_size = np.linalg.norm(acc_reading) * (self.error_acc/100)
                
        acc_reading += get_random_vector_normal(self.rng, error_size)
        acc_reading += self.calibration_error

        return acc_reading
