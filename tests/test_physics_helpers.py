import unittest
import numpy.testing
from util.physics_helpers import *
import numpy as np

class TestVectorHelpers(unittest.TestCase):

    def test_calculate_acceleration(self):
        up = np.asarray([0,1]).reshape((2,1))
        left = np.asarray([-1,0]).reshape((2,1))
        zero = np.asarray([0,0]).reshape((2,1))
        
        numpy.testing.assert_array_almost_equal(calculate_acceleration(0, 0, 0, up, 1), zero)
        numpy.testing.assert_array_almost_equal(calculate_acceleration(up * 10000, 8, 0, up, 1), up * 8)
        numpy.testing.assert_array_almost_equal(calculate_acceleration(up * 10000, 8, 0, up, 0), zero)
        numpy.testing.assert_array_almost_equal(calculate_acceleration(left * 10000, 8, 0, left, 10), left*8)
    
    def test_calculate_velocity(self):
        up = np.asarray([0,1]).reshape((2,1))
        left = np.asarray([-1,0]).reshape((2,1))
        zero = np.asarray([0,0]).reshape((2,1))
        
        numpy.testing.assert_array_almost_equal(calculate_velocity(zero, zero, 100, 100, 1), zero)
        numpy.testing.assert_array_almost_equal(calculate_velocity(left * 1000, left, 100, 100, 1), left * 100)
        numpy.testing.assert_array_almost_equal(calculate_velocity(left * 1000, left, 100, 100, 0), left)
        numpy.testing.assert_array_almost_equal(calculate_velocity(up * 1000, up, 100, 100, 1), up * 100)
        
    def test_get_numerical_velocity(self):
        left = np.asarray([-1,0]).reshape((2,1))
        zero = np.asarray([0,0]).reshape((2,1))
        
        self.assertEqual(get_numerical_velocities(left), (-1, 0, 1))
        self.assertEqual(get_numerical_velocities(zero), (0, 0, 0))
        self.assertEqual(get_numerical_velocities(left * 100), (-100, 0, 100))        

if __name__ == '__main__':
    unittest.main()