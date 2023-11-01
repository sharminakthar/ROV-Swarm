import unittest
import numpy.testing
from util.vector_helpers import *
import numpy as np

class TestVectorHelpers(unittest.TestCase):

    def test_angle_radians(self):
        up = np.asarray([0,1]).reshape((2,1))
        left = np.asarray([-1,0]).reshape((2,1))
        right = np.asarray([1,0]).reshape((2,1))
        down = np.asarray([0,-1]).reshape((2,1))
        
        self.assertEqual(angle_radians(up, left), (np.pi / 2, 1))
        self.assertEqual(angle_radians(up, right), (np.pi / 2, -1))
        self.assertEqual(angle_radians(up, down), (np.pi, -1))

    def test_rotate(self):
        up = np.asarray([0,1]).reshape((2,1))
        left = np.asarray([-1,0]).reshape((2,1))
        right = np.asarray([1,0]).reshape((2,1))
                
        numpy.testing.assert_array_almost_equal(rotate(up, np.pi / 2), right)
        numpy.testing.assert_array_almost_equal(rotate(up, -np.pi / 2), left)
        numpy.testing.assert_array_almost_equal(rotate(left, np.pi), right)

    def test_bearing_degrees_to_radians(self):
        self.assertEqual(bearing_degrees_to_direction_radians(0), np.pi / 2)
        self.assertEqual(bearing_degrees_to_direction_radians(360), np.pi / 2)
        self.assertEqual(bearing_degrees_to_direction_radians(90), 0)
        self.assertEqual(bearing_degrees_to_direction_radians(-90), np.pi)
        
    def test_direction_radians_to_bearing_degrees(self):
        self.assertEqual(direction_radians_to_bearing_degrees(np.pi / 2), 0)
        self.assertEqual(direction_radians_to_bearing_degrees(5 * np.pi / 2), 0)
        self.assertEqual(direction_radians_to_bearing_degrees(0), 90)
        self.assertEqual(direction_radians_to_bearing_degrees(np.pi), 270)

    def test_polar_to_cartesian(self):
        up = np.asarray([0,1]).reshape((2,1))
        left = np.asarray([-1,0]).reshape((2,1))
        right = np.asarray([1,0]).reshape((2,1))
        
        numpy.testing.assert_array_almost_equal(polar_to_cartesian(1, 0), right)
        numpy.testing.assert_array_almost_equal(polar_to_cartesian(1, np.pi / 2), up)
        numpy.testing.assert_array_almost_equal(polar_to_cartesian(7, np.pi / 2), 7 * up)
        numpy.testing.assert_array_almost_equal(polar_to_cartesian(1, np.pi), left)
        
        
    def test_bearing_to_vector(self):        
        up = np.asarray([0,1]).reshape((2,1))
        left = np.asarray([-1,0]).reshape((2,1))
        right = np.asarray([1,0]).reshape((2,1))
        
        numpy.testing.assert_array_almost_equal(bearing_to_vector(0), up)
        numpy.testing.assert_array_almost_equal(bearing_to_vector(360), up)
        numpy.testing.assert_array_almost_equal(bearing_to_vector(-360), up)
        numpy.testing.assert_array_almost_equal(bearing_to_vector(90), right)
        numpy.testing.assert_array_almost_equal(bearing_to_vector(270), left)
        
    def test_cartesian_to_polar(self):
        up = np.asarray([0,1]).reshape((2,1))
        left = np.asarray([-1,0]).reshape((2,1))
        right = np.asarray([1,0]).reshape((2,1))
        
        self.assertEqual(cartesian_to_polar(right), (1,0))
        self.assertEqual(cartesian_to_polar(up), (1, np.pi / 2))
        self.assertEqual(cartesian_to_polar(left * 10), (10, np.pi))
        
    def test_vector_to_bearing(self):
        up = np.asarray([0,1]).reshape((2,1))
        left = np.asarray([-1,0]).reshape((2,1))
        right = np.asarray([1,0]).reshape((2,1))
        
        self.assertEqual(vector_to_bearing(up), 0)
        self.assertEqual(vector_to_bearing(right), 90)
        self.assertEqual(vector_to_bearing(left), 270)
        
    def test_normalise(self):
        up = np.asarray([0,1]).reshape((2,1))
        zero = np.asarray([0,0]).reshape((2,1))
        
        numpy.testing.assert_array_almost_equal(normalize(up), up)
        numpy.testing.assert_array_almost_equal(normalize(up * 300), up)
        numpy.testing.assert_array_almost_equal(normalize(up / 300), up)
        numpy.testing.assert_array_almost_equal(normalize(zero), zero)
        
    def test_get_distance(self):
        up = np.asarray([0,1]).reshape((2,1))
        left = np.asarray([-1,0]).reshape((2,1))
        right = np.asarray([1,0]).reshape((2,1))
        
        self.assertEqual(get_distance(left, right), 2)
        self.assertEqual(get_distance(right, left * 20), 21)
        self.assertEqual(get_distance(-left, right), 0)
        self.assertEqual(get_distance(up, left), np.sqrt(2))

if __name__ == '__main__':
    unittest.main()