import unittest
import numpy.testing
from simulation.sensors.acceleration_sensor import AccelerationSensor
from simulation.sensors.bearing_sensor import BearingSensor
from simulation.sensors.heading_sensor import HeadingSensor
from simulation.sensors.range_sensor import RangeSensor
from simulation.sensors.velocity_sensor import VelocitySensor
import numpy as np
from numpy.random import default_rng

class TestSimulator(unittest.TestCase):
    
    def test_acceleration_sensor(self):        
        s = AccelerationSensor(default_rng(0), 0, 0)        
        numpy.testing.assert_almost_equal(s.get_reading([[0],[10]]), [[0],[10]])

    def test_bearing_sensor(self):        
        s = BearingSensor(default_rng(0), 0, 0)        
        self.assertEqual(s.get_reading(np.asarray([[0],[0]]),
                                       np.asarray([[0],[-10]])), 
                         180)

    def test_heading_sensor(self):        
        s = HeadingSensor(default_rng(0), 0, 0)     
        self.assertEqual(s.get_reading(np.asarray([[0],[100]])), 0)   
        self.assertEqual(s.get_reading(np.asarray([[1],[0]])), 90)
        
    def test_range_sensor(self):        
        s = RangeSensor(default_rng(0), 0, 0)     
        self.assertEqual(s.get_reading(np.asarray([[0],[0]]),
                                       np.asarray([[3],[4]])), 
                         5)
        
    def test_velocity_sensor(self):        
        s = VelocitySensor(default_rng(0), 0, 0)        
        numpy.testing.assert_almost_equal(s.get_reading(10, 180), [[0],[-10]])
        
    

    
if __name__ == '__main__':
    unittest.main()