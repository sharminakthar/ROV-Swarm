import unittest
import numpy.testing
from flock_settings import FlockSettings
from simulation.drone_controller import DroneController
import numpy as np

from simulation.message import Message

class TestSimulator(unittest.TestCase):

    def test_drone_controller_creation(self):
        controller = DroneController(0, FlockSettings(), 
                                     np.asarray([[0.0],[0.0]]), 0, 
                                     np.asarray([[0.0],[0.0]]))           
        self.assertEqual(controller.my_id, 0)
        
    def test_drone_controller_velocity_reading(self):
        controller = DroneController(0, FlockSettings(), 
                                     np.asarray([[0.0],[0.0]]), 0, 
                                     np.asarray([[0.0],[0.0]]))     
        
        controller.update_readings(np.asarray([[0],[10]]), 0, np.asarray([[0],[0]]), 1)
        controller.update(1)
            
        numpy.testing.assert_array_equal(controller.last_velocity_reading, [[0],[10]])
        
    def test_drone_controller_dead_reckoning(self):
        controller = DroneController(0, FlockSettings(), 
                                     np.asarray([[0.0],[0.0]]), 0, 
                                     np.asarray([[0.0],[0.0]]))           
        for i in range(0, 10):
            controller.update_readings(np.asarray([[0.0],[10.0]]), 0, np.asarray([[0.0],[0.0]]), 1)
            controller.update(1)
            
        numpy.testing.assert_array_equal(controller.get_approximated_position(), [[0],[100]])

    def test_drone_position_approximation(self):
        controller = DroneController(0, FlockSettings(), 
                                     np.asarray([[0.0],[0.0]]), 0, 
                                     np.asarray([[0.0],[0.0]]))     
        
        controller.receive_message(1, 10, 0, Message(1, 0, 0))
        
        numpy.testing.assert_almost_equal(controller.flock_info[1].get_position(), [[0],[10]])

if __name__ == '__main__':
    unittest.main()