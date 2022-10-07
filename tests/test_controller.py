import unittest
import numpy.testing
from controller import Controller
import numpy as np

class TestSimulator(unittest.TestCase):

    def test_controller_creation(self):
        controller = Controller()            
        self.assertEqual(controller.simulator.get_step(), 0)
    
    def test_controller_updates(self):
        controller = Controller()    
        for i in range(0,10):    
            controller.update()        
        self.assertEqual(controller.simulator.get_step(), 10)
        
    def test_controller_speed(self):
        controller = Controller()   
        controller.sim_speed = 2 
        for i in range(0,10):    
            controller.update() 
        self.assertEqual(controller.simulator.get_step(), 20)
        

if __name__ == '__main__':
    unittest.main()