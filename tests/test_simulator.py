import unittest
import numpy.testing
from flock_settings import FlockSettings, Setting
from simulation.flock import Flock
from simulator import Simulator
import numpy as np

class TestSimulator(unittest.TestCase):

    def test_flock_creation(self):
        settings = FlockSettings()
        
        sim = Simulator(settings)
        
        self.assertEqual(sim.get_flock().get_size(), settings.get(Setting.FLOCK_SIZE))
        
    def test_simulator_updates(self):
        sim = Simulator(FlockSettings())
        
        for i in range(0,10):
            sim.update(False)
        
        self.assertEqual(sim.get_step(), 10)
        
    def test_metrics_log(self):
        sim = Simulator(FlockSettings())
        
        for i in range(0,10):
            sim.update(True)
        
        self.assertEqual(sim.get_metrics_log().shape[0], 10)
        

if __name__ == '__main__':
    unittest.main()