import unittest
import numpy.testing
from flock_settings import FlockSettings, Setting
from simulation.flock import Flock
from simulator import Simulator
import numpy as np

class TestSimulator(unittest.TestCase):

    def test_flock_creation(self):
        settings = FlockSettings()
        
        settings.set(Setting.FLOCK_SIZE, 3)
        
        flock = Flock(settings, np.zeros((2, 3)), np.zeros((2, 3)))
        
        self.assertEqual(flock.get_size(), 3)
        
    def test_multiple_collisions(self):
        positions = np.zeros((2, 3))
        
        settings = FlockSettings()        
        settings.set(Setting.FLOCK_SIZE, 3)    
            
        flock = Flock(settings, positions, np.zeros((2, 3)))        
        for i in range(0,3):
            flock.update(i)                
        self.assertEqual(flock.get_num_collisions(), 3)        
        
    def test_single_collision(self):
        positions = np.zeros((2, 3))
        
        settings = FlockSettings()        
        settings.set(Setting.FLOCK_SIZE, 3) 
        positions[:,0] = [10000,10000]
        
        flock = Flock(settings, positions, np.zeros((2, 3)))        
        flock.update(0)                
        self.assertEqual(flock.get_num_collisions(), 1)
        
    def test_no_collisions(self):
        positions = np.zeros((2, 3))
        
        settings = FlockSettings()        
        settings.set(Setting.FLOCK_SIZE, 3) 
        positions[:,0] = [10000,10000]
        positions[:,1] = [-10000,-10000]
        
        flock = Flock(settings, positions, np.zeros((2, 3)))        
        flock.update(0)                
        self.assertEqual(flock.get_num_collisions(), 0)
        
    def test_flock_moves(self):
        positions = np.zeros((2, 3))        
        settings = FlockSettings()    
        settings.set(Setting.SEPARATION_DISTANCE, 50)    
        settings.set(Setting.FLOCK_SIZE, 3) 
        positions[:,0] = [100,100]
        positions[:,1] = [-100,-100]
        
        flock = Flock(settings, positions, np.zeros((2, 3)))
        
        for i in range(0,10):
            flock.update(i)   
            
        self.assertGreater(max(flock.get_speeds()), 0)
        
    def test_many_flock_groups(self):        
        positions = np.zeros((2, 5))        
        settings = FlockSettings()    
        settings.set(Setting.FLOCK_SIZE, 5) 
        positions[:,0] = [10000,10000]
        positions[:,1] = [10050,10000]
        positions[:,2] = [-10000,-10000]
        positions[:,3] = [-10050,-10000]        
        
        flock = Flock(settings, positions, np.zeros((2, 5)))
        flock.update(0)   
        self.assertEqual(flock.calculate_flock_groups(), 3)
        
    def test_single_flock_group(self):        
        positions = np.zeros((2, 3))        
        settings = FlockSettings()    
        settings.set(Setting.FLOCK_SIZE, 3) 
        positions[:,0] = [10,10]
        positions[:,1] = [-10,-10]
        
        flock = Flock(settings, positions, np.zeros((2, 3)))
        flock.update(0)   
        self.assertEqual(flock.calculate_flock_groups(), 1)
        
    def test_flock_center(self):
        positions = np.zeros((2, 3))        
        settings = FlockSettings()    
        settings.set(Setting.FLOCK_SIZE, 3) 
        positions[:,0] = [50,20]
        positions[:,1] = [100,10]
        
        flock = Flock(settings, positions, np.zeros((2, 3))) 
        numpy.testing.assert_array_equal(flock.get_flock_center(), [[50],[10]])
        
if __name__ == '__main__':
    unittest.main()