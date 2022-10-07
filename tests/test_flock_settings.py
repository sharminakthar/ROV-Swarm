from typing import Set
import unittest
import numpy.testing
from flock_settings import FlockSettings, Setting
import numpy as np

class TestSimulator(unittest.TestCase):

    def test_settings_creation(self):
        settings = FlockSettings()    
        self.assertEqual(settings.get(Setting.FLOCK_SIZE), 
                         settings.get_default(Setting.FLOCK_SIZE)) 
        
    def test_settings_default_change(self):
        settings = FlockSettings()            
        settings.set_as_default(Setting.SEPARATION_DISTANCE, 123456)        
        self.assertEqual(settings.get_default(Setting.SEPARATION_DISTANCE), 123456) 
        
    def test_settings_set(self):
        settings = FlockSettings()            
        settings.set(Setting.BANDWIDTH, 12)        
        self.assertEqual(settings.get(Setting.BANDWIDTH), 12)
        
    def test_settings_caching(self):
        settings = FlockSettings()            
        settings.set(Setting.MESSAGE_SIZE, 1000)
        settings.cache_active_settings()        
        self.assertEqual(settings.get_cached(Setting.MESSAGE_SIZE), 1000)

    def test_settings_reset(self):
        settings = FlockSettings()           
         
        settings.set_as_default(Setting.PACKET_LOSS, 42)        
        self.assertEqual(settings.get(Setting.PACKET_LOSS), 42)
        
        settings.set(Setting.PACKET_LOSS, 43)        
        self.assertEqual(settings.get(Setting.PACKET_LOSS), 43)
        
        settings.reset(Setting.PACKET_LOSS)
        self.assertEqual(settings.get(Setting.PACKET_LOSS), 42)
        
if __name__ == '__main__':
    unittest.main()