import unittest
import numpy.testing
from flock_settings import FlockSettings
from headless.parser import Parser
import numpy as np

class TestSimulator(unittest.TestCase):

    def test_parser_steps(self):
        s = FlockSettings()
        p = Parser(s)
        
        v = p.parser.parse_args(["--steps", "997"])
        
        self.assertEqual(vars(v)["STEPS"], 997)
        
    def test_parser_misc(self):
        s = FlockSettings()
        p = Parser(s)
        
        v = p.parser.parse_args(["--steps", "997", "--drone_objective", "NONE"])
        
        self.assertEqual(vars(v)["DRONE_OBJECTIVE"], "NONE")

if __name__ == '__main__':
    unittest.main()