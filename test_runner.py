from json import load
import unittest

import tests.test_vector_helpers
import tests.test_physics_helpers
import tests.test_simulator
import tests.test_controller
import tests.test_flock
import tests.test_flock_settings
import tests.test_drone_controller
import tests.test_parser
import tests.test_sensors

loader = unittest.TestLoader()
suite  = unittest.TestSuite()

suite.addTests(loader.loadTestsFromModule(tests.test_vector_helpers))
suite.addTests(loader.loadTestsFromModule(tests.test_physics_helpers))
suite.addTests(loader.loadTestsFromModule(tests.test_simulator))
suite.addTests(loader.loadTestsFromModule(tests.test_controller))
suite.addTests(loader.loadTestsFromModule(tests.test_flock))
suite.addTests(loader.loadTestsFromModule(tests.test_flock_settings))
suite.addTests(loader.loadTestsFromModule(tests.test_drone_controller))
suite.addTests(loader.loadTestsFromModule(tests.test_parser))
suite.addTests(loader.loadTestsFromModule(tests.test_sensors))

runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)