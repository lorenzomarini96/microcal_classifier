"""Unittest"""
import sys
import os
from pathlib import Path

import unittest
from microcal_classifier.module_template import func1

# Get the absolute path to the parent dir.
sys.path.insert(0, str(Path(os.getcwd()).parent))


class TestFunc1(unittest.TestCase):
    """Unittest for the module_template"""

    def test_int(self):
        """Test with integers"""
        self.assertAlmostEqual(func1(2, 2), 4)

    def test_float(self):
        """Test with floats"""
        self.assertAlmostEqual(func1(2.5, 2.5), 5.0)


if __name__ == "__main__":
    unittest.main()
