__author__ = 'alexflint'

import unittest

from polynomial import parse
from ring import *

class RingTest(unittest.TestCase):
    def test_gcd(self):
        f,g,h = parse('(x**2 + x) * 3',
                      '(x**2 + x)**2 * (x + 2)',
                      '(x**2 + x)')
        self.assertEqual(gcd(f,g).normalized(), h)


if __name__ == '__main__':
    unittest.main()
