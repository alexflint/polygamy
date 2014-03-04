__author__ = 'Alex Flint'

import unittest

from polynomial import parse
from multivariate import *

class MultivariateTest(unittest.TestCase):
    def test_polish_multivariate_root(self):
        # the following system has a root at x=1, y=2
        fs = parse('(x + y - 3) * (x*y**2 - 1)',
                   '(x - 1) * (x - 10)',
                   '(y - 2)**2')
        v = polish_root(fs, (1.1, 2.2), method='lm')


if __name__ == '__main__':
    unittest.main()
