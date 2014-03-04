__author__ = 'alexflint'

import unittest

from modulo import *

class ModuloTest(unittest.TestCase):
    def test_comparisons(self):
        a = ModuloInteger(3, 11)
        self.assertEqual(a, 3)
        self.assertEqual(3, a)
        self.assertEqual(a, ModuloInteger(3,11))
        self.assertLess(a, 4)
        self.assertGreater(a, 0)

    def test_addition(self):
        a = ModuloInteger(4, 7)
        self.assertEqual(a + 2, 6)
        self.assertEqual(a + 4, 1)
        self.assertEqual(6 + a, 3)
        self.assertEqual(a + ModuloInteger(2,5), 6)
        self.assertEqual(a + ModuloInteger(4,5), 1)

    def test_subtraction(self):
        a = ModuloInteger(4, 7)
        self.assertEqual(a - 1, 3)
        self.assertEqual(a - 4, 0)
        self.assertEqual(a - ModuloInteger(1,5), 3)
        self.assertEqual(a - ModuloInteger(4,5), 0)

    def test_subtraction(self):
        a = ModuloInteger(4, 7)
        self.assertEqual(a - 1, 3)
        self.assertEqual(a - 4, 0)
        self.assertEqual(5 - a, 1)
        self.assertEqual(3 - a, 6)
        self.assertEqual(a - ModuloInteger(1,5), 3)
        self.assertEqual(a - ModuloInteger(4,5), 0)

    def test_multiplication(self):
        a = ModuloInteger(2,5)
        self.assertEqual(a*2, 4)
        self.assertEqual(a*3, 1)
        self.assertEqual(a*0, 0)
        self.assertEqual(2*a, 4)
        self.assertEqual(3*a, 1)
        self.assertEqual(0*a, 0)

    def test_inverse(self):
        self.assertEqual(multiplicative_inverse(2, 5), 3)
        self.assertEqual(multiplicative_inverse(9, 17), 2)
        self.assertEqual(multiplicative_inverse(5, 11), 9)
        self.assertEqual(ModuloInteger(2,5).inverse, 3)
        self.assertEqual(ModuloInteger(9,17).inverse, 2)
        self.assertEqual(ModuloInteger(5,11).inverse, 9)

    def test_division(self):
        a = ModuloInteger(2,5)
        self.assertEqual(a / 1, a)
        self.assertEqual(a / 1, 2)
        self.assertEqual(1 / a, 3)
        self.assertEqual(4 / a, 2)
        self.assertEqual(3 / a, 4)
        self.assertEqual(a / 2, 1)
        self.assertEqual(2 / a, 1)


if __name__ == '__main__':
    unittest.main()
