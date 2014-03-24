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


class ModuloTypeTest(unittest.TestCase):
    def test_construct(self):
        Z37 = ModuloInteger[37]
        x = Z37(5)
        self.assertEqual(x.r, 5)
        self.assertEqual(x.n, 37)
        self.assertIsInstance(x, Z37)

    def test_equality(self):
        Z37 = ModuloInteger[37]
        also_Z37 = ModuloInteger[37]
        Z39 = ModuloInteger[39]
        self.assertEqual(Z37, also_Z37)
        self.assertNotEqual(Z37, Z39)

    def test_equality_vs_base(self):
        self.assertNotEqual(ModuloInteger, ModuloInteger[37])
        self.assertNotEqual(ModuloInteger[37], ModuloInteger)
        self.assertEqual(ModuloInteger, ModuloInteger)

    def test_base_class(self):
        Z37 = ModuloInteger[37]
        x = Z37(1)
        self.assertIsInstance(x, Z37)
        self.assertIsInstance(x, ModuloInteger)

    def test_cross_isinstance(self):
        self.assertIsInstance(ModuloInteger(2, 37), ModuloInteger[37])

    def test_str(self):
        self.assertEqual(str(ModuloInteger[37]), 'ModuloInteger[37]')
        self.assertEqual(repr(ModuloInteger[37]), 'ModuloInteger[37]')

    def test_integral(self):
        self.assertIsInstance(ModuloInteger(1,4), numbers.Integral)
        self.assertTrue(issubclass(ModuloInteger, numbers.Integral))
        self.assertTrue(issubclass(ModuloInteger[5], numbers.Integral))


if __name__ == '__main__':
    unittest.main()
