from __future__ import division

import unittest

from polynomial import *
from modulo import ModuloInteger

def first(xs):
    return iter(xs).next()

class TermTest(unittest.TestCase):
    pass

class PolynomialTest(unittest.TestCase):
    def test_constructor(self):
        f = Polynomial(3, float)
        self.assertEqual(f.num_vars, 3)
        self.assertEqual(f.ctype, float)

    def test_constructor2(self):
        f = Polynomial(4)
        self.assertEqual(f.ctype, fractions.Fraction)

    def test_create(self):
        f = Polynomial.create([], 5, int)
        self.assertEqual(len(f), 0)
        self.assertFalse(f)
        self.assertEqual(f.num_vars, 5)
        self.assertEqual(f.ctype, int)

    def test_create2(self):
        f = Polynomial.create([Term(7.1, (1,2,3), float)])
        self.assertEqual(len(f), 1)
        self.assertTrue(f)
        self.assertEqual(f.num_vars, 3)
        self.assertEqual(f.ctype, float)
        self.assertEqual(first(f).ctype, float)

    def test_create3(self):
        f = Polynomial.create([Term(7.1, (1,2,3), float)], ctype=int)
        self.assertEqual(len(f), 1)
        self.assertTrue(f)
        self.assertEqual(f.num_vars, 3)
        self.assertEqual(f.ctype, int)
        self.assertEqual(first(f).ctype, int)
        self.assertEqual(first(f).coef, 7)

    def test_astype(self):
        f = Polynomial.create([Term(7.1, (1,2,3), float)])
        self.assertEqual(f.ctype, float)
        g = f.astype(long)
        self.assertEqual(first(g).ctype, long)
        self.assertEqual(first(f).ctype, float)

    def test_astype_modulo(self):
        f = parse('2*x**2 + 7*x + 1').astype(ModuloInteger[37])
        g = parse('2*x - 1').astype(ModuloInteger[37])
        h1 = f ** 5
        self.assertEqual(h1.ctype, ModuloInteger[37])
        h2 = f // g
        self.assertEqual(h2.ctype, ModuloInteger[37])
        h3 = f % g
        self.assertEqual(h3.ctype, ModuloInteger[37])

    def test_type_upgrade1(self):
        f = parse('x').astype(int)
        g = parse('x').astype(float)
        self.assertEqual((f+g).ctype, float)
        self.assertEqual((f-g).ctype, float)
        self.assertEqual((f*g).ctype, float)

    def test_type_upgrade2(self):
        f = parse('x').astype(fractions.Fraction)
        g = parse('x').astype(fractions.Fraction)
        self.assertEqual((f+g).ctype, fractions.Fraction)
        self.assertEqual((f-g).ctype, fractions.Fraction)
        self.assertEqual((f*g).ctype, fractions.Fraction)

    def test_type_upgrade3(self):
        f = parse('x').astype(int)
        g = parse('x').astype(fractions.Fraction)
        self.assertEqual((f+g).ctype, fractions.Fraction)
        self.assertEqual((f-g).ctype, fractions.Fraction)
        self.assertEqual((f*g).ctype, fractions.Fraction)

    def test_type_upgrade4(self):
        f = parse('x').astype(fractions.Fraction)
        g = parse('x').astype(float)
        self.assertEqual((f+g).ctype, float)
        self.assertEqual((f-g).ctype, float)
        self.assertEqual((f*g).ctype, float)

    def test_getitem(self):
        f = parse('2*x + 11*x*y**2 - 1')
        self.assertEqual(f[1,0], 2)
        self.assertEqual(f[1,2], 11)
        self.assertEqual(f[0,0], -1)
        self.assertEqual(f[1,3], 0)
        self.assertEqual(f[0,1], 0)

    def test_getitem(self):
        f = parse('2*x + 11*x*y**2 - 1')
        f.coefficients[1,2] = 12
        self.assertEqual(f.coefficients[1,2], 12)
        f.coefficients[0,0] = 2
        self.assertEqual(f.coefficients[0,0], 2)
        f.coefficients[1,3] -= 4
        self.assertEqual(f.coefficients[1,3], -4)
        f.coefficients[0,0] -= 2
        self.assertTrue((0,0) not in f)
        self.assertEqual(f.coefficients[0,0], 0)

    def test_contains(self):
        f = parse('2*x + 11*x*y**2 - 1')
        self.assertTrue((1,0) in f)
        self.assertTrue((1,2) in f)
        self.assertTrue((0,0) in f)
        self.assertTrue((1,3) not in f)
        self.assertTrue((0,1) not in f)

    def test_normalized(self):
        f,g = parse('2*x**2 + 4*x + 8',
                    'x**2 + 2*x + 4')
        self.assertEqual(f.normalized(), g)
    
    def test_remainder(self):
        h1,h2,f,rem = parse('x**2 + z**2 - 1',
                            'x**2 + y**2 + (z-1)**2 - 4',
                            'x**2 + y**2*z/2 - z - 1',
                            'y**2*z/2 - z**2 - z')
        self.assertEqual(remainder(f, [h1,h2], LexOrdering()), rem)

    def test_derivative(self):
        f, J_f_wrt_x, J_f_wrt_y = parse('2*x + 3*x*y**2 + 8*y**6 + 6',
                                        '2   + 3*y**2',
                                        '      6*x*y    + 48*y**5')
        self.assertEqual(f.partial_derivative(0), J_f_wrt_x)
        self.assertEqual(f.partial_derivative(1), J_f_wrt_y)

    def test_squeeze(self):
        f = parse('2*x + 3*x**2 - 1', variable_order=('w','x','y'))
        g = parse('2*x + 3*x**2 - 1')
        assert f.squeezed() == g

    def test_evaluate(self):
        f = parse('2*x + 3*x**2*y + 6*y**5 - 1')
        self.assertEqual(f(2,10), 4+3*4*10+6*1e+5-1)
        self.assertEqual(f(-1,0), -3)
        self.assertEqual(f(0, 1.5), 44.5625)

    def test_evaluate_partial(self):
        f,g,h = parse('3*x**2*y + 2*x + y**5 - 1',
                      'y**5 + 12*y + 3',
                      '6*x**2 + 2*x + 31')
        self.assertEqual(f.evaluate_partial(0,2), g)
        self.assertEqual(f.evaluate_partial(1,2), h)
        self.assertEqual(f.evaluate_partial(0,0).evaluate_partial(1,0), -1)

    def test_evaluate_partial2(self):
        f,g,h = parse('3*x + y + 1',
                      'y + 7',
                      '3*x + 3')
        self.assertEqual(f.evaluate_partial(0,2), g)
        self.assertEqual(f.evaluate_partial(1,2), h)

    def test_compile(self):
        f = parse('2*x + 3*x**2*y + 6*y**5 - 1')
        ff = f.compile()
        self.assertEqual(ff(2,10), 4+3*4*10+6*1e+5-1)
        self.assertEqual(ff(-1,0), -3)
        self.assertEqual(ff(0, 1.5), 44.5625)

    def test_ndarray_of_polynomials(self):
        f, g, h = parse('x+y', '2*x**2', '4*x**4 + x**2 + 2*x*y + y**2')
        v = np.array([f, g])
        self.assertEqual(np.dot(v,v), h)


class IdealTest(unittest.TestCase):
    def test_ideal_from_variety(self):
        zeros = np.array([[-2, -1],
                          [3,  2],
                          [4,  5],
                          [6,  7]], dtype=fractions.Fraction)
        F = ideal_from_variety(zeros, fractions.Fraction)
        for f in F:
            self.assertNotEqual(f(10, 20), 0)
            self.assertNotEqual(f(0, 0), 0)
            for zero in zeros:
                self.assertEqual(f(*zero), 0)


class GrobnerBasisTest(unittest.TestCase):
    def test_equivalent_remainders(self):
        # This test checks that remainders computed with respect to Grobner
        # bases in different orders are equivalent.

        fs = parse('x**2*y + x + 3',
                   'y**2 * y*x + 2',
                   'x*y**2 + x*y + 1',
                   ctype=fractions.Fraction)

        p = fs[-1]
        F = fs[:-1]
        G_lex = gbasis(F, LexOrdering())

        rem1 = remainder(p, G_lex, LexOrdering())
        rem2 = remainder(p, G_lex[::-1], LexOrdering())
        rem3 = remainder(p, G_lex[::2] + G_lex[1::2], LexOrdering())

        self.assertEqual(rem1, rem2)
        self.assertEqual(rem1, rem3)

    def test_ideal_membership(self):
        # This test checks that remainders computed with respect to Grobner
        # bases with different monomial orderings all correctly identify
        # ideal members

        F = parse('x**2*y + x + 3',
                  'y**2 * x*y + 2',
                  'x*y**2 + x*y + 1',
                  ctype=fractions.Fraction)

        G_lex = gbasis(F, LexOrdering())
        G_grlex = gbasis(F, GrlexOrdering())
        G_grevlex = gbasis(F, GrevlexOrdering())

        x = Polynomial.coordinate(0, 2)
        y = Polynomial.coordinate(0, 2)

        p1 = F[0] + F[1] + F[2]
        p2 = F[0] - x*F[1]

        self.assertEqual(remainder(p1, G_lex, LexOrdering()), 0)
        self.assertEqual(remainder(p1, G_lex, LexOrdering()), 0)

        self.assertEqual(remainder(p1, G_grlex, GrlexOrdering()), 0)
        self.assertEqual(remainder(p1, G_grlex, GrlexOrdering()), 0)

        self.assertEqual(remainder(p1, G_grevlex, GrevlexOrdering()), 0)
        self.assertEqual(remainder(p1, G_grevlex, GrevlexOrdering()), 0)


if __name__ == '__main__':
    unittest.main()
