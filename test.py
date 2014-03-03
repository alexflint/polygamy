from __future__ import division

import fractions
import unittest

from polysolve import *
from modulo import *
from ring import *

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

    def test_getitem(self):
        f = parse('2*x + 11*x*y**2 - 1')
        self.assertEqual(f[1,0], 2)
        self.assertEqual(f[1,2], 11)
        self.assertEqual(f[0,0], -1)
        self.assertEqual(f[1,3], 0)
        self.assertEqual(f[0,1], 0)

    def test_getitem(self):
        f = parse('2*x + 11*x*y**2 - 1')
        f[1,2] = 12
        self.assertEqual(f[1,2], 12)
        f[0,0] = 2
        self.assertEqual(f[0,0], 2)
        f[1,3] -= 4
        self.assertEqual(f[1,3], -4)
        f[0,0] -= 2
        self.assertTrue((0,0) not in f)
        self.assertEqual(f[0,0], 0)

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
        assert f.squeeze() == g

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

    def test_newton_raphson(self):
        f = parse('(x-1)*(x-2)*(x-3)*(x-4)')
        self.assertAlmostEqual(polish_univariate_root(f, 0.0), 1.0, places=8)
        self.assertAlmostEqual(polish_univariate_root(f, 1.0), 1.0, places=8)
        self.assertAlmostEqual(polish_univariate_root(f, 1.1), 1.0, places=8)
        self.assertAlmostEqual(polish_univariate_root(f, 1.9), 2.0, places=8)
        self.assertAlmostEqual(polish_univariate_root(f, 3.2), 3.0, places=8)
        self.assertAlmostEqual(polish_univariate_root(f, 6.5), 4.0, places=8)

    def _test_evaluation_speed(self):
        import timeit
        f = parse('(x-1)*(x-2)*(x-3)*(x-4)+x*x**2*x**3*x**4*x**5')
        ff = f.compile()
        print 'Time to evaluate uncompiled: ',timeit.timeit(lambda:f(10), number=10000)/10000
        print 'Time to evaluate compiled: ',timeit.timeit(lambda:ff(10), number=10000)/10000
        print 'Time to compile: ',timeit.timeit(lambda:f.compile(), number=10000)/10000

    def test_sturm(self):
        p = parse('(x-0.5)*(x-1.5)*(x-2.5)*(x-3.5)', ctype=float)
        s = SturmChain(p)
        self.assertEqual(s.count_roots(), 4)
        self.assertEqual(s.count_roots_between(0, 10), 4)
        self.assertEqual(s.count_roots_between(0, 0), 0)
        self.assertEqual(s.count_roots_between(0, 3), 3)
        self.assertEqual(s.count_roots_between(0, 4), 4)
        self.assertEqual(s.count_roots_between(1, 4), 3)
        self.assertEqual(s.count_roots_between(2, 2), 0)
        self.assertEqual(s.count_roots_between(2, 4), 2)
        self.assertEqual(s.count_roots_between(3, 3), 0)
        self.assertEqual(s.count_roots_between(3, 4), 1)
        self.assertEqual(s.count_roots_between(4, 4), 0)
        self.assertEqual(s.count_roots_between(.5-1e-8, .5+1e-8), 1)
        self.assertEqual(s.count_roots_between(-1e+15, 1e+15), 4)

    def test_sturm_with_multiple_roots(self):
        s = SturmChain(parse('(x-1)**2').astype(float))
        self.assertEqual(s.count_roots(), 1)
        self.assertEqual(s.count_roots_between(0, 10), 1)

        s = SturmChain(parse('(x-1)**3').astype(float))
        self.assertEqual(s.count_roots(), 1)
        self.assertEqual(s.count_roots_between(0, 10), 1)

        s = SturmChain(parse('(x-1)**3 * (x-2)').astype(float))
        self.assertEqual(s.count_roots(), 2)
        self.assertEqual(s.count_roots_between(0, 10), 2)

        s = SturmChain(parse('(x-1)**3 * (x-2)**5').astype(float))
        self.assertEqual(s.count_roots(), 2)
        self.assertEqual(s.count_roots_between(0, 10), 2)

    def validate_brackets(self, brackets, zeros):
        self.assertEqual(len(brackets), len(zeros)+1)
        for i,z in enumerate(zeros):
            self.assertLess(brackets[i], z)
            self.assertGreater(brackets[i+1], z)

    def test_isolate_univariate_roots(self):
        # TODO: figure out why this doesn't work in rational arithmetic
        f = parse('(x-1)*(x-2)*(x-3)')
        brackets = isolate_univariate_roots(f.astype(float))
        self.validate_brackets(brackets, (1,2,3))

    def test_isolate_univariate_roots2(self):
        # TODO: figure out why this doesn't work in rational arithmetic
        f = parse('(x-1)**2*(x-2)*(x-3)')
        brackets = isolate_univariate_roots(f.astype(float))
        self.validate_brackets(brackets, (1,2,3))

    # TODO: figure out why this test fails and fix it (it's to do with coefficient typing)
    def _test_isolate_univariate_roots3(self):
        f = parse('(x-1)*(x-1.000001)*(x-1e+8)')
        brackets = isolate_univariate_roots(f.astype(fractions.Fraction))
        self.validate_brackets(brackets, (1, 1.000001, 1e+8))

    def test_bisect_bracket(self):
        f = parse('(x-1)*(x-2)*(x-3)')
        bracket = bisect_bracket(f, -1.2, 1.8, 1e-5)
        self.validate_brackets(bracket, [1])

    def assert_all_near(self, xs, ys, places=8):
        self.assertEqual(len(xs), len(ys))
        for x,y in zip(xs,ys):
            self.assertAlmostEqual(x,y,places=places)

    def test_solve_univariate_via_sturm(self):
        # TODO: figure out why this doesn't work in rational arithmetic
        f = parse('(x-1)*(x-2)*(x-3)')
        roots,brackets = solve_univariate_via_sturm(f.astype(float), tol=1e-8)
        self.assert_all_near(roots, (1,2,3), 8)

        f = parse('(x-10)*(x-200)*(x-3000)**3')
        roots,brackets = solve_univariate_via_sturm(f.astype(float), tol=1e-8)
        self.assert_all_near(roots, (10,200,3000), 2)

    def test_solve_univariate_via_companion(self):
        # TODO: figure out why this doesn't work in rational arithmetic
        f = parse('(x-1)*(x-2)*(x-3)')
        roots = solve_univariate_via_companion(f.astype(float))
        self.assert_all_near(roots, (1,2,3), 8)

        f = parse('(x-10)*(x-200)*(x-3000)**3')
        roots = solve_univariate_via_companion(f.astype(float))
        self.assert_all_near(roots, (10,200,3000), 1)

    def test_polish_multivariate_root(self):
        # the following system has a root at x=1, y=2
        fs = parse('(x + y - 3) * (x*y**2 - 1)',
                   '(x - 1) * (x - 10)',
                   '(y - 2)**2')
        v = polish_multivariate_root(fs, (1.1, 2.2), method='lm')

    def test_double_root(self):
        f = parse('-63*x**2 + -2700 + 1*x**3 + 1080*x**1')
        #print isolate_univariate_roots(f)

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


class RingTest(unittest.TestCase):
    def test_gcd(self):
        f,g,h = parse('(x**2 + x) * 3',
                      '(x**2 + x)**2 * (x + 2)',
                      '(x**2 + x)')
        self.assertEqual(ring.gcd(f,g).normalized(), h)


if __name__ == '__main__':
    unittest.main()
        
