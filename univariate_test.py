__author__ = 'alexflint'

import unittest

from polynomial import parse
from univariate import *

class UnivariateTest(unittest.TestCase):
    def test_newton_raphson(self):
        f = parse('(x-1)*(x-2)*(x-3)*(x-4)')
        self.assertAlmostEqual(polish_root(f, 0.0), 1.0, places=8)
        self.assertAlmostEqual(polish_root(f, 1.0), 1.0, places=8)
        self.assertAlmostEqual(polish_root(f, 1.1), 1.0, places=8)
        self.assertAlmostEqual(polish_root(f, 1.9), 2.0, places=8)
        self.assertAlmostEqual(polish_root(f, 3.2), 3.0, places=8)
        self.assertAlmostEqual(polish_root(f, 6.5), 4.0, places=8)

    def _test_evaluation_speed(self):
        import timeit
        f = parse('(x-1)*(x-2)*(x-3)*(x-4)+x*x**2*x**3*x**4*x**5')
        ff = f.compile()
        print('Time to evaluate uncompiled: ',timeit.timeit(lambda:f(10), number=10000)/10000)
        print('Time to evaluate compiled: ',timeit.timeit(lambda:ff(10), number=10000)/10000)
        print('Time to compile: ',timeit.timeit(lambda:f.compile(), number=10000)/10000)

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

    def test_isolate_roots(self):
        # TODO: figure out why this doesn't work in rational arithmetic
        f = parse('(x-1)*(x-2)*(x-3)')
        brackets = isolate_roots(f.astype(float))
        self.validate_brackets(brackets, (1,2,3))

    def test_isolate_roots2(self):
        # TODO: figure out why this doesn't work in rational arithmetic
        f = parse('(x-1)**2*(x-2)*(x-3)')
        brackets = isolate_roots(f.astype(float))
        self.validate_brackets(brackets, (1,2,3))

    # TODO: figure out why this test fails and fix it (it's to do with coefficient typing)
    def _test_isolate_roots3(self):
        f = parse('(x-1)*(x-1.000001)*(x-1e+8)')
        brackets = isolate_roots(f.astype(fractions.Fraction))
        self.validate_brackets(brackets, (1, 1.000001, 1e+8))

    def test_bisect_bracket(self):
        f = parse('(x-1)*(x-2)*(x-3)')
        bracket = bisect_bracket(f, -1.2, 1.8, 1e-5)
        self.validate_brackets(bracket, [1])

    def assert_all_near(self, xs, ys, places=8):
        self.assertEqual(len(xs), len(ys))
        for x,y in zip(xs,ys):
            self.assertAlmostEqual(x,y,places=places)

    def test_solve_via_sturm(self):
        # TODO: figure out why this doesn't work in rational arithmetic
        f = parse('(x-1)*(x-2)*(x-3)')
        roots,brackets = solve_via_sturm(f.astype(float), tol=1e-8)
        self.assert_all_near(roots, (1,2,3), 8)

        f = parse('(x-10)*(x-200)*(x-3000)**3')
        roots,brackets = solve_via_sturm(f.astype(float), tol=1e-8)
        self.assert_all_near(roots, (10,200,3000), 2)

    def test_solve_via_companion(self):
        # TODO: figure out why this doesn't work in rational arithmetic
        f = parse('(x-1)*(x-2)*(x-3)')
        roots = solve_via_companion(f.astype(float))
        self.assert_all_near(roots, (1,2,3), 8)

        f = parse('(x-10)*(x-200)*(x-3000)**3')
        roots = solve_via_companion(f.astype(float))
        self.assert_all_near(roots, (10,200,3000), 1)


if __name__ == '__main__':
    unittest.main()
