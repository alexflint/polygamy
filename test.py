import unittest

from polysolve import *

class PolynomialTest(unittest.TestCase):
    def test_add(self):
        p = Polynomial.create( [ Term(10,(0,0,0)), Term(3,(1,0,0)), Term(1,(0,2,0)), Term(1,(1,1,2)) ] )
        q = Polynomial.create( [ Term(3,(1,0,0)), Term(1,(0,2,0)), Term(1,(1,1,2)) ] )
        print p + q
        print p - q
        print p + 100

    def test_mul(self):
        p = Polynomial.create( [ Term(2,(1,0)), Term(5,(2,0)) ] )
        q = Polynomial.create( [ Term(1,(0,1)), Term(10,(0,2)) ] )
        print p
        print p*10
        return
        print p*Term(1,(0,1))
        print p*Term(1,(1,1))
        print p*q
        print p*p

    def test_remainder(self):
        h1,h2,f = parse('x**2 + z**2 - 1',
                        'x**2 + y**2 + (z-1)**2 - 4',
                        'x**2 + y**2*z/2 - z - 1')

        print remainder(f, [h1,h2], LexOrdering())

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
        f = parse('(x-0.5)*(x-1.5)*(x-2.5)*(x-3.5)')
        s = SturmChain(f)

        self.assertEqual(s.count_roots(), 4)
        self.assertEqual(s.count_roots_between(0, 10), 4)

        self.assertEqual(s.count_roots_between(0, 0), 0)
        self.assertEqual(s.count_roots_between(0, 1), 1)
        self.assertEqual(s.count_roots_between(0, 2), 2)
        self.assertEqual(s.count_roots_between(0, 3), 3)
        self.assertEqual(s.count_roots_between(0, 4), 4)

        self.assertEqual(s.count_roots_between(1, 1), 0)
        self.assertEqual(s.count_roots_between(1, 2), 1)
        self.assertEqual(s.count_roots_between(1, 3), 2)
        self.assertEqual(s.count_roots_between(1, 4), 3)

        self.assertEqual(s.count_roots_between(2, 2), 0)
        self.assertEqual(s.count_roots_between(2, 3), 1)
        self.assertEqual(s.count_roots_between(2, 4), 2)

        self.assertEqual(s.count_roots_between(3, 3), 0)
        self.assertEqual(s.count_roots_between(3, 4), 1)
        self.assertEqual(s.count_roots_between(4, 4), 0)

        self.assertEqual(s.count_roots_between(.5-1e-8, .5+1e-8), 1)
        self.assertEqual(s.count_roots_between(-1e+15, 1e+15), 4)

    def test_sturm_with_multiple_roots(self):
        s = SturmChain(parse('(x-1)**2'))
        self.assertEqual(s.count_roots(), 1)
        self.assertEqual(s.count_roots_between(0, 10), 1)

        s = SturmChain(parse('(x-1)**3'))
        self.assertEqual(s.count_roots(), 1)
        self.assertEqual(s.count_roots_between(0, 10), 1)

        s = SturmChain(parse('(x-1)**3 * (x-2)'))
        self.assertEqual(s.count_roots(), 2)
        self.assertEqual(s.count_roots_between(0, 10), 2)

        s = SturmChain(parse('(x-1)**3 * (x-2)**5'))
        self.assertEqual(s.count_roots(), 2)
        self.assertEqual(s.count_roots_between(0, 10), 2)

    def validate_brackets(self, brackets, zeros):
        self.assertEqual(len(brackets), len(zeros)+1)
        for i,z in enumerate(zeros):
            self.assertLess(brackets[i], z)
            self.assertGreater(brackets[i+1], z)

    def test_bracket_univariate_roots(self):
        f = parse('(x-1)*(x-2)*(x-3)')
        brackets = bracket_univariate_roots(f)
        self.validate_brackets(brackets, (1,2,3))

    def test_bracket_univariate_roots2(self):
        f = parse('(x-1)**2*(x-2)*(x-3)')
        brackets = bracket_univariate_roots(f)
        self.validate_brackets(brackets, (1,2,3))

    def test_bracket_univariate_roots3(self):
        f = parse('(x-1)*(x-1.000001)*(x-1e+8)')
        brackets = bracket_univariate_roots(f.rationalize())
        self.validate_brackets(brackets, (1, 1.000001, 1e+8))

    def test_bisect_bracket(self):
        f = parse('(x-1)*(x-2)*(x-3)')
        bracket = bisect_bracket(f, -1.2, 1.8, 1e-5)
        self.validate_brackets(bracket, [1])


if __name__ == '__main__':
    unittest.main()
        
