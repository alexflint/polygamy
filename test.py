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

    def test_divide(self):
        p = Polynomial.create( [ Term(2,(1,0)), Term(5,(2,0)) ] )
        q = Polynomial.create( [ Term(2,(1,0)), Term(8,(0,0)) ] )
        print p.divide_by(Term(2,(0,0)))
        print p.divide_by(q)

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

    def test_evaluation_speed(self):
        import timeit
        f = parse('(x-1)*(x-2)*(x-3)*(x-4)+x*x**2*x**3*x**4*x**5')
        ff = f.compile()
        print 'Time to evaluate uncompiled: ',timeit.timeit(lambda:f(10), number=10000)/10000
        print 'Time to evaluate compiled: ',timeit.timeit(lambda:ff(10), number=10000)/10000


if __name__ == '__main__':
    unittest.main()
        
