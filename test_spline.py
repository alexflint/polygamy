import unittest
import numdifftools

import numpy as np

import spline

class BezierTest(unittest.TestCase):
    def test_derivative(self):
        w = np.array([1.7, 2.8, 1.4, -3.6])
        f = lambda t: spline.bezier(w, t)
        g = numdifftools.Gradient(f)
        np.testing.assert_array_almost_equal(g(.9),
                                             spline.bezier_deriv(w, .9))

    def test_derivative_multidimensional(self):
        np.random.seed(123)
        w = np.random.rand(4, 3)
        f = lambda t: spline.bezier(w, t)
        g = numdifftools.Jacobian(f)
        np.testing.assert_array_almost_equal(np.squeeze(g(.9)),
                                             spline.bezier_deriv(w, .9))

    def test_second_derivative(self):
        w = np.array([1.7, 2.8, 1.4, -3.6])
        f = lambda t: spline.bezier(w, t)
        g2 = numdifftools.Hessdiag(f)
        np.testing.assert_array_almost_equal(g2(.9),
                                             spline.bezier_second_deriv(w, .9))

    def test_derivative_multidimensional(self):
        np.random.seed(123)
        w = np.random.rand(4, 3)
        f = lambda t: spline.bezier(w, t)

        g2 = [numdifftools.Derivative((lambda t: f(t)[i]), n=2)(.9)
              for i in range(len(w[0]))]

        g2 = np.squeeze(g2)
        np.testing.assert_array_almost_equal(g2,
                                             spline.bezier_second_deriv(w, .9))

