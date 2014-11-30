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

    def test_second_derivative_multidimensional(self):
        np.random.seed(123)
        w = np.random.rand(4, 3)
        f = lambda t: spline.bezier(w, t)

        g2 = [numdifftools.Derivative((lambda t: f(t)[i]), n=2)(.9)
              for i in range(len(w[0]))]

        g2 = np.squeeze(g2)
        np.testing.assert_array_almost_equal(g2,
                                             spline.bezier_second_deriv(w, .9))

    def test_zero_offset_second_derivative(self):
        w = np.array([1.7, 2.8, 1.4, -3.6])
        f = lambda t: spline.zero_offset_bezier(w, t)
        g2 = numdifftools.Hessdiag(f)
        np.testing.assert_array_almost_equal(g2(.9),
                                             spline.zero_offset_bezier_second_deriv(w, .9))

    def test_zero_offset_bezier_mat(self):
        bezier_order = 4
        t = .5
        f = lambda x: spline.zero_offset_bezier(x.reshape((bezier_order, 3)), t)

        J_numeric = numdifftools.Jacobian(f)(np.zeros(bezier_order*3))
        J_analytic = spline.zero_offset_bezier_mat(t, bezier_order, 3)

        np.testing.assert_array_almost_equal(J_numeric, J_analytic)

    def test_zero_offset_bezier_second_deriv_mat(self):
        bezier_order = 4
        t = .5
        f = lambda x: spline.zero_offset_bezier_second_deriv(x.reshape((bezier_order, 3)), t)

        J_numeric = numdifftools.Jacobian(f)(np.zeros(bezier_order*3))
        J_analytic = spline.zero_offset_bezier_second_deriv_mat(t, bezier_order, 3)

        np.testing.assert_array_almost_equal(J_numeric, J_analytic)
