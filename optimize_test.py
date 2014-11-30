import numpy as np
import unittest

from polynomial import Polynomial
import utils
import solvers
import optimize


class OptimizationTestCase(unittest.TestCase):
    def test_orientation_9params(self):
        np.random.seed(0)

        true_s = np.array([.1, .2, -.3])
        true_r = utils.cayley(true_s)
        true_vars = true_r.flatten()

        observed_xs = np.random.rand(8, 3)
        observed_ys = np.dot(observed_xs, true_r.T)

        sym_vars = Polynomial.coordinates(9, ctype=float)
        sym_r = np.reshape(sym_vars, (3, 3))

        residuals = (np.dot(observed_xs, sym_r.T) - observed_ys).flatten()
        constraints = (np.dot(sym_r.T, sym_r) - np.eye(3)).flatten()

        cost = sum(r**2 for r in residuals)
        gradients = cost.partial_derivatives()

        print 'Cost:', cost
        print 'Constraints:'
        for constraint in constraints:
            print '  ', constraint

        print 'At ground truth:'
        print '  Cost = ', cost(*true_vars)
        print '  Constraints = ', utils.evaluate_array(constraints, *true_vars)
        print '  Gradients = ', [p(*true_vars) for p in gradients]
        expansions = [solvers.all_monomials(sym_vars, 2) for _ in range(cost.num_vars)]
        minima = optimize.minimize_globally(cost,
                                            constraints,
                                            expansions=expansions,
                                            #diagnostic_solutions=[true_vars],
                                            )

        estimated_r = np.reshape(minima, (3, 3))
        error = np.linalg.norm(estimated_r - true_r)

        print 'Minima:\n', estimated_r
        print 'Ground truth:\n', true_r
        print 'Error:', error

    def test_orientation_reprojection(self):
        true_s = np.array([.1, .2, -.3])
        true_r = utils.cayley(true_s)
        true_k = 1. / (1. + np.dot(true_s, true_s))
        true_vars = np.r_[true_s, true_k]

        xs = np.random.rand(8, 3)
        true_ys = np.dot(xs, true_r.T)

        sym_vars = Polynomial.coordinates(4, ctype=float)
        x, y, z, w = sym_vars
        sym_s = sym_vars[:3]
        sym_k = sym_vars[3]
        sym_r = utils.cayley_mat(sym_s)
        sym_rd = utils.cayley_denom(sym_s)

        residuals = (np.dot(xs, sym_r.T) - true_ys * sym_rd).flatten()
        cost = sum(r**2 for r in residuals)
        gradients = cost.partial_derivatives()
        constraint = sym_k * (1 + np.dot(sym_s, sym_s)) - 1

        print 'Cost:', cost
        print 'Constraint:', constraint
        print 'At ground truth:'
        print '  Cost = ', cost(*true_vars)
        print '  Constraint = ', constraint(*true_vars)
        print '  Gradients = ', [p(*true_vars) for p in gradients]
        expansions = [solvers.all_monomials(sym_vars, 2) for _ in range(cost.num_vars)]
        for a in expansions:
            a.extend([z*z*w, x*x*w, y*y*w, z*z*w*w, z*w*w])
        minima = optimize.minimize_globally(cost,
                                            [constraint],
                                            expansions=expansions,
                                            diagnostic_solutions=[true_vars])
        print 'Minima: ', minima

    def test_orientation_angular(self):
        true_s = np.array([.1, .2, -.3])
        true_r = utils.cayley(true_s)
        true_k = 1. / (1. + np.dot(true_s, true_s))
        true_vars = np.r_[true_s, true_k]

        observed_xs = np.random.randn(10, 3)
        observed_ys = np.dot(observed_xs, true_r.T)

        sym_vars = Polynomial.coordinates(4, ctype=float)
        x, y, z, w = sym_vars
        sym_s = sym_vars[:3]
        sym_k = sym_vars[3]
        sym_r = utils.cayley_mat(sym_s)

        residuals = [np.dot(x, np.dot(sym_r, y)) for x, y in zip(observed_xs, observed_ys)]
        objective = sum(r**2 for r in residuals)
        gradients = objective.partial_derivatives()
        constraint = sym_k * (1 + np.dot(sym_s, sym_s)) - 1

        print 'Cost:', objective
        print 'Constraint:', constraint
        print 'At ground truth:'
        print '  Objective = ', objective(*true_vars)
        print '  Constraint = ', constraint(*true_vars)
        print '  Gradients = ', [p(*true_vars) for p in gradients]
        expansions = [solvers.all_monomials(sym_vars, 2) for _ in range(objective.num_vars)]
        for a in expansions:
            a.extend([z*z*w, x*x*w, y*y*w, z*z*w*w])
        minima = optimize.minimize_globally(-objective,
                                            [constraint],
                                            expansions=expansions,
                                            diagnostic_solutions=[true_vars])
        print 'Minima: ', minima

