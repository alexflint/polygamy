import numpy as np
import unittest

from polynomial import Polynomial, polynomial_jacobian
import utils
import solvers
import optimize


def make_langrangian(objective, constraints):
    ex_vars = Polynomial.coordinates(objective.num_vars + len(constraints))
    orig_vars = ex_vars[:objective.num_vars]
    lg_vars = ex_vars[objective.num_vars:]
    return objective(*orig_vars) + sum(lg * c(*orig_vars) for lg, c in zip(lg_vars, constraints))


class OptimizationTestCase(unittest.TestCase):
    def test_optimize_2d(self):
        np.random.seed(0)
        x, y = Polynomial.coordinates(2, float)
        fs = [
            x+1,
            y+1,
            x*x,
        ]
        true_solution = np.array([2., 5.])
        cost = sum((f(x, y) - f(*true_solution)) ** 2 for f in fs)

        import matplotlib.pyplot as plt
        dX, dY = np.meshgrid(np.linspace(-.1, .1, 50), np.linspace(-.1, .1, 50))
        X = true_solution[0] + dX
        Y = true_solution[1] + dY * 1j
        Z = np.abs(cost(X, Y))
        print(np.min(Z), np.max(Z))
        plt.contourf(dX, dY, Z, levels=np.logspace(np.log10(np.min(Z)), np.log10(np.max(Z)), 16))
        plt.plot(0, 0, 'mx')
        plt.show()
        return

        gradients = cost.partial_derivatives()
        print('Cost:', cost)
        print('Residuals:')
        for f in fs:
            print(f(x, y) - f(*true_solution))
        print('Gradients:')
        for gradient in gradients:
            print('  ', gradient(*true_solution))
        print('Jacobian:')
        print(polynomial_jacobian(fs)(*true_solution))
        expansions = 3  # solvers.all_monomials(sym_vars, 2) + [sym_vars[2]**3, sym_vars[0]**2*sym_vars[2]]
        minima = optimize.minimize_globally(cost, expansions=expansions, verbosity=2, constraints=fs[:2],
                                            #diagnostic_solutions=[true_vars]
        )
        np.testing.assert_array_almost_equal(minima, true_solution)

    def test_range_optimization_2d(self):
        np.random.seed(0)

        bases = np.round(np.random.randn(2, 3) * 6.)
        true_position = np.array([-2., 1., 3.])
        true_ranges = np.array([np.linalg.norm(base - true_position) for base in bases])
        true_vars = np.hstack((true_position, true_ranges))

        sym_vars = Polynomial.coordinates(3 + len(bases), float)
        sym_position = sym_vars[:3]
        sym_ranges = sym_vars[3:]

        cost = sum((zz - z)**2 for zz, z in zip(sym_ranges, true_ranges))
        constraints = [np.sum(np.square(base - sym_position)) - sym_range*sym_range
                       for base, sym_range in zip(bases, sym_ranges)]

        lagrangian = make_langrangian(-cost, constraints)
        gradients = lagrangian.partial_derivatives()

        print('Cost:', cost)
        print('Lagrangian:', lagrangian)
        print('Constraints:')
        for constraint in constraints:
            print('  ', constraint)
        print('Gradients:')
        for gradient in gradients:
            print('  ', gradient)

        print('At ground truth:')
        print('  Cost = ', cost(*true_vars))
        print('  Constraints = ', utils.evaluate_array(constraints, *true_vars))
        #print '  Gradients = ', [p(*true_vars) for p in gradients]

        expansions = 3  # solvers.all_monomials(sym_vars, 2) + [sym_vars[2]**3, sym_vars[0]**2*sym_vars[2]]
        minima = optimize.minimize_globally(-lagrangian, [], expansions=expansions, verbosity=2,
                                            diagnostic_solutions=[], include_grobner=False)

        error = np.linalg.norm(minima - true_position)
        print(minima)
        print(true_position)
        print('Error:', error)

    def test_estimate_orientation_9params(self):
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

        print('Cost:', cost)
        print('Constraints:')
        for constraint in constraints:
            print('  ', constraint)

        print('At ground truth:')
        print('  Cost = ', cost(*true_vars))
        print('  Constraints = ', utils.evaluate_array(constraints, *true_vars))
        print('  Gradients = ', [p(*true_vars) for p in gradients])
        expansions = [solvers.all_monomials(sym_vars, 2) for _ in range(cost.num_vars)]
        minima = optimize.minimize_globally(cost,
                                            constraints,
                                            expansions=expansions,
                                            #diagnostic_solutions=[true_vars],
                                            )

        estimated_r = np.reshape(minima, (3, 3))
        error = np.linalg.norm(estimated_r - true_r)

        print('Minima:\n', estimated_r)
        print('Ground truth:\n', true_r)
        print('Error:', error)

    def test_estimate_pose_and_depths(self):
        np.random.seed(0)
        num_landmarks = 5

        # Generate ground truth
        true_cayleys = np.array([.1, .2, -.3])
        true_orientation = utils.cayley(true_cayleys)
        true_position = np.array([2., 3., 10.])

        # Generate observed quantities
        true_landmarks = np.random.randn(num_landmarks, 3)
        true_pfeatures = np.dot(true_landmarks - true_position, true_orientation.T)
        true_depths = np.sqrt(np.sum(np.square(true_pfeatures), axis=1))
        true_features = true_pfeatures / true_depths[:, None]
        true_vars = np.hstack((true_orientation.flatten(), true_position, true_depths))

        # Create symbolic quantities
        sym_vars = Polynomial.coordinates(12 + num_landmarks, ctype=float)
        sym_orientation = np.reshape(sym_vars[:9], (3, 3))
        sym_position = np.array(sym_vars[9:12])
        sym_depths = np.array(sym_vars[12:])

        residuals = []
        for i, (landmark, feature, sym_depth) in enumerate(zip(true_landmarks, true_features, sym_depths)):
            residual = np.dot(sym_orientation, landmark - sym_position) - sym_depth * feature
            residuals.extend(residual)

        constraints = (np.dot(sym_orientation.T, sym_orientation) - np.eye(3)).flatten()

        cost = sum(r**2 for r in residuals)
        gradients = cost.partial_derivatives()

        print('Cost:', cost)
        print('Constraints:')
        for constraint in constraints:
            print('  ', constraint)

        print('At ground truth:')
        print('  Cost = ', cost(*true_vars))
        print('  Constraints = ', utils.evaluate_array(constraints, *true_vars))
        print('  Gradients = ', [p(*true_vars) for p in gradients])
        expansions = solvers.all_monomials(sym_vars, 2)
        minima = optimize.minimize_globally(cost,
                                            constraints,
                                            expansions=expansions,
                                            #diagnostic_solutions=[true_vars],
                                            )

        estimated_r = np.reshape(minima, (3, 3))
        error = np.linalg.norm(estimated_r - true_orientation)

        print('Minima:\n', estimated_r)
        print('Ground truth:\n', true_orientation)
        print('Error:', error)

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

        print('Cost:', cost)
        print('Constraint:', constraint)
        print('At ground truth:')
        print('  Cost = ', cost(*true_vars))
        print('  Constraint = ', constraint(*true_vars))
        print('  Gradients = ', [p(*true_vars) for p in gradients])
        expansions = [solvers.all_monomials(sym_vars, 2) for _ in range(cost.num_vars)]
        for a in expansions:
            a.extend([z*z*w, x*x*w, y*y*w, z*z*w*w, z*w*w])
        minima = optimize.minimize_globally(cost,
                                            [constraint],
                                            expansions=expansions,
                                            diagnostic_solutions=[true_vars])
        print('Minima: ', minima)

