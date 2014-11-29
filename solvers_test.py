import itertools
import unittest
import numpy as np

from polynomial import Polynomial, ideal_from_variety, product
import solvers


def all_monomials(variables, degree):
    return map(product, itertools.product(list(variables)+[1], repeat=degree))


class SolverTestCase(unittest.TestCase):
    def assert_solutions_found(self, result, expected_solutions, tol=1e-8):
        for x in expected_solutions:
            found = False
            for solution in result.solutions:
                if np.linalg.norm(solution - x) <= tol:
                    found = True
                    break
            if not found:
                self.fail('expected %s to be in solution set but was not found' % x)

    def test_two_vars(self):
        x, y = Polynomial.coordinates(2)
        equations = [x**2 + y**2 - 1,
                     x-y]
        expansion_monomials = [
            [x, y],
            [x, y, x*y, x*x, y*y]
        ]
        result = solvers.solve_via_basis_selection(equations, expansion_monomials, x+y+1)
        expected_solutions = [np.sqrt([.5, .5]), -np.sqrt([.5, .5])]
        self.assert_solutions_found(result, expected_solutions)

    def test_two_circles(self):
        x, y = Polynomial.coordinates(2)
        equations = [
            (x+2)**2 + (y+2)**2 - 25,
            (x-6)**2 + (y+2)**2 - 25,
        ]
        expansion_monomials = [
            [x, y, x*y, x*x, y*y, x*x*y],
            [x, y, x*y, x*x, y*y, x*x*y]
        ]
        expected_solutions = [(2, 1), (2, -5)]
        result = solvers.solve_via_basis_selection(equations,
                                                   expansion_monomials,
                                                   x+2*y-3,
                                                   solutions=expected_solutions)
        self.assert_solutions_found(result, expected_solutions)

    def test_three_circles(self):
        x, y, z = Polynomial.coordinates(3)
        equations = [
            (x-6)**2 + (y-1)**2 + (z-1)**2 - 25,
            (x-1)**2 + (y-6)**2 + (z-1)**2 - 25,
            (x-1)**2 + (y-1)**2 + (z-6)**2 - 25,
        ]
        expansion_monomials = [
            all_monomials((x, y, z), degree=2),
            all_monomials((x, y, z), degree=2),
            all_monomials((x, y, z), degree=2),
        ]
        expected_solutions = [(1, 1, 1)]
        lambda_poly = x + 2*y + 3*z + 4
        result = solvers.solve_via_basis_selection(equations,
                                                   expansion_monomials,
                                                   lambda_poly,
                                                   solutions=expected_solutions)
        self.assert_solutions_found(result, expected_solutions)

    def test_synthetic_ideal(self):
        zeros = [[-2., -3., 5., 6.], [4.5, 5., -1., 8.]]
        equations = ideal_from_variety(zeros, ctype=float)
        coords = Polynomial.coordinates(len(zeros[0]))
        expansion_monomials = [
            all_monomials(coords, degree=1),
            all_monomials(coords, degree=1),
            all_monomials(coords, degree=1),
        ]
        lambda_poly = sum(xi * (i + 1) for i, xi in enumerate(coords)) + 1
        result = solvers.solve_via_basis_selection(
            equations,
            expansion_monomials,
            lambda_poly,
            solutions=[zeros[0]])
        self.assert_solutions_found(result, zeros)

    def test_three_vars(self):
        x, y, z = Polynomial.coordinates(3)
        equations = [
            x**2 + y**2 + z**2 - 1,
            x - y,
            x - z
        ]
        expansion_monomials = [
            [],
            [x, y, z],
            [x, y, z]
        ]

        result = solvers.solve_via_basis_selection(equations, expansion_monomials, x)

        point = np.sqrt(np.ones(3) / 3.)
        expected_solutions = [point, -point]
        self.assert_solutions_found(result, expected_solutions)


if __name__ == '__main__':
    unittest.main()
