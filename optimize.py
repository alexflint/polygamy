import numbers
import random
import numpy as np

from polynomial import Polynomial
import solvers


class GlobalOptimizationError(Exception):
    pass


def find_critical_points(polynomial, constraints=None, expansions=2, lambda_poly=None, seed=0, **kwargs):
    """Find all local minima, local maxima, and saddle points of the given polynomial by solving the first order
    conditions."""
    system = polynomial.astype(float).partial_derivatives()
    system = [p for p in system if p.total_degree > 0]
    if constraints is not None:
        system.extend(constraints)
    coords = Polynomial.coordinates(polynomial.num_vars, ctype=float)
    if lambda_poly is None:
        rng = random.Random(seed)  # use this rng for repeatability
        lambda_poly = sum(p * rng.random() for p in coords) + rng.random()
    if isinstance(expansions, numbers.Integral):
        expansions = [solvers.all_monomials(coords, expansions)] * len(system)
    return solvers.solve_via_basis_selection(system, expansions, lambda_poly, **kwargs)


def minimize_globally(polynomial, *args, **kwargs):
    """Find the global minimizer of the given polynomial."""
    result = find_critical_points(polynomial, *args, **kwargs)
    if len(result.solutions) == 0:
        raise GlobalOptimizationError("No critical points found by solver")
    min_cost = None
    best_solution = None
    for solution in result.solutions:
        cost = polynomial(*solution)
        if min_cost is None or cost < min_cost:
            min_cost = cost
            best_solution = solution
    return best_solution
