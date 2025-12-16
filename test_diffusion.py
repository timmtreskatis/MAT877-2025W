"""
Unit tests for the diffusion module
"""

import numpy as np
from diffusion import solve_diffusion_equation


def test_sin_solution():
    """
    Test that u(x) = sin(x) solves the second-order boundary-value problem
        -u"(x) = f(x) in ]2, 4pi[
        u(x) = sin(x) on {2, 4pi}
    for f(x) = sin(x).
    """

    # Parameters
    def u_manufactured(x):
        return np.sin(x)

    domain = [2.0, 4.0 * np.pi]

    def source_term(x):
        return np.sin(x)

    # Numerical solution
    num_subintervals = 100
    x_vals, u_numerical = solve_diffusion_equation(
        source_term, u_manufactured, domain, num_subintervals
    )
    u_analytical = u_manufactured(x_vals)

    assert np.max(np.abs(u_numerical - u_analytical)) <= 1e-1


def test_linear_solution():
    """
    Test that u(x) = x + 4 solves the second-order boundary-value problem
        -u"(x) = f(x) in ]2, 4pi[
        u(x) = x + 4 on {2, 4pi}
    for f(x) = 0.
    """

    # Parameters
    def u_manufactured(x):
        return x + 4.0

    domain = [2.0, 4.0 * np.pi]

    def source_term(x):
        return np.zeros_like(x)

    # Numerical solution
    num_subintervals = 100
    x_vals, u_numerical = solve_diffusion_equation(
        source_term, u_manufactured, domain, num_subintervals
    )
    u_analytical = u_manufactured(x_vals)

    assert np.max(np.abs(u_numerical - u_analytical)) <= 1e-1
