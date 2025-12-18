import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt


def solve_diffusion_equation(source_term, u_dirichlet, domain, num_subintervals):
    """
    Solves the diffusion problem
        -u"(x) = f(x) in ]a, b[
        u(x) = uD(x) on {a, b}
    Args:
        source_term: a function that computes f(x)
        u_dirichlet: a function that computes uD(x)
        domain: a list or an array [a, b]
        num_subintervals: number of subintervals to use in the finite-difference discretisation
    Returns:
        x_vals: Array of grid points from a to b
        u_numerical: Array of numerical solution values at these grid points
    """

    dx = (domain[1] - domain[0]) / num_subintervals
    x_vals = np.linspace(domain[0], domain[1], num_subintervals + 1)

    u_numerical = np.zeros_like(x_vals)
    u_numerical[0] = u_dirichlet(domain[0])
    u_numerical[-1] = u_dirichlet(domain[1])

    rhs_vector = source_term(x_vals[1:-1])
    rhs_vector[0] += u_numerical[0] / dx**2
    rhs_vector[-1] += u_numerical[-1] / dx**2

    lhs_matrix = sparse.diags_array(
        [-1 / dx**2, 2 / dx**2, -1 / dx**2],
        offsets=[-1, 0, 1],
        shape=(num_subintervals - 1, num_subintervals - 1),
        format="csr",
    )

    u_numerical[1:-1] = sparse.linalg.spsolve(lhs_matrix, rhs_vector)

    return x_vals, u_numerical


def plot_numerical_solution(x_vals, u_vals):
    plt.plot(x_vals, u_vals)
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.show()


if __name__ == "__main__":

    def source_term(x): return x**2

    def u_dirichlet(x): return 4.0 * x

    x_vals, u_vals = solve_diffusion_equation(source_term, u_dirichlet, [0.0, 4.0], 10)
    plot_numerical_solution(x_vals, u_vals)
