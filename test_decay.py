"""
Unit tests for the decay module
"""
import numpy as np
from decay import forward_euler

def test_forward_euler():
    """
    Test convergence rate of the forward Euler implementation for the test problem
        du/dt + r*u = f(t) with u(0) = u0
    with the manufactured solution u(t) = sin(t).
    """

    # Parameters
    decay_rate = 2.
    u_init = 0.
    t_final = 10.
    def source_term(t):
        return np.cos(t) + 2. * np.sin(t)

    def u_manufactured(t):
        return np.sin(t)

    # Numerical solution on coarser mesh
    num_steps = 400
    t_vals, u_numerical = forward_euler(decay_rate, source_term, u_init, t_final, num_steps)
    u_analytical = u_manufactured(t_vals)
    err_coarse = np.max(np.abs(u_numerical - u_analytical))

    # Numerical solution on finer mesh dt -> dt/2
    num_steps = 800
    t_vals, u_numerical = forward_euler(decay_rate, source_term, u_init, t_final, num_steps)
    u_analytical = u_manufactured(t_vals)
    err_fine = np.max(np.abs(u_numerical - u_analytical))

    eoc = (np.log(err_coarse) - np.log(err_fine))/np.log(2)

    assert np.abs(eoc - 1.) <= .01