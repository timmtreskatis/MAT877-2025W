"""
Solve the ODE
    du/dt + r*u = f in ]0,T[
    u(0) = u0
for exponential decay.
"""

import numpy as np
import matplotlib.pyplot as plt


def forward_euler(decay_rate, source_term, u_init, t_final, num_steps):
    dt = t_final / num_steps
    t_vals = np.linspace(0., t_final, num_steps + 1)

    # Initialise entire array with initial value
    u_vals = u_init * np.ones_like(t_vals)

    for n in range(num_steps):
        u_vals[n+1] = dt*source_term(t_vals[n]) + (1. - decay_rate*dt)*u_vals[n]

    return t_vals, u_vals

def plot_numerical_solution(t_vals, u_vals):
    plt.plot(t_vals, u_vals)
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.show()

if __name__ == "__main__":
    # Parameters
    decay_rate = 2.
    u_init = 0.
    t_final = 10.
    num_steps = 400

    def source_term(t):
        return np.cos(t) + 2. * np.sin(t)

    def u_manufactured(t):
        return np.sin(t)

    t_vals, u_numerical = forward_euler(decay_rate, source_term, u_init, t_final, num_steps)


    u_analytical = u_manufactured(t_vals)
    err = np.max(np.abs(u_numerical - u_analytical))

    print("Maximum error:", err)

    plot_numerical_solution(t_vals, u_numerical)