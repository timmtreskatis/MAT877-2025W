"""
Solve the ODE
    du/dt + r*u = 0 in ]0,T[
    u(0) = u0
for exponential decay.
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
decay_rate = 2.
u_init = 6.
t_final = 5.

num_steps = 20
dt = t_final / num_steps

t_vals = np.linspace(0., t_final, num_steps + 1)
u_vals = u_init * np.ones_like(t_vals)

for n in range(num_steps):
    u_vals[n+1] = (1. - decay_rate*dt)*u_vals[n]

plt.plot(t_vals, u_vals)
plt.xlabel("t")
plt.ylabel("u(t)")
plt.show()