"""
Solves the Allen-Cahn equation
    ∂u/∂t - Δu + 1/ɛ² (u³ - u) = 0
in a rectangular domain with initial condition
    u = u0
and boundary conditions
    ∂u/∂n = 0
using finite elements in space and the Crank-Nicolson method in time.
The initial condition is interpolated from random noise with function values uniformly drawn from u(x) ∈ [-1E-5, 1E-5].
"""

from mpi4py import MPI
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import NonlinearProblem
from ufl import (
    dot,
    dx,
    grad,
    TestFunction,
)
from pathlib import Path
import numpy as np

# Mesh
bottom_left = [0.0, -1.0]
top_right = [10.0, 1.0]
resolution = [100, 20]
domain = mesh.create_rectangle(
    MPI.COMM_WORLD, [bottom_left, top_right], resolution, mesh.CellType.triangle
)

# Parameter
epsilon = 1e-2

# Time stepping
t_final = 0.01
num_steps = 100
dt = t_final / num_steps
t = 0.0

# Function space
degree = 1
V = fem.functionspace(domain, ("Lagrange", degree))

# Define initial condition
uh = fem.Function(V)
uh_old = fem.Function(V)
uh.x.array[:] = np.random.uniform(-1e-5, 1e-5, uh.x.array.shape)
uh_old.x.array[:] = uh.x.array
uh.name = "Phase"

# Export initial condition
results_folder = Path("results")
results_folder.mkdir(exist_ok=True, parents=True)
vtx = io.VTXWriter(
    domain.comm,
    (results_folder / "allen_cahn").with_suffix(".bp"),
    [uh],
)  # without 'with' statement, we now have to remember to manually call vtx.close() at the end
vtx.write(t)

# Define weak formulation: for nonlinear problems, the solution appears as a Function (not TrialFunction) in a nonlinear form F
v = TestFunction(V)

F = (
    fem.Constant(domain, 1.0 / dt) * uh * v * dx
    + 1 / 2 * dot(grad(uh), grad(v)) * dx
    + 1 / (2 * epsilon**2) * (uh**3 - uh) * v * dx
    + 1 / 2 * dot(grad(uh_old), grad(v)) * dx
    + 1 / (2 * epsilon**2) * (uh_old**3 - uh_old) * v * dx
    - fem.Constant(domain, 1.0 / dt) * uh_old * v * dx
)

problem = NonlinearProblem(  # NB: FEniCSx has automatic differentiation built in to compute the Jacobian DF(uh). No need to compute it by hand!
    F,  # at each time step, solve F = 0
    uh,  # solve for uh
    petsc_options={
        "snes_type": "newtonls",  # Newton's method with line search for more robust convergence
        "snes_monitor": None,
        "ksp_type": "cg",  # The linear system of equations in each Newton iteration is symmetric positive definite and can be solved with CG
        "ksp_monitor": None,
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
    },
    petsc_options_prefix="newton_",
)

for time_step in range(num_steps):
    t += dt

    # Solve nonlinear problem at this time step
    if domain.comm.rank == 0:
        print("Time step", time_step, "out of", num_steps - 1)
    uh = problem.solve()
    # Export solution
    vtx.write(t)

    # Update old values with current values
    uh_old.x.array[:] = uh.x.array

vtx.close()
