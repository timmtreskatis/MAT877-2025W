"""
Solves the boundary value problem
    -Δu + 3u = f in Ω
    ∂u/∂n + 5u = g on ∂Ω
on Ω = ]0,1[² with finite elements.

Based on the DOLFINx tutorial by Jørgen Schartum Dokken
https://jsdokken.com/dolfinx-tutorial/chapter1/fundamentals_code.html
"""

from mpi4py import MPI
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem
from ufl import (
    as_vector,
    dot,
    ds,
    dx,
    grad,
    FacetNormal,
    SpatialCoordinate,
    TrialFunction,
    TestFunction,
)
from pathlib import Path
import numpy


# Manufactured solution
def u_manufactured(x):
    return 1 + x[0] ** 5 + 2 * x[1] ** 6


# Mesh
N = 16  # divide the unit square into an N x N mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, mesh.CellType.triangle)

# Function space
degree = 1  # 1 for linear finite elements, 2 for quadratic finite elements, ...
V = fem.functionspace(domain, ("Lagrange", degree))


# Variational formulation
u = TrialFunction(V)
v = TestFunction(V)

x = SpatialCoordinate(domain)  # x[0] is the x-variable, x[1] is the y-variable
f = (
    3 - 20 * x[0] ** 3 - 60 * x[1] ** 4 + 3 * x[0] ** 5 + 6 * x[1] ** 6
)  # RHS of the PDE to reproduce the manufactured solution

n = FacetNormal(domain)  # outward pointing unit normal vector
g = (
    dot(as_vector([5 * x[0] ** 4, 12 * x[1] ** 5]), n)
    + 5
    + 5 * x[0] ** 5
    + 10 * x[1] ** 6
)  # RHS of the Robin BC to reproduce the manufactured solution

a = dot(grad(u), grad(v)) * dx + 3 * u * v * dx + 5 * u * v * ds
L = f * v * dx + g * v * ds

# Linear system of equations
problem = LinearProblem(
    a,
    L,
    petsc_options={"ksp_type": "none", "pc_type": "cholesky"},
    petsc_options_prefix="Poisson",
)
uh = problem.solve()

# Method of manufactured solutions
u_exact = u_manufactured(x)

L2_error = fem.form((uh - u_exact) * (uh - u_exact) * dx)
error_local = fem.assemble_scalar(L2_error)
error_L2 = numpy.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

H1_error = fem.form(
    (uh - u_exact) * (uh - u_exact) * dx
    + dot(grad(uh - u_exact), grad(uh - u_exact)) * dx
)
error_local = fem.assemble_scalar(H1_error)
error_H1 = numpy.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

if domain.comm.rank == 0:  # Only print the error on one process
    print("Using finite elements of degree", degree, "and a mesh with N =", N)
    print(f"   L²-error: {error_L2:.2e}")
    print(f"   H¹-error: {error_H1:.2e}")

# Export numerical results
results_folder = Path("results")
results_folder.mkdir(exist_ok=True, parents=True)
filename = results_folder / "fundamentals"
with io.VTXWriter(domain.comm, filename.with_suffix(".bp"), [uh]) as vtx:
    vtx.write(0.0)
