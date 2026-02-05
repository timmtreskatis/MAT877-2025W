"""
Solves the Stokes problem
    - ν Δu + ∇p = 0
    - div u     = 0
in the 2-D DFG benchmark configuration with boundary conditions
    u = u_D on the inlet
    u = 0 on the walls
    ν ∂u/∂n - pn = 0 on the outlet
using finite elements.
"""

from mpi4py import MPI
from dolfinx import fem, io
from dolfinx.fem.petsc import LinearProblem
from ufl import (
    div,
    dx,
    grad,
    inner,
    TrialFunction,
    TestFunction,
)
from pathlib import Path

# Mesh
mesh_data = io.gmsh.read_from_msh("dfg_benchmark_2d.msh", MPI.COMM_WORLD, gdim=2)
domain = mesh_data.mesh
facet_tags = mesh_data.facet_tags

# Function spaces
degree = 1
V = fem.functionspace(
    domain, ("Lagrange", degree + 1, (2,))
)  # the velocity space is vector-valued
Q = fem.functionspace(domain, ("Lagrange", degree))


# Variational formulation
u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)

nu = fem.Constant(domain, 1e-3)
f = fem.Constant(domain, (0.0, 0.0))

a = [
    [nu * inner(grad(u), grad(v)) * dx, -p * div(v) * dx],
    [-q * div(u) * dx, None],
]  # the zero block can be specified as None instead of 0 * p * q * dx
L = [inner(f, v) * dx, fem.Constant(domain, 0.0) * q * dx]


# Boundary conditions
def inflow_velocity(x):
    U = 0.3
    return (4.0 * U * x[1] * (0.41 - x[1]) / 0.41**2, 0.0 * x[1])


u_in = fem.Function(V)
u_in.interpolate(inflow_velocity)
inlet_dofs = fem.locate_dofs_topological(
    V, 1, facet_tags.find(1)
)  # The inlet is marked with facet tag 1 in the GEO file
bc_in = fem.dirichletbc(u_in, inlet_dofs)

u_wall = fem.Constant(domain, (0.0, 0.0))
wall_dofs = fem.locate_dofs_topological(
    V, 1, facet_tags.find(2)
)  # The top and bottom walls are marked with facet tag 2 in the GEO file
bc_wall = fem.dirichletbc(u_wall, wall_dofs, V)

obstacle_dofs = fem.locate_dofs_topological(
    V, 1, facet_tags.find(3)
)  # The obstacle is marked with facet tag 3 in the GEO file
bc_obstacle = fem.dirichletbc(u_wall, obstacle_dofs, V)

bcs = [bc_in, bc_wall, bc_obstacle]

# Linear system of equations
problem = LinearProblem(
    a,
    L,
    bcs=bcs,
    kind="nest",  # the problem is specified blockwise = "nest" for PETSc
    petsc_options={"ksp_type": "none", "pc_type": "lu"},
    petsc_options_prefix="stokes_",
)

# # Block-diagonal bilinear form for preconditioning an iterative solver: Lapacian (uu-block) and mass matrix (pp-block)
# a_p = [[a[0][0], None], [None, p * q * dx]]
# problem = LinearProblem(
#     a,
#     L,
#     bcs=bcs,
#     kind="nest",
#     P=a_p,  # block-diagonal preconditioner
#     petsc_options={
#         "ksp_type": "gmres",
#         "ksp_monitor": None,
#         "pc_type": "fieldsplit",
#         "pc_fieldsplit_type": "additive",  # fieldsplit of type additive means: block-diagonal preconditioning
#         "fieldsplit_0_ksp_type": "preonly",  # block 0 = the uu-block uses 1 AMG iteration as preconditioner
#         "fieldsplit_0_pc_type": "hypre",
#         "fieldsplit_0_pc_hypre_type": "boomeramg",
#         "fieldsplit_1_ksp_type": "none",  # block 1 = the pp-block uses 1 Jacobi iteration as preconditioner
#         "fieldsplit_1_pc_type": "jacobi",
#     },
#     petsc_options_prefix="stokes_",
# )

uh, ph = problem.solve()
uh.name = "Velocity"
ph.name = "Pressure"

# Export numerical results
results_folder = Path("results")
results_folder.mkdir(exist_ok=True, parents=True)
filename = results_folder / "stokes"
with io.VTXWriter(domain.comm, filename.with_suffix(".bp"), [uh]) as vtx:
    vtx.write(0.0)

filename = results_folder / "stokes_pressure"
with io.VTXWriter(domain.comm, filename.with_suffix(".bp"), [ph]) as vtx:
    vtx.write(0.0)
