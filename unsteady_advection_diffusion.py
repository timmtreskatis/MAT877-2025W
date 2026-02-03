"""
Solves the unsteady 2D advection-diffusion problem
    ∂u/∂t -div (D ∇u) + a·∇u = 0
in a rectangular channel with initial condition
    u = 0
and boundary conditions
    u = u_D on the inlet
    -D ∂u/∂n + u a·n = 0 on the walls
    -D ∂u/∂n = 0 on the outlet
using finite elements in space and the Crank-Nicolson method in time.
"""

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    create_vector,
    set_bc,
)
from ufl import (
    as_vector,
    dot,
    dx,
    grad,
    FacetNormal,
    Measure,
    SpatialCoordinate,
    TrialFunction,
    TestFunction,
)
from pathlib import Path
import numpy as np


def mark_boundaries(domain):
    """
    Mark boundary facets with integer tags to distinguish different pieces of the boundary.

    The following boundary markers are used:
        1 for the inflow boundary (x = 0)
        2 for the top and bottom walls (y = ±1)
        3 for the outflow boundary (x = 10)

    Args:
        domain: Computational mesh

    Returns:
        facet_tags: Mesh tags identifying each triangle edge on the boundary with 1, 2 or 3
    """
    tdim = domain.topology.dim  # topological dimension of the mesh
    fdim = tdim - 1  # topological dimension of the boundary facets

    def dirichlet_boundary(x):
        return np.isclose(x[0], 0.0)

    def robin_boundary(x):
        return np.isclose(x[1], -1.0) | np.isclose(x[1], 1.0)

    def neumann_boundary(x):
        return np.isclose(x[0], 10.0)

    boundaries = [
        (1, dirichlet_boundary),
        (2, robin_boundary),
        (3, neumann_boundary),
    ]

    facet_indices, facet_markers = [], []
    for marker, locator in boundaries:
        facets = mesh.locate_entities(domain, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))

    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)

    facet_tags = mesh.meshtags(
        domain,
        fdim,
        facet_indices[sorted_facets],
        facet_markers[sorted_facets],
    )

    return facet_tags


# Mesh
bottom_left = [0.0, -1.0]
top_right = [10.0, 1.0]
resolution = [100, 20]
domain = mesh.create_rectangle(
    MPI.COMM_WORLD, [bottom_left, top_right], resolution, mesh.CellType.triangle
)

# Time stepping
t_final = 20.0
num_steps = 200
dt = t_final / num_steps
t = 0.0

# Function space
degree = 1
V = fem.functionspace(domain, ("Lagrange", degree))

# Define initial condition
uh = fem.Function(V)
uh_old = fem.Function(V)
uh.x.array[:] = 0.0
uh_old.x.array[:] = uh.x.array
uh.name = "Concentration"

# Export initial condition
results_folder = Path("results")
results_folder.mkdir(exist_ok=True, parents=True)
vtx = io.VTXWriter(
    domain.comm,
    (results_folder / "unsteady_advection_diffusion").with_suffix(".bp"),
    [uh],
)  # without 'with' statement, we now have to remember to manually call vtx.close() at the end
vtx.write(t)

# Tags labelling boundaries
facet_tags = mark_boundaries(domain)

# Define weak formulation
u = TrialFunction(V)
v = TestFunction(V)

x = SpatialCoordinate(domain)
n = FacetNormal(domain)

diffusivity = fem.Constant(domain, 1e-2)
advection_velocity = as_vector([1.0 - x[1] ** 2, 0.0])

ds = Measure("ds", domain, subdomain_data=facet_tags)
# ds = integral over the entire boundary
# ds(1) = integral over the boundary segments labelled 1

a = (
    fem.Constant(domain, 1.0 / dt) * u * v * dx
    + 1/2 * diffusivity * dot(grad(u), grad(v)) * dx
    - 1/2 * u * dot(advection_velocity, grad(v)) * dx
    + 1/2 * u * dot(advection_velocity, n) * v * ds(3)
)

L = (
    fem.Constant(domain, 1.0 / dt) * uh_old * v * dx
    - 1/2 * diffusivity * dot(grad(uh_old), grad(v)) * dx
    + 1/2 * uh_old * dot(advection_velocity, grad(v)) * dx
    - 1/2 * uh_old * dot(advection_velocity, n) * v * ds(3)
)


# Dirichlet boundary condition
def u_inflow(x):
    return np.where(np.abs(x[1]) <= 0.5, 1.0, 0.0)


u_D = fem.Function(V)
u_D.interpolate(u_inflow)

tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
dirichlet_dofs = fem.locate_dofs_topological(V, fdim, facet_tags.find(1))

bc = fem.dirichletbc(u_D, dirichlet_dofs)

# Compile weak formulation into bilinear and linear forms
bilinear_form = fem.form(a)
linear_form = fem.form(L)

# Create PETSc matrix and vector for the linear system of equations
A = assemble_matrix(bilinear_form, bcs=[bc])
A.assemble()
b = create_vector(fem.extract_function_spaces(linear_form))

# Create and configure a linear solver
advection_diffusion_solver = PETSc.KSP().create(domain.comm)
advection_diffusion_solver.setOperators(A)
opts = PETSc.Options()
opts["ksp_type"] = "gmres"
opts["pc_type"] = "hypre"
opts["pc_hypre_type"] = "boomeramg"
opts["pc_hypre_boomeramg_smooth_type"] = "ilu"
opts["ksp_monitor"] = None
opts["ksp_initial_guess_nonzero"] = True # "Warm start": don't use the default zero vector as initial guess for GMRES, but uh from the previous time step
advection_diffusion_solver.setFromOptions()


for time_step in range(num_steps):
    t += dt

    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(
            0
        )  # as the assemble_vector command adds to (not overwrites) the existing vector entries, we first have to set b to zero
    assemble_vector(b, linear_form)

    # Apply Dirichlet boundary condition to the vector
    apply_lifting(b, [bilinear_form], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])

    # Solve linear problem
    if domain.comm.rank == 0:
        print("Time step", time_step, "out of", num_steps - 1)
    advection_diffusion_solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    # Export solution
    vtx.write(t)

    # Update old values with current values
    uh_old.x.array[:] = uh.x.array

vtx.close()
