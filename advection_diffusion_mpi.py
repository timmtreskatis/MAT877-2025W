"""
Solves the steady 2D advection-diffusion problem
    -div (D ∇u) + a·∇u = 0
in a rectangular channel with boundary conditions
    u = u_D on the inlet
    -D ∂u/∂n + u a·n = 0 on the walls
    -D ∂u/∂n = 0 on the outlet
using finite elements.

This code does the same as advection_diffusion.py, but LinearProblem has been replaced with a more low-level, own implementation to
    (1) compile the bilinear and linear forms
    (2) allocate a PETSc matrix and vector
    (3) assemble the matrix and the vector from the bilinear and linear forms
    (4) apply Dirichlet boundary conditions by lifting
    (5) configure the linear solver options
    (6) solve the linear system of equations
in a fully MPI-compatible way.
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

# Function space
degree = 1
V = fem.functionspace(domain, ("Lagrange", degree))

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
    diffusivity * dot(grad(u), grad(v)) * dx
    - u * dot(advection_velocity, grad(v)) * dx
    + u * dot(advection_velocity, n) * v * ds(3)
)

L = fem.Constant(domain, 0.0) * v * dx


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
A = assemble_matrix(bilinear_form, bcs=[bc]) # adds local contributions to the matrix (from own rank)
A.assemble() # adds contibutions from other ranks
b = create_vector(fem.extract_function_spaces(linear_form)) # creates a PETSc vector of the right length initialised with zeros
assemble_vector(b, linear_form) # adds all contributions (owned and from other processes) to the vector

# Apply Dirichlet boundary condition to the vector
apply_lifting(b, [bilinear_form], [[bc]]) # modifies the vector in rows where no Dirichlet data are imposed; matrix-vector products end up partially in owned entries, partially in ghost entries of b
b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE) # ghost entries of b are added to owned entries of b; owned entries now have the correct data
set_bc(b, [bc]) # modifies the vector in rows where Dirichlet data are imposed: previous vector entries are overwritten with Dirichlet data

# Create and configure a linear solver
advection_diffusion_solver = PETSc.KSP().create(domain.comm)
advection_diffusion_solver.setOperators(A)
opts = PETSc.Options()
opts["ksp_type"] = "gmres"
opts["pc_type"] = "hypre"
opts["pc_hypre_type"] = "boomeramg"
opts["pc_hypre_boomeramg_smooth_type"] = "ilu"
opts["ksp_monitor"] = None
advection_diffusion_solver.setFromOptions()

# Solve linear problem
uh = fem.Function(V)
uh.name = "Concentration"
advection_diffusion_solver.solve(b, uh.x.petsc_vec)
uh.x.scatter_forward() # solution values are copied from owned entries to ghost entries; ghost entries now have the correct data

# Export solution
results_folder = Path("results")
results_folder.mkdir(exist_ok=True, parents=True)

with io.VTXWriter(
    domain.comm, (results_folder / "advection_diffusion").with_suffix(".bp"), [uh]
) as vtx:
    vtx.write(0.0)
