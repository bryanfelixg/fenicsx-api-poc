from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem.petsc import LinearProblem

app = FastAPI(title="FEniCSx Poisson API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PoissonParams(BaseModel):
    lx: float = Field(1.0, gt=0, description="Domain size in x")
    ly: float = Field(1.0, gt=0, description="Domain size in y")
    nx: int = Field(48, ge=4, le=256, description="Cells in x (cap for PoC)")
    ny: int = Field(48, ge=4, le=256, description="Cells in y (cap for PoC)")
    f_const: float = Field(1.0, description="RHS constant f")
    bc_value: float = Field(0.0, description="Dirichlet boundary value on ∂Ω")
    return_plot: bool = Field(True, description="(unused/deprecated: kept for compatibility)")

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/solve")
def solve(params: PoissonParams):
    """
    Solve -Δu = f on [0,lx]x[0,ly] with u = bc on ∂Ω.
    Returns summary stats and mesh arrays for interactive 3D plotting.
    """
    comm = MPI.COMM_WORLD

    # Domain & mesh
    domain = mesh.create_rectangle(
        comm,
        points=[np.array([0.0, 0.0]), np.array([params.lx, params.ly])],
        n=[params.nx, params.ny],
        cell_type=mesh.CellType.triangle,
    )

    V = fem.functionspace(domain, ("Lagrange", 1))
    v = ufl.TestFunction(V)
    u_trial = ufl.TrialFunction(V)
    f = fem.Constant(domain, params.f_const)
    a = ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    def on_boundary(x):
        return np.isclose(x[0], 0.0) | np.isclose(x[0], params.lx) | \
               np.isclose(x[1], 0.0) | np.isclose(x[1], params.ly)

    facets = mesh.locate_entities_boundary(domain, 1, on_boundary)
    dofs = fem.locate_dofs_topological(V, 1, facets)
    bc = fem.dirichletbc(fem.Constant(domain, params.bc_value), dofs, V)

    problem = LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )

    uh = problem.solve()
    coords = V.tabulate_dof_coordinates()  # shape (num_nodes, 2)
    values = uh.x.array  # shape (num_nodes,)

    center = np.array([params.lx/2, params.ly/2])
    center_idx = np.argmin((coords[:,0]-center[0])**2 + (coords[:,1]-center[1])**2)
    center_u = float(values[center_idx])

    # Extract triangle connectivity
    domain.topology.create_connectivity(2, 0)
    tri = domain.topology.connectivity(2, 0).array.reshape(-1, 3)  # (num_tri, 3)

    # Prepare data to send as lists
    x = coords[:, 0].tolist()
    y = coords[:, 1].tolist()
    u = values.tolist()
    triangles = tri.tolist()  # Each row is a [i, j, k]

    # Return only from rank 0
    if comm.rank == 0:
        return {
            "domain": {"lx": params.lx, "ly": params.ly},
            "mesh": {"nx": params.nx, "ny": params.ny, "ndofs": int(len(values))},
            "rhs_const": params.f_const,
            "bc_value": params.bc_value,
            "center_u": center_u,
            "x": x,
            "y": y,
            "u": u,
            "triangles": triangles,
        }
    else:
        return {"status": "ok"}