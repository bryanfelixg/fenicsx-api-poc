from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem.petsc import LinearProblem

import io as sysio
import base64
import matplotlib.pyplot as plt

app = FastAPI(title="FEniCSx Poisson API", version="0.1.0")

# CORS so a Shiny (Python) app on Posit Connect can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PoissonParams(BaseModel):
    # Domain
    lx: float = Field(1.0, gt=0, description="Domain size in x")
    ly: float = Field(1.0, gt=0, description="Domain size in y")
    # Mesh
    nx: int = Field(48, ge=4, le=256, description="Cells in x (cap for PoC)")
    ny: int = Field(48, ge=4, le=256, description="Cells in y (cap for PoC)")
    # Forcing and BC
    f_const: float = Field(1.0, description="RHS constant f")
    bc_value: float = Field(0.0, description="Dirichlet boundary value on \u2202\u03A9")
    # Output controls
    return_plot: bool = Field(True, description="Return base64 PNG of trisurface plot")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/solve")
def solve(params: PoissonParams):
    """Solve -Δu = f on [0,lx]x[0,ly] with u = bc on ∂Ω.
    Returns a few summary stats and optionally a base64 PNG trisurface plot.
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

    # Dirichlet BC on the outer boundary
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

    # Gather simple summary on rank 0
    coords = V.tabulate_dof_coordinates()
    values = uh.x.array

    # probe at center
    center = np.array([params.lx/2, params.ly/2])
    center_idx = np.argmin((coords[:,0]-center[0])**2 + (coords[:,1]-center[1])**2)
    center_u = float(values[center_idx])

    # Optional plot (rank 0)
    plot_b64 = None
    if params.return_plot and comm.rank == 0:
        # triangles for trisurf
        domain.topology.create_connectivity(2, 0)
        tri = domain.topology.connectivity(2, 0).array.reshape(-1, 3)

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(coords[:, 0], coords[:, 1], values, triangles=tri, linewidth=0.2, antialiased=True)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("u")
        ax.set_title("Poisson solution (FEniCSx)")
        buf = sysio.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        plot_b64 = base64.b64encode(buf.getvalue()).decode()

    plot_b64 = comm.bcast(plot_b64, root=0)

    # Return only from rank 0 to avoid JSON issues
    if comm.rank == 0:
        return {
            "domain": {"lx": params.lx, "ly": params.ly},
            "mesh": {"nx": params.nx, "ny": params.ny, "ndofs": int(len(values))},
            "rhs_const": params.f_const,
            "bc_value": params.bc_value,
            "center_u": center_u,
            "plot_png_base64": plot_b64,
        }
    else:
        return {"status": "ok"}