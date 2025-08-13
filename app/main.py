from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from dolfinx.fem.petsc import LinearProblem

app = FastAPI(title="FEniCSx Poisson API", version="0.2.0")

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
    nx: int = Field(48, ge=4, le=256, description="Cells in x")
    ny: int = Field(48, ge=4, le=256, description="Cells in y")
    # Point source
    x0: float = Field(0.5, description="Point source x position")
    y0: float = Field(0.5, description="Point source y position")
    sigma: float = Field(0.1, gt=0, description="Point source width (Gaussian sigma)")
    amplitude: float = Field(1.0, description="Point source amplitude")
    # Dirichlet BC
    bc_value: float = Field(0.0, description="Dirichlet boundary value")
    # Output
    return_plot: bool = Field(True, description="(deprecated; for compatibility)")

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/solve")
def solve(params: PoissonParams):
    """
    Solve -Δu = f(x,y) with Dirichlet BC, Gaussian source.
    Returns mesh, solution, and summary statistics.
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
    
    # Gaussian source at (x0, y0)
    x0, y0, sigma, amp = params.x0, params.y0, params.sigma, params.amplitude

    class GaussianSource:
        def __init__(self, x0, y0, sigma, amplitude):
            self.x0 = x0
            self.y0 = y0
            self.sigma = sigma
            self.amplitude = amplitude

        def __call__(self, x):
            # x has shape (2, N)
            dx = x[0] - self.x0
            dy = x[1] - self.y0
            return self.amplitude * np.exp(-((dx**2 + dy**2)/(2*self.sigma**2)))

    f_gauss = fem.Function(V)
    f_gauss.interpolate(GaussianSource(x0, y0, sigma, amp))

    a = ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
    L = f_gauss * v * ufl.dx

    # Dirichlet boundary
    def on_boundary(x):
        return np.isclose(x[0], 0.0) | np.isclose(x[0], params.lx) | \
               np.isclose(x[1], 0.0) | np.isclose(x[1], params.ly)

    facets = mesh.locate_entities_boundary(domain, 1, on_boundary)
    dofs = fem.locate_dofs_topological(V, 1, facets)
    bc = fem.dirichletbc(fem.Constant(domain, params.bc_value), dofs, V)

    problem = LinearProblem(
        a, L, bcs=[bc],
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
    tri = domain.topology.connectivity(2, 0).array.reshape(-1, 3)

    x = coords[:, 0].tolist()
    y = coords[:, 1].tolist()
    u = values.tolist()
    triangles = tri.tolist()

    if comm.rank == 0:
        return {
            "domain": {"lx": params.lx, "ly": params.ly},
            "mesh": {"nx": params.nx, "ny": params.ny, "ndofs": int(len(values))},
            "source": {
                "x0": x0, "y0": y0, "sigma": sigma, "amplitude": amp
            },
            "bc_value": params.bc_value,
            "center_u": center_u,
            "x": x,
            "y": y,
            "u": u,
            "triangles": triangles,
        }
    else:
        return {"status": "ok"}