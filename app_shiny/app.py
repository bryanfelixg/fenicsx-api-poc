from shiny import App, ui, render, reactive
import httpx
import plotly.graph_objs as go

from shinywidgets import output_widget, render_plotly

API_BASE = "https://fenicsx-api-poc.onrender.com"

app_ui = ui.page_fluid(
    ui.h2("FEniCSx Poisson with Gaussian Source"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_numeric("nx", "mesh: nx", 20, min=4, max=256),
            ui.input_numeric("ny", "mesh: ny", 20, min=4, max=256),
            ui.input_slider("x0", "source x0", 0, 1, 0.8, step=0.01),
            ui.input_slider("y0", "source y0", 0, 1, 0.8, step=0.01),
            ui.input_slider("sigma", "source width (sigma)", 0.01, 0.5, 0.1, step=0.01),
            ui.input_numeric("amplitude", "source amplitude", 1.0),
            ui.input_numeric("bc", "Dirichlet boundary value", 0.0),
            ui.input_action_button("solve", "Solve"),
        ),
        ui.card(
            ui.h4("Poisson Solution"),
            output_widget("plot_surface", height="480px"),
            ui.h4("Mesh Visualization"),
            output_widget("plot_mesh", height="360px"),
            ui.output_text_verbatim("center_u"),
        ),
    ),
)

def server(input, output, session):

    def build_payload():
        return {
            "lx": 1.0,
            "ly": 1.0,
            "nx": int(input.nx()),
            "ny": int(input.ny()),
            "x0": float(input.x0()),
            "y0": float(input.y0()),
            "sigma": float(input.sigma()),
            "amplitude": float(input.amplitude()),
            "bc_value": float(input.bc()),
            "return_plot": True,
        }

    @output
    @render_plotly
    @reactive.event(input.solve)
    def plot_surface():
        payload = build_payload()
        r = httpx.post(f"{API_BASE}/solve", json=payload, timeout=60.0)
        r.raise_for_status()
        data = r.json()
        if not all(key in data for key in ("x", "y", "u", "triangles")) or not data["triangles"]:
            return go.Figure()
        x, y, z, triangles = data["x"], data["y"], data["u"], data["triangles"]
        i, j, k = zip(*triangles)
        surface = go.Mesh3d(
            x=x, y=y, z=z, i=i, j=j, k=k,
            intensity=z, colorscale="Viridis", showscale=True
        )
        fig = go.Figure(data=[surface])
        fig.update_layout(
            title="Solution u(x, y)",
            scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="u"),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        return fig

    @output
    @render_plotly
    @reactive.event(input.solve)
    def plot_mesh():
        payload = build_payload()
        r = httpx.post(f"{API_BASE}/solve", json=payload, timeout=60.0)
        r.raise_for_status()
        data = r.json()
        if not all(key in data for key in ("x", "y", "triangles")) or not data["triangles"]:
            return go.Figure()
        x, y, triangles = data["x"], data["y"], data["triangles"]

        lines_x = []
        lines_y = []
        lines_z = []
        for tri in triangles:
            idx = [tri[0], tri[1], tri[2], tri[0]]  # back to start
            lines_x.extend([x[i] for i in idx] + [None])
            lines_y.extend([y[i] for i in idx] + [None])
            lines_z.extend([0 for _ in idx] + [None])

        mesh_lines = go.Scatter3d(
            x=lines_x,
            y=lines_y,
            z=lines_z,
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False,
            hoverinfo='skip',
        )

        node_markers = go.Scatter3d(
            x=x,
            y=y,
            z=[0]*len(x),
            mode='markers',
            marker=dict(size=2, color='red'),
            showlegend=False,
            hoverinfo="skip",
        )

        fig = go.Figure(data=[mesh_lines, node_markers])
        fig.update_layout(
            title="Finite Element Mesh",
            scene=dict(
                xaxis_title="x", yaxis_title="y", zaxis_title="",
                zaxis=dict(showticklabels=False, visible=False)
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        return fig

    @output
    @render.text
    @reactive.event(input.solve)
    def center_u():
        payload = build_payload()
        # For efficiency, optionally set `"return_plot": False` for center_u
        r = httpx.post(f"{API_BASE}/solve", json=payload, timeout=60.0)
        r.raise_for_status()
        data = r.json()
        return f"u(center) = {data.get('center_u', 'N/A'):.4f}"

app = App(app_ui, server)