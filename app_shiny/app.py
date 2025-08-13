from shiny import App, ui, render, reactive
import httpx
import plotly.graph_objs as go

from shinywidgets import output_widget, render_plotly

API_BASE = "https://fenicsx-api-poc.onrender.com"

app_ui = ui.page_fluid(
    ui.h2("FEniCSx Poisson PoC"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_numeric("nx", "nx", 64, min=4, max=256),
            ui.input_numeric("ny", "ny", 64, min=4, max=256),
            ui.input_numeric("f", "f_const", 1.0),
            ui.input_numeric("bc", "bc_value", 0.0),
            ui.input_action_button("solve", "Solve"),
        ),
        ui.card(
            output_widget("plot_surface", height="480px"),   # << USE output_widget FOR ALL WIDGETS
            ui.output_text_verbatim("center_u"),
        ),
    ),
)

def server(input, output, session):
    @output
    @render_plotly
    @reactive.event(input.solve)
    def plot_surface():
        payload = {
            "lx": 1.0,
            "ly": 1.0,
            "nx": int(input.nx()),
            "ny": int(input.ny()),
            "f_const": float(input.f()),
            "bc_value": float(input.bc()),
            "return_plot": True,
        }
        r = httpx.post(f"{API_BASE}/solve", json=payload, timeout=60.0)
        r.raise_for_status()
        data = r.json()

        # Defensive: skip if any expected array missing or empty
        if not all(key in data for key in ("x", "y", "u", "triangles")) or not data["triangles"]:
            return go.Figure()

        x = data["x"]
        y = data["y"]
        z = data["u"]
        triangles = data["triangles"]

        i, j, k = zip(*triangles)
        surface = go.Mesh3d(
            x=x, y=y, z=z, i=i, j=j, k=k,
            intensity=z,
            colorscale="Viridis",
            showscale=True,
        )
        fig = go.Figure(data=[surface])
        fig.update_layout(
            title="Poisson solution (interactive)",
            scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="u"),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        return fig

    @output
    @render.text
    @reactive.event(input.solve)
    def center_u():
        payload = {
            "lx": 1.0,
            "ly": 1.0,
            "nx": int(input.nx()),
            "ny": int(input.ny()),
            "f_const": float(input.f()),
            "bc_value": float(input.bc()),
            "return_plot": False,
        }
        r = httpx.post(f"{API_BASE}/solve", json=payload, timeout=60.0)
        r.raise_for_status()
        data = r.json()
        return f"u(center) = {data.get('center_u', 'N/A')}"

app = App(app_ui, server)