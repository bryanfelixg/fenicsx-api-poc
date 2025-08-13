from shiny import App, ui, render, reactive  # Add reactive import here!
import httpx, base64, tempfile

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
            ui.output_image("plot", height="480px"),
            ui.output_text_verbatim("center_u"),
        ),
    ),
)

def server(input, output, session):
    @output
    @render.image(delete_file=True)
    @reactive.event(input.solve)  # <--- Only triggers on "Solve" clicks!
    def plot():
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
        png_bytes = base64.b64decode(data["plot_png_base64"])
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(png_bytes)
            tmp_path = tmp.name
        return {"src": tmp_path, "alt": "Poisson solution", "height": "480px"}

    @output
    @render.text
    @reactive.event(input.solve)
    def center_u():
        # Run only when 'Solve' is clicked.
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