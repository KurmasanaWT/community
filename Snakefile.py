# sidebar.py
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "overflow": "scroll"
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "display": "inline-block"
}

sidebar = html.Div(
    [
        html.H2("Sidebar", className="display-4"),
        html.Hr(),
        html.P(
            "A simple sidebar", className="lead"
        ),
    ],
    style=SIDEBAR_STYLE,
)

maindiv = html.Div(
    id="first-div",
    children=[
        # first row
        html.Div([
            html.H2("First Row"),
            html.Hr(),
            html.P(
                "First row stuff", className="lead"
            )
        ]),

        # second row
        html.Div([
            html.H2("Second Row"),
            html.Hr(),
            html.P(
                "Second row stuff", className="lead"
            )
        ])
    ],
    style=CONTENT_STYLE
)

app.layout = html.Div([sidebar, maindiv])

if __name__ == "__main__":
    app.run_server(port=8888)