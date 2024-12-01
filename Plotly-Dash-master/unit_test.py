import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
from dash import dash_table

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Load data into a DataFrame
df = pd.read_csv('yokyo.log', sep=" ", header=None,
                 names=["Timestamp", "IP Address", "HTTP Method",
                        "Path", "Status Code","HTTP Version", "Traffic Source", "User Agent", "Country"],
                 usecols=["Timestamp", "IP Address", "HTTP Method",
                        "Path", "Status Code","HTTP Version","Traffic Source", "User Agent", "Country"])

# Create a DataTable component to display the data
table = dash_table.DataTable(
    id='table',
    columns=[{"name": i, "id": i} for i in df.columns],
    data=df.to_dict('records'),
)

# Create a Modal component to display the table
modal = dbc.Modal(
    [
        dbc.ModalHeader("Data Table"),
        dbc.ModalBody(table),
        dbc.ModalFooter(
            dbc.Button("Close", id="close", className="ml-auto")
        ),
    ],
    id="modal",
    size="xl",
)

# Create a button to open the modal
button = dbc.Button("View Data", id="open")

app.layout = html.Div([button, modal])

# Callback to open the modal
@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

if __name__ == "__main__":
    app.run_server(debug=True)
