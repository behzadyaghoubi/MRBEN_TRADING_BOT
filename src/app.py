import json
import logging
import os

import dash
import dash.dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output, State, dcc, html

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- CONFIG ---
TRADE_LOG = "live_trades_log.csv"
SETTINGS_FILE = "settings.json"
SYMBOLS = ["XAUUSD", "GBPUSD", "USDJPY", "EURUSD"]
TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]


# --- DATA ACCESS LAYER ---
def read_settings():
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE) as f:
                return json.load(f)
    except Exception as e:
        logging.warning(f"Failed to read settings: {e}")
    return {"symbol": "XAUUSD", "timeframe": "M15", "enabled": True, "volume": 0.1}


def write_settings(settings):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
        logging.info("Settings updated.")
    except Exception as e:
        logging.error(f"Failed to write settings: {e}")


def load_trade_log():
    if os.path.exists(TRADE_LOG):
        df = pd.read_csv(TRADE_LOG)
        # Parse datetime columns if present
        for col in ['timestamp', 'time', 'entry_time']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        if (
            "profit" not in df.columns
            and "exit_price" in df.columns
            and "entry_price" in df.columns
        ):
            df["profit"] = df["exit_price"] - df["entry_price"]
        if "balance" not in df.columns:
            df["balance"] = 100_000 + df.get("profit", 0).cumsum()
        df["profit"] = pd.to_numeric(df["profit"], errors="coerce").fillna(0)
        df["balance"] = pd.to_numeric(df["balance"], errors="coerce").ffill().fillna(100_000)
        for col in df.columns:
            df[col] = df[col].fillna("")
        return df
    return pd.DataFrame([])


# --- DASH APP ---
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY, dbc.icons.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "MR BEN AI Trading Dashboard"
settings = read_settings()


# --- UI COMPONENTS ---
def settings_panel(settings):
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Trade Settings", className="card-title text-primary"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Symbol:"),
                                dcc.Dropdown(
                                    SYMBOLS,
                                    value=settings.get("symbol", "XAUUSD"),
                                    id="symbol-select",
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                html.Label("Timeframe:"),
                                dcc.Dropdown(
                                    TIMEFRAMES,
                                    value=settings.get("timeframe", "M15"),
                                    id="timeframe-select",
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                html.Label("Auto Trading:"),
                                dcc.RadioItems(
                                    options=[
                                        {"label": "ON", "value": True},
                                        {"label": "OFF", "value": False},
                                    ],
                                    value=settings.get("enabled", True),
                                    id="autotrade-switch",
                                    inline=True,
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                html.Label("Volume:"),
                                dcc.Input(
                                    type="number",
                                    value=settings.get("volume", 0.1),
                                    id="volume-input",
                                    min=0.01,
                                    step=0.01,
                                    style={"width": "100%"},
                                ),
                            ],
                            width=3,
                        ),
                    ],
                    align="center",
                    className="mb-2",
                ),
                dbc.Button(
                    "Apply Settings", id="apply-settings", color="primary", className="me-2"
                ),
                html.Span(id="save-msg"),
            ]
        ),
        className="mb-4 shadow",
    )


def stats_panel():
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Account Overview", className="card-title text-success"),
                html.Div(id="account-info"),
                html.Div(id="account-stats"),
            ]
        ),
        className="mb-4 shadow",
    )


def trade_log_table():
    return dash.dash_table.DataTable(
        id="trade-log-table",
        page_size=10,
        style_table={"overflowX": "auto"},
        style_cell={
            "fontFamily": "monospace",
            "fontSize": 13,
            "backgroundColor": "#2b2b2b",
            "color": "#f2f2f2",
        },
        style_header={"fontWeight": "bold", "backgroundColor": "#222"},
        sort_action="native",
        filter_action="native",
        export_format="csv",
    )


def profit_chart():
    return dcc.Graph(id="profit-curve", config={"displayModeBar": True})


# --- MAIN LAYOUT ---
app.layout = dbc.Container(
    [
        html.H1("MR BEN AI Trading Dashboard", className="mb-3 mt-2 text-primary"),
        settings_panel(settings),
        html.Hr(className="mb-4"),
        dbc.Row(
            [
                dbc.Col(stats_panel(), width=4),
                dbc.Col(profit_chart(), width=8),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H5("Latest Trade", className="text-info"),
                        html.Div(id="latest-trade"),
                    ],
                    width=5,
                ),
                dbc.Col(
                    [
                        html.H5("Trade Log (Last 50)", className="text-info"),
                        dcc.Loading(trade_log_table(), color="primary"),
                    ],
                    width=7,
                ),
            ]
        ),
        dcc.Interval(id="refresh-interval", interval=60 * 1000, n_intervals=0),
        html.Br(),
        html.Footer(
            [
                html.Small(
                    [
                        "Made with ",
                        html.I(className="bi bi-lightning-fill text-warning"),
                        " by MR BEN Team · ",
                        html.A(
                            "GitHub",
                            href="https://github.com/",
                            target="_blank",
                            className="text-info",
                        ),
                    ]
                )
            ],
            className="text-center text-muted mt-5 mb-2",
        ),
    ],
    fluid=True,
)


# --- CALLBACKS ---
@app.callback(
    Output("save-msg", "children"),
    Output("symbol-select", "value"),
    Output("timeframe-select", "value"),
    Output("autotrade-switch", "value"),
    Output("volume-input", "value"),
    Input("apply-settings", "n_clicks"),
    State("symbol-select", "value"),
    State("timeframe-select", "value"),
    State("autotrade-switch", "value"),
    State("volume-input", "value"),
    prevent_initial_call=True,
)
def save_settings(n, symbol, timeframe, enabled, volume):
    try:
        settings = {"symbol": symbol, "timeframe": timeframe, "enabled": enabled, "volume": volume}
        write_settings(settings)
        return (
            dbc.Alert("Settings updated!", color="success", duration=2500),
            symbol,
            timeframe,
            enabled,
            volume,
        )
    except Exception as e:
        logging.error(f"Settings update failed: {e}")
        return (
            dbc.Alert(f"Failed to update settings: {e}", color="danger", duration=4000),
            symbol,
            timeframe,
            enabled,
            volume,
        )


@app.callback(
    Output("account-info", "children"),
    Output("account-stats", "children"),
    Output("profit-curve", "figure"),
    Output("latest-trade", "children"),
    Output("trade-log-table", "data"),
    Output("trade-log-table", "columns"),
    Input("refresh-interval", "n_intervals"),
)
def update_dashboard(n):
    df = load_trade_log()
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No trade data available", template="plotly_dark")
        return "No trades yet.", "", fig, "No trades.", [], []
    try:
        last = df.iloc[-1]
        bal = last.get("balance", "-")
        profit = (
            float(last.get("balance", 0)) - float(df["balance"].iloc[0])
            if "balance" in df.columns
            else "-"
        )
        info = [
            html.P(f"Balance: {bal:,.2f}" if isinstance(bal, (float, int)) else f"Balance: {bal}"),
            html.P(f"Net Profit: {profit:,.2f}"),
            html.P(f"Total Trades: {len(df)}"),
        ]
        # Advanced Stats
        wins = df[df["profit"] > 0].shape[0]
        losses = df[df["profit"] < 0].shape[0]
        win_rate = (wins / len(df)) * 100 if len(df) > 0 else 0
        avg_profit = df["profit"].mean() if "profit" in df.columns else 0
        max_drawdown = (
            (df["balance"].cummax() - df["balance"]).max() if "balance" in df.columns else 0
        )
        stats = [
            html.P(f"Win Rate: {win_rate:.1f}%"),
            html.P(f"Avg Profit: {avg_profit:.2f}"),
            html.P(f"Max Drawdown: {max_drawdown:.2f}"),
            html.P(f"Wins: {wins} | Losses: {losses}"),
        ]
        # Profit Curve
        x_axis = (
            df["timestamp"]
            if "timestamp" in df.columns
            else (
                df["entry_time"]
                if "entry_time" in df.columns
                else df["time"] if "time" in df.columns else df.index
            )
        )
        fig = go.Figure()
        if "balance" in df.columns:
            fig.add_trace(
                go.Scatter(x=x_axis, y=df["balance"], mode="lines+markers", name="Balance")
            )
        if "profit" in df.columns:
            fig.add_trace(go.Bar(x=x_axis, y=df["profit"], name="Profit", opacity=0.3))
        fig.update_layout(
            height=350,
            template="plotly_dark",
            margin=dict(t=40, l=10, r=10, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        # Latest Trade
        trade = [html.P(f"{col}: {last[col]}") for col in df.columns]
        latest_trade = html.Div(trade)
        # Trade Log (Last 50)
        last50 = df.tail(50)
        data = last50.to_dict("records")
        columns = [{"name": c, "id": c} for c in last50.columns]
        return info, stats, fig, latest_trade, data, columns
    except Exception as e:
        logging.error(f"Dashboard update error: {e}")
        fig = go.Figure()
        fig.update_layout(title=f"Data Error: {e}", template="plotly_dark")
        return [f"Data Error: {e}"], "", fig, f"Data Error: {e}", [], []


# --- MAIN ENTRY ---
if __name__ == "__main__":
    app.run(debug=True, port=8050)
