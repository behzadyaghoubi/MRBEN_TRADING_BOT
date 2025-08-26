"""
Web Dashboard for MR BEN Trading Bot.
"""

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback_context
import dash.dash_table
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json

from ..core.logger import get_logger
from ..config.settings import settings
from ..core.database import db_manager
from ..trading.trading_engine import trading_engine
from ..trading.position_manager import position_manager

logger = get_logger("web.dashboard")


def create_dashboard_app():
    """Create and configure the Dash dashboard application."""
    
    # Initialize Dash app
    app = dash.Dash(
        __name__, 
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
            dbc.icons.FONT_AWESOME
        ],
        suppress_callback_exceptions=True
    )
    
    app.title = "MR BEN AI Trading Dashboard"
    app.layout = create_dashboard_layout()
    
    # Register callbacks
    register_dashboard_callbacks(app)
    
    return app


def create_dashboard_layout():
    """Create the main dashboard layout."""
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1([
                    html.I(className="fas fa-chart-line me-2"),
                    "MR BEN AI Trading Dashboard"
                ], className="text-primary mb-3 mt-2"),
                html.Hr()
            ])
        ]),
        
        # Control Panel
        dbc.Row([
            dbc.Col([
                html.Label("Symbol:", className="fw-bold"),
                dcc.Dropdown(
                    id="symbol-select",
                    options=[
                        {"label": "XAUUSD", "value": "XAUUSD"},
                        {"label": "GBPUSD", "value": "GBPUSD"},
                        {"label": "USDJPY", "value": "USDJPY"},
                        {"label": "EURUSD", "value": "EURUSD"}
                    ],
                    value=settings.trading.symbol,
                    clearable=False
                )
            ], width=2),
            dbc.Col([
                html.Label("Timeframe:", className="fw-bold"),
                dcc.Dropdown(
                    id="timeframe-select",
                    options=[
                        {"label": "M1", "value": "M1"},
                        {"label": "M5", "value": "M5"},
                        {"label": "M15", "value": "M15"},
                        {"label": "M30", "value": "M30"},
                        {"label": "H1", "value": "H1"},
                        {"label": "H4", "value": "H4"},
                        {"label": "D1", "value": "D1"}
                    ],
                    value=settings.trading.timeframe,
                    clearable=False
                )
            ], width=2),
            dbc.Col([
                html.Label("Auto Trading:", className="fw-bold"),
                dcc.RadioItems(
                    id="autotrade-switch",
                    options=[
                        {"label": "ON", "value": True},
                        {"label": "OFF", "value": False}
                    ],
                    value=settings.trading.enabled,
                    inline=True,
                    className="mt-2"
                )
            ], width=2),
            dbc.Col([
                html.Br(),
                dbc.Button(
                    [html.I(className="fas fa-save me-1"), "Apply Settings"],
                    id="apply-settings",
                    color="primary",
                    size="sm"
                ),
                html.Div(id="save-msg", className="mt-2")
            ], width=2),
            dbc.Col([
                html.Br(),
                dbc.Button(
                    [html.I(className="fas fa-sync me-1"), "Refresh"],
                    id="refresh-btn",
                    color="secondary",
                    size="sm"
                )
            ], width=2)
        ], className="mb-4", align="center"),
        
        # Status Cards
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Account Balance", className="card-title text-success"),
                        html.H2(id="balance-display", className="text-success"),
                        html.P(id="profit-display", className="text-muted")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Open Positions", className="card-title text-info"),
                        html.H2(id="positions-display", className="text-info"),
                        html.P(id="exposure-display", className="text-muted")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Total Trades", className="card-title text-warning"),
                        html.H2(id="trades-display", className="text-warning"),
                        html.P(id="winrate-display", className="text-muted")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("AI Status", className="card-title text-primary"),
                        html.H2(id="ai-status-display", className="text-primary"),
                        html.P(id="ai-confidence-display", className="text-muted")
                    ])
                ])
            ], width=3)
        ], className="mb-4"),
        
        # Charts Row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Equity Curve", className="mb-0")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(
                            id="equity-chart",
                            config={"displayModeBar": True, "displaylogo": False}
                        )
                    ])
                ])
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Trade Distribution", className="mb-0")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(
                            id="trade-distribution-chart",
                            config={"displayModeBar": False, "displaylogo": False}
                        )
                    ])
                ])
            ], width=4)
        ], className="mb-4"),
        
        # Tables Row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Latest Trades", className="mb-0")
                    ]),
                    dbc.CardBody([
                        dcc.Loading([
                            dash.dash_table.DataTable(
                                id="trades-table",
                                page_size=10,
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "fontFamily": "monospace",
                                    "fontSize": 12,
                                    "textAlign": "center"
                                },
                                style_header={
                                    "fontWeight": "bold",
                                    "backgroundColor": "#f8f9fa"
                                },
                                style_data_conditional=[
                                    {
                                        "if": {"column_id": "profit", "filter_query": "{profit} > 0"},
                                        "color": "green",
                                        "fontWeight": "bold"
                                    },
                                    {
                                        "if": {"column_id": "profit", "filter_query": "{profit} < 0"},
                                        "color": "red",
                                        "fontWeight": "bold"
                                    }
                                ],
                                sort_action="native",
                                filter_action="native"
                            )
                        ])
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Open Positions", className="mb-0")
                    ]),
                    dbc.CardBody([
                        dcc.Loading([
                            dash.dash_table.DataTable(
                                id="positions-table",
                                page_size=5,
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "fontFamily": "monospace",
                                    "fontSize": 12,
                                    "textAlign": "center"
                                },
                                style_header={
                                    "fontWeight": "bold",
                                    "backgroundColor": "#f8f9fa"
                                },
                                style_data_conditional=[
                                    {
                                        "if": {"column_id": "unrealized_profit", "filter_query": "{unrealized_profit} > 0"},
                                        "color": "green",
                                        "fontWeight": "bold"
                                    },
                                    {
                                        "if": {"column_id": "unrealized_profit", "filter_query": "{unrealized_profit} < 0"},
                                        "color": "red",
                                        "fontWeight": "bold"
                                    }
                                ]
                            )
                        ])
                    ])
                ])
            ], width=6)
        ], className="mb-4"),
        
        # Auto-refresh interval
        dcc.Interval(
            id="refresh-interval",
            interval=30 * 1000,  # 30 seconds
            n_intervals=0
        ),
        
        # Store for settings
        dcc.Store(id="settings-store"),
        
        # Footer
        dbc.Row([
            dbc.Col([
                html.Hr(),
                html.P([
                    "MR BEN AI Trading Bot v2.0 | ",
                    html.Small("Powered by Dash & Plotly")
                ], className="text-muted text-center")
            ])
        ])
        
    ], fluid=True)


def register_dashboard_callbacks(app):
    """Register all dashboard callbacks."""
    
    @app.callback(
        Output("save-msg", "children"),
        Output("symbol-select", "value"),
        Output("timeframe-select", "value"),
        Output("autotrade-switch", "value"),
        Input("apply-settings", "n_clicks"),
        State("symbol-select", "value"),
        State("timeframe-select", "value"),
        State("autotrade-switch", "value"),
        prevent_initial_call=True
    )
    def save_settings(n_clicks, symbol, timeframe, enabled):
        """Save trading settings."""
        if n_clicks:
            try:
                # Update settings
                settings.update_trading_config(
                    symbol=symbol,
                    timeframe=timeframe,
                    enabled=enabled
                )
                settings.save_settings()
                
                return (
                    dbc.Alert("Settings updated successfully!", color="success", duration=3000),
                    symbol, timeframe, enabled
                )
            except Exception as e:
                logger.error(f"Error saving settings: {e}")
                return (
                    dbc.Alert(f"Error saving settings: {e}", color="danger", duration=3000),
                    symbol, timeframe, enabled
                )
        
        return "", symbol, timeframe, enabled
    
    @app.callback(
        Output("balance-display", "children"),
        Output("profit-display", "children"),
        Output("positions-display", "children"),
        Output("exposure-display", "children"),
        Output("trades-display", "children"),
        Output("winrate-display", "children"),
        Output("ai-status-display", "children"),
        Output("ai-confidence-display", "children"),
        Input("refresh-interval", "n_intervals"),
        Input("refresh-btn", "n_clicks")
    )
    def update_status_cards(n_intervals, n_clicks):
        """Update status cards with current data."""
        try:
            # Get account info
            account_info = trading_engine.executor.get_account_info()
            if account_info:
                balance = f"${account_info['balance']:,.2f}"
                profit = f"PnL: ${account_info['profit']:,.2f}"
            else:
                balance = "N/A"
                profit = "N/A"
            
            # Get positions info
            positions = position_manager.get_open_positions()
            positions_count = len(positions)
            exposure = position_manager.get_total_exposure()
            exposure_text = f"Volume: {exposure['total_volume']:.2f}"
            
            # Get trades info
            performance = trading_engine.get_performance_summary()
            total_trades = performance.get('total_trades', 0)
            win_rate = f"Win Rate: {performance.get('win_rate', 0):.1%}"
            
            # Get AI status
            ai_info = ai_filter.get_model_info()
            if ai_info['model_loaded']:
                ai_status = "Active"
                ai_confidence = f"Model: {ai_info['model_type']}"
            else:
                ai_status = "Fallback"
                ai_confidence = "Using rule-based logic"
            
            return (
                balance, profit,
                positions_count, exposure_text,
                total_trades, win_rate,
                ai_status, ai_confidence
            )
            
        except Exception as e:
            logger.error(f"Error updating status cards: {e}")
            return "Error", "Error", "Error", "Error", "Error", "Error", "Error", "Error"
    
    @app.callback(
        Output("equity-chart", "figure"),
        Input("refresh-interval", "n_intervals"),
        Input("refresh-btn", "n_clicks")
    )
    def update_equity_chart(n_intervals, n_clicks):
        """Update equity curve chart."""
        try:
            # Get trades data
            trades_df = db_manager.get_trades()
            
            if trades_df.empty:
                return create_empty_chart("No trade data available")
            
            # Calculate equity curve
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            trades_df = trades_df.sort_values('timestamp')
            
            # Calculate cumulative profit
            trades_df['cumulative_profit'] = trades_df['profit'].fillna(0).cumsum()
            trades_df['equity'] = settings.trading.start_balance + trades_df['cumulative_profit']
            
            # Create chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=trades_df['timestamp'],
                y=trades_df['equity'],
                mode='lines+markers',
                name='Equity',
                line=dict(color='#28a745', width=2),
                marker=dict(size=4)
            ))
            
            fig.update_layout(
                title="Account Equity Curve",
                xaxis_title="Time",
                yaxis_title="Equity ($)",
                template="plotly_white",
                height=400,
                margin=dict(l=50, r=50, t=50, b=50),
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error updating equity chart: {e}")
            return create_empty_chart("Error loading chart")
    
    @app.callback(
        Output("trade-distribution-chart", "figure"),
        Input("refresh-interval", "n_intervals"),
        Input("refresh-btn", "n_clicks")
    )
    def update_trade_distribution_chart(n_intervals, n_clicks):
        """Update trade distribution chart."""
        try:
            # Get trades data
            trades_df = db_manager.get_trades()
            
            if trades_df.empty:
                return create_empty_chart("No trade data available")
            
            # Count trades by action
            action_counts = trades_df['action'].value_counts()
            
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=action_counts.index,
                values=action_counts.values,
                hole=0.3,
                marker_colors=['#28a745', '#dc3545', '#ffc107']
            )])
            
            fig.update_layout(
                title="Trade Distribution",
                template="plotly_white",
                height=400,
                margin=dict(l=20, r=20, t=50, b=20),
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error updating trade distribution chart: {e}")
            return create_empty_chart("Error loading chart")
    
    @app.callback(
        Output("trades-table", "data"),
        Output("trades-table", "columns"),
        Input("refresh-interval", "n_intervals"),
        Input("refresh-btn", "n_clicks")
    )
    def update_trades_table(n_intervals, n_clicks):
        """Update trades table."""
        try:
            # Get recent trades
            trades_df = db_manager.get_trades(limit=50)
            
            if trades_df.empty:
                return [], []
            
            # Format data for table
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            trades_df['profit'] = trades_df['profit'].fillna(0).round(2)
            trades_df['entry_price'] = trades_df['entry_price'].round(5)
            
            # Select columns for display
            display_columns = [
                'timestamp', 'symbol', 'action', 'entry_price', 
                'lot_size', 'profit', 'status'
            ]
            
            data = trades_df[display_columns].to_dict('records')
            columns = [
                {"name": col.replace('_', ' ').title(), "id": col}
                for col in display_columns
            ]
            
            return data, columns
            
        except Exception as e:
            logger.error(f"Error updating trades table: {e}")
            return [], []
    
    @app.callback(
        Output("positions-table", "data"),
        Output("positions-table", "columns"),
        Input("refresh-interval", "n_intervals"),
        Input("refresh-btn", "n_clicks")
    )
    def update_positions_table(n_intervals, n_clicks):
        """Update positions table."""
        try:
            # Get open positions
            positions = position_manager.get_open_positions()
            
            if not positions:
                return [], []
            
            # Format data for table
            for pos in positions:
                pos['time'] = pos['time'].strftime('%Y-%m-%d %H:%M')
                pos['price_open'] = round(pos['price_open'], 5)
                pos['price_current'] = round(pos['price_current'], 5)
                pos['unrealized_profit'] = round(pos.get('unrealized_profit', 0), 2)
                pos['volume'] = round(pos['volume'], 2)
            
            # Select columns for display
            display_columns = [
                'ticket', 'symbol', 'type', 'volume', 'price_open',
                'price_current', 'unrealized_profit', 'time'
            ]
            
            data = [{col: pos[col] for col in display_columns} for pos in positions]
            columns = [
                {"name": col.replace('_', ' ').title(), "id": col}
                for col in display_columns
            ]
            
            return data, columns
            
        except Exception as e:
            logger.error(f"Error updating positions table: {e}")
            return [], []


def create_empty_chart(message: str):
    """Create an empty chart with a message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray")
    )
    fig.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=50, r=50, t=50, b=50),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    return fig 