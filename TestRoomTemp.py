import dash
from dash import dcc, html, ctx, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import plotly.colors
import pandas as pd
from datetime import time, date
import numpy as np
import os
import glob
import dash_bootstrap_components as dbc
import colorsys

def adjust_color_brightness(hex_color, factor=0.4):
    """
    Lightens a hex color by a factor.
    Factor > 0 will lighten, < 0 will darken.
    For example, factor=0.5 will make it 50% lighter.
    """
    try:
        # Convert hex to RGB (0-255)
        h = hex_color.lstrip('#')
        r, g, b = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        # Convert RGB to HLS (0-1 scale)
        h, l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
        
        # Adjust lightness
        if factor > 0:
            l = l + (1 - l) * factor
        else:
            l = l + l * factor # Use a different formula for darkening
            
        # Convert back to RGB
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        # Convert back to hex
        return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
    except:
        # Fallback in case of an error
        return hex_color

# --- Fast Data Loading from Feather ---
try:
    df_original = pd.read_feather('cleaned_data.feather')
    df_original.set_index('datetime', inplace=True)
except FileNotFoundError:
    print("Error: 'cleaned_data.feather' not found.")
    print("Please run the 'preprocess_data.py' script first to generate it.")
    df_original = pd.DataFrame()


# --- DYNAMICALLY FIND AVAILABLE TEMPERATURE COLUMNS ---
all_possible_sensors = [
    'T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08',
    'T09', 'T10', 'T11', 'T12', 'L-Env', 'R-Env'
]

temperature_columns = []
if not df_original.empty:
    temperature_columns = [col for col in all_possible_sensors if col in df_original.columns]

if not df_original.empty:
    max_date = df_original.index.max().date()
    # Set the start date to the first day of the month of the latest data point
    start_date_default = max_date.replace(day=1)
    end_date_default = max_date
else:
    start_date_default = None
    end_date_default = None

# --- 2. Initialize the Dash App with Bootstrap Theme ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.title = 'Temperature Logger Data'

# --- 3. Define the App Layout ---
app.layout = html.Div(children=[
    dcc.Store(id='removed-rows-storage', data=[]),
    dcc.Store(id='event-storage', data=[], storage_type='local'),
    dcc.Store(id='mode-storage', data='pan'),
    dcc.Store(id='hvac-filter-storage', data='ALL'),
    
    html.H1(
        children='Temperature Logger Analysis',
        style={'textAlign': 'center', 'color': '#1f77b4', 'paddingTop': '20px'}
    ),
    html.Div(id='no-data-message', children=[
        dbc.Alert("No data found. Please run the preprocess_data.py script.", color="danger")
    ]) if df_original.empty else html.Div(),
    
    html.Div(id='controls-and-graph', children=[
        html.Div(style={
            'textAlign': 'center', 'marginBottom': '20px'
        }),
        # Filter controls
        html.Div([
            html.Div([
                html.Label('Select Date Range:'),
                dcc.DatePickerRange(
                    id='date-picker-range',
                    min_date_allowed=df_original.index.min().date() if not df_original.empty else None,
                    max_date_allowed=df_original.index.max().date() if not df_original.empty else None,
                    initial_visible_month=start_date_default,
                    start_date=start_date_default,
                    end_date=end_date_default,
                    style={'marginLeft': '10px'}
                ),
            ], style={'display': 'inline-block', 'marginRight': '40px', 'verticalAlign': 'middle'}),
            
            html.Div([
                html.Label('Filter by HVAC Schedule:', style={'marginRight': '10px'}),
                dbc.ButtonGroup([
                    dbc.Button("Show All", id="hvac-all-btn", outline=False, color="primary"),
                    dbc.Button("HVAC On", id="hvac-on-btn", outline=True, color="primary"),
                    dbc.Button("HVAC Off", id="hvac-off-btn", outline=True, color="primary")
                ])
            ], style={'display': 'inline-block', 'verticalAlign': 'middle'}),
        ], style={'textAlign': 'center', 'marginBottom': '20px'}),
        html.Div([
            html.Label('Select Temperature Sensors:', style={'fontSize': '18px'}),
            dcc.Checklist(
                id='temperature-checklist',
                options=[{'label': col, 'value': col} for col in temperature_columns],
                value=['T01','T02','T03','T04','T05','T06','T07','T08','T09','T10','T11','T12'],
                inline=True,
                style={'padding': '10px'},
                labelStyle={'display': 'inline-block', 'marginRight': '10px'}
            )
        ], style={'textAlign': 'center', 'marginBottom': '5px'}),
        
        dcc.Graph(
            id='temperature-graph',
            style={'height': '70vh'},
            config={
                'scrollZoom': True,
                'displaylogo': False
            }
        ),

        html.Div([
            html.Div([
                html.Label("Interaction Mode:", style={'marginRight': '10px'}),
                dbc.ButtonGroup([
                    dbc.Button("Pan/Zoom", id="pan-mode-btn", outline=False, color="primary"),
                    dbc.Button("Delete Point", id="delete-mode-btn", outline=True, color="primary")
                ])
            ], style={'display': 'inline-block', 'marginRight': '40px'}),
            html.Div(
                dbc.Button("Reset Graph Data", id="reset-button", color="danger"),
                style={'display': 'inline-block'}
            ),
        ], style={'textAlign': 'center', 'marginTop': '20px'}),

        html.Div([
            html.Label("Log an Event:", style={'fontWeight': 'bold', 'marginRight': '15px'}),
            dcc.DatePickerSingle(id='event-date-picker', date=date.today(), style={'marginRight': '5px'}),
            dcc.Input(id='event-time-input', type='text', placeholder='HH:MM:SS', style={'marginRight': '5px'}),
            dcc.Input(id='event-description-input', type='text', placeholder='Event description...', style={'width': '300px', 'marginRight': '5px'}),
            dbc.Button("Add Event", id="add-event-btn", color="success", n_clicks=0)
        ], style={'textAlign': 'center', 'marginTop': '20px', 'padding': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px'}),
        
        html.Hr(),
        html.H2("Statistical Summary", style={'textAlign': 'center', 'marginTop': '40px'}),
        dash_table.DataTable(
            id='stats-table',
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '5px'},
            style_header={'fontWeight': 'bold', 'border': '1px solid black'},
            style_data={'border': '1px solid grey'},
        ),

        html.H2("Event Log", style={'textAlign': 'center', 'marginTop': '40px'}),
        dash_table.DataTable(
            id='event-log-table',
            style_table={'overflowX': 'auto', 'marginTop': '20px'},
            style_cell={'textAlign': 'left', 'padding': '5px'},
            style_header={'fontWeight': 'bold', 'border': '1px solid black'},
            style_data={'border': '1px solid grey'},
        )

    ]) if not df_original.empty else html.Div()
], style={'padding': '20px'})

# --- 4. Define Callbacks ---

# Callbacks for Interaction Mode buttons
@app.callback(
    Output('mode-storage', 'data'),
    Input('pan-mode-btn', 'n_clicks'),
    Input('delete-mode-btn', 'n_clicks'),
    prevent_initial_call=True
)
def update_mode_storage(pan_clicks, delete_clicks):
    if ctx.triggered_id == 'pan-mode-btn':
        return 'pan'
    return 'delete'

@app.callback(
    Output('pan-mode-btn', 'outline'),
    Output('delete-mode-btn', 'outline'),
    Input('mode-storage', 'data')
)
def update_mode_button_styles(active_mode):
    if active_mode == 'pan':
        return False, True
    return True, False

# Callbacks for HVAC filter buttons
@app.callback(
    Output('hvac-filter-storage', 'data'),
    Input('hvac-all-btn', 'n_clicks'),
    Input('hvac-on-btn', 'n_clicks'),
    Input('hvac-off-btn', 'n_clicks'),
    prevent_initial_call=True
)
def update_hvac_storage(all_clicks, on_clicks, off_clicks):
    if ctx.triggered_id == 'hvac-on-btn':
        return 'ON'
    if ctx.triggered_id == 'hvac-off-btn':
        return 'OFF'
    return 'ALL'

@app.callback(
    Output('hvac-all-btn', 'outline'),
    Output('hvac-on-btn', 'outline'),
    Output('hvac-off-btn', 'outline'),
    Input('hvac-filter-storage', 'data')
)
def update_hvac_button_styles(active_filter):
    if active_filter == 'ON':
        return True, False, True
    if active_filter == 'OFF':
        return True, True, False
    return False, True, True
    
@app.callback(
    Output('event-storage', 'data'),
    Input('add-event-btn', 'n_clicks'),
    State('event-date-picker', 'date'),
    State('event-time-input', 'value'),
    State('event-description-input', 'value'),
    State('event-storage', 'data'),
    prevent_initial_call=True
)
def store_event_data(n_clicks, event_date, event_time, description, existing_events):
    if not all([event_date, event_time, description]):
        raise PreventUpdate # Don't do anything if inputs are empty

    # Combine date and time strings into a single datetime object
    datetime_str = f"{event_date} {event_time}"
    event_datetime = pd.to_datetime(datetime_str)
    
    # Add the new event to the list of existing events
    existing_events.append({
        'datetime': event_datetime.isoformat(),
        'description': description
    })
    
    return existing_events
    
# --- Main callback for the graph ---
@app.callback(
    Output('temperature-graph', 'figure'),
    Output('removed-rows-storage', 'data'),
    Output('stats-table', 'data'),      
    Output('stats-table', 'columns'),
    Output('event-log-table', 'data'),  
    Output('event-log-table', 'columns'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date'),
    Input('hvac-filter-storage', 'data'),
    Input('temperature-checklist', 'value'),
    Input('temperature-graph', 'selectedData'),
    Input('temperature-graph', 'clickData'),
    Input('reset-button', 'n_clicks'),
    Input('mode-storage', 'data'),
    Input('event-storage', 'data'),
    State('removed-rows-storage', 'data')
)
def update_graph(start_date, end_date, hvac_status, selected_columns, selectedData, clickData, reset_clicks, current_mode, events, removed_indices):
    triggered_id = ctx.triggered_id

    if triggered_id == 'reset-button':
        removed_indices = []

    if triggered_id == 'temperature-graph' and selectedData and 'points' in selectedData:
        for point in selectedData['points']:
            timestamp = pd.to_datetime(point['x'])
            if timestamp not in removed_indices:
                removed_indices.append(timestamp)

    if triggered_id == 'temperature-graph' and clickData and current_mode == 'delete':
        point = clickData['points'][0]
        timestamp = pd.to_datetime(point['x'])
        if timestamp not in removed_indices:
            removed_indices.append(timestamp)
            
    if not all([start_date, end_date, selected_columns]):
        return go.Figure(), [], [], []
        
    df = df_original
    
    if removed_indices:
        indices_to_drop = pd.to_datetime(removed_indices)
        df = df.drop(indices_to_drop, errors='ignore')

    end_date_inclusive = pd.to_datetime(end_date) + pd.Timedelta(days=1)
    mask = (df.index >= start_date) & (df.index < end_date_inclusive)
    date_filtered_df = df.loc[mask]

    #Statistical Calcucation
    if not date_filtered_df.empty and selected_columns:
        stats_df = date_filtered_df[selected_columns].describe().T.reset_index()

        stats_df['Range'] = stats_df['max'] - stats_df['min'] #calculate range
        
        stats_df = stats_df.rename(columns={
            'index': 'Sensor',
            '25%': 'Q1',
            '50%': 'Q2 (Median)',
            '75%': 'Q3'})
        stats_df['IQR'] = stats_df['Q3'] - stats_df['Q1']
        cols = list(stats_df.columns)
        max_position = cols.index('max')
        cols.insert(max_position + 1, cols.pop(cols.index('Range')))
        q3_position = cols.index('Q3')
        cols.insert(q3_position + 1, cols.pop(cols.index('IQR')))
        stats_df = stats_df[cols]
        stats_df = stats_df.round(2) # Round values for cleaner display
        stats_data = stats_df.to_dict('records')
        stats_columns = [{"name": i, "id": i} for i in stats_df.columns]
    else:
        stats_data = []
        stats_columns = []

    if events:
        # Format the data for the table
        event_data = []
        for event in events:
            # Make the timestamp more human-readable
            dt_object = pd.to_datetime(event['datetime'])
            event_data.append({
                'Timestamp': dt_object.strftime('%Y-%m-%d %H:%M:%S'),
                'Description': event['description']
            })
        
        # Sort events by timestamp (newest first)
        event_data.sort(key=lambda x: x['Timestamp'], reverse=True)

        event_columns = [
            {"name": "Timestamp", "id": "Timestamp"},
            {"name": "Description", "id": "Description"},
        ]
    else:
        event_data = []
        event_columns = []
    
    # --- PERFORMANCE: Downsample for long date ranges ---
    df_to_plot = date_filtered_df # Default to the full filtered data
    start_date_obj = date.fromisoformat(start_date.split('T')[0])
    end_date_obj = date.fromisoformat(end_date.split('T')[0])
    duration_days = (end_date_obj - start_date_obj).days

    # Downsample if the range is ~4 months (120 days) or more
    if duration_days >= 120:
        DOWNSAMPLE_THRESHOLD = 20000
        if len(date_filtered_df) > DOWNSAMPLE_THRESHOLD:
            step = max(1, len(date_filtered_df) // DOWNSAMPLE_THRESHOLD)
            df_to_plot = date_filtered_df.iloc[::step]
            print(f"Downsampling data: showing 1 of every {step} points.")

    if not df_to_plot.empty:
        first_timestamp = df_to_plot.index[0]
        print("\n--- HVAC FILTER DEBUG ---")
        print(f"Sample Timestamp from Data: {first_timestamp}")
        print(f"Time component being used: {first_timestamp.time()}")
        print(f"Schedule start time for comparison: {time(6, 0)}")
        print(f"Is data time >= schedule start time? {(first_timestamp.time() >= time(6, 0))}")
        print(f"Is data day a weekday? {(first_timestamp.dayofweek <= 4)}")
        print("-------------------------\n")
    
    fig = go.Figure()

    weekday_mask = (df_to_plot.index.dayofweek <= 4) & (df_to_plot.index.time >= time(6, 0)) & (df_to_plot.index.time < time(18, 0))
    saturday_mask = (df_to_plot.index.dayofweek == 5) & (df_to_plot.index.time >= time(8, 0)) & (df_to_plot.index.time < time(13, 0))
    hvac_on_mask = weekday_mask | saturday_mask

    should_connect_gaps = (hvac_status == 'ALL')
    colors = plotly.colors.qualitative.Plotly + plotly.colors.qualitative.G10

    for i, col in enumerate(selected_columns):
        # Assign a color from the default palette, wrapping around if needed
        color = colors[i % len(colors)]
        y_values = df_to_plot[col].copy()
        if hvac_status == 'ON':
            y_values[~hvac_on_mask] = np.nan
        elif hvac_status == 'OFF':
            y_values[hvac_on_mask] = np.nan
        
        fig.add_trace(go.Scattergl(
            x=df_to_plot.index, 
            y=y_values, 
            mode='lines', 
            name=col,
            connectgaps=should_connect_gaps,
            legendgroup=col,
            line=dict(color=color)
        ))  
        # Create a clean dataframe for the regression calculation (no NaNs)
        trace_df = pd.DataFrame({'y': y_values}).dropna()
        if len(trace_df) > 1:
            trend_color = adjust_color_brightness(color, factor=-0.4) #changes trendline color to a lighter or darker shade    
            x_numeric = trace_df.index.astype(np.int64)  # Convert datetime index to a numeric format for np.polyfit
            m, b = np.polyfit(x_numeric, trace_df['y'], 1) # Calculate the slope (m) and intercept (b) of the trendline
            x_full_numeric = df_to_plot.index.astype(np.int64) # Calculate the y-values for the trendline over the entire plotted range
            y_trend = m * x_full_numeric + b
            
            fig.add_trace(go.Scattergl(
                x=df_to_plot.index,
                y=y_trend,
                mode='lines',
                name=f'{col} Trend',
                line=dict(dash='dash', color=trend_color),
                legendgroup=col,
                showlegend=False
            ))
            
    fig.update_layout(
        title='Temperature Readings Over Time',
        xaxis_title='Date and Time',
        yaxis_title='Temperature (°C)',
        legend_title='Sensors',
        transition_duration=100,
    )
    
    return fig, removed_indices, stats_data, stats_columns, event_data, event_columns

# --- 5. Run the App ---
if __name__ == '__main__':
    app.run(debug=False)