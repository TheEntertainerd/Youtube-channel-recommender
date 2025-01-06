# app.py
import dash
from dash import dcc, html, Input, Output, State, callback_context
import os
import dash_ag_grid as dag
import requests

styles = {
    'container': {
        'width': '90%',
        'max-width': '1200px',
        'margin': '0 auto',
        'padding': '20px',
        'font-family': 'Roboto, sans-serif',
        'background-color': '#f4f4f4',
    },
    'header': {
        'backgroundColor': 'linear-gradient(90deg, #4CAF50 0%, #66BB6A 100%)',
        'padding': '20px',
        'color': 'white',
        'margin-bottom': '20px',
        'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
        'border-radius': '10px',
        'text-align': 'center',
        'font-size': '1.5em',
        'text-shadow': '1px 1px 2px rgba(0, 0, 0, 0.2)',
    },
    'link': {
        'color': '#1E88E5',
        'text-decoration': 'none',
        'margin-right': '15px',
        'font-weight': 'bold',
        'font-size': '1em',
    },
    'link-hover': {
        'color': '#0D47A1',
        'text-decoration': 'underline',
    },
    'result-card': {
        'display': 'flex',
        'align-items': 'center',
        'padding': '15px',
        'border': '1px solid #ddd',
        'border-radius': '10px',
        'margin-bottom': '15px',
        'background-color': 'white',
        'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
        'transition': 'transform 0.2s ease-in-out',
        'cursor': 'pointer',
    },
    'result-card-hover': {
        'transform': 'scale(1.02)',
        'box-shadow': '0 6px 12px rgba(0, 0, 0, 0.2)',
    },
    'thumbnail': {
        'width': '100px',
        'height': '100px',
        'border-radius': '10%',
        'margin-right': '15px',
        'box-shadow': '0 2px 4px rgba(0, 0, 0, 0.1)',
    },
    'result-text': {
        'flex-grow': '1',
        'text-align': 'left',
    },
    'input-field': {
        'width': '100%',
        'padding': '10px',
        'margin': '5px 0',
        'border-radius': '5px',
        'border': '1px solid #ddd',
    },
    'button': {
        'padding': '10px 20px',
        'border-radius': '5px',
        'border': 'none',
        'background-color': '#4CAF50',
        'color': 'white',
        'cursor': 'pointer',
        'font-size': '1em',
        'transition': 'background-color 0.3s ease-in-out',
    },
    'button-hover': {
        'background-color': '#388E3C',
    },
}


FLASK_SERVER = os.getenv('FLASK_SERVER', 'http://localhost:5000')

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)


def create_filters(filter_data):
    return [
        html.Div([
            html.Label("Filter by Country:"),
            dcc.Dropdown(
                id='filter-country',
                options=filter_data['country_options'],
                multi=True,
                placeholder='Select Country'
            )
        ]),
        html.Div([
            html.Label("Filter by Subscribers:"),
            dcc.RangeSlider(
                id='filter-subscribers',
                min=filter_data['subscribers_min'],
                max=filter_data['subscribers_max'],
                step=1,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ]),
        html.Div([
            html.Label("Filter by Total Views:"),
            dcc.RangeSlider(
                id='filter-total-views',
                min=filter_data['total_views_min'],
                max=filter_data['total_views_max'],
                step=1,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ]),
        html.Div([
            html.Label("Filter by Videos per Week:"),
            dcc.RangeSlider(
                id='filter-videos-per-week',
                min=filter_data['videos_per_week_min'],
                max=filter_data['videos_per_week_max'],
                step=0.1,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ])
    ]

# Define the layout
app.layout = html.Div(style=styles['container'], children=[
    html.Div(style=styles['header'], children=[
        html.H1('YouTube Channel Recommender'),
        html.Div([
            dcc.Link('Home', href='/', style=styles['link']),
            dcc.Link('Channel Search & Similar Channels', href='/combined-search', style=styles['link']),
        ], id='nav-links')
    ]),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    dcc.Store(id='all-tags-store')
])


# Home page
home_layout = html.Div([
    html.H2('Welcome to the YouTube Channel Recommender'),
    html.P('Use the links above to navigate to different features of the app.')
])

# Create the combined layout for the page
def combined_page_layout(filter_data):
    return html.Div([
        html.H2('YouTube Channel Recommender'),
        html.Div(create_filters(filter_data)),
        html.Div([
            html.Label("Enter YouTube Channel Link:"),
            dcc.Input(id='channel-link-input', type='text', placeholder='Enter a YouTube channel link', style=styles['input-field']),
        ]),
        html.Div([
            html.Label("Select up to 2 YouTube Channels:"),
            dcc.Dropdown(
                id='channel-dropdown',
                options=[],  # This will be dynamically populated based on `channel-link-input`
                placeholder='Select up to 2 channels',
                multi=True,
                maxHeight=300,
                style=styles['input-field'],
            ),
            dcc.Store(id='selected-channels-store'),  # Store to keep track of selected channels
        ]),
        html.Div([
            html.Label("Enter a Search Query:"),
            dcc.Input(id='search-query-input', type='text', placeholder='Enter your search query', style=styles['input-field']),
        ]),
        html.Button('Find Channels', id='find-channels-button', n_clicks=0, style=styles['button']),
        html.Div(id='results-output'),
        html.Button('Show More', id='show-more-button', n_clicks=0, style={'display': 'none', **styles['button']}),
        dcc.Store(id='results-list'),
        dcc.Store(id='results-index', data=0)
    ])


@app.callback(
    Output('all-tags-store', 'data'),
    Input('url', 'pathname'),
    State('all-tags-store', 'data')  # Check the current state of the store
)
def load_filters(pathname, existing_filter_data):
    if pathname == '/combined-search':
        # If the filter data already exists, return it
        if existing_filter_data is not None:
            return existing_filter_data

        # Otherwise, fetch the filter data from the Flask service
        try:
            response = requests.get(f'{FLASK_SERVER}/filters')
            
            if response.status_code == 200:
                filter_data = response.json()  # Parse the JSON response

                return filter_data
            else:
                print(f"Error fetching filters: Received status code {response.status_code}")
                return dash.no_update

        except Exception as e:
            print(f"Error fetching filters: {e}")
            return dash.no_update

    # If not on the combined search page, do not update the store
    return dash.no_update



# Callback to update page content
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname'),
     Input('all-tags-store', 'data')],
    State('all-tags-store', 'data')  # Check the current state of the store
)
def display_page(pathname, filter_data, existing_filter_data):
    if pathname == '/combined-search':
        if filter_data is None and existing_filter_data is None:
            # If the filter data is not yet loaded, show a loading message
            return html.Div("Loading filters...")
        elif filter_data is None:
            # If the callback is triggered with no new filter data, use the existing data
            filter_data = existing_filter_data
        return combined_page_layout(filter_data)  # Pass filter_data directly
 
    elif pathname == '/data-viz':
        return html.Div("Data Visualization Placeholder")
            
    else:
        return home_layout


@app.callback(
    [Output('channel-dropdown', 'options'),
     Output('channel-dropdown', 'value')],
    [Input('channel-dropdown', 'search_value')],
    [State('channel-dropdown', 'value')]
)
def update_channel_options(search_value, selected_values):
    # Initialize selected_values as an empty list if it's None
    if selected_values is None:
        selected_values = []

    # Ensure selected_values are strings (id|name)
    selected_values = [value if isinstance(value, str) else f"{value[0]}|{value[1]}" for value in selected_values]

    # If the user has already selected 2 channels, prevent further selection
    if len(selected_values) >= 2:
        # Return only the currently selected options
        return [{'label': value.split('|')[1], 'value': value} for value in selected_values], dash.no_update

    if not search_value or len(search_value) < 3:
        # If no search value or less than 3 characters, return only existing selected options
        return [{'label': value.split('|')[1], 'value': value} for value in selected_values], dash.no_update

    try:
        # Fetch channel options matching the search prefix from the backend service
        response = requests.get(f'{FLASK_SERVER}/search_channels?prefix={search_value}')

        if response.status_code == 200:
            channel_options = response.json()  # Expected to be a list of dicts [{'label': ..., 'value': ...}, ...]

            # Convert the response to the expected format (value is now "id|name")
            channel_options = [{'label': opt['label'], 'value': f"{opt['value']}|{opt['label']}"} for opt in channel_options]

            # Filter out already selected values from the new options
            channel_options = [opt for opt in channel_options if opt['value'] not in selected_values]

            # Keep currently selected options in the dropdown and add new ones
            combined_options = [{'label': value.split('|')[1], 'value': value} for value in selected_values] + channel_options

            return combined_options, dash.no_update

        else:
            print(f"Error fetching channels: Received status code {response.status_code}")
            return [{'label': value.split('|')[1], 'value': value} for value in selected_values], dash.no_update

    except Exception as e:
        print(f"Error fetching channels: {e}")
        return [{'label': value.split('|')[1], 'value': value} for value in selected_values], dash.no_update

@app.callback(
    [Output('results-output', 'children'),
     Output('show-more-button', 'style'),
     Output('results-list', 'data'),
     Output('results-index', 'data')],
    [Input('find-channels-button', 'n_clicks'),
     Input('show-more-button', 'n_clicks')],
    [State('channel-link-input', 'value'),
     State('channel-dropdown', 'value'),
     State('search-query-input', 'value'),
     State('filter-country', 'value'),
     State('filter-subscribers', 'value'),
     State('filter-total-views', 'value'),
     State('filter-videos-per-week', 'value'),
     State('results-output', 'children'),
     State('results-list', 'data'),
     State('results-index', 'data')]
)
def find_channels(n_clicks, show_more_clicks, channel_link, selected_channels, search_query, 
                  country, subscribers, total_views, videos_per_week, 
                  current_output, results_list, results_index):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update, {'display': 'none'}, [], 0
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'find-channels-button' and n_clicks > 0:
        filters = {
            'country': country,
            'subscribers': subscribers,
            'total_views': total_views,
            'videos_per_week': videos_per_week,
        }

        # Ensure selected_channels is a list (even if empty)
        if selected_channels is None:
            selected_channels = []
        selected_channels = [channel.split('|')[0] for channel in selected_channels]

        selected_channel_ids = selected_channels 

        print(f"Selected Channel IDs: {selected_channel_ids}")

        payload = {
            'channel_link': channel_link,
            'selected_channels': selected_channel_ids,
            'search_query': search_query,
            'filters': filters
        }

        try:
            response = requests.post(f'{FLASK_SERVER}/query', json=payload)
            response_data = response.json()
        except Exception as e:
            return html.Div(f"An error occurred while querying the service: {e}"), {'display': 'none'}, [], 0
        
        if 'error' in response_data:
            return html.Div(f"Service error: {response_data['error']}"), {'display': 'none'}, [], 0

        if not response_data:
            return html.Div("No results found matching your query. Please try different search criteria."), {'display': 'none'}, [], 0

        high_quality_results = [res for res in response_data if res['score'] > 0.62]
        decent_quality_results = [res for res in response_data if 0.55 <= res['score'] <= 0.62]

        all_results = high_quality_results + decent_quality_results

        if high_quality_results:
            initial_results = high_quality_results[:3]
            return display_results(initial_results), {'display': 'block'} if len(all_results) > 3 else {'display': 'none'}, all_results, 3
        elif decent_quality_results:
            initial_results = decent_quality_results[:3]
            initial_results_display = display_results(initial_results)
            warning_message = html.Div("Results below this point may be less precise.")
            return [warning_message] + initial_results_display, {'display': 'block'} if len(all_results) > 3 else {'display': 'none'}, all_results, 3

    elif button_id == 'show-more-button' and show_more_clicks > 0:
        if results_list:
            next_results = results_list[results_index:results_index + 3]
            more_results = display_results(next_results)

            if results_index + 3 < len(results_list):
                if results_list[results_index + 3]['score'] <= 0.62 and results_list[results_index]['score'] > 0.62:
                    warning_message = html.Div("Results below this point may be less precise.")
                    return current_output + [warning_message] + more_results, {'display': 'block'}, results_list, results_index + 3
                return current_output + more_results, {'display': 'block'}, results_list, results_index + 3
            else:
                return current_output + more_results, {'display': 'none'}, results_list, results_index + 3
    
    return dash.no_update, {'display': 'none'}, [], 0

def display_results(results):
    """Helper function to format the results for display."""
    
    return [
        html.Div([
            html.Img(src=result['thumbnail'], style=styles['thumbnail'],**{'referrerPolicy': 'no-referrer'}),  # Use the thumbnail URL from the Flask response
            html.Div([
                html.H4(f"{result['title']}"),
                html.P(f"{result['description']}"),
                html.P(f"Similarity Score: {result['score']:.2f}"),
                html.A("Visit Channel", href=result['channel_link'], target="_blank")
            ], style=styles['result-text'])
        ], style=styles['result-card'])
        for result in results
    ]

if __name__ == '__main__':
    # Suppress default Dash server messages
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("\n=== YouTube Recommender ===")
    print("→ Dash UI: http://localhost:8050")
    print("→ Flask API: http://localhost:5000")
    print("Press CTRL+C to quit\n")
    
    app.run_server(
        debug=True,
        host='0.0.0.0',
        port=8050,
        use_reloader=True,
        dev_tools_hot_reload=True,
        dev_tools_ui=True,
        dev_tools_silence_routes_logging=True
    )