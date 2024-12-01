# Import all the libraries
import dash
from dash import dcc, html
from flask import Flask
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
from dash_bootstrap_templates import load_figure_template
from sklearn.preprocessing import LabelEncoder
import pickle
from dash import dash_table
import io
import base64
from PIL import Image



templates = [
    "LUX"
]

load_figure_template(templates)

# # Load the saved model
# with open('lr_yokyo_model', 'rb') as f:
#     lr = pickle.load(f)

# initialise the App
server = Flask(__name__)
# app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.LUX])

# read files
df = pd.read_csv("yokyo.log", sep=" ", header=None,
                 names=["Timestamp", "IP Address", "HTTP Method",
                        "Path", "Status Code","HTTP Version", "Traffic Source", "User Agent", "Country"],
                 usecols=["Timestamp", "IP Address", "HTTP Method",
                        "Path", "Status Code","HTTP Version","Traffic Source", "User Agent", "Country"])
# df = df.sample(frac=1)

    # Define aliases for User Agent and Path
user_agent_aliases = {
    # Desktop devices
    "Mozilla/5.0-(Windows-NT-10.0;-Win64;-x64)-AppleWebKit/537.36-(KHTML,-like-Gecko)-Chrome/90.0.4430.212-Safari/537.36": "Windows Chrome",
    "Mozilla/5.0-(Windows-NT-10.0;-Win64;-x64)-AppleWebKit/537.36-(KHTML,-like-Gecko)-Edge/90.0.818.62-Safari/537.36": "Windows Edge",
    "Mozilla/5.0-(Macintosh;-Intel-Mac_OS-X-10_15_7)-AppleWebKit/537.36-(KHTML,-like-Gecko)-Chrome/90.0.4430.212-Safari/537.36": "Mac Chrome",
    "Mozilla/5.0-(Macintosh;-Intel-Mac_OS-X-10_15_7)-AppleWebKit/537.36-(KHTML,-like-Gecko)-Version/14.1-Safari/537.36": "Mac Safari",
    "Mozilla/5.0-(Windows-NT-10.0;-Win64;-x64)-AppleWebKit/537.36-(KHTML,-like-Gecko)-Chrome/90.0.4430.93-YaBrowser/21.5.1.174-Yowser/2.5-Safari/537.36": "Windows Yandex",
    "Mozilla/5.0-(Windows-NT-10.0;-Win64;-x64;-rv:88.0)-Gecko/20100101-Firefox/88.0": "Windows Firefox",
    
    # Mobile Devices
    "Mozilla/5.0-(Linux;-Android-11;-SM-G975F)-AppleWebKit/537.36-(KHTML,-like-Gecko)-Chrome/88.0.4324.181-Mobile-Safari/537.36": ["Android 11 - Samsung Galaxy S10"],
    "Mozilla/5.0-(iPhone;-CPU-iPhone-OS-14_4-like-Mac-OS-X)-AppleWebKit/605.1.15-(KHTML,-like-Gecko)-Version/14.0.3-Mobile/15E148-Safari/604.1": ["iOS 14 - iPhone"],
    "Mozilla/5.0-(Linux;-Android-11;-SM-G986B)-AppleWebKit/537.36-(KHTML,-like-Gecko)-Chrome/90.0.4430.210-Mobile-Safari/537.36-EdgA/46.3.2.5155": ["Android 11 - Samsung Galaxy S20"],
    "Mozilla/5.0-(iPad;-CPU-OS-14_4-like-Mac-OS-X)-AppleWebKit/605.1.15-(KHTML,-like-Gecko)-Version/14.0-Mobile/15E148-Safari/604.1": ["iOS 14 - iPad"],
    "Mozilla/5.0-(Linux;-Android-11;-SM-A515F)-AppleWebKit/537.36-(KHTML,-like-Gecko)-Chrome/89.0.4389.105-Mobile-Safari/537.36": ["Android 11 - Samsung Galaxy A51"],

    # TV devices
    "Mozilla/5.0-(SMART-TV;-Linux;-Tizen-5.0)-AppleWebKit/537.36-(KHTML,-like-Gecko)-SamsungBrowser/2.2-Chrome/63.0.3239.84-TV-Safari/537.36": ["Samsung Smart TV - Tizen 5.0"],
    "Mozilla/5.0-(SMART-TV;-X11;-Linux-x86_64)-AppleWebKit/537.36-(KHTML,-like-Gecko)-Version/4.0-Chrome/78.0.3904.108-Safari/537.36": ["Smart TV - Linux x86_64"],
    "Mozilla/5.0-(SMART-TV;-Linux;-Tizen-3.0)-AppleWebKit/538.1-(KHTML,-like-Gecko)-Version/3.0-TV-Safari/538.1": ["Samsung Smart TV - Tizen 3.0"],
    "Mozilla/5.0-(SMART-TV;-Linux;-Tizen-5.5)-AppleWebKit/537.36-(KHTML,-like-Gecko)-SamsungBrowser/2.2-Chrome/63.0.3239.84-TV-Safari/537.36": ["Samsung Smart TV - Tizen 5.5"],
    "Mozilla/5.0-(SMART-TV;-X11;-Linux-armv7l)-AppleWebKit/537.42-(KHTML,-like-Gecko)-Safari/537.42": ["Smart TV - Linux armv7l"],

 }
    # Replace User Agent values with aliases
df["User Agent"] = df["User Agent"].replace(user_agent_aliases)

# Create a DataTable component to display the data
table = dash_table.DataTable(
    id='table',
    columns=[{"name": i, "id": i} for i in df.columns],
    data=df.to_dict('records'),
)

# Create a Modal component to display the table
modal = dbc.Modal(
    [
        dbc.ModalHeader("Yokyo Webserver Log Data"),
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

report_button =  dbc.Button("Report", id="report", color= "danger")

# @app.callback(
#     Output('report', 'children'),
#     Input('report', 'n_clicks'),
#     prevent_initial_call=True
# )
# def generate_report(n_clicks):
#     # Create in-memory binary streams for each chart
#     pie_stream = io.BytesIO()
#     hist_stream = io.BytesIO()
#     line_stream = io.BytesIO()

#     # Save static images of each chart to the corresponding stream
#     pie_fig.write_image(pie_stream, format='png')
#     hist_fig.write_image(hist_stream, format='png')
#     line_fig.write_image(line_stream, format='png')

#     # Reset the stream positions
#     pie_stream.seek(0)
#     hist_stream.seek(0)
#     line_stream.seek(0)

#     # Create Image objects from the streams
#     pie_image = Image.open(pie_stream)
#     hist_image = Image.open(hist_stream)
#     line_image = Image.open(line_stream)

#     # Calculate the dimensions of the final image
#     width = max(pie_image.width, hist_image.width, line_image.width)
#     height = pie_image.height + hist_image.height + line_image.height

#     # Create a new Image object with the calculated dimensions
#     report_image = Image.new('RGB', (width, height))

#     # Paste the chart images into the report image
#     y_offset = 0
#     for image in [pie_image, hist_image, line_image]:
#         report_image.paste(image, (0, y_offset))
#         y_offset += image.height

#     # Save the report image to a file
#     report_image.save('report.png')

#     # Return a message indicating that the report was generated successfully
#     return 'Report generated successfully!'

# Add callback to open the modal
@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


Header_component = html.Div([
        html.H1(
            "Yokyo Olympics Dashboard",
            style={"color": "#503D36", "font-size": 50},
        ),
        html.Div([button, report_button]),
    ],
    style={
        "display": "flex",
        "justify-content": "space-between",
        "align-items": "center",
    },)


status_meanings = {
    200: 'OK',
    404: 'Not Found',
    500: 'Internal Server Error'
}

df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%b/%Y:%H:%M:%S')
df.set_index('Timestamp', inplace=True)


# calculate the stats
max = df['Traffic Source'].max()
min = df['Traffic Source'].min()
total_requests = len(df)
status_counts = df['Status Code'].value_counts()

# create a row of more cards
# create cards
cards = []
for status, count in status_counts.items():
    meaning = status_meanings.get(status, 'Unknown')
    card = dbc.Card(
        dbc.CardBody([
            html.H4(f"Status {status}", className="card-title"),
            html.P(f"{count} ({meaning})", className="card-text")
        ]),
        color='primary',
        inverse=True
    )
    cards.append(card)


# # add the new card to the layout in the left column
# app.layout.children[-1].children[0].children.append(new_page_card)


# create the cards
card_mean = dbc.Card(
    dbc.CardBody([
        html.H4("Maximum Traffic Source", className="card-title"),
        html.P(f"{max}", className="card-text")
    ]),
    color='primary',
    inverse=True
)

card_median = dbc.Card(
    dbc.CardBody([
        html.H4("Minimum Traffic Source", className="card-title"),
        html.P(f"{min}", className="card-text")
    ]),
    color='primary',
    inverse=True
)

card_total_requests = dbc.Card(
    dbc.CardBody([
        html.H4("Total Requests", className="card-title"),
        html.P(f"{total_requests}", className="card-text")
    ]),
    color='primary',
    inverse=True
)

# create pie chart
traffic_counts = df['Traffic Source'].value_counts()
pie_fig = px.pie(
    traffic_counts,
    values=traffic_counts.values,
    names=traffic_counts.index,
    title='User Traffic Sources'
)

# create histogram
hist_fig = px.histogram(
    df,
    x='Status Code',
    title='Status Code Distribution Histogram'
)

date_slider = dcc.DatePickerRange(
    id='date-slider',
    min_date_allowed=df.index.min().date(),
    max_date_allowed=df.index.max().date(),
    start_date=df.index.min().date(),
    end_date=df.index.max().date(),
    style={'width': '100%'}
)

# Group the data by country and count the number of occurrences for each country
country_counts = df.groupby("Country").size()

# Create a line graph of the country counts using Plotly Express instead of Matplotlib to keep it consistent with other graphs in the app.
line_fig = px.line(country_counts, x=country_counts.index, y=country_counts.values, labels={"x": "Country", "y": "Number of Requests"}, title="Line Graph of Web Traffic by Country")

line_graph = dcc.Graph(figure=line_fig)

line_card = dbc.Card(
    dbc.CardBody([
        html.H4('Web Traffic by Country', className='card-title'),
        line_graph
    ]),
    style={
    'box-shadow': '2px 2px 2px lightgrey',
    'border-radius': '5px'
    }
)


# App layout
app.layout = html.Div(
    style={'width': '100%', 'height': '100%', 'padding': '35px'},
    children=[
        dbc.Row([
            dbc.Col([
                Header_component
            ]),
        ]),
        dbc.Row([
            dbc.Col(card_mean),
            dbc.Col(card_median),
            dbc.Col(card_total_requests),
            # dbc.Col(new_page_card)
        ]),
        modal
    ]
)

# add horizontal line to layout
app.layout.children.insert(
    2,
    dbc.Row([
        dbc.Col([
            html.Hr()
        ])
    ])
)

# create select filter
country_select = dcc.Dropdown(
    id='country-select',
    options=[{'label': country, 'value': country} for country in df['Country'].unique()],
    value=[],
    multi=True,
    clearable=False
)

# create card for select filter
select_card = dbc.Card(
    dbc.CardBody([
        html.H4('Select Countries', className='card-title'),
        country_select
    ]),
    style={
        'box-shadow': '2px 2px 2px lightgrey',
        'border-radius': '5px',
        'margin-bottom': '20px'
    }
)

# create card for pie chart
pie_card = dbc.Card(
    dbc.CardBody([
        dcc.Graph(id='pie-chart', figure=pie_fig)
    ]),
    style={
        'box-shadow': '2px 2px 2px lightgrey',
        'border-radius': '5px'
    }
)


# create card for user agent chart
user_agent_card = dbc.Card(
    dbc.CardBody([
        dcc.Graph(id='user-agent-chart')
    ]),
    style={
        'box-shadow': '2px 2px 2px lightgrey',
        'border-radius': '5px'
    }
)

# create card for histogram
hist_card = dbc.Card(
    dbc.CardBody([
        dcc.Graph(id='hist-chart', figure=hist_fig)
    ]),
    style={
        'box-shadow': '2px 2px 2px lightgrey',
        'border-radius': '5px'
    }
)

# create card for path chart
path_card = dbc.Card(
    dbc.CardBody([
        dcc.Graph(id='path-chart')
    ]),
    style={
        'box-shadow': '2px 2px 2px lightgrey',
        'border-radius': '5px'
    }
)

date_slider_card = dbc.Card(
    dbc.CardBody([
        html.H4('Select Date Range', className='card-title'),
        date_slider
    ]),
    style={
        'box-shadow': '2px 2px 2px lightgrey',
        'border-radius': '5px',
        'margin-bottom': '20px'
    }
)

# create button
button = dbc.Button('Prediction Model', id='prediction-button', color='primary')

# create input containers
country_input = dbc.Input(id='country-input', placeholder='Country')
traffic_input = dbc.Input(id='traffic-input', placeholder='Traffic Sources')
user_agent_input = dbc.Input(id='user-agent-input', placeholder='Device')

# create results badge
results_badge = dbc.Badge(id='results-badge', color='primary', className='mr-1')


# create card for button
button_card = dbc.Card(
    dbc.CardBody([
        html.H4('Prediction Model', className='card-title'),
        dbc.Row([
            dbc.Col(country_input),
            dbc.Col(traffic_input)
        ]),
        dbc.Row([
            dbc.Col(user_agent_input),
            dbc.Col(dbc.Button('Make Prediction', id='prediction-button', color='primary'))
        ]),
        html.Hr(),
        html.H4('Prediction Results'),
        results_badge
    ]),
    style={
        'box-shadow': '2px 2px 2px lightgrey',
        'border-radius': '5px',
        'margin-bottom': '20px'
    }
)

# load the saved model
with open('lr_yokyo_model', 'rb') as f:
    lr = pickle.load(f)

# create label encoders
le_country = LabelEncoder()
le_country.fit(['USA', 'United-Kingdom', 'France', 'China', 'Japan', 'Canada', 'Australia', 'Spain', 'Germany', 'Italy'])

le_traffic_source = LabelEncoder()
le_traffic_source.fit(['Facebook', 'TikTok', 'Twitter', 'Reddit'])

le_user_agent = LabelEncoder()
le_user_agent.fit(['Windows Firefox', 'Windows Yandex', 'Smart TV - Linux x86_64',
                    'Windows Chrome', 'Windows Edge', 'Android 11 - Samsung Galaxy S20',
                      'Mac Chrome', 'Mac Safari', 'Smart TV - Linux armv7l',
                      'Android 11 - Samsung Galaxy S10', 'iOS 14 - Iphone', 'iOS 14 - Ipad', 'Samsung Smart TV - Tizen 5.5',
                       'Samsung Smart TV - Tizen 5.0', 'Android 11 - Samsung Galaxy A51'])

# add sidebar to layout
app.layout.children.append(
    dbc.Row([
        dbc.Col([
            html.H4('Filters'),
            select_card,
            html.Div(style={'height': '5px'}),  # add empty space here
            date_slider_card,
            line_card,
            html.Div(style={'height': '2px'}), # add empty space here
            button_card
        ], width=3, style={'background-color': '#f0f0f0'}),
        dbc.Col([
            hist_card,  # add path_card here
            html.Div(style={'width': '50px'}),  # add empty space here
            pie_card
        ], width=3),
        dbc.Col([
            path_card,
            user_agent_card,
        ], width=6),
    ])
)

# add cards to layout
app.layout.children[1].children.extend(
    [dbc.Col(card) for card in cards]
)



# __________________________________________________________________________________________________________________________
# Callback function to update charts
@app.callback(
    [Output('user-agent-chart', 'figure'), Output('pie-chart', 'figure'),
     Output('hist-chart', 'figure'), Output('path-chart', 'figure')],
    [Input('country-select', 'value'), Input('date-slider', 'start_date'), Input('date-slider', 'end_date')]
)
def update_charts(countries, start_date, end_date):
    filtered_df = df.loc[start_date:end_date]
    
    if countries:
        filtered_df = filtered_df[filtered_df['Country'].isin(countries)]
    


    user_agent_counts = filtered_df['User Agent'].value_counts()
    user_agent_fig = px.bar(
        user_agent_counts,
        x=user_agent_counts.index,
        y=user_agent_counts.values,
        labels={'x': 'User Agent', 'y': 'Count'},
        title='User Devices Histogram'
    )
    # user_agent_fig.update_layout(template='darkly')

    # update pie chart
    traffic_counts = filtered_df['Traffic Source'].value_counts()
    pie_fig = px.pie(
        traffic_counts,
        values=traffic_counts.values,
        names=traffic_counts.index,
        title='User Traffic Sources'
    )
    pie_fig.update_traces(
    marker=dict(
        colors=['#003f5c', '#58508d', '#bc5090', '#ff6361', '#ffa600'],
        line=dict(width=0)
    ),
    hole=0.4
)


    pie_fig.update_layout(
    font=dict(
        family='lux'
    )
)

    # update histogram
    hist_fig = px.histogram(
        filtered_df,
        x='Status Code',
        title='Status Code Distribution Histogram'
    )

    # update path chart
    path_aliases = {
      "/": "home",
      "/athletes": "athletes",
      "/sports": "sports",
      "/medals": "medals",
      "/schedule": "schedule",
      "/results": "results",
      "/sports/basketball": "basketball",
      "/sports/cycling": "cycling",
      "/sports/diving": "diving",
      "/sports/gymnastics": "gymnastics",
      "/sports/rowing": "rowing",
      "/sports/soccer": "soccer",
      "/sports/swimming": "swimming",
      "/sports/table-tennis": "table-tennis",
      "/sports/tennis": "tennis",
      "/sports/track-and-field": "track-and-field",
      "/sports/volleyball": "volleyball",
      "/sports/water-polo": "water-polo",
      "/sports/wrestling": "wrestling",
      "/about": "about"
   }

    path_counts = filtered_df['Path'].value_counts()
    path_data = []
    for path, count in path_counts.items():
       alias = path_aliases.get(path, path)
       path_data.append({'Path': alias, 'Count': count})

    path_fig = px.line(
       pd.DataFrame(path_data),
       x='Path',
       y='Count',
       labels={'x': 'Pages', 'y': 'Count'},
       title='Popular User Access Pages '
   )

    return user_agent_fig, pie_fig, hist_fig, path_fig

@app.callback(
    Output('results-badge', 'children'),
    [Input('prediction-button', 'n_clicks')],
    [State('country-input', 'value'), State('traffic-input', 'value'), State('user-agent-input', 'value')]
)
def make_prediction(n_clicks, country, traffic_source, user_agent):
    if n_clicks:
        # encode input data
        country_encoded = le_country.transform([country])[0]
        traffic_source_encoded = le_traffic_source.transform([traffic_source])[0]
        user_agent_encoded = le_user_agent.transform([user_agent])[0]

        # format input data
        input_data = [[country_encoded, traffic_source_encoded, user_agent_encoded]]

        # make prediction
        output = lr.predict(input_data)

        # return prediction result
        return output[0]

# run the app
if __name__ == "__main__":
   app.run_server(debug=True)
