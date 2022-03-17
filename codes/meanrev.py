from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
import yfinance as yf

from app import app

np.seterr(divide='ignore')
pd.options.display.float_format = '{:,.2f}'.format

# FORMATA E CONFIGURA GRÁFICOS
pio.templates["draft"] = go.layout.Template(
    layout=go.Layout(
        title_x = 0.0, 
        title_pad = dict(l=10, t=10),
        margin = dict(l=70,t=50, b=70, r=30, pad=10, autoexpand=False),
        font = dict(family="Arial", size=10), 
        autosize=False,
        ),
    layout_annotations=[
        dict(
            name="draft watermark",
            text="KWT-Community",
            textangle=-30,
            opacity=0.03,
            font=dict(family="Arial", color="black", size=80),
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
    ]
)

pio.templates.default = "seaborn+draft"

plotres:dict = dict(width=1920, height=1080)

config1 = {
    "displaylogo": False,
    "toImageButtonOptions": plotres, 
    "modeBarButtonsToAdd": [
        "drawline",
        "drawopenpath",
        "drawclosedpath",
        "drawcircle",
        "drawrect",
        "eraseshape",
        "hoverClosestCartesian",
        "hoverCompareCartesian"

    ]
}

# INPUTS PARA DROPDOWN MENU
tickers = pd.read_csv('db/tickers.csv', delimiter=';') # ativos da na bolsa brasileira
tickers['label'] = tickers['value']+" - "+tickers['label']
tickers['value'] = tickers['value']+".SA"
tickers = tickers.to_dict('records')

periods = pd.read_csv('db/periods.csv', delimiter=';').to_dict('records') # períodos de análise

intervals = pd.read_csv('db/intervals.csv', delimiter=';').to_dict('records') # intervalos entre dados do período

# LAYOUT
layout = dbc.Container(
        children=[
            dcc.Loading(
                id="load_o1",
                color='#ff9800',
                style={'vertical-align':'bottom'},
                parent_style={'vertical-align':'bottom'},
                children=html.Span(id="load_o1"),
                type="dot",
                ),
            dbc.Row([
                html.Div(className='kwtdrops', children=[
                        "ATIVO", dcc.Dropdown( id="ticker", options=tickers, value='OIBR3.SA', clearable=False, style={'width':'30rem', 'margin':'5px'} ), 
                        "PERÍODO", dcc.Dropdown( id="periods", options=periods, value='1y', clearable=False, style={'width':'10rem', 'margin':'5px'} ),
                        "INTERVALO", dcc.Dropdown( id="intervals", options=intervals, value='1d', clearable=False, style={'width':'10rem', 'margin':'5px'} ),
                        dbc.Button(id='submitb', n_clicks=0, children='Atualizar')
                ]),
            ]),
            html.Br(),
            dbc.Row([
                dbc.Card(dbc.CardBody(dcc.Graph(id="graph", config=config1)), style={"width": "70rem", "margin-right": "10px", "margin-bottom": "10px"}),                             
            ]),                      
            ], fluid=True)

def get():
    return html.Div(layout)

####### CALLBACKS #######

####### CALLBACK PAINEL MERCADO
#
@app.callback(

    [ Output("graph", "figure"),
    Output("load_o1", "children") ],

    [ Input('submitb', 'n_clicks') ],

    [ State("ticker", "value"), 
    State("periods", "value"), 
    State("intervals", "value") ],
    
    )

###### FUNC. CALLBACK PAINEL MERCADO
#
def display(sutb, tkr, prd, itv):
    ####### DOWNLOAD DE PREÇO E VOLUME DO ATIVO SELECIONADO
    #

    #start='2017-01-01'
    #end='2021-12-31'
    df = yf.download(tkr, interval=itv, period=prd)
    df = df.resample(rule='B').bfill()
    df.fillna( method ='ffill', inplace = True)
    df.fillna( method ='bfill', inplace = True)

    
    df['PrevClose']=df.Close.shift(1)
    df['VarClose']=((df.Close - df.Close.shift(1))/df.Close.shift(1))*100
    df['VarOpen']=((df.Open - df.Open.shift(1))/df.Open.shift(1))*100
    df['VarHigh']=((df.High - df.High.shift(1))/df.High.shift(1))*100
    df['VarLow']=((df.Low - df.Low.shift(1))/df.Low.shift(1))*100
    df['VarIntra']=((df.Close - df.Open)/df.Open)*100
    df['VarHighLow']=((df.High - df.Low)/df.Low)*100
    df['VarCloseOpen']=((df.Open - df.Close.shift(1))/df.Close.shift(1))*100
    df['VarHL']=(np.log(df.High / df.Low.shift(1))+np.log(df.Low / df.High.shift(1)) ) / 2 * 100
    df['VarAcum'] = ((df.Close/df.Close.iloc[0])-1)*100
    df["VAColor"] = np.where(df.VarAcum < 0, 'red', 'green')
    
    per=21
    df['VarCMax']=df['VarClose'].rolling(per).max()
    df['VarCMin']=df['VarClose'].rolling(per).min()  
    df['Overshoot'] = ((abs(df['VarCMax'].rolling(per).mean())/abs(df['VarCMin'].rolling(per).mean()))-1)*100

    df['CMax']=df['High'].rolling(per).max()
    df['CMin']=df['Low'].rolling(per).min()
    df['CDrawdown']=np.log(df['CMin']/df['CMax'])*100

    df['Returns'] = (np.log(df.Close / df.Close.shift(1)))*100
    df['AnnualVol'] = df.Returns.rolling(21).std() * 252 ** 0.5

    ####### CONSTROI GRÁFICOS
    #
    fig = go.Figure()
    #fig.add_trace( go.Scatter(x=df.index, y=df.Close, mode='lines', name=tkr, connectgaps=True, showlegend=True, line_shape='spline') )
    
    fig.add_trace( go.Candlestick ( 
        x=df.index,
        open=df.Open,
        high=df.High,
        low=df.Low,
        close=df.Close,
        name='COTAÇÃO') )

    fig.add_trace( go.Scatter(x=df.index, y=df.Close.rolling(21).mean(), mode='lines', name='MMA21', line_color='navy')  )
    fig.add_trace( go.Scatter(x=df.index, y=df.Close.rolling(50).mean(), mode='lines', name='MMA50', line_color='orangered')  )
    fig.add_trace( go.Scatter(x=df.index, y=df.Close.ewm(span=200, adjust=False).mean(), mode='lines', name='EWA200', line_color='purple')  )
        
    ####### ATUALIZA LAYOUT, TRACES E AXES DOS GRÁFICOS
    #
    fig.update_layout( title='<b>EVOLUÇÃO DO PREÇO</b>', xaxis_title='', yaxis_title='<b>Preço</b>', xaxis_rangeslider_visible=False, legend=dict(orientation="h") )

    return fig, ""