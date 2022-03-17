from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
                        html.H5("ATIVO"), dcc.Dropdown( id="ticker", options=tickers, value='OIBR3.SA', clearable=False, style={'width':'600px'} ), 
                        html.H5("PERÍODO"), dcc.Dropdown( id="periods", options=periods, value='1y', clearable=False, style={'width':'10rem'} ),
                        html.H5("INTERVALO"), dcc.Dropdown( id="intervals", options=intervals, value='1d', clearable=False, style={'width':'10rem'} ),
                        dbc.Button(className="kwtchartbtn",id='submitb', n_clicks=0, children='Atualizar')
                ]),
            ]),
            html.Br(),
            dbc.Row([
                dbc.Card(dbc.CardBody(dcc.Graph(id="graph", config=config1)), style={"margin-right": "5px", "margin-bottom": "5px"}),                             
                dbc.Card(dbc.CardBody(dcc.Graph(id="graph1", config=config1)), style={"margin-right": "5px", "margin-bottom": "5px"}),                             
                dbc.Card(dbc.CardBody(dcc.Graph(id="graph2", config=config1)), style={"margin-right": "5px", "margin-bottom": "5px"}),                             
            ]),                      
            ], fluid=True)

def get():
    return html.Div(layout)

####### CALLBACKS #######

####### CALLBACK PAINEL MERCADO
#
@app.callback(

    [ Output("graph", "figure"),
    Output("graph1", "figure"),
    Output("graph2", "figure"),
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

    ### REVERSÃO À MÉDIA ARITIMÉTICA DE 21 DIAS
    per21dd=21
    df['CSMA21dd']=df.Close.rolling(per21dd).mean()
    df['RSMA21dd']=((df.Close/df['CSMA21dd'])-1)*100
    df["RSMA21dd_Color"] = np.where(df.RSMA21dd < 0, 'red', 'green')

    ### REVERSÃO À MÉDIA ARITIMÉTICA DE 50 DIAS
    per50dd=50
    df['CSMA50dd']=df.Close.rolling(per50dd).mean()
    df['RSMA50dd']=((df.Close/df['CSMA50dd'])-1)*100
    df["RSMA50dd_Color"] = np.where(df.RSMA50dd < 0, 'red', 'green')

    ### REVERSÃO À MÉDIA EXPONENCIAL DE 200 DIAS
    per200dd=200
    df['CEMA200dd']=df.Close.ewm(span=per200dd, min_periods=per200dd, adjust=True).mean()
    df['REMA200dd']=((df.Close/df['CEMA200dd'])-1)*100
    df["REMA200dd_Color"] = np.where(df.REMA200dd < 0, 'red', 'green')

    '''
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
    df["VAColor"] = np.where(df.VarAcum < 0, 'red', 'green'
    df['Returns'] = (np.log(df.Close / df.Close.shift(1)))*100
    
    #21dd
    df['CMax']=df.Close.rolling(per21dd).max()
    df['CMin']=df.Close.rolling(per21dd).min()  
    df['VarCMax']=df['VarClose'].rolling(per21dd).max()
    df['VarCMin']=df['VarClose'].rolling(per21dd).min()  
    df['Overshoot'] = ((abs(df['VarCMax'].rolling(per21dd).mean())/abs(df['VarCMin'].rolling(per21dd).mean()))-1)*100  
    df['HMax']=df.High.rolling(per21dd).max()
    df['LMin']=df.Low.rolling(per21dd).min()
    df['CDrawdown']=np.log(df['LMin']/df['HMax'])*100
    df['AnnualVol'] = df.Returns.rolling(per21dd).std() * 252 ** 0.5
    '''

    ####### CONSTROI GRÁFICOS
    #
    fig = go.Figure()  
    fig.add_trace( go.Candlestick ( 
        x=df.index,
        open=df.Open,
        high=df.High,
        low=df.Low,
        close=df.Close,
        name='COTAÇÃO') )
    fig.add_trace( go.Scatter(x=df.index, y=df.CSMA21dd, mode='lines', name='MMA21', line_color='orange')  )
    fig.add_trace( go.Scatter(x=df.index, y=df.CSMA50dd, mode='lines', name='MMA50', line_color='navy')  )
    fig.add_trace( go.Scatter(x=df.index, y=df.CEMA200dd, mode='lines', name='EMA200', line_color='purple')  )
    
    fig1 = go.Figure()  
    fig1.add_trace( go.Scatter(x=df.index, y=df.RSMA21dd, mode='lines', name='R_MMA21', line_color='orange')  )
    fig1.add_trace( go.Scatter(x=df.index, y=df.RSMA50dd, mode='lines', name='R_MMA50', line_color='navy')  )
    fig1.add_trace( go.Scatter(x=df.index, y=df.REMA200dd, mode='lines', name='R_EMA200', line_color='purple')  )
    fig1.add_hline(y=0, 
        line_color='black', line_dash='dot', line_width=1,
        annotation_text="Centro da Média", 
        annotation_position="bottom left")

    fig2 = make_subplots(
        rows=3, cols=1,
        column_widths=[1],
        row_heights=[0.33, 0.33, 0.33],
        specs=[
           [{}],
           [{}],
           [{}],
           ]
        )
    fig2.add_trace( go.Scatter(x=df.index, y=df.RSMA21dd, mode='lines', name='R_MMA21', line_color='orange') , row=1, col=1  )
    fig2.add_trace( go.Scatter(x=df.index, y=df.RSMA50dd, mode='lines', name='R_MMA50', line_color='navy')  , row=2, col=1  )
    fig2.add_trace( go.Scatter(x=df.index, y=df.REMA200dd, mode='lines', name='R_EMA200', line_color='purple')  , row=3, col=1  )
    fig2.add_hline(y=0, 
        line_color='black', line_dash='dot', line_width=1,
        row='all', col='all',
        annotation_text="Centro da Média", 
        annotation_position="bottom left")
    
    ####### ATUALIZA LAYOUT, TRACES E AXES DOS GRÁFICOS
    #
    fig.update_layout( title='<b>EVOLUÇÃO DO PREÇO</b>', xaxis_title='', yaxis_title='<b>Preço</b>', xaxis_rangeslider_visible=False, legend=dict(orientation="h") )
    fig1.update_layout( title='<b>REVERSÃO À MÉDIA - AGRUPADO</b>', xaxis_title='', yaxis_title='<b>Valor</b>', xaxis_rangeslider_visible=False, legend=dict(orientation="h") )
    fig2.update_layout( title='<b>REVERSÃO À MÉDIA - INDIVIDUAL</b>', xaxis_title='', yaxis_title='<b>Valor</b>', xaxis_rangeslider_visible=False, legend=dict(orientation="h") )

    return fig, fig1, fig2, ""