from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import math
from sklearn.linear_model import LinearRegression

from app import app

np.seterr(divide='ignore')
pd.options.display.float_format = '{:,.2f}'.format

# FORMATA E CONFIGURA GRÁFICOS
pio.templates["draft"] = go.layout.Template(
    layout=go.Layout(
        title_x = 0.0, 
        title_pad = dict(l=10, t=10),
        margin = dict(l=50,t=50, b=50, r=50, pad=0, autoexpand=True),
        font = dict(family="Arial", size=10), 
        autosize=True,
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

other = pd.read_csv('db/other.csv', delimiter=';') # outros ativos e índices
other['label'] = other['value']+" - "+other['label']

tickers=pd.concat([tickers,other])
tickers = tickers.to_dict('records')

periods = pd.read_csv('db/periods.csv', delimiter=';').to_dict('records') # períodos de análise

intervals = pd.read_csv('db/intervals.csv', delimiter=';').to_dict('records') # intervalos entre dados do período

def market_beta(X,Y,N):
    """ 
    X = The independent variable which is the Market
    Y = The dependent variable which is the Stock
    N = The length of the Window
     
    It returns the alphas and the betas of
    the rolling regression
    """
     
    # all the observations
    obs = len(X)
     
    # initiate the betas with null values
    betas = np.full(obs, np.nan)
     
    # initiate the alphas with null values
    alphas = np.full(obs, np.nan)
     
     
    for i in range((obs-N)):
        regressor = LinearRegression()
        regressor.fit(X.to_numpy()[i : i + N+1].reshape(-1,1), Y.to_numpy()[i : i + N+1])
         
        betas[i+N]  = regressor.coef_[0]
        alphas[i+N]  = regressor.intercept_
 
    return(alphas, betas)

# LAYOUT
layout = dbc.Container(
        children=[
            dcc.Loading(
                #className="kwtload",
                id="load_o1",
                color='#0a0',
                style={'background-color':'rgba(0, 0, 0, 0.5)'},
                parent_style={},
                fullscreen=True,
                children=html.Span(id="correl_load_o1", children=["LOADING..."]),
                type="default",
                ),
            dbc.Row([
                html.Div(className='kwtdrops', children=[
                        html.H5("ATIVO"), dcc.Dropdown( id="ticker", options=tickers, value='VALE3.SA', clearable=False, style={'width':'300px'} ), 
                        html.H5("BENCHMARK"), dcc.Dropdown( id="indexer", options=tickers, value='^BVSP', clearable=False, style={'width':'300px'} ), 
                        html.H5("PERÍODO"), dcc.Dropdown( id="periods", options=periods, value='1y', clearable=False, style={'width':'10rem'} ),
                        html.H5("INTERVALO"), dcc.Dropdown( id="intervals", options=intervals, value='1d', clearable=False, style={'width':'10rem'} ),
                        dbc.Button(className="kwtchartbtn",id='submitb', n_clicks=0, children='Atualizar')
                ]),
            ]),
            html.Br(),
            dbc.Row([
                dcc.Graph(id="correl_graph", config=config1),                             
                dcc.Graph(id="correl_graph1", config=config1),
                dcc.Graph(id="correl_graph2", config=config1),          
                dcc.Graph(id="correl_graph3", config=config1),                     
            ]),                      
            ], fluid=True)

def get():
    return html.Div(layout)

####### CALLBACKS #######

####### CALLBACK PAINEL MERCADO
#
@app.callback(

    [ Output("correl_graph", "figure"),
    Output("correl_graph1", "figure"),
    Output("correl_graph2", "figure"),
    Output("correl_graph3", "figure"),
    Output("correl_load_o1", "children") ],

    [ Input('submitb', 'n_clicks') ],

    [ State("ticker", "value"), 
    State("indexer", "value"), 
    State("periods", "value"), 
    State("intervals", "value") ],
    
    )

###### FUNC. CALLBACK PAINEL MERCADO
#
def display(sutb, tkr, idx, prd, itv):

    per21dd=21
    per50dd=50
    per200dd=200

    ####### DOWNLOAD DE PREÇO E VOLUME DO ATIVO ANALISADO

    df = yf.download(tkr, interval=itv, period=prd)
    df = pd.DataFrame(df)
    df = df[df.index.dayofweek < 5]
    df.dropna(inplace=True)
    #df.fillna( method ='ffill', inplace = True)
    #df.fillna( method ='bfill', inplace = True)

    ### VARIAÇÃO E RETORNO
    #df['Return'] = (np.log(df.Close / df.Close.shift(1)))*100
    df['Return'] = df.Close.pct_change()
    print(df['Return'].isnull().sum())
    df.dropna(inplace=True)
    print(df['Return'].isnull().sum())

    df['PrevClose']=df.Close.shift(1)
    df['VarClose']=((df.Close - df.Close.shift(1))/df.Close.shift(1))*100
    #df['VarClose'] = df.Close.pct_change()

    df['Return21dd'] = (np.log(df.Close / df.Close.shift(per21dd)))*100
    df['Return50dd'] = (np.log(df.Close / df.Close.shift(per50dd)))*100
    df['Return200dd'] = (np.log(df.Close / df.Close.shift(per200dd)))*100

    df['VarAcum'] = ((df.Close/df.Close.iloc[0])-1)*100

    df['RetAcum'] = (np.log(df.Close / df.Close.iloc[0]))*100
    df["RetAcumColor"] = np.where(df.RetAcum < 0, 'red', 'green')

    ### REVERSÃO À MÉDIA ARITIMÉTICA DE 21 DIAS

    df['CSMA21dd']=df.Close.rolling(per21dd).mean()
    df['RSMA21dd']=((df.Close/df['CSMA21dd'])-1)*100
    df["RSMA21dd_Color"] = np.where(df.RSMA21dd < 0, 'red', 'green')

    ### REVERSÃO À MÉDIA ARITIMÉTICA DE 50 DIAS

    df['CSMA50dd']=df.Close.rolling(per50dd).mean()
    df['RSMA50dd']=((df.Close/df['CSMA50dd'])-1)*100
    df["RSMA50dd_Color"] = np.where(df.RSMA50dd < 0, 'red', 'green')

    ### REVERSÃO À MÉDIA EXPONENCIAL DE 200 DIAS

    df['CEMA200dd']=df.Close.ewm(span=per200dd, min_periods=per200dd, adjust=True).mean()
    df['REMA200dd']=((df.Close/df['CEMA200dd'])-1)*100
    df["REMA200dd_Color"] = np.where(df.REMA200dd < 0, 'red', 'green')

    #print(df['Return'].isnull().sum())
    #df.Return

    ####### DOWNLOAD DE PREÇO E VOLUME DO ATIVO DE REFERÊNCIA (BENCHMARK)

    dfi = yf.download(idx, interval=itv, period=prd)
    dfi = pd.DataFrame(dfi)
    dfi = dfi[dfi.index.dayofweek < 5]
    dfi.dropna(inplace=True)
    #dfi.fillna( method ='ffill', inplace = True)
    #dfi.fillna( method ='bfill', inplace = True)

    ### VARIAÇÃO E RETORNO

    #dfi['Return'] = (np.log(dfi.Close / dfi.Close.shift(1)))*100
    dfi['Return'] = dfi.Close.pct_change()
    print(dfi['Return'].isnull().sum())
    dfi.dropna(inplace=True)
    print(dfi['Return'].isnull().sum())

    dfi['PrevClose']=dfi.Close.shift(1)
    dfi['VarClose']=((dfi.Close - dfi.Close.shift(1))/dfi.Close.shift(1))*100
    #dfi['VarClose'] = dfi.Close.pct_change()

    dfi['Return21dd'] = (np.log(dfi.Close / dfi.Close.shift(per21dd)))*100
    dfi['Return50dd'] = (np.log(dfi.Close / dfi.Close.shift(per50dd)))*100
    dfi['Return200dd'] = (np.log(dfi.Close / dfi.Close.shift(per200dd)))*100

    dfi['VarAcum'] = ((dfi.Close/dfi.Close.iloc[0])-1)*100

    dfi['RetAcum'] = (np.log(dfi.Close / dfi.Close.iloc[0]))*100
    dfi["RetAcumColor"] = np.where(dfi.RetAcum < 0, 'red', 'green')

    ### REVERSÃO À MÉDIA ARITIMÉTICA DE 21 DIAS

    dfi['CSMA21dd']=dfi.Close.rolling(per21dd).mean()
    dfi['RSMA21dd']=((dfi.Close/dfi['CSMA21dd'])-1)*100
    dfi["RSMA21dd_Color"] = np.where(dfi.RSMA21dd < 0, 'red', 'green')

    ### REVERSÃO À MÉDIA ARITIMÉTICA DE 50 DIAS

    dfi['CSMA50dd']=dfi.Close.rolling(per50dd).mean()
    dfi['RSMA50dd']=((dfi.Close/dfi['CSMA50dd'])-1)*100
    dfi["RSMA50dd_Color"] = np.where(dfi.RSMA50dd < 0, 'red', 'green')

    ### REVERSÃO À MÉDIA EXPONENCIAL DE 200 DIAS

    dfi['CEMA200dd']=dfi.Close.ewm(span=per200dd, min_periods=per200dd, adjust=True).mean()
    dfi['REMA200dd']=((dfi.Close/dfi['CEMA200dd'])-1)*100
    dfi["REMA200dd_Color"] = np.where(dfi.REMA200dd < 0, 'red', 'green')

    #print(dfi['Return'].isnull().sum())
    #dfi.Return

    ### ROLLING CORRELATION

    df['RCorr21dd'] = df['VarClose'].rolling(per21dd).corr(dfi['VarClose'])
    df['RCorr50dd'] = df['VarClose'].rolling(per50dd).corr(dfi['VarClose'])
    df['RCorr200dd'] = df['VarClose'].rolling(per200dd).corr(dfi['VarClose'])

    ### RETORNO COMPARADO

    df['RetComp'] = df['RetAcum'] / dfi['RetAcum']

    ### CALCULA ALPHA E BETA

    df['Alpha21dd'],df['Beta21dd'] = market_beta(df.Return, dfi.Return, 21)
    df['Alpha50dd'],df['Beta50dd'] = market_beta(df.Return, dfi.Return, 50)
    df['Alpha200dd'],df['Beta200dd'] = market_beta(df.Return, dfi.Return, 200)

    ####### CONSTROI GRÁFICOS
    #

    ### FIG 0 ---------------------------------------------------------------------------
    fig = go.Figure()
    fig.add_trace( go.Candlestick ( 
        x=df.index,
        open=df.Open,
        high=df.High,
        low=df.Low,
        close=df.Close,
        name=tkr) )
    fig.add_trace( go.Scatter(x=df.index, y=df.CSMA21dd, mode='lines', name='MMA21', line_width=1,line_color='orange') )
    fig.add_trace( go.Scatter(x=df.index, y=df.CSMA50dd, mode='lines', name='MMA50', line_width=1,line_color='navy') )
    fig.add_trace( go.Scatter(x=df.index, y=df.CEMA200dd, mode='lines', name='EMA200', line_width=1,line_color='purple') )
    
    ### FIG 1 ---------------------------------------------------------------------------
    fig1 = make_subplots(
        rows=1, cols=2,
        column_widths=[.85,.15],
        subplot_titles=("", "Histograma (Percent)")
        )
    fig1.add_trace( go.Scatter(x=df.index, y=df.RSMA21dd, mode='lines', name='R_MMA21', line_width=1, line_color='orange'), col=1, row=1  )
    fig1.add_trace( go.Scatter(x=df.index, y=df.RSMA50dd, mode='lines', name='R_MMA50', line_width=1,line_color='navy'), col=1, row=1  )
    fig1.add_trace( go.Scatter(x=df.index, y=df.REMA200dd, mode='lines', name='R_EMA200', line_width=1,line_color='purple'), col=1, row=1  )
    fig1.add_hline(y=0, 
        line_color='black', line_dash='dot', line_width=1,
        annotation_text="Centro da Média", 
        annotation_position="bottom left", col=1, row=1)
    
    fig1.add_trace( go.Histogram(x=df.RSMA21dd, name='R_MMA21', histnorm='percent', offsetgroup=0), col=2, row=1  )
    fig1.add_trace( go.Histogram(x=df.RSMA50dd, name='R_MMA50', histnorm='percent', offsetgroup=0), col=2, row=1  )
    fig1.add_trace( go.Histogram(x=df.REMA200dd, name='R_EMA200', histnorm='percent', offsetgroup=0), col=2, row=1  )

    fig1.update_layout(
        xaxis=dict(showgrid=False),
        xaxis2=dict(showgrid=False)
    )

    fig1.update_traces(bingroup='overlay', nbinsx=20, opacity=0.5, col=2, row=1, cumulative_enabled=False) 

    ### FIG 2 ---------------------------------------------------------------------------
    fig2 = make_subplots(
        rows=3, cols=2,
        #subplot_titles=("Reversão à Média", "Indicador"),
        column_widths=[0.85,.15],
        row_heights=[.33, .33, .33],
        specs= [
            [{'type' : 'xy'}, {'type' : 'indicator'}],
            [{'type' : 'xy'}, {'type' : 'indicator'}],
            [{'type' : 'xy'}, {'type' : 'indicator'}],
        ],
        #subplot_titles=('Mercedes', 'Ford', 'BMW')
        #specs=[
        #   [{}],
        #   [{}],
        #   [{}],
        #   ]
        )
    fig2.add_trace( go.Scatter(x=df.index, y=df.RSMA21dd, mode='lines', line_width=1, name='R_MMA21', line_color='orange') , row=1, col=1  ),
    
    fig2.add_trace( 
        go.Indicator( 
            mode = "gauge+number+delta", 
            value = df.RSMA21dd[-1], 
            #title = {'text': "Reversão MMA21"}, 
            delta = {'reference': df.RSMA21dd.mean(), 'relative': True,'valueformat':'.2%'}, 
            gauge={
                'axis':{
                    'range':[math.floor(df.RSMA21dd.min()),math.ceil(df.RSMA21dd.max())],
                    'dtick': ( math.ceil(df.RSMA21dd.max()) - math.floor(df.RSMA21dd.min()) )/10,
                    'tickformat':'0.1f'
                },
                'steps' : [
                    {'range': [math.floor(df.RSMA21dd.min()), (math.floor(df.RSMA21dd.min())*0.5)], 'color': "rgba(50,50,200,0.55)"},
                    {'range': [(math.ceil(df.RSMA21dd.max())*0.5), math.ceil(df.RSMA21dd.max())], 'color': "rgba(200,50,50,0.55)"}],
                'threshold' : {'line': {'color': "red", 'width': 4}, 
                            'thickness': 1, 
                            'value': df.RSMA21dd.mean()},
                'bar': {'color': "black"}
            }
        ), row=1, col=2  ),

    fig2.add_trace( go.Scatter(x=df.index, y=df.RSMA50dd, mode='lines', line_width=1, name='R_MMA50', line_color='navy')  , row=2, col=1  )

    fig2.add_trace( 
        go.Indicator( 
            mode = "gauge+number+delta", 
            value = df.RSMA50dd[-1], 
            #title = {'text': "Reversão MMA50"}, 
            delta = {'reference': df.RSMA50dd.mean(), 'relative': True, 'valueformat':'.2%'}, 
            gauge={
                'axis':{
                    'range':[math.floor(df.RSMA50dd.min()),math.ceil(df.RSMA50dd.max())],
                    'dtick': ( math.ceil(df.RSMA50dd.max()) - math.floor(df.RSMA50dd.min()) )/10,
                    'tickformat':'0.1f'
                },
                'steps' : [
                    {'range': [math.floor(df.RSMA50dd.min()), (math.floor(df.RSMA50dd.min())*0.5)], 'color': "rgba(50,50,200,0.55)"},
                    {'range': [(math.ceil(df.RSMA50dd.max())*0.5), math.ceil(df.RSMA50dd.max())], 'color': "rgba(200,50,50,0.55)"}],
                'threshold' : {'line': {'color': "red", 'width': 4}, 
                            'thickness': 1, 
                            'value': df.RSMA50dd.mean()},
                'bar': {'color': "black"}
            }
        ), row=2, col=2  ),

    fig2.add_trace( go.Scatter(x=df.index, y=df.REMA200dd, mode='lines', line_width=1, name='R_EMA200', line_color='purple')  , row=3, col=1  )

    fig2.add_trace( 
        go.Indicator( 
            mode = "gauge+number+delta", 
            value = df.REMA200dd[-1], 
            #title = {'text': "Reversão EMA200"}, 
            delta = {'reference': df.REMA200dd.mean(), 'relative': True, 'valueformat':'.2%'}, 
            gauge={
                'axis':{
                    'range':[math.floor(df.REMA200dd.min()),math.ceil(df.REMA200dd.max())],
                    'dtick': ( math.ceil(df.REMA200dd.max()) - math.floor(df.REMA200dd.min()) )/10,
                    'tickformat':'0.1f'
                },
                'steps' : [
                    {'range': [math.floor(df.REMA200dd.min()), (math.floor(df.REMA200dd.min())*0.5)], 'color': "rgba(50,50,200,0.55)"},
                    {'range': [(math.ceil(df.REMA200dd.max())*0.5), math.ceil(df.REMA200dd.max())], 'color': "rgba(200,50,50,0.55)"}],
                'threshold' : {'line': {'color': "red", 'width': 4}, 
                            'thickness': 1, 
                            'value': df.REMA200dd.mean()},
                'bar': {'color': "black"}
            }
      ), row=3, col=2  ),

    #fig2.add_hline(y=0, 
    #    line_color='black', line_dash='dot', line_width=1,
    #    annotation_text="Centro da Média", 
    #    annotation_position="bottom left", 
    #    row=1, col=1,)

    ### FIG 3 ---------------------------------------------------------------------------

    fig3 = make_subplots(
        rows=1, cols=3,
        column_widths=[.33, .33, .33],
        subplot_titles=("MÉDIA vs RSMA21dd", "MÉDIA vs RSMA50dd", "MÉDIA vs REMA200dd"),
    )
    fig3.add_trace( go.Scatter(name='', x=df.RSMA21dd, y=df.CSMA21dd, text=df.index.strftime("%d/%m/%Y"),
        mode='markers',
        marker=dict(
            size=7,
            color=df.RSMA21dd, #set color equal to a variable
            colorscale='Bluered', # one of plotly colorscales
            opacity=0.5,
            showscale=False),
        hovertemplate = "%{text} <br> RSMA21dd : %{x:.2f} </br> MÉDIA PREÇO : %{y:,.2f}"
        ), row=1, col=1 
    ) 
    fig3.add_trace( go.Scatter(name='', x=df.RSMA50dd, y=df.CSMA50dd, text=df.index.strftime("%d/%m/%Y"),
        mode='markers',
        marker=dict(
            size=7,
            color=df.RSMA50dd, #set color equal to a variable
            colorscale='Bluered', # one of plotly colorscales
            opacity=0.5,
            showscale=False),
        hovertemplate = "%{text} <br> RSMA50dd : %{x:.2f} </br> MÉDIA PREÇO : %{y:,.2f}"
        ), row=1, col=2 
    ) 
    fig3.add_trace( go.Scatter(name='', x=df.REMA200dd, y=df.CEMA200dd, text=df.index.strftime("%d/%m/%Y"),
        mode='markers',
        marker=dict(
            size=7,
            color=df.REMA200dd, #set color equal to a variable
            colorscale='Bluered', # one of plotly colorscales
            opacity=0.5,
            showscale=False),
        hovertemplate = "%{text} <br> REMA200dd : %{x:.2f} </br> MÉDIA PREÇO : %{y:,.2f}"
        ), row=1, col=3 
    ) 
    


    ####### ATUALIZA LAYOUT, TRACES E AXES DOS GRÁFICOS
    #
    fig.update_layout( title='<b>EVOLUÇÃO DO PREÇO</b>', xaxis_title='',yaxis_title='<b>Preço</b>', xaxis_rangeslider_visible=False, hovermode='x unified', legend=dict(orientation="h") )
    
    fig1.update_layout(title_text='REVERSÃO À MÉDIA - Agrupado', yaxis_title='<b>Valor</b>', xaxis_rangeslider_visible=False, hovermode='x unified', legend=dict(orientation="h") )

    fig2.update_layout(title_text='REVERSÃO À MÉDIA', yaxis_title='<b>Valor</b>', xaxis_rangeslider_visible=False, hovermode='x unified', legend=dict(orientation="h") )

    fig3.update_layout( showlegend=False )

    fig.update_xaxes( rangebreaks=[ dict(bounds=["sat", "mon"]) ] )
    fig1.update_xaxes( rangebreaks=[ dict(bounds=["sat", "mon"]) ] )
    fig2.update_xaxes( rangebreaks=[ dict(bounds=["sat", "mon"]) ] )
    fig3.update_xaxes( rangebreaks=[ dict(bounds=["sat", "mon"]) ] )

    return fig, fig2, fig1, fig3, ""