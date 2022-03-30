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
    if len(X)==len(Y):
        print("Yes") 
    else:
        print("No")

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
                children=html.Span(id="load_o1", children=["LOADING..."]),
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
            dcc.Tabs([
                dcc.Tab(label='Reversão à Média', children=[
                    dbc.Row([
                        dcc.Graph(id="graph", config=config1),                             
                        dcc.Graph(id="graph1", config=config1),
                        dcc.Graph(id="graph2", config=config1),          
                        dcc.Graph(id="graph3", config=config1),                     
                        ])
                    ]),   

                dcc.Tab(label='Retorno, Correlação, Alpha e Beta', children=[
                    dbc.Row([
                        dcc.Graph(id="cor_graph", config=config1),                             
                        dcc.Graph(id="cor_graph1", config=config1),
                        dcc.Graph(id="cor_graph2", config=config1),          
                        dcc.Graph(id="cor_graph3", config=config1),                     
                        ])
                    ]),      
                ])
            ], fluid=True)

def get():
    return html.Div(layout)

####### CALLBACKS #######

####### CALLBACK PAINEL MERCADO
#
@app.callback(

    [ 
    Output("graph", "figure"),
    Output("graph1", "figure"),
    Output("graph2", "figure"),
    Output("graph3", "figure"),

    Output("cor_graph", "figure"),
    Output("cor_graph1", "figure"),
    Output("cor_graph2", "figure"),
    Output("cor_graph3", "figure"),

    Output("load_o1", "children") ],

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

    ####### DOWNLOAD DE PREÇO E VOLUME
    df = yf.download([ tkr , idx ], interval=itv, period=prd)
    df = pd.DataFrame(df)
    df = df[df.index.dayofweek < 5]
    print("DF : " + str(len(df)))

    for n in df.Close:
        df['Return',n] = df.Close[n].pct_change()
        df.fillna(method="bfill",inplace=True)

    for n in df.Close:
        df['RetAcum',n] = (np.log(df.Close[n] / df.Close[n].iloc[0]))*100
        df['Return21dd',n] = (np.log(df.Close[n] / df.Close[n].shift(per21dd)))*100
        df['Return50dd',n] = (np.log(df.Close[n] / df.Close[n].shift(per50dd)))*100
        df['Return200dd',n] = (np.log(df.Close[n] / df.Close[n].shift(per200dd)))*100
        ##divide by zero encountered in log

        df['PrevClose',n]=df.Close[n].shift(1)

        df['VarClose',n]=((df.Close[n] - df.Close[n].shift(1))/df.Close[n].shift(1))*100
        df['VarAcum',n] = ((df.Close[n]/df.Close[n].iloc[0])-1)*100

        ### REVERSÃO À MÉDIA ARITIMÉTICA DE 21 DIAS
        df['CSMA21dd',n]=df.Close[n].rolling(per21dd).mean()
        df['RSMA21dd',n]=((df.Close[n]/df['CSMA21dd',n])-1)*100

        ### REVERSÃO À MÉDIA ARITIMÉTICA DE 50 DIAS
        df['CSMA50dd',n]=df.Close[n].rolling(per50dd).mean()
        df['RSMA50dd',n]=((df.Close[n]/df['CSMA50dd',n])-1)*100

        ### REVERSÃO À MÉDIA EXPONENCIAL DE 200 DIAS
        df['CEMA200dd',n]=df.Close[n].ewm(span=per200dd, min_periods=per200dd, adjust=True).mean()
        df['REMA200dd',n]=((df.Close[n]/df['CEMA200dd',n])-1)*100

    df = df.sort_index(axis=1)

    ### ROLLING CORRELATION

    df['RCorr21dd'] = df['Return'][tkr].rolling(per21dd).corr(df['Return'][idx])
    df['RCorr50dd'] = df['Return'][tkr].rolling(per50dd).corr(df['Return'][idx])
    df['RCorr200dd'] = df['Return'][tkr].rolling(per200dd).corr(df['Return'][idx])

    ### RETORNO COMPARADO

    df['RetComp'] = df['RetAcum'][tkr] / df['RetAcum'][idx]
    df['RetCompDif'] = df['RetAcum'][tkr] - df['RetAcum'][idx]
    df["RetCompDif_Color"] = np.where(df.RetCompDif < 0, 'red', 'green')
    df['TBENCK'] = df['Close'][tkr] / df['Close'][idx]

    ### CALCULA ALPHA E BETA

    df['Alpha21dd'],df['Beta21dd'] = market_beta(df.Return[tkr], df.Return[idx], 21)
    df['Alpha50dd'],df['Beta50dd'] = market_beta(df.Return[tkr], df.Return[idx], 50)
    df['Alpha200dd'],df['Beta200dd'] = market_beta(df.Return[tkr], df.Return[idx], 200)

    '''
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

    ### REVERSÃO À MÉIA ###
    ### FIG 0 ---------------------------------------------------------------------------
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[.85,.15],
        #subplot_titles=("", "")
        specs= [
            [{'type' : 'candlestick'}, {'type' : 'histogram'}],
        ],
        )
    fig.add_trace( go.Candlestick ( x=df.index, open=df.Open[tkr], high=df.High[tkr], low=df.Low[tkr], close=df.Close[tkr], name=tkr), col=1, row=1)
    fig.add_trace( go.Scatter(x=df.index, y=df.CSMA21dd[tkr], mode='lines', name='MMA21', line_width=1,line_color='orange', line_shape='linear'), col=1, row=1 )
    fig.add_trace( go.Scatter(x=df.index, y=df.CSMA50dd[tkr], mode='lines', name='MMA50', line_width=1,line_color='navy', line_shape='linear'), col=1, row=1 )
    fig.add_trace( go.Scatter(x=df.index, y=df.CEMA200dd[tkr], mode='lines', name='EMA200', line_width=1,line_color='purple', line_shape='linear'), col=1, row=1 )

    fig.add_trace( go.Histogram(x=df.VarClose[tkr], name=tkr, histnorm='percent', offsetgroup=0), col=2, row=1 )

    fig.update_layout( title='<b>EVOLUÇÃO DO PREÇO DO ATIVO</b>' )
    fig.update_layout( xaxis_title='', yaxis_title='Preço', xaxis2_title='Var. %', yaxis2_title='Percent. Ocorrências', xaxis_rangeslider_visible=False, hovermode='x unified', legend=dict(orientation="h") )

    fig.update_traces(bingroup='overlay', nbinsx=20, marker_line_color='rgb(0,0,0)', marker_line_width=1.5, opacity=0.5, col=2, row=1, cumulative_enabled=False) 
    fig.update_xaxes( rangebreaks=[ dict(bounds=["sat", "mon"]) ]), 
    
    ### FIG 1 ---------------------------------------------------------------------------
    fig1 = make_subplots(
        rows=1, cols=2,
        column_widths=[.85,.15],
        subplot_titles=("", "Histograma (Percent)")
        )
    fig1.add_trace( go.Scatter(x=df.index, y=df.RSMA21dd[tkr], mode='lines', name='R_MMA21', line_width=1, line_color='orange', line_shape='linear'), col=1, row=1  )
    fig1.add_trace( go.Scatter(x=df.index, y=df.RSMA50dd[tkr], mode='lines', name='R_MMA50', line_width=1,line_color='navy', line_shape='linear'), col=1, row=1  )
    fig1.add_trace( go.Scatter(x=df.index, y=df.REMA200dd[tkr], mode='lines', name='R_EMA200', line_width=1,line_color='purple', line_shape='linear'), col=1, row=1  )
    fig1.add_hline(y=0, 
        line_color='black', line_dash='dot', line_width=1,
        annotation_text="Centro da Média", 
        annotation_position="bottom left", col=1, row=1)
    
    fig1.add_trace( go.Histogram(x=df.RSMA21dd[tkr], name='R_MMA21', histnorm='percent', offsetgroup=0), col=2, row=1  )
    fig1.add_trace( go.Histogram(x=df.RSMA50dd[tkr], name='R_MMA50', histnorm='percent', offsetgroup=0), col=2, row=1  )
    fig1.add_trace( go.Histogram(x=df.REMA200dd[tkr], name='R_EMA200', histnorm='percent', offsetgroup=0), col=2, row=1  )

    fig1.update_layout(title_text='<b>REVERSÃO À MÉDIA - Agrupada</b>', yaxis_title='<b>Valor</b>', xaxis_rangeslider_visible=False, hovermode='x unified', legend=dict(orientation="h") )
    fig1.update_layout(
        xaxis=dict(showgrid=False),
        xaxis2=dict(showgrid=False)
    )

    fig1.update_traces(bingroup='overlay', nbinsx=20, marker_line_color='rgb(0,0,0)', marker_line_width=1.5, opacity=0.5, col=2, row=1, cumulative_enabled=False) 

    fig1.update_xaxes( rangebreaks=[ dict(bounds=["sat", "mon"]) ] )


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
    fig2.add_trace( go.Scatter(x=df.index, y=df.RSMA21dd[tkr], mode='lines', line_width=1, name='R_MMA21', line_color='orange', line_shape='linear') , row=1, col=1  ),
    
    fig2.add_trace( 
        go.Indicator( 
            mode = "gauge+number+delta", 
            value = df.RSMA21dd[tkr][-1], 
            #title = {'text': "Reversão MMA21"}, 
            delta = {'reference': df.RSMA21dd[tkr].mean(), 'relative': True,'valueformat':'.2%'}, 
            gauge={
                'axis':{
                    'range':[math.floor(df.RSMA21dd[tkr].min()),math.ceil(df.RSMA21dd[tkr].max())],
                    'dtick': ( math.ceil(df.RSMA21dd[tkr].max()) - math.floor(df.RSMA21dd[tkr].min()) )/10,
                    'tickformat':'0.1f'
                },
                'steps' : [
                    {'range': [math.floor(df.RSMA21dd[tkr].min()), (math.floor(df.RSMA21dd[tkr].min())*0.5)], 'color': "rgba(50,50,200,0.55)"},
                    {'range': [(math.ceil(df.RSMA21dd[tkr].max())*0.5), math.ceil(df.RSMA21dd[tkr].max())], 'color': "rgba(200,50,50,0.55)"}],
                'threshold' : {'line': {'color': "red", 'width': 4}, 
                            'thickness': 1, 
                            'value': df.RSMA21dd[tkr].mean()},
                'bar': {'color': "black"}
            }
        ), row=1, col=2  ),

    fig2.add_trace( go.Scatter(x=df.index, y=df.RSMA50dd[tkr], mode='lines', line_width=1, name='R_MMA50', line_color='navy')  , row=2, col=1  )

    fig2.add_trace( 
        go.Indicator( 
            mode = "gauge+number+delta", 
            value = df.RSMA50dd[tkr][-1], 
            #title = {'text': "Reversão MMA50"}, 
            delta = {'reference': df.RSMA50dd[tkr].mean(), 'relative': True, 'valueformat':'.2%'}, 
            gauge={
                'axis':{
                    'range':[math.floor(df.RSMA50dd[tkr].min()),math.ceil(df.RSMA50dd[tkr].max())],
                    'dtick': ( math.ceil(df.RSMA50dd[tkr].max()) - math.floor(df.RSMA50dd[tkr].min()) )/10,
                    'tickformat':'0.1f'
                },
                'steps' : [
                    {'range': [math.floor(df.RSMA50dd[tkr].min()), (math.floor(df.RSMA50dd[tkr].min())*0.5)], 'color': "rgba(50,50,200,0.55)"},
                    {'range': [(math.ceil(df.RSMA50dd[tkr].max())*0.5), math.ceil(df.RSMA50dd[tkr].max())], 'color': "rgba(200,50,50,0.55)"}],
                'threshold' : {'line': {'color': "red", 'width': 4}, 
                            'thickness': 1, 
                            'value': df.RSMA50dd[tkr].mean()},
                'bar': {'color': "black"}
            }
        ), row=2, col=2  ),

    fig2.add_trace( go.Scatter(x=df.index, y=df.REMA200dd[tkr], mode='lines', line_width=1, name='R_EMA200', line_color='purple', line_shape='linear')  , row=3, col=1  )

    fig2.add_trace( 
        go.Indicator( 
            mode = "gauge+number+delta", 
            value = df.REMA200dd[tkr][-1], 
            #title = {'text': "Reversão EMA200"}, 
            delta = {'reference': df.REMA200dd[tkr].mean(), 'relative': True, 'valueformat':'.2%'}, 
            gauge={
                'axis':{
                    'range':[math.floor(df.REMA200dd[tkr].min()),math.ceil(df.REMA200dd[tkr].max())],
                    'dtick': ( math.ceil(df.REMA200dd[tkr].max()) - math.floor(df.REMA200dd[tkr].min()) )/10,
                    'tickformat':'0.1f'
                },
                'steps' : [
                    {'range': [math.floor(df.REMA200dd[tkr].min()), (math.floor(df.REMA200dd[tkr].min())*0.5)], 'color': "rgba(50,50,200,0.55)"},
                    {'range': [(math.ceil(df.REMA200dd[tkr].max())*0.5), math.ceil(df.REMA200dd[tkr].max())], 'color': "rgba(200,50,50,0.55)"}],
                'threshold' : {'line': {'color': "red", 'width': 4}, 
                            'thickness': 1, 
                            'value': df.REMA200dd[tkr].mean()},
                'bar': {'color': "black"}
            }
      ), row=3, col=2  ),

    #fig2.add_hline(y=0, 
    #    line_color='black', line_dash='dot', line_width=1,
    #    annotation_text="Centro da Média", 
    #    annotation_position="bottom left", 
    #    row=1, col=1,)

    fig2.update_layout(title_text='<b>REVERSÃO À MÉDIA - Individualizada</b>', yaxis_title='<b>Valor</b>', xaxis_rangeslider_visible=False, hovermode='x unified', legend=dict(orientation="h") )
    fig2.update_xaxes( rangebreaks=[ dict(bounds=["sat", "mon"]) ] )

    ### FIG 3 ---------------------------------------------------------------------------

    fig3 = make_subplots(
        rows=1, cols=3,
        column_widths=[.33, .33, .33],
        subplot_titles=("MÉDIA vs RSMA21dd", "MÉDIA vs RSMA50dd", "MÉDIA vs REMA200dd"),
    )
    fig3.add_trace( go.Scatter(name='', x=df.RSMA21dd[tkr], y=df.CSMA21dd[tkr], text=df.index.strftime("%d/%m/%Y"),
        mode='markers',
        marker=dict(
            size=7,
            color=df.RSMA21dd[tkr], #set color equal to a variable
            colorscale='Bluered', # one of plotly colorscales
            opacity=0.5,
            showscale=False),
        hovertemplate = "%{text} <br> RSMA21dd : %{x:.2f} </br> MÉDIA PREÇO : %{y:,.2f}"
        ), row=1, col=1 
    ) 
    fig3.add_trace( go.Scatter(name='', x=df.RSMA50dd[tkr], y=df.CSMA50dd[tkr], text=df.index.strftime("%d/%m/%Y"),
        mode='markers',
        marker=dict(
            size=7,
            color=df.RSMA50dd[tkr], #set color equal to a variable
            colorscale='Bluered', # one of plotly colorscales
            opacity=0.5,
            showscale=False),
        hovertemplate = "%{text} <br> RSMA50dd : %{x:.2f} </br> MÉDIA PREÇO : %{y:,.2f}"
        ), row=1, col=2 
    ) 
    fig3.add_trace( go.Scatter(name='', x=df.REMA200dd[tkr], y=df.CEMA200dd[tkr], text=df.index.strftime("%d/%m/%Y"),
        mode='markers',
        marker=dict(
            size=7,
            color=df.REMA200dd[tkr], #set color equal to a variable
            colorscale='Bluered', # one of plotly colorscales
            opacity=0.5,
            showscale=False),
        hovertemplate = "%{text} <br> REMA200dd : %{x:.2f} </br> MÉDIA PREÇO : %{y:,.2f}"
        ), row=1, col=3 
    ) 
    
    fig3.update_layout(title_text='<b>REVERSÃO À MÉDIA COMPARADA: MÉDIA</b>', showlegend=False, hovermode='x unified' )
    fig3.update_xaxes( rangebreaks=[ dict(bounds=["sat", "mon"]) ] )

    ### ALPHA, BETA, CORRELAÇÃO E RETORNO ###
    ### FIG 4 ---------------------------------------------------------------------------    
    fig4 = make_subplots(
        rows=1, cols=2,
        column_widths=[.85,.15],
        #subplot_titles=("", "")
        )
    
    fig4.add_trace( go.Scatter(x=df.index, y=df.RetAcum[tkr], mode="lines", line_width=1.5, line_color="blue", name=tkr, connectgaps=True, line_shape='linear') , col=1, row=1 ) 
    fig4.add_trace( go.Scatter(x=df.index, y=df.RetAcum[idx], mode="lines", line_width=1.5, line_dash='dot', line_color="black", name=idx, connectgaps=True, line_shape='linear') , col=1, row=1 ) 
    
    fig4.add_trace( go.Bar( x=df.index, y=df.RetCompDif, marker_color=df.RetCompDif_Color, opacity=0.5, name=tkr), col=1, row=1 ) 

    fig4.add_hline(y=0, 
        line_color='black', line_dash='dot', line_width=1,
        annotation_text="", 
        annotation_position="bottom left", col=1, row=1)
    
    fig4.add_trace( go.Histogram(x=df.RetAcum[tkr], name=tkr, histnorm='percent', offsetgroup=0), col=2, row=1  )
    fig4.add_trace( go.Histogram(x=df.RetAcum[idx], name=idx, histnorm='percent', offsetgroup=0), col=2, row=1  )

    fig4.update_layout( title='<b>RETORNO ACUMULADO</b>' )
    fig4.update_layout( xaxis_title='', yaxis_title='Var. Percentual Acumulada', xaxis2_title='Var. %', yaxis2_title='Percent. Ocorrências', hovermode='x unified', legend=dict(orientation="h") )

    fig4.update_traces(bingroup='overlay', nbinsx=20, marker_line_color='rgb(0,0,0)', marker_line_width=1.5, opacity=0.5, col=2, row=1, cumulative_enabled=False) 
    fig4.update_xaxes( rangebreaks=[ dict(bounds=["sat", "mon"]) ]), 

    ### FIG 5 ---------------------------------------------------------------------------    
    fig5 = make_subplots(
        rows=3, cols=2,
        column_widths=[0.85,.15],
        row_heights=[.33, .33, .33],
        specs= [
            [{'type' : 'xy'}, {'type' : 'indicator'}],
            [{'type' : 'xy'}, {'type' : 'indicator'}],
            [{'type' : 'xy'}, {'type' : 'indicator'}],
        ],)
    fig5.add_trace( go.Scatter(x=df.index, y=df.RCorr21dd, mode="lines", line_width=1, line_color="orange", name="RCorr21dd", connectgaps=True, line_shape='linear') , row=1, col=1 ) 
    fig5.add_trace( 
        go.Indicator( 
            mode = "gauge+number+delta", 
            value = df.RCorr21dd[-1], 
            #title = {'text': "Reversão MMA21"}, 
            delta = {'reference': df.RCorr21dd.mean(), 'relative': True,'valueformat':'.2%'}, 
            gauge={
                'axis':{
                    'range':[math.floor(df.RCorr21dd.min()),math.ceil(df.RCorr21dd.max())],
                    'dtick': ( math.ceil(df.RCorr21dd.max()) - math.floor(df.RCorr21dd.min()) )/10,
                    'tickformat':'0.1f'
                },
                'steps' : [
                    {'range': [math.floor(df.RCorr21dd.min()), (math.floor(df.RCorr21dd.min())*0.5)], 'color': "rgba(50,50,200,0.55)"},
                    {'range': [(math.ceil(df.RCorr21dd.max())*0.5), math.ceil(df.RCorr21dd.max())], 'color': "rgba(200,50,50,0.55)"}],
                'threshold' : {'line': {'color': "red", 'width': 4}, 
                            'thickness': 1, 
                            'value': df.RCorr21dd.mean()},
                'bar': {'color': "black"}
            }
        ), row=1, col=2  ),

    fig5.add_trace( go.Scatter(x=df.index, y=df.RCorr50dd, mode="lines", line_width=1, line_color="navy", name="RCorr50dd", connectgaps=True, line_shape='linear') , row=2, col=1 ) 
    fig5.add_trace( 
        go.Indicator( 
            mode = "gauge+number+delta", 
            value = df.RCorr50dd[-1], 
            #title = {'text': "Reversão MMA21"}, 
            delta = {'reference': df.RCorr50dd.mean(), 'relative': True,'valueformat':'.2%'}, 
            gauge={
                'axis':{
                    'range':[math.floor(df.RCorr50dd.min()),math.ceil(df.RCorr50dd.max())],
                    'dtick': ( math.ceil(df.RCorr50dd.max()) - math.floor(df.RCorr50dd.min()) )/10,
                    'tickformat':'0.1f'
                },
                'steps' : [
                    {'range': [math.floor(df.RCorr50dd.min()), (math.floor(df.RCorr50dd.min())*0.5)], 'color': "rgba(50,50,200,0.55)"},
                    {'range': [(math.ceil(df.RCorr50dd.max())*0.5), math.ceil(df.RCorr50dd.max())], 'color': "rgba(200,50,50,0.55)"}],
                'threshold' : {'line': {'color': "red", 'width': 4}, 
                            'thickness': 1, 
                            'value': df.RCorr50dd.mean()},
                'bar': {'color': "black"}
            }
        ), row=2, col=2  ),
        
    fig5.add_trace( go.Scatter(x=df.index, y=df.RCorr200dd, mode="lines", line_width=1, line_color="purple", name="RCorr200dd", connectgaps=True, line_shape='linear') , row=3, col=1 ) 
    fig5.add_trace( 
        go.Indicator( 
            mode = "gauge+number+delta", 
            value = df.RCorr200dd[-1], 
            #title = {'text': "Reversão MMA21"}, 
            delta = {'reference': df.RCorr200dd.mean(), 'relative': True,'valueformat':'.2%'}, 
            gauge={
                'axis':{
                    'range':[math.floor(df.RCorr200dd.min()),math.ceil(df.RCorr200dd.max())],
                    'dtick': ( math.ceil(df.RCorr200dd.max()) - math.floor(df.RCorr200dd.min()) )/10,
                    'tickformat':'0.1f'
                },
                'steps' : [
                    {'range': [math.floor(df.RCorr200dd.min()), (math.floor(df.RCorr200dd.min())*0.5)], 'color': "rgba(50,50,200,0.55)"},
                    {'range': [(math.ceil(df.RCorr200dd.max())*0.5), math.ceil(df.RCorr200dd.max())], 'color': "rgba(200,50,50,0.55)"}],
                'threshold' : {'line': {'color': "red", 'width': 4}, 
                            'thickness': 1, 
                            'value': df.RCorr200dd.mean()},
                'bar': {'color': "black"}
            }
        ), row=3, col=2  ),

    fig5.update_layout( title='<b>CORRELAÇÃO MÓVEL</b>', xaxis_title='', yaxis_title='<b>Valor', legend=dict(orientation="h"), hovermode='x unified' )
    fig5.update_xaxes( rangebreaks=[ dict(bounds=["sat", "mon"]) ]), 

    ### FIG 6 ---------------------------------------------------------------------------    
    fig6 = make_subplots(
        rows=3, cols=2,
        column_widths=[0.85,.15],
        row_heights=[.33, .33, .33],
        specs= [
            [{'type' : 'xy'}, {'type' : 'indicator'}],
            [{'type' : 'xy'}, {'type' : 'indicator'}],
            [{'type' : 'xy'}, {'type' : 'indicator'}],
        ],)
    fig6.add_trace( go.Scatter(x=df.index, y=df.Alpha21dd, mode="lines", line_width=1, line_color="orange", name="Alpha21dd", connectgaps=True, line_shape='linear') , row=1, col=1) 
    fig6.add_trace( 
        go.Indicator( 
            mode = "gauge+number+delta", 
            value = df.Alpha21dd[-1], 
            #title = {'text': "Reversão MMA21"}, 
            delta = {'reference': df.Alpha21dd.mean(), 'relative': True,'valueformat':'.2%'}, 
            gauge={
                'axis':{
                    'range':[math.floor(df.Alpha21dd.min()),math.ceil(df.Alpha21dd.max())],
                    'dtick': ( math.ceil(df.Alpha21dd.max()) - math.floor(df.Alpha21dd.min()) )/10,
                    'tickformat':'0.1f'
                },
                'steps' : [
                    {'range': [math.floor(df.Alpha21dd.min()), (math.floor(df.Alpha21dd.min())*0.5)], 'color': "rgba(50,50,200,0.55)"},
                    {'range': [(math.ceil(df.Alpha21dd.max())*0.5), math.ceil(df.Alpha21dd.max())], 'color': "rgba(200,50,50,0.55)"}],
                'threshold' : {'line': {'color': "red", 'width': 4}, 
                            'thickness': 1, 
                            'value': df.Alpha21dd.mean()},
                'bar': {'color': "black"}
            }
        ), row=1, col=2  ),

    fig6.add_trace( go.Scatter(x=df.index, y=df.Alpha50dd, mode="lines", line_width=1, line_color="navy", name="Alpha50dd", connectgaps=True, line_shape='linear') , row=2, col=1) 
    fig6.add_trace( 
        go.Indicator( 
            mode = "gauge+number+delta", 
            value = df.Alpha50dd[-1], 
            #title = {'text': "Reversão MMA21"}, 
            delta = {'reference': df.Alpha50dd.mean(), 'relative': True,'valueformat':'.2%'}, 
            gauge={
                'axis':{
                    'range':[math.floor(df.Alpha50dd.min()),math.ceil(df.Alpha50dd.max())],
                    'dtick': ( math.ceil(df.Alpha50dd.max()) - math.floor(df.Alpha50dd.min()) )/10,
                    'tickformat':'0.1f'
                },
                'steps' : [
                    {'range': [math.floor(df.Alpha50dd.min()), (math.floor(df.Alpha50dd.min())*0.5)], 'color': "rgba(50,50,200,0.55)"},
                    {'range': [(math.ceil(df.Alpha50dd.max())*0.5), math.ceil(df.Alpha50dd.max())], 'color': "rgba(200,50,50,0.55)"}],
                'threshold' : {'line': {'color': "red", 'width': 4}, 
                            'thickness': 1, 
                            'value': df.Alpha50dd.mean()},
                'bar': {'color': "black"}
            }
        ), row=2, col=2  ),

    fig6.add_trace( go.Scatter(x=df.index, y=df.Alpha200dd, mode="lines", line_width=1, line_color="purple", name="Alpha200dd", connectgaps=True, line_shape='linear') , row=3, col=1) 
    fig6.add_trace( 
        go.Indicator( 
            mode = "gauge+number+delta", 
            value = df.Alpha200dd[-1], 
            #title = {'text': "Reversão MMA21"}, 
            delta = {'reference': df.Alpha200dd.mean(), 'relative': True,'valueformat':'.2%'}, 
            gauge={
                'axis':{
                    'range':[math.floor(df.Alpha200dd.min()),math.ceil(df.Alpha200dd.max())],
                    'dtick': ( math.ceil(df.Alpha200dd.max()) - math.floor(df.Alpha200dd.min()) )/10,
                    'tickformat':'0.1f'
                },
                'steps' : [
                    {'range': [math.floor(df.Alpha200dd.min()), (math.floor(df.Alpha200dd.min())*0.5)], 'color': "rgba(50,50,200,0.55)"},
                    {'range': [(math.ceil(df.Alpha200dd.max())*0.5), math.ceil(df.Alpha200dd.max())], 'color': "rgba(200,50,50,0.55)"}],
                'threshold' : {'line': {'color': "red", 'width': 4}, 
                            'thickness': 1, 
                            'value': df.Alpha200dd.mean()},
                'bar': {'color': "black"}
            }
        ), row=3, col=2  ),
    
    fig6.update_layout( title='<b>ALPHA MÓVEL</b>', xaxis_title='', yaxis_title='<b>Valor', legend=dict(orientation="h"), hovermode='x unified')
    fig6.update_xaxes( rangebreaks=[ dict(bounds=["sat", "mon"]) ]), 

    ### FIG 7 ---------------------------------------------------------------------------    
    fig7 = make_subplots(
        rows=3, cols=2,
        column_widths=[0.85,.15],
        row_heights=[.33, .33, .33],
        specs= [
            [{'type' : 'xy'}, {'type' : 'indicator'}],
            [{'type' : 'xy'}, {'type' : 'indicator'}],
            [{'type' : 'xy'}, {'type' : 'indicator'}],
        ],)
    fig7.add_trace( go.Scatter(x=df.index, y=df.Beta21dd, mode="lines", line_width=1, line_color="orange", name="Beta21dd", connectgaps=True, line_shape='linear') , row=1, col=1) 
    fig7.add_trace( 
        go.Indicator( 
            mode = "gauge+number+delta", 
            value = df.Beta21dd[-1], 
            #title = {'text': "Reversão MMA21"}, 
            delta = {'reference': df.Beta21dd.mean(), 'relative': True,'valueformat':'.2%'}, 
            gauge={
                'axis':{
                    'range':[math.floor(df.Beta21dd.min()),math.ceil(df.Beta21dd.max())],
                    'dtick': ( math.ceil(df.Beta21dd.max()) - math.floor(df.Beta21dd.min()) )/10,
                    'tickformat':'0.1f'
                },
                'steps' : [
                    {'range': [math.floor(df.Beta21dd.min()), (math.floor(df.Beta21dd.min())*0.5)], 'color': "rgba(50,50,200,0.55)"},
                    {'range': [(math.ceil(df.Beta21dd.max())*0.5), math.ceil(df.Beta21dd.max())], 'color': "rgba(200,50,50,0.55)"}],
                'threshold' : {'line': {'color': "red", 'width': 4}, 
                            'thickness': 1, 
                            'value': df.Beta21dd.mean()},
                'bar': {'color': "black"}
            }
        ), row=1, col=2  ),

    fig7.add_trace( go.Scatter(x=df.index, y=df.Beta50dd, mode="lines", line_width=1, line_color="navy", name="Beta50dd", connectgaps=True, line_shape='linear') , row=2, col=1) 
    fig7.add_trace( 
        go.Indicator( 
            mode = "gauge+number+delta", 
            value = df.Beta50dd[-1], 
            #title = {'text': "Reversão MMA21"}, 
            delta = {'reference': df.Beta50dd.mean(), 'relative': True,'valueformat':'.2%'}, 
            gauge={
                'axis':{
                    'range':[math.floor(df.Beta50dd.min()),math.ceil(df.Beta50dd.max())],
                    'dtick': ( math.ceil(df.Beta50dd.max()) - math.floor(df.Beta50dd.min()) )/10,
                    'tickformat':'0.1f'
                },
                'steps' : [
                    {'range': [math.floor(df.Beta50dd.min()), (math.floor(df.Beta50dd.min())*0.5)], 'color': "rgba(50,50,200,0.55)"},
                    {'range': [(math.ceil(df.Beta50dd.max())*0.5), math.ceil(df.Beta50dd.max())], 'color': "rgba(200,50,50,0.55)"}],
                'threshold' : {'line': {'color': "red", 'width': 4}, 
                            'thickness': 1, 
                            'value': df.Beta50dd.mean()},
                'bar': {'color': "black"}
            }
        ), row=2, col=2  ),

    fig7.add_trace( go.Scatter(x=df.index, y=df.Beta200dd, mode="lines", line_width=1, line_color="purple", name="Beta200dd", connectgaps=True, line_shape='linear') , row=3, col=1) 
    fig7.add_trace( 
        go.Indicator( 
            mode = "gauge+number+delta", 
            value = df.Beta200dd[-1], 
            #title = {'text': "Reversão MMA21"}, 
            delta = {'reference': df.Beta200dd.mean(), 'relative': True,'valueformat':'.2%'}, 
            gauge={
                'axis':{
                    'range':[math.floor(df.Beta200dd.min()),math.ceil(df.Beta200dd.max())],
                    'dtick': ( math.ceil(df.Beta200dd.max()) - math.floor(df.Beta200dd.min()) )/10,
                    'tickformat':'0.1f'
                },
                'steps' : [
                    {'range': [math.floor(df.Beta200dd.min()), (math.floor(df.Beta200dd.min())*0.5)], 'color': "rgba(50,50,200,0.55)"},
                    {'range': [(math.ceil(df.Beta200dd.max())*0.5), math.ceil(df.Beta200dd.max())], 'color': "rgba(200,50,50,0.55)"}],
                'threshold' : {'line': {'color': "red", 'width': 4}, 
                            'thickness': 1, 
                            'value': df.Beta200dd.mean()},
                'bar': {'color': "black"}
            }
        ), row=3, col=2  ),

    fig7.update_layout( title='<b>BETA MÓVEL</b>', xaxis_title='', yaxis_title='<b>Valor', legend=dict(orientation="h"), hovermode='x unified')
    fig7.update_xaxes( rangebreaks=[ dict(bounds=["sat", "mon"]) ]), 

    return fig, fig2, fig1, fig3, fig4, fig5, fig6, fig7, ""