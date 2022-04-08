from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
#import statistics
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
#import math
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
        font = dict(family="Roboto", size=10), 
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
        "hoverCompareCartesian",
        #"toggleSpikelines"
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

def speed_dial(df,item,tkr):
    
    if tkr==None:
        q1=df[item].quantile(q=0.10)
        q4=df[item].quantile(q=0.90)
        lmax=df[item].max()
        lmin=df[item].min()
        media=df[item].mean()
        lastv=df[item][-1]
        sticks=(lmax-lmin)/10
        titulo=str(item)
    else:
        q1=df[item][tkr].quantile(q=0.10)
        q4=df[item][tkr].quantile(q=0.90)
        lmax=df[item][tkr].max() #math.ceil()
        lmin=df[item][tkr].min() #math.floor()
        media=df[item][tkr].mean()
        lastv=df[item][tkr][-1]
        sticks=(lmax-lmin)/10
        titulo=str(item)+" - "+str(tkr)
    
    return go.Indicator( 
        mode = "gauge+number+delta", 
        title = titulo,
        #title_align = 'left',
        value = lastv, 
        number = {"valueformat": ".2f"},
        delta = {'reference': media, 'relative': True,'valueformat':'.2%'}, 
        gauge={
            #'shape':'bullet',
            'axis':{
                'range':[lmin,lmax],
                'dtick': sticks,
                'tickformat':'0.1f'
            },
            'steps' : [
                {'range': [lmin, q1], 'color': "rgba(50,50,200,0.55)"},
                {'range': [q4,lmax], 'color': "rgba(200,50,50,0.55)"}],
            'threshold' : {'line': {'color': "red", 'width': 4}, 
                        'thickness': 1, 
                        'value': media
                        },
            'bar': {'color': "black"}
        }
    )

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
                html.P(className='kwtdrops', children=[
                        html.H5("ATIVO"), dcc.Dropdown( id="ticker", options=tickers, value='VALE3.SA', clearable=False, style={'width':'300px'} ), 
                        html.H5("BENCHMARK"), dcc.Dropdown( id="indexer", options=tickers, value='^BVSP', clearable=False, style={'width':'300px'} ), 
                        html.H5("PERÍODO"), dcc.Dropdown( id="periods", options=periods, value='1y', clearable=False, style={'width':'10rem'} ),
                        html.H5("INTERVALO"), dcc.Dropdown( id="intervals", options=intervals, value='1d', clearable=False, style={'width':'10rem'} ),
                        dbc.Button(className="kwtchartbtn",id='submitb', n_clicks=0, children='Atualizar'),
                ]),
            ]),
            dcc.Tabs([
                dcc.Tab(label='Reversão à Média', children=[

                    dbc.Row([
                        html.Div([
                            html.P("Normalização de médias e a distância dos preços em relação à mesma para avaliar as bandas de cima e de baixo. A REVERSÃO À MÉDIA é uma teoria usada em finanças que sugere que a volatilidade dos preços dos ativos e os retornos históricos eventualmente reverterão para a média de longo prazo ou o nível médio de todo o conjunto de dados. Esse nível médio pode aparecer em vários contextos, como o crescimento econômico, a volatilidade de uma ação, a relação preço/lucro de uma ação (relação P/L) ou o retorno médio de uma indústria.", className="p-3"),
                            html.P([                   
                                dbc.Badge("Versão Beta 2022-004"),
                                dbc.Badge("Médias"),
                                dbc.Badge("FinTwit"),
                                dbc.Badge("RAFI"),
                                ], className="p-3"),
                            html.Span([
                                html.Span(""),
                                dbc.Button([html.I(className="fab fa-github")," GitHub "], href='https://github.com/KurmasanaWT/community/blob/main/codes/meanrev.py', target="new"),
                                ], className="p-3"),
                            ], className="d-flex w-100 p-3 justify-content-between")# className=('d-grid w-100 p-3 text-center justify-content-center'))
                        ]),

                    dbc.Row([
                        dcc.Graph(id="graph", config=config1),                             
                        dcc.Graph(id="graph1", config=config1),
                        dcc.Graph(id="graph2", config=config1),          
                        dcc.Graph(id="graph3", config=config1),
                        ]),

                    dbc.Row([
                        html.P([
                            "Conteúdo adicional sobre o tópico: ", 
                            ], className=('d-flex w-100 p-0 m-0 text-center justify-content-center')),
                        html.P([
                            dbc.ListGroup([

                            dbc.ListGroupItem(
                                html.Span([
                                    html.A('Investopedia',href='https://www.investopedia.com/terms/m/meanreversion.asp', target="new"),
                                    dbc.Badge("Definitions"),
                                    ],
                                    className="d-flex w-100 justify-content-between")),

                            dbc.ListGroupItem(
                                html.Span([
                                    html.A('FactSet',href='https://insight.factset.com/how-much-alpha-can-be-derived-from-a-mean-reversion-strategy', target="new"),
                                    dbc.Badge("Articles"),
                                    ],
                                    className="d-flex w-100 justify-content-between")),

                            dbc.ListGroupItem(
                                html.Span([
                                    html.A('SciELO',href='https://search.scielo.org/?fb=&q=%22mean+reversion%22&lang=pt&where=&filter%5Bin%5D%5B%5D=*', target="new"),
                                    dbc.Badge("Academic Papers"),
                                    ],
                                    className="d-flex w-100 justify-content-between")),      
                                    
                            dbc.ListGroupItem(
                                html.Span([
                                    html.A('Google Scholar',href='https://scholar.google.com.br/scholar?hl=pt-BR&as_sdt=0,5&q=%22mean+reversion%22+%22stock+market%22', target="new"),
                                    dbc.Badge("Academic Papers"),
                                    ],
                                    className="d-flex w-100 justify-content-between")),      
                                                                                                                                
                                ], className='d-grid p-3 justify-content-center')
                            ], className=('')),
                        
                        ]),

                    ]),   

                dcc.Tab(label='Retorno, Correlação, Alpha e Beta', children=[

                    dbc.Row([
                        html.Div([
                            html.P("RETORNO, também conhecido como retorno financeiro, de uma maneira simples, é o dinheiro ganho ou perdido em um investimento durante algum período de tempo. A CORRELAÇÃO, nas indústrias de finanças e investimentos, é uma estatística que mede o grau em que dois títulos se movem em relação um ao outro. As correlações são utilizadas na gestão avançada de carteiras, calculadas como o coeficiente de correlação, que tem um valor que deve situar-se entre -1,0 e +1,0. ALPHA (α) é um termo usado em investimentos para descrever a capacidade de uma estratégia de investimento de vencer o mercado, ou sua 'vantagem'. BETA é uma medida da volatilidade - ou risco sistemático - de um título ou portfólio em comparação com o mercado como um todo.", className="p-3"),
                            html.P([                   
                                dbc.Badge("Versão Beta 2022-004"),
                                dbc.Badge("Retorno"),
                                dbc.Badge("Correlação"),
                                dbc.Badge("Alpha"),
                                dbc.Badge("Beta"),
                                ], className="p-3"),    
                            html.Span([
                                html.Span(""),
                                dbc.Button([html.I(className="fab fa-github")," GitHub "], href='https://github.com/KurmasanaWT/community/blob/main/codes/meanrev.py', target="new"),
                                ], className="p-3"),
                            ], className="d-flex w-100 p-3 justify-content-between")# className=('d-grid w-100 p-3 text-center justify-content-center'))
                        ]),

                    dbc.Row([
                        dcc.Graph(id="cor_graph", config=config1),                             
                        dcc.Graph(id="cor_graph1", config=config1),
                        dcc.Graph(id="cor_graph2", config=config1),          
                        dcc.Graph(id="cor_graph3", config=config1),                     
                        ]),

                    dbc.Row([
                        html.P([
                            "Conteúdo adicional sobre o tópico: ", 
                            ], className=('d-flex w-100 p-0 m-0 text-center justify-content-center')),
                        html.P([
                            dbc.ListGroup([

                            dbc.ListGroupItem([
                                html.Span([html.Span("Investopedia", style={'color':'black !important'}), dbc.Badge("Definitions")], className="d-flex w-100 justify-content-between"),
                                html.Span([
                                    html.Li(html.A('Retorno',href='https://www.investopedia.com/terms/r/return.asp', target="new")),
                                    html.Li(html.A('Correlação',href='https://www.investopedia.com/terms/c/correlation.asp', target="new")),
                                    html.Li(html.A('Alpha',href='https://www.investopedia.com/terms/a/alpha.asp', target="new")),
                                    html.Li(html.A('Beta',href='https://www.investopedia.com/terms/b/beta.asp', target="new")),
                                    ])
                            ],className="d-grid w-100 justify-content-between"),

                            dbc.ListGroupItem(
                                html.Span([
                                    html.A('Investopedia',href='https://www.investopedia.com/ask/answers/051315/how-can-i-use-correlation-coefficient-predict-returns-stock-market.asp', target="new"),
                                    dbc.Badge("Articles"),
                                    ],
                                    className="d-flex w-100 justify-content-between")),

                            dbc.ListGroupItem(
                                html.Span([
                                    html.A('SciELO',href='https://search.scielo.org/?q=%22mean+reversion%22&lang=pt&count=15&from=0&output=site&sort=&format=summary&fb=&page=1&filter%5Bin%5D%5B%5D=*&q=%28return%0D%0A%29+OR+%28correlation%29+OR+%28alpha%29+OR+%28beta%29+AND+%28stock+market%29&lang=pt&page=1', target="new"),
                                    dbc.Badge("Academic Papers"),
                                    ],
                                    className="d-flex w-100 justify-content-between")),      
                                    
                            dbc.ListGroupItem(
                                html.Span([
                                    html.A('Google Scholar',href='https://scholar.google.com.br/scholar?q=%22return%22+%22correlation%22+%22alpha%22+%22beta%22+%22stock+market%22&hl=pt-BR&as_sdt=0,5', target="new"),
                                    dbc.Badge("Academic Papers"),
                                    ],
                                    className="d-flex w-100 justify-content-between")),      
                                                                                                                                
                                ], className='d-grid p-3 justify-content-center')
                            ], className=('')),
                        
                        ]),

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

    for n in df.Close:
        df.fillna(method="bfill",inplace=True)

        df['PrevClose',n]=df.Close[n].shift(1)
        df['VarClose',n] = df.Close[n].pct_change()
        #df['VarClose',n]=((df.Close[n]/df.PrevClose[n])-1)*100
        df['Return',n] = (np.log(df.Close[n] / df.Close[n].shift(1)))*100

        df.fillna(method="bfill",inplace=True)
        df.fillna(method="ffill",inplace=True)

    date_ini=df.index.min()
    date_end=df.index.max()

    for n in df.Close:

        df['MaxVal']=df.High[n].max()
        df['MinVal']=df.Low[n].min()

        df['RetAcum',n] = (np.log(df.Close[n] / df.Close[n].iloc[0]))*100
        df['Return21dd',n] = (np.log(df.Close[n] / df.Close[n].shift(per21dd)))*100
        df['Return50dd',n] = (np.log(df.Close[n] / df.Close[n].shift(per50dd)))*100
        df['Return200dd',n] = (np.log(df.Close[n] / df.Close[n].shift(per200dd)))*100
        ##divide by zero encountered in log

        df['VarAcum',n] = ((df.Close[n]/df.Close[n].iloc[0])-1)*100
        df["VarACum_Color",n] = np.where(df.VarAcum[n] < 0, 'red', 'green')

        ### REVERSÃO À MÉDIA ARITIMÉTICA DE 21 DIAS
        df['CSMA21dd',n]=df.Close[n].rolling(per21dd).mean()
        df['RSMA21dd',n]=((df.Close[n]/df['CSMA21dd',n])-1)*100

        ### REVERSÃO À MÉDIA ARITIMÉTICA DE 50 DIAS
        df['CSMA50dd',n]=df.Close[n].rolling(per50dd).mean()
        df['RSMA50dd',n]=((df.Close[n]/df['CSMA50dd',n])-1)*100

        ### REVERSÃO À MÉDIA EXPONENCIAL DE 200 DIAS
        df['CEMA200dd',n]=df.Close[n].ewm(span=per200dd, min_periods=per200dd, adjust=True).mean()
        df['REMA200dd',n]=((df.Close[n]/df['CEMA200dd',n])-1)*100

        df['VarOpen',n]=((df.Open[n]/df.Open[n].shift(1))-1)*100
        df['VarHigh',n]=((df.High[n]/df.High[n].shift(1))-1)*100
        df['VarLow',n]=((df.Low[n]/df.Low[n].shift(1))-1)*100
        df['VarIntra',n]=((df.Close[n]/df.Open[n])-1)*100
        df['VarHighLow',n]=((df.High[n]/df.Low[n])-1)*100
        df['VarCloseOpen',n]=((df.Open[n]/df.PrevClose[n])-1)*100
        df['VarHL',n]=(np.log(df.High[n] / df.Low[n].shift(1))+np.log(df.Low[n] / df.High[n].shift(1)) ) / 2 * 100

    df = df.sort_index(axis=1)

    ### ROLLING CORRELATION

    df['RCorr21dd'] = df['VarClose'][tkr].rolling(per21dd).corr(df['VarClose'][idx])
    df['RCorr50dd'] = df['VarClose'][tkr].rolling(per50dd).corr(df['VarClose'][idx])
    df['RCorr200dd'] = df['VarClose'][tkr].rolling(per200dd).corr(df['VarClose'][idx])

    ### RETORNO COMPARADO

    df['RetComp'] = df['VarAcum'][tkr] / df['VarAcum'][idx]
    df['RetCompDif'] = df['VarAcum'][tkr] - df['VarAcum'][idx]
    df["RetCompDif_Color"] = np.where(df.RetCompDif < 0, 'red', 'green')
    df['TBENCK'] = df['Close'][tkr] / df['Close'][idx]

    ### CALCULA ALPHA E BETA

    df['Alpha21dd'],df['Beta21dd'] = market_beta(df.VarClose[tkr], df.VarClose[idx], 21)
    df['Alpha50dd'],df['Beta50dd'] = market_beta(df.VarClose[tkr], df.VarClose[idx], 50)
    df['Alpha200dd'],df['Beta200dd'] = market_beta(df.VarClose[tkr], df.VarClose[idx], 200)

    '''    
    #21dd
    df['CMax']=df.Close.rolling(per21dd).max()
    df['CMin']=df.Close.rolling(per21dd).min()  
    df['VarCMax']=df['VarClose'].rolling(per21dd).max()
    df['VarCMin']=df['VarClose'].rolling(per21dd).min()  
    df['Overshoot'] = ((abs(df['VarCMax'].rolling(per21dd).mean())/abs(df['VarCMin'].rolling(per21dd).mean()))-1)*100  

    '''

    ####### CONSTROI GRÁFICOS
    #

    ### REVERSÃO À MÉIA ###
    ### FIG 0 ---------------------------------------------------------------------------
    fig = make_subplots(
        rows=3, cols=2,
        column_widths=[.85,.15],
        #row_heights=[.40,.60],
        horizontal_spacing=0.05, vertical_spacing=0.15,
        #subplot_titles=("", "")
        specs= [
            [ {'type' : 'candlestick', 'rowspan':3}, {'type' : 'indicator'} ],
            [ None, {'type' : 'indicator'} ],
            [ None, {'type' : 'histogram'} ]
        ],
        )
    fig.add_trace( go.Candlestick ( x=df.index, open=df.Open[tkr], high=df.High[tkr], low=df.Low[tkr], close=df.Close[tkr], name=tkr), col=1, row=1)
    fig.add_trace( go.Scatter(x=df.index, y=df.CSMA21dd[tkr], mode='lines', name='MMA21', line_width=2,line_color='darkorange', line_shape='linear'), col=1, row=1 )
    fig.add_trace( go.Scatter(x=df.index, y=df.CSMA50dd[tkr], mode='lines', name='MMA50', line_width=2,line_color='blue', line_shape='linear'), col=1, row=1 )
    fig.add_trace( go.Scatter(x=df.index, y=df.CEMA200dd[tkr], mode='lines', name='EMA200', line_width=2,line_color='mediumvioletred', line_shape='linear'), col=1, row=1 )
    
    fig.add_trace( speed_dial(df,"Close",tkr), row=1, col=2 )

    fig.add_trace( speed_dial(df,"VarClose",tkr), row=2, col=2 )

    fig.add_trace( go.Histogram(x=df.VarClose[tkr], name=tkr, histnorm='percent', offsetgroup=0), col=2, row=3 )

    fig.update_layout( title='<b>EVOLUÇÃO DO PREÇO DO ATIVO</b>',  xaxis_rangeslider_visible=False, hovermode='x unified', legend=dict(orientation="h") )

    fig.update_layout( xaxis_title='', xaxis2_title='Var. %' )
    fig.update_layout( xaxis_showspikes=True, xaxis_spikemode='across' )
    
    fig.update_layout( yaxis_title='Preço', yaxis2_title='Percent. Ocorrências' )
    fig.update_layout( yaxis_showspikes=True, yaxis_spikemode='across', yaxis_spikesnap="cursor" )

    fig.update_traces(bingroup='overlay', nbinsx=20, marker_line_color='rgb(0,0,0)', marker_line_width=1.5, opacity=0.5, col=2, row=3, cumulative_enabled=False) 
    fig.update_xaxes( rangebreaks=[ dict(bounds=["sat", "mon"]) ]), 

    ### FIG 2 - MOD: POSIÇÃO 2 NO GRID --------------------------------------------------------------------------
    
    fig2_rows=3
    fig2_cols=2
    fig2_values=[('RSMA21dd',df.RSMA21dd[tkr],'darkorange'),('RSMA50dd',df.RSMA50dd[tkr],'blue'),('REMA200dd',df.REMA200dd[tkr],'mediumvioletred')]

    fig2 = make_subplots(
        rows=fig2_rows, cols=fig2_cols,
        column_widths=[0.85,.15],
        #row_heights=[0.30,0.30,0.30],
        horizontal_spacing=0.05, vertical_spacing=0.15,
        specs= [
            [{'type' : 'xy'}, {'type' : 'indicator'}],
            [{'type' : 'xy'}, {'type' : 'indicator'}],
            [{'type' : 'xy'}, {'type' : 'indicator'}],
        ])

    contador=0
    for n in fig2_values:
        contador += 1
        fig2.add_trace( go.Scatter(x=df.index, y=n[1].values, mode='lines', line_width=2, name=n[0], line_color=n[2], line_shape='linear') , row=contador, col=1  )
        fig2.add_trace( speed_dial(df,n[0],tkr), row=contador, col=2 )
        fig2.add_shape( x0=date_ini, x1=date_end, y0=n[1].values[-1], y1=n[1].values[-1], type="line", line_color='red', line_dash='dash', line_width=1, row=contador, col=1 ) 
        fig2.add_shape( x0=date_ini, x1=date_end, y0=0, y1=0, line_color='black', line_width=1, line_dash='dot', row=contador, col=1)
 
    fig2.update_layout(title_text='<b>REVERSÃO À MÉDIA - Individualizada</b>', yaxis_title='Valor', yaxis2_title='Valor', yaxis3_title='Valor', xaxis_rangeslider_visible=False, hovermode='x unified', legend=dict(orientation="h") )

    fig2.update_xaxes( rangebreaks=[ dict(bounds=["sat", "mon"]) ] )

    ### FIG 1 ---------------------------------------------------------------------------
    fig1 = make_subplots(
        rows=1, cols=2,
        column_widths=[.85,.15],
        horizontal_spacing=0.05, vertical_spacing=0.15,
        #subplot_titles=("", "Histograma (Percent)")
        )
    fig1.add_trace( go.Scatter(x=df.index, y=df.RSMA21dd[tkr], mode='lines', name='R_MMA21', line_width=2, line_color='darkorange', line_shape='linear'), col=1, row=1  )
    fig1.add_trace( go.Scatter(x=df.index, y=df.RSMA50dd[tkr], mode='lines', name='R_MMA50', line_width=2,line_color='blue', line_shape='linear'), col=1, row=1  )
    fig1.add_trace( go.Scatter(x=df.index, y=df.REMA200dd[tkr], mode='lines', name='R_EMA200', line_width=2,line_color='mediumvioletred', line_shape='linear'), col=1, row=1  )
    fig1.add_hline(y=0, 
        line_color='black', line_dash='dot', line_width=1,
        annotation_text="Centro da Média", 
        annotation_position="bottom left", col=1, row=1)
    
    fig1.add_trace( go.Histogram(x=df.RSMA21dd[tkr], name='R_MMA21', histnorm='percent', offsetgroup=0), col=2, row=1  )
    fig1.add_trace( go.Histogram(x=df.RSMA50dd[tkr], name='R_MMA50', histnorm='percent', offsetgroup=0), col=2, row=1  )
    fig1.add_trace( go.Histogram(x=df.REMA200dd[tkr], name='R_EMA200', histnorm='percent', offsetgroup=0), col=2, row=1  )

    fig1.update_layout(title_text='<b>REVERSÃO À MÉDIA - Agrupada</b>', xaxis2_title='Var. %', yaxis2_title='Percent. Ocorrências' , xaxis_rangeslider_visible=False, hovermode='x unified', legend=dict(orientation="h") )

    fig1.update_layout(
        xaxis=dict(showgrid=False),
        xaxis2=dict(showgrid=False)
    )

    fig1.update_traces(bingroup='overlay', nbinsx=20, marker_line_color='rgb(0,0,0)', marker_line_width=1.5, opacity=0.5, col=2, row=1, cumulative_enabled=False) 

    fig1.update_xaxes( rangebreaks=[ dict(bounds=["sat", "mon"]) ] )

    ### FIG 3 ---------------------------------------------------------------------------

    fig3_rows=1
    fig3_cols=3
    fig3 = make_subplots(
        rows=fig3_rows, cols=fig3_cols,
        column_widths=[.33, .33, .33],
        horizontal_spacing=0.05, vertical_spacing=0.15,
        subplot_titles=("MÉDIA vs RSMA21dd", "MÉDIA vs RSMA50dd", "MÉDIA vs REMA200dd"),
    )
    fig3.add_trace( go.Scatter(name='', x=df.RSMA21dd[tkr], y=df.CSMA21dd[tkr], text=df.index.strftime("%d/%m/%Y"),
        mode='markers',
        marker=dict(
            size=9,
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
            size=9,
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
            size=9,
            color=df.REMA200dd[tkr], #set color equal to a variable
            colorscale='Bluered', # one of plotly colorscales
            opacity=0.5,
            showscale=False),
        hovertemplate = "%{text} <br> REMA200dd : %{x:.2f} </br> MÉDIA PREÇO : %{y:,.2f}"
        ), row=1, col=3 
    )  
    
    fig3.update_layout(title_text='<b>REVERSÃO À MÉDIA COMPARADA: MÉDIA</b>', showlegend=False, hovermode='closest' )

    for n in range(1, fig3_cols+1):
        q=n/fig3_cols
        fig3.add_vline(x=0, 
            line_color='black', line_width=1, line_dash='dot',
            annotation_text="", 
            annotation_position="bottom left", col=n, row=1)

    #fig3.update_layout( xaxis_showspikes=True, xaxis_spikemode='across', xaxis_mirror=True)
    fig3.update_layout( xaxis_mirror=True, xaxis2_mirror=True, xaxis3_mirror='all' )
    fig3.update_layout( yaxis_mirror=True, yaxis2_mirror=True, yaxis3_mirror='all' )

    #for xaxis in fig3['layout']['xaxis']: 
    #    xaxis['showspikes']= 'True',
    #    xaxis['spikemode']= 'across'

    annonformat = dict(
        font_size=11, 
        align="left",
        bgcolor="rgba(255,255,255,.60)",
        bordercolor="green", borderpad=3,
        showarrow=True, arrowhead=2, arrowwidth=1, arrowcolor="green",
        xref="x", yref="y",
        ax=50, ay=-50)

    fig3.add_annotation(
        annonformat,
        x=df.RSMA21dd[tkr][-1], 
        y=df.CSMA21dd[tkr][-1],
        text= ("<b>VALOR ATUAL</b><br>Data: "+
            df.RSMA21dd.index[-1].strftime("%d/%m/%y"))+
            f"<br>RSMA21dd: {df.RSMA21dd[tkr][-1]:,.2f}"+
            f"<br>Média: {df.CSMA21dd[tkr][-1]:,.2f}",
        row=1, col=1 )

    fig3.add_annotation(
        annonformat,
        x=df.RSMA50dd[tkr][-1], 
        y=df.CSMA50dd[tkr][-1],
        text= ("<b>VALOR ATUAL</b><br>Data: "+
            df.RSMA50dd.index[-1].strftime("%d/%m/%y"))+
            f"<br>RSMA50dd: {df.RSMA50dd[tkr][-1]:,.2f}"+
            f"<br>Média: {df.CSMA50dd[tkr][-1]:,.2f}",
        row=1, col=2 )

    fig3.add_annotation(
        annonformat,
        x=df.REMA200dd[tkr][-1], 
        y=df.CEMA200dd[tkr][-1],
        text= ("<b>VALOR ATUAL</b><br>Data: "+
            df.REMA200dd.index[-1].strftime("%d/%m/%y"))+
            f"<br>REMA200dd: {df.REMA200dd[tkr][-1]:,.2f}"+
            f"<br>Média: {df.CEMA200dd[tkr][-1]:,.2f}",
        row=1, col=3 )

    fig3.update_xaxes( rangebreaks=[ dict(bounds=["sat", "mon"]) ] )

    ### ALPHA, BETA, CORRELAÇÃO E RETORNO ###

    ### FIG 4 ---------------------------------------------------------------------------    
    
    fig4 = make_subplots(
        rows=3, cols=2,
        column_widths=[.85,.15],
        row_heights=[.33, .33, .33],
        horizontal_spacing=0.05, vertical_spacing=0.15,
        specs= [
            [ {'type' : 'xy', 'rowspan':3}, {'type' : 'indicator'} ],
            [ None, {'type' : 'indicator'} ],
            [ None, {'type' : 'histogram'} ]
        ],
        )
    
    fig4.add_trace( go.Scatter(x=df.index, y=df.VarAcum[tkr], mode="lines", line_width=2, line_color="blue", name=tkr, connectgaps=True, line_shape='linear') , col=1, row=1 ) 
    fig4.add_trace( go.Scatter(x=df.index, y=df.VarAcum[idx], mode="lines", line_width=2, line_dash='dot', line_color="black", name=idx, connectgaps=True, line_shape='linear') , col=1, row=1 ) 
    fig4.add_trace( go.Bar( x=df.index, y=df.RetCompDif, marker_color=df.RetCompDif_Color, opacity=0.5, name='Delta p.p.'), col=1, row=1 ) 

    fig4.add_hline(y=0, 
        line_color='black', line_dash='dot', line_width=1,
        annotation_text="", 
        annotation_position="bottom left", col=1, row=1)

    fig4.add_trace( speed_dial(df,"VarAcum",tkr), row=1, col=2 )

    fig4.add_trace( speed_dial(df,"VarAcum",idx), row=2, col=2 )
    
    fig4.add_trace( go.Histogram(x=df.VarAcum[tkr], name=tkr, histnorm='percent', offsetgroup=0), col=2, row=3  )
    fig4.add_trace( go.Histogram(x=df.VarAcum[idx], name=idx, histnorm='percent', offsetgroup=0), col=2, row=3  )

    fig4.update_layout( title='<b>RETORNO ACUMULADO</b>', legend=dict(orientation="h"), hovermode='x unified')
    #fig4.update_layout( xaxis_title='', yaxis_title='Var. Percentual Acumulada', xaxis2_title='Var. %', yaxis2_title='Percent. Ocorrências', hovermode='x unified', legend=dict(orientation="h") )
    fig4.update_layout( xaxis_title='', xaxis2_title='Var. %' )
    fig4.update_layout( xaxis_showspikes=True, xaxis_spikemode='across' )
    
    fig4.update_layout( yaxis_title='Var. Percentual Acumulada', yaxis2_title='Percent. Ocorrências' )
    fig4.update_layout( yaxis_showspikes=True, yaxis_spikemode='across', yaxis_spikesnap='cursor')

    fig4.update_traces(bingroup='overlay', nbinsx=20, marker_line_color='rgb(0,0,0)', marker_line_width=1.5, opacity=0.5, col=2, row=3, cumulative_enabled=False) 
    fig4.update_xaxes( rangebreaks=[ dict(bounds=["sat", "mon"]) ])

    ### FIG 5 ---------------------------------------------------------------------------    
    
    fig5 = make_subplots(
        rows=3, cols=2,
        column_widths=[0.85,.15],
        row_heights=[.33, .33, .33],
        horizontal_spacing=0.05, vertical_spacing=0.15,
        specs= [
            [{'type' : 'xy'}, {'type' : 'indicator'}],
            [{'type' : 'xy'}, {'type' : 'indicator'}],
            [{'type' : 'xy'}, {'type' : 'indicator'}],
        ],)

    fig5.add_trace( go.Scatter(x=df.index, y=df.RCorr21dd, mode="lines", line_width=2, line_color="darkorange", name="RCorr21dd", connectgaps=True, line_shape='linear') , row=1, col=1 ) 
    fig5.add_trace( speed_dial(df,"RCorr21dd",None), row=1, col=2 )

    fig5.add_trace( go.Scatter(x=df.index, y=df.RCorr50dd, mode="lines", line_width=2, line_color="blue", name="RCorr50dd", connectgaps=True, line_shape='linear') , row=2, col=1 ) 
    fig5.add_trace( speed_dial(df,"RCorr50dd",None), row=2, col=2 )
        
    fig5.add_trace( go.Scatter(x=df.index, y=df.RCorr200dd, mode="lines", line_width=2, line_color="mediumvioletred", name="RCorr200dd", connectgaps=True, line_shape='linear') , row=3, col=1 ) 
    fig5.add_trace( speed_dial(df,"RCorr200dd",None), row=3, col=2 )

    fig5.update_layout( title='<b>CORRELAÇÃO MÓVEL</b>', xaxis_title='', yaxis_title='<b>Valor', legend=dict(orientation="h"), hovermode='x unified' )

    fig5.add_shape( x0=date_ini, x1=date_end, y0=0, y1=0, type="line", line_color='black', line_dash='dot', line_width=1, row=1, col=1 ) 
    fig5.add_shape( x0=date_ini, x1=date_end, y0=0, y1=0, type="line", line_color='black', line_dash='dot', line_width=1, row=2, col=1 ) 
    fig5.add_shape( x0=date_ini, x1=date_end, y0=0, y1=0, type="line", line_color='black', line_dash='dot', line_width=1, row=3, col=1 ) 
 
    fig5.update_xaxes( rangebreaks=[ dict(bounds=["sat", "mon"]) ]) 

    ### FIG 6 ---------------------------------------------------------------------------    
    fig6 = make_subplots(
        rows=3, cols=2,
        column_widths=[0.85,.15],
        row_heights=[.33, .33, .33],
        horizontal_spacing=0.05, vertical_spacing=0.15,
        specs= [
            [{'type' : 'xy'}, {'type' : 'indicator'}],
            [{'type' : 'xy'}, {'type' : 'indicator'}],
            [{'type' : 'xy'}, {'type' : 'indicator'}],
        ],)
    fig6.add_trace( go.Scatter(x=df.index, y=df.Alpha21dd, mode="lines", line_width=2, line_color="darkorange", name="Alpha21dd", connectgaps=True, line_shape='linear') , row=1, col=1) 
    fig6.add_trace( speed_dial(df,"Alpha21dd",None), row=1, col=2 )

    fig6.add_trace( go.Scatter(x=df.index, y=df.Alpha50dd, mode="lines", line_width=2, line_color="blue", name="Alpha50dd", connectgaps=True, line_shape='linear') , row=2, col=1) 
    fig6.add_trace( speed_dial(df,"Alpha50dd",None), row=2, col=2 )

    fig6.add_trace( go.Scatter(x=df.index, y=df.Alpha200dd, mode="lines", line_width=2, line_color="mediumvioletred", name="Alpha200dd", connectgaps=True, line_shape='linear') , row=3, col=1) 
    fig6.add_trace( speed_dial(df,"Alpha200dd",None), row=3, col=2 )
    
    fig6.update_layout( title='<b>ALPHA MÓVEL</b>', xaxis_title='', yaxis_title='<b>Valor', legend=dict(orientation="h"), hovermode='x unified')

    fig6.add_shape( x0=date_ini, x1=date_end, y0=0, y1=0, type="line", line_color='black', line_dash='dot', line_width=1, row=1, col=1 ) 
    fig6.add_shape( x0=date_ini, x1=date_end, y0=0, y1=0, type="line", line_color='black', line_dash='dot', line_width=1, row=2, col=1 ) 
    fig6.add_shape( x0=date_ini, x1=date_end, y0=0, y1=0, type="line", line_color='black', line_dash='dot', line_width=1, row=3, col=1 ) 

    fig6.update_xaxes( rangebreaks=[ dict(bounds=["sat", "mon"]) ])

    ### FIG 7 ---------------------------------------------------------------------------    
    fig7 = make_subplots(
        rows=3, cols=2,
        column_widths=[0.85,.15],
        row_heights=[.33, .33, .33],
        horizontal_spacing=0.05, vertical_spacing=0.15,
        specs= [
            [{'type' : 'xy'}, {'type' : 'indicator'}],
            [{'type' : 'xy'}, {'type' : 'indicator'}],
            [{'type' : 'xy'}, {'type' : 'indicator'}],
        ],)
    fig7.add_trace( go.Scatter(x=df.index, y=df.Beta21dd, mode="lines", line_width=2, line_color="darkorange", name="Beta21dd", connectgaps=True, line_shape='linear') , row=1, col=1) 
    fig7.add_trace( speed_dial(df,"Beta21dd",None), row=1, col=2 ),

    fig7.add_trace( go.Scatter(x=df.index, y=df.Beta50dd, mode="lines", line_width=2, line_color="blue", name="Beta50dd", connectgaps=True, line_shape='linear') , row=2, col=1) 
    fig7.add_trace( speed_dial(df,"Beta50dd",None), row=2, col=2 ),

    fig7.add_trace( go.Scatter(x=df.index, y=df.Beta200dd, mode="lines", line_width=2, line_color="mediumvioletred", name="Beta200dd", connectgaps=True, line_shape='linear') , row=3, col=1) 
    fig7.add_trace( speed_dial(df,"Beta200dd",None), row=3, col=2 ),

    fig7.update_layout( title='<b>BETA MÓVEL</b>', xaxis_title='', yaxis_title='<b>Valor', legend=dict(orientation="h"), hovermode='x unified')

    fig7.add_shape( x0=date_ini, x1=date_end, y0=0, y1=0, type="line", line_color='black', line_dash='dot', line_width=1, row=1, col=1 ) 
    fig7.add_shape( x0=date_ini, x1=date_end, y0=0, y1=0, type="line", line_color='black', line_dash='dot', line_width=1, row=2, col=1 ) 
    fig7.add_shape( x0=date_ini, x1=date_end, y0=0, y1=0, type="line", line_color='black', line_dash='dot', line_width=1, row=3, col=1 ) 

    fig7.update_xaxes( rangebreaks=[ dict(bounds=["sat", "mon"]) ]), 

    return fig, fig2, fig1, fig3, fig4, fig5, fig6, fig7, ""