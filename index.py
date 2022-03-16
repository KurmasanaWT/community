# -*- coding: utf-8 -*-  

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import dash_trich_components as dtc
from dash.dependencies import Input, Output, State
import pandas as pd

#from app import app 

app = dash.Dash(__name__, suppress_callback_exceptions=False, title='KWT-Community', external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])
server = app.server   

#theme_toggle = dtc.ThemeToggle()

navbar = html.Div(className='topnav',
    children=[ 
        html.A( html.Img(src="static/logo.png", height="47"), href="../") 
    ]) 

desc="Normalização de médias e a distância dos preços em relação à mesma para avaliar as bandas de cima e de baixo. &amp; &amp \n &#13; &#013; &lt;br&gt; \A Sponsor: <b>RAFI</b> (Twitter @RaphaFigueredo)."

cards = dbc.Container(dbc.Row(
    [
     
        dbc.Card([
            dbc.CardHeader([html.H6("REVERSÃO À MÉDIA"), dbc.Badge("16/03/2022")]),
            dbc.CardImg(src='static/mean.jpg'),
            dbc.CardBody([

                dbc.Button(html.I(className="fas fa-play"), href='../meanreversion.py', target="new"),
                dbc.Button(html.I(className="fab fa-github"), href='https://github.com/KurmasanaWT', target="new"),

                html.P("Normalização de médias e a distância dos preços em relação à mesma para avaliar as bandas de cima e de baixo."),

                html.P([
                    "Sponsor: ", 
                    html.A([" RAFI | ",html.I(className="fab fa-twitter"), " @RaphaFigueredo)"], href='https://twitter.com/RaphaFigueredo', target="new")
                ]),

                dbc.Badge("Médias"),
                dbc.Badge("FinTwit"),
                dbc.Badge("RAFI"),

            ], style={'text-align':'center', 'justify-content':'center'}),
        ], className="cardSize", ),
     
      ### incluir mais cards aqui

    ], className='content'), fluid=True)

comm_cards = dbc.Container(dbc.Row(
    [
        dbc.Card([
            dbc.CardHeader(html.Img(src="static/logo.png", height="30"), style={'text-align':'center'}),
            dbc.CardBody([
                html.P("Bem vindo!"),
                html.P([
                    html.Span("Este é o "),
                    html.P("Kurmasana Wealth Tech Community hub.", style={'color':'#0a0', 'font-weight':'bolder'})
                    ], style={'text-align':'center'}),
                html.P("Como forma de contribuir para o desenvolvimento do mercado de capitais no Brasil, decidimos colaborar mais ativamente com a comunidade, desenvolvendo e pubicando gratuitamente códigos em Python a partir de sugestões de especialistas em finanças e economia."),
                html.P("As primeiras sugestões já estão chegando e em breve vamos disponibilizá-las gratuitamente para todos!"),
                html.P("Notícias e anúncios serão publicados nessa seção do site. Não deixe de visitá-la frequentemente."),
                html.P(["Siga a KWT no ", html.A([html.I(className="fab fa-twitter"), " Twitter"], href='https://twitter.com/KurmasanaWT', target="new")," ou no ", html.A([html.I(className="fab fa-github")," GitHub"], href='https://github.com/KurmasanaWT', target="new")]),
                html.P("Ficaremos muito felizes com suas observações e contribuições."),
                html.P("Um abraço, Nós!", style={'text-align':'right'}),
                dbc.Badge("16/03/2022")
            ]),
        ], className="cardSize", ),
      
      ### incluir mais cards aqui

    ], className='content'), fluid=True)

kwt_cards = dbc.Container(dbc.Row(
    [
        dbc.Card([
            dbc.CardHeader(html.H6("QUEM SOMOS NÓS ???")),
            dbc.CardBody([
                html.P(html.Img(src="static/logo_only.png", height="60"), style={'text-align':'center'}),
                html.P("Nós somos parte do universo das Start-Ups, operando no espaço existente entre os experts do mercado de capitais e os geeks do eco-sistema de inovação."),
                html.P("Somos uma WEALTHTECH."),
                html.P("As palavras “riqueza” e “tecnologia” se juntaram para dar origem a uma nova geração de empresas de tecnologia financeira que criam soluções digitais para transformar o setor de investimentos e gestão de ativos."),
                html.P([
                    dbc.Badge("WealthTech"),
                    dbc.Badge("Artificiall Intelligence"),
                    dbc.Badge("Machine Learning"),
                ], style={'text-align':'center'}),
            ], style={}),
        ], className="cardSize", ),

        dbc.Card([
            dbc.CardHeader(html.H6("FUNÇÃO, CRENÇA E SONHO")),
            dbc.CardBody([
                html.P("Nossa FUNÇÃO: Intensificar a integração no espaço entre experts e geeks."),
                html.P("Nossa CRENÇA: Total democratização do conhecimento e da informação."),
                html.P("Nosso SONHO: Liberté, Égalité, Fraternité!"),
            ], style={}),
        ], className="cardSize", ),

        dbc.Card([
            dbc.CardHeader(html.H6("ANIMAIS DE PODER")),
            dbc.CardBody([
                html.P("Nosso ANIMAL DE PODER: A Tartaruga, afinal devagar se vai longe!"),
                html.P("Nos ensinamentos nativos americanos, a tartaruga é o símbolo mais velho do planeta Terra. É a personificação da deusa da energia e a Mãe eterna na qual nossa vida evolui. A tartaruga é considerada para o xamã como sendo o chão ou piso da Mãe Terra. Ela, com sua lentidão e grossa carapaça, nos ensina como podemos nos proteger."),
                html.P("Nossos Cachorros: Filomena (Genérico) e Apollo (Golden Retriever)."),
            ], style={}),
        ], className="cardSize", ),


         dbc.Card([
            dbc.CardHeader(html.H6("SÂNSCRITO")),
            dbc.CardBody([
                html.P("Kúrmásana dêvanágari कूर्मासन IAST kūrmāsana."),
                html.P("Em sânscrito kúrma é tartaruga. Também dá nome a uma nádí do corpo energético ou um sub-prana."),
                html.Span("Kūrma vāyu - responsável pelo piscar dos olhos. "),
                html.I(className="fas fa-grin-wink")
            ], style={}),
        ], className="cardSize", ),

        dbc.Card([
            dbc.CardHeader(html.H6("MITOLOGIA HINDU")),
            dbc.CardBody([
                html.P("Na mitologia hindu, a Tartaruga Mundial, chamada Kurma ou Kacchapa, sustenta quatro elefantes nas costas; eles, por sua vez, carregam nas costas o peso do mundo inteiro."),
                html.P("Conta a lenda hindu, que Vishnu tomou a forma de uma tartaruga, Kurma, para salvar a humanidade em uma batalha entre Devas (seres não humanos poderosos, tidos como divindades) e os Asuras (inimigos dos deuses ou demônios).")
            ], style={}),
        ], className="cardSize", ),
      
        dbc.Card([
            dbc.CardHeader(html.H6("CHINA")),
            dbc.CardBody([
                html.P("Na China, a tartaruga era um dos quatro animais sagrados no confucionismo."),
                 html.P("Durante o período Han, estelas eram montadas em cima de tartarugas de pedra, mais tarde ligadas a Bixi, o filho de casco de tartaruga do Rei Dragão."),
            ], style={}),
        ], className="cardSize", ),

        dbc.Card([
            dbc.CardHeader(html.H6("ROMA ANTIGA")),
            dbc.CardBody([
                html.P("Na Roma Antiga o exército usava a formação testudo ('tartaruga') onde os soldados formavam uma parede de escudos para proteção."),
            ], style={}),
        ], className="cardSize", ),

        dbc.Card([
            dbc.CardHeader(html.H6("ESOPO")),
            dbc.CardBody([
                html.P("Nas Fábulas de Esopo, 'A Tartaruga e a Lebre' conta como uma corrida desigual pode ser vencida pelo parceiro mais lento."),
            ], style={}),
        ], className="cardSize", ),
      
      ### incluir mais cards aqui

    ], className='content'), fluid=True)

sidebar = dtc.SideBar(className='sidenav',
    children=[
        dtc.SideBarItem(id='id_1', label="Python Codes", icon="fab fa-python"),
        dtc.SideBarItem(id='id_2', label="A Comunidade", icon="fas fa-users"),
        dtc.SideBarItem(id='id_3', label="Sobre Nós", icon="fas fa-user-astronaut"),
        #dtc.SideBarItem(id='id_4', label="GitHub", icon="fab fa-github"),
        #dtc.SideBarItem(id='id_5', label="Twitter", icon="fab fa-twitter"),
    ])

main_content = html.Div([
    sidebar,
    html.Div([], id="page_content"),
])

alert=html.Div(
    dbc.Alert(
        "*** VERSÃO BETA - 20220315 ***",
        id="alert-fade",
        dismissable=True,
        is_open=True,
        duration=4000
    ),
    style={'text-align':'center'})

content_1 = html.Div( [cards] )
content_2 = html.Div( [comm_cards] )
content_3 = html.Div( [kwt_cards] )

app.layout=dbc.Container(
    children=[
        alert,
        dbc.Row(className='Top', children=[navbar], style={'width':'100%', 'background-color':'#000'}),
        dbc.Row(className='Main', children=[main_content]),
    ], fluid=True)
            

@app.callback(
    Output("page_content", "children"),
    [
        Input("id_1", "n_clicks_timestamp"),
        Input("id_2", "n_clicks_timestamp"),
        Input("id_3", "n_clicks_timestamp"),
        #Input("id_4", "n_clicks_timestamp"),
        #Input("id_5", "n_clicks_timestamp")
    ]
)

def toggle_collapse(input1, input2, input3):#, input4, input5):
    btn_df = pd.DataFrame({"input1": [input1], "input2": [input2],
                           "input3": [input3]})#, "input4": [input4], 
                           #"input5": [input5]})
    
    btn_df = btn_df.fillna(0)

    if btn_df.idxmax(axis=1).values == "input1":
        return content_1
    if btn_df.idxmax(axis=1).values == "input2":
        return content_2
    if btn_df.idxmax(axis=1).values == "input3":
        return content_3
    #if btn_df.idxmax(axis=1).values == "input4":
    #    return html.Div([
    #        content_1,
    #        #dcc.Link(href='https://github.com/KurmasanaWT')
    #    ])
    #if btn_df.idxmax(axis=1).values == "input5":
    #    return html.Div([
    #        content_1,
    #        #dcc.Location(href='https://twitter.com/KurmasanaWT', refresh=True)
    #    ])

if __name__ == '__main__':
    app.run_server(debug=True)