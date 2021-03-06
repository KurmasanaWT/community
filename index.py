# -*- coding: utf-8 -*-  

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import dash_trich_components as dtc
from dash.dependencies import Input, Output, State
import pandas as pd

from app import app 
from app import server

from codes import meanrev, correl, mosaic #returnlog

navbar = html.Div(className='topnav',
    children=[ 
        html.A( html.Img(src="static/logo.png", height="30"), href="../") 
    ]) 

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
                html.P("Como forma de contribuir para o desenvolvimento do mercado de capitais no Brasil, decidimos colaborar mais ativamente com a comunidade, desenvolvendo e publicando gratuitamente códigos em Python a partir de sugestões de especialistas em finanças e economia."),
                html.P("As primeiras sugestões já estão chegando e em breve vamos disponibilizá-las gratuitamente para todos!"),
                html.P("Notícias e anúncios serão publicados nessa seção do site. Não deixe de visitá-la frequentemente."),

                html.P([
                    "Siga a KWT no ", 
                    html.A(
                        [html.I(className="fab fa-twitter"), " Twitter"], 
                        href='https://twitter.com/KurmasanaWT', 
                        target="new"),
                    " ou no ", 
                    html.A(
                        [html.I(className="fab fa-github")," GitHub"], 
                        href='https://github.com/KurmasanaWT', 
                        target="new")
                    ]),

                html.P("Ficaremos muito felizes com suas observações e contribuições."),
                html.P("Um abraço!", style={'text-align':'right'}),
                dbc.Badge("16/03/2022")
            ]),
        ], className="cardSize", ),

      ### incluir mais cards aqui

    ], className='content'), fluid=True)

cards = dbc.Container(dbc.Row(
    [
     
        dbc.Card([
            dbc.CardHeader([html.H6(children=["INCEPTION CODEX"]),
            html.P(className='card-header-sub', children="Primeiros códigos desenvolvidos para a comunidade e com inputs da comunidade!"),
            #html.P(dbc.Button([html.I(className="fab fa-github")," GitHub "], href='https://github.com/KurmasanaWT/community/blob/main/codes/meanrev.py', target="new")),
            ]),
            #html.Br(),
            
            meanrev.get(), ############## IMPORTANT
            
            dbc.CardBody([

            ], style={'text-align':'center', 'justify-content':'center'}),
        ], className="cardSize-code", ),
     
    ### incluir mais cards aqui

    ], className='content'), fluid=True)

home_cards = dbc.Container(dbc.Row(
    [
        dbc.Card([
            dbc.CardHeader(html.Img(src="static/logo.png", height="30"), style={'text-align':'center'}),
            dbc.CardBody([
              
                html.P("A idéia aqui é democratizar o conhecimento e unir forças."),
                html.P("Nós acreditamos na capacidade de ir mais longe juntos, de exergar em grupo algo que nunca veríamos sozinhos."),
                html.P("Acreditamos no potencial ilimitado da inteligência e creatividade humanas."),
                html.P("Na força e simplicidade de um objetivo comum."),
                html.P("Se você é um especialista do mercado financeiro, um desenvolvedor Python ou alguém interessado na união do mundo das finanças com a tecnologia, junte-se a nós!"),
                html.P(["Siga a KWT no ", html.A([html.I(className="fab fa-twitter"), " Twitter"], href='https://twitter.com/KurmasanaWT', target="new")," ou no ", html.A([html.I(className="fab fa-github")," GitHub"], href='https://github.com/KurmasanaWT', target="new")]),
                html.P(["Se achar melhor, entre em contato por email:", html.Br(),html.A([" kwt-community@1971ventures.com"], href='mailto:kwt-community@1971ventures.com', target="new")]),
                dbc.Badge("22/03/2022")
            ]),
        ], className="cardSize", ),

      ### incluir mais cards aqui

    ], className='content'), fluid=True)

kwt_cards = dbc.Container(dbc.Row(
    [
        dbc.Card([
            dbc.CardHeader(html.H6("QUEM SOMOS NÓS")),
            dbc.CardImg(src="static/kwt_logowide.png"),
            dbc.CardBody([
                html.P("Nós somos parte do universo das Start-Ups, operando no espaço existente entre os experts do mercado de capitais e os geeks do eco-sistema de inovação."),
                html.P("Somos uma WEALTHTECH."),
                html.P("As palavras “riqueza” e “tecnologia” se juntaram para dar origem a uma nova geração de empresas de tecnologia financeira que criam soluções digitais para transformar o setor de investimentos e gestão de ativos."),
                html.P([
                    dbc.Badge("WealthTech"),
                    dbc.Badge("Artificial Intelligence"),
                    dbc.Badge("Machine Learning"),
                ], style={'text-align':'center'}),
            ], style={}),
        ], className="cardSize", ),

         dbc.Card([
            dbc.CardHeader(html.H6("REFERÊNCIAS")),
            dbc.CardBody([
                html.Li([html.B("FOCO"), " : Desenvolvimento de sistemas de Inteligência Artificial eficazes em aplicações no mercado de capitais."]),html.Br(),
                html.Li([html.B("ESTRATÉGIA"), " : Inteligência não é estatística. Inteligência é capacidade de percepção, interpretação, adaptação e ação."]),html.Br(),
                html.Li([html.B("INOVAÇÃO"), " : Superar a eficiência e eficácia da gestão humana no mercado de capitais em operações de swing-trade, democratizando a gestão de ativos e alavancando estratégias institucionais."]),html.Br(),
                html.Li([html.B("TRANSFORMAÇÃO"), " : Nossas soluções devem diminuir o gap entre o especialista em finanças e o especialista em tecnologia."]),html.Br(),
            ], style={}),
        ], className="cardSize", ),

        dbc.Card([
            dbc.CardHeader(html.H6("INVESTIDORES")),
            dbc.CardImg(src="static/1971v.png"),
            dbc.CardBody([
                html.P("A KURMASANA WEALTHTECH faz parte do portfolio de investimentos da 1971 VENTURES."),
                html.P([
                    dbc.Badge("1971 Ventures"),
                    dbc.Badge("Venture Capital"),
                ], style={'text-align':'center'}),
            ], style={}),
        ], className="cardSize", )

      ### incluir mais cards aqui

    ], className='content'), fluid=True)

mosaic = dbc.Container(dbc.Row(
    [
        dbc.Card([
            #dbc.CardHeader("WORLD NEWS COVERAGE", style={'text-align':'center'}),
            dbc.CardBody([
                mosaic.get(),
            ]),
        ], className="cardSize-vidbkg" ),

      ### incluir mais cards aqui

    ], className='content'), fluid=True)

sidebar = dtc.SideBar(className='sidenav',
    children=[
        dtc.SideBarItem(id='id_0', label="Página Inicial", icon="fas fa-home"),
        dtc.SideBarItem(id='id_1', label="Códigos Python", icon="fab fa-python"),
        dtc.SideBarItem(id='id_2', label="A Comunidade", icon="fas fa-users"),
        dtc.SideBarItem(id='id_3', label="Sobre Nós", icon="fas fa-address-card"),
        dtc.SideBarItem(id='id_4', label="World News (Beta)", icon="fas fa-video"),
        #dtc.SideBarItem(id='id_5', label="Twitter", icon="fab fa-twitter"),
    ])

main_content = html.Div([
    sidebar,
    html.Div([], id="page_content"),
])

alert=html.Div(
    dbc.Alert(
        id="alert-beta",
        className="alerts",
        children="*** VERSÃO BETA - 20220330 ***",
        dismissable=True,
        is_open=True,
        fade=True,
        duration=4000
    ))

content_0 = html.Div( [home_cards] )
content_1 = html.Div( [cards] )
content_2 = html.Div( [comm_cards] )
content_3 = html.Div( [kwt_cards] )
content_4 = html.Div( [mosaic] )

app.layout=dbc.Container(
    children=[
        alert,
        dbc.Row(className='Top', children=[navbar], style={'width':'100%', }),
        dbc.Row(className='Main', children=[main_content]),
    ], fluid=True)
            

@app.callback(
    Output("page_content", "children"),
    [
        Input("id_0", "n_clicks_timestamp"),
        Input("id_1", "n_clicks_timestamp"),
        Input("id_2", "n_clicks_timestamp"),
        Input("id_3", "n_clicks_timestamp"),
        Input("id_4", "n_clicks_timestamp"),
        #Input("PlayBtn001", "n_clicks"),
        #Input("id_4", "n_clicks_timestamp"),
        #Input("id_5", "n_clicks_timestamp")
    ]
)

def toggle_collapse(input0, input1, input2, input3, input4):
    btn_df = pd.DataFrame({"input0": [input0], "input1": [input1], "input2": [input2],
                           "input3": [input3], "input4": [input4]})
    
    btn_df = btn_df.fillna(0)

    if btn_df.idxmax(axis=1).values == "input0":
        return content_0
    if btn_df.idxmax(axis=1).values == "input1":
        return content_1
    if btn_df.idxmax(axis=1).values == "input2":
        return content_2
    if btn_df.idxmax(axis=1).values == "input3":
        return content_3
    if btn_df.idxmax(axis=1).values == "input4":
        return content_4
    #if btn_df.idxmax(axis=1).values == "input4":
    #    if btn001 is None:
    #        return "Not clicked."
    #    else:
    #        return codes.meanrev.get()

if __name__ == '__main__':
    app.run_server(debug=True)