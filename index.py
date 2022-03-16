# -*- coding: utf-8 -*-  

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import dash_trich_components as dtc
from dash.dependencies import Input, Output, State
import pandas as pd
import webbrowser

app = dash.Dash(__name__, suppress_callback_exceptions=False, title='KWT-Community', external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])
server = app.server   

#theme_toggle = dtc.ThemeToggle()

navbar = html.Div(className='topnav',
    children=[ 
        html.A( html.Img(src="static/logo.png", height="47"), href="../") 
    ]) 

cards = dbc.Container(dbc.Row(
    [

        dtc.Card(
            className='content',
            link='https://github.com/KurmasanaWT/community/ram_raphi',
            image='static/mean.jpg',
            title='Reversão à Média',
            description="Normalização de médias e a distância dos preços em relação à mesma para avaliar as bandas de cima e de baixo. Veja também <https://www.investopedia.com/terms/m/meanreversion.asp>",
            badges=['Python', 'FinTwit', 'Raphi'],
            git='https://github.com/KurmasanaWT/community',
            dark=True,
        ),
      
      ### incluir mais cards aqui

    ]), fluid=True)

comm_cards = dbc.Container(dbc.Row(
    [
        dbc.Card([
            dbc.CardHeader(html.Img(src="static/logo.png", height="30"), style={'text-align':'center'}),
            dbc.CardBody([
                "Bem vindo!",
                html.Br(),
                "Este é o ",
                html.Span("Kurmasana Wealth Tech Community hub.", style={'color':'#0a0', 'font-weight':'bolder'}),
                html.P(),
                "Como forma de contribuir para o desenvolvimento do mercado de capitais no Brasil, decidimos colaborar mais ativamente com a comunidade, desenvolvendo e pubicando gratuitamente códigos em Python a partir de sugestões de especialistas em finanças e economia.",
                html.P(),
                "As primeiras sugestões já estão chegando e em breve vamos disponibilizá-las gratuitamente para todos!",
                html.P(),
                "Notícias e anúncios serão publicados nessa seção do site. Não deixe de visitá-la frequentemente.",
                html.P(),
                "Siga a KWT no Twitter ou no GitHub, ficaremos muito felizes com suas obersavções e contribuições.",
                html.P(),
                "Um abraço, Nós!"
            ], style={'text-align':'center', 'justify-content':'center'}),
        ], className="cardSize", ),
      
      ### incluir mais cards aqui

    ], className='content'), fluid=True)

kwt_cards = dbc.Container(dbc.Row(
    [
        dbc.Card([
            dbc.CardHeader(html.H5("Quem somos Nós???", style={'text-align':'center', 'color':'#0a0', 'font-weight':'bolder'})),
            dbc.CardBody([
                html.Img(src="static/logo.png", height="30"),
                html.Br(),
                "Este é o ",
                html.Span("Kurmasana Wealth Tech Community hub.", style={'color':'#0a0', 'font-weight':'bolder'}),
                html.P(),
                "",
                html.P(),
                "",
                html.P(),
                "",
                html.P(),
                "",
                html.P(),
                "Um abraço, Nós!"
            ], style={'text-align':'center', 'justify-content':'center'}),
        ], className="cardSize", ),
      
      ### incluir mais cards aqui

    ], className='content'), fluid=True)

sidebar = dtc.SideBar(className='sidenav',
    children=[
        dtc.SideBarItem(id='id_1', label="Python Codes", icon="fab fa-python"),
        dtc.SideBarItem(id='id_2', label="A Comunidade", icon="fas fa-users"),
        dtc.SideBarItem(id='id_3', label="Sobre Nós", icon="fas fa-user-astronaut"),
        dtc.SideBarItem(id='id_4', label="GitHub", icon="fab fa-github"),
        dtc.SideBarItem(id='id_5', label="Twitter", icon="fab fa-twitter"),
    ], bg_color="#000")

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
        Input("id_4", "n_clicks_timestamp"),
        Input("id_5", "n_clicks_timestamp")
    ]
)

def toggle_collapse(input1, input2, input3, input4, input5):
    btn_df = pd.DataFrame({"input1": [input1], "input2": [input2],
                           "input3": [input3], "input4": [input4], 
                           "input5": [input5]})
    
    btn_df = btn_df.fillna(0)

    if btn_df.idxmax(axis=1).values == "input1":
        return content_1
    if btn_df.idxmax(axis=1).values == "input2":
        return content_2
    if btn_df.idxmax(axis=1).values == "input3":
        return content_3
    if btn_df.idxmax(axis=1).values == "input4":
        return html.Div([
            content_1,
            webbrowser.open('https://github.com/KurmasanaWT', new=2, autoraise=True)
        ])
    if btn_df.idxmax(axis=1).values == "input5":
        return html.Div([
            content_1,
            webbrowser.open('https://twitter.com/KurmasanaWT', new=2, autoraise=True)
        ])

if __name__ == '__main__':
    app.run_server(debug=False)