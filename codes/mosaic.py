import dash_bootstrap_components as dbc
from dash import html

from app import app

layout = dbc.Container(
    children=[ 

        dbc.Card([
            dbc.CardHeader("EURONEWS (União Européia)"),
            dbc.CardBody([html.Iframe(className="ytvid", width="420", height="315", src="https://www.youtube.com/embed/sPgqEHsONK8?&autoplay=1&mute=1")]),
            ], className="cardSize-vid"),

        dbc.Card([
            dbc.CardHeader("SKY NEWS (Reino Unido)"),
            dbc.CardBody([html.Iframe(className="ytvid", width="420", height="315", src="https://www.youtube.com/embed/9Auq9mYxFEE?&autoplay=1&mute=1")]),
            ], className="cardSize-vid"),    

        dbc.Card([
            dbc.CardHeader("FRANCE 24 (França)"),
            dbc.CardBody([html.Iframe(className="ytvid", width="420", height="315", src="https://www.youtube.com/embed/jNhh-OLzWlE?&autoplay=1&mute=1")]),
            ], className="cardSize-vid"),

        dbc.Card([
            dbc.CardHeader("DEUTSCH WELLE (Alemanha)"),
            dbc.CardBody([html.Iframe(className="ytvid", width="420", height="315", src="https://www.youtube.com/embed/V9KZGs1MtP4?&autoplay=1&mute=1")]),
            ], className="cardSize-vid"),

        dbc.Card([
            dbc.CardHeader("RT (Rússia)"),
            dbc.CardBody([html.Iframe(className="ytvid", width="420", height="315", src="https://odysee.com/$/embed/RTlivestream/8c06ebe369b6ecf6ad383e4a32bfca34c0168d79?r=RfLjh5uDhbZHt8SDkQFdZyKTmCbSCpWH&autoplay=1&mute=1")]),
            ], className="cardSize-vid"),
 
         dbc.Card([
            dbc.CardHeader("TRT WORLD (Turquia)"),
            dbc.CardBody([html.Iframe(className="ytvid", width="420", height="315", src="https://www.youtube.com/embed/CV5Fooi8YJA?&autoplay=1&mute=1")]),
            ], className="cardSize-vid"),

        dbc.Card([
            dbc.CardHeader("AL JAZEERA (Catar)"),
            dbc.CardBody([html.Iframe(className="ytvid", width="420", height="315", src="https://www.youtube.com/embed/-upyPouRrB8?&autoplay=1&mute=1")]),
            ], className="cardSize-vid"),
  
        dbc.Card([
            dbc.CardHeader("CGTN EUROPE (China)"),
            dbc.CardBody([html.Iframe(className="ytvid", width="420", height="315", src="https://www.youtube.com/embed/9Y-LKliWJ4U?&autoplay=1&mute=1")]),
            ], className="cardSize-vid"),     

        html.Div(dbc.Badge(children=[
            html.Span("* Todos os vídeos são transmitidos pelo "),
            html.A(href='http://www.youtube.com', target='new', children='YouTube'),
            html.Span(" exceto o canal RT, transmitido pelo "),
            html.A(href='http://www.odysee.com', target='new', children='Odysee'),
            html.Span(" .")
        ], className="badge-link",
        ))

    ], fluid=True
)

def get():
    return html.Div(layout)

'''
RT RUSSIA
https://rumble.com/embed/vtp5hp/?pub=4&autoplay=2
https://odysee.com/$/embed/RTlivestream/8c06ebe369b6ecf6ad383e4a32bfca34c0168d79?r=RfLjh5uDhbZHt8SDkQFdZyKTmCbSCpWH
'''