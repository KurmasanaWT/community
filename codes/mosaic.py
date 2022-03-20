import dash_bootstrap_components as dbc
from dash import html

from app import app

layout = dbc.Container(
    children=[
        
        dbc.Card([
            dbc.CardHeader("NBC NEWS (Estados Unidos)"),
            dbc.CardBody([html.Iframe(className="ytvid", width="420", height="315", src="https://www.youtube.com/embed/3UaQIMGmoDU?&autoplay=1")]),
            ], className="cardSize-vid"),

        dbc.Card([
            dbc.CardHeader("EURONEWS (União Européia)"),
            dbc.CardBody([html.Iframe(className="ytvid", width="420", height="315", src="https://www.youtube.com/embed/sPgqEHsONK8?&autoplay=1")]),
            ], className="cardSize-vid"),

        dbc.Card([
            dbc.CardHeader("SKY NEWS (Reino Unido)"),
            dbc.CardBody([html.Iframe(className="ytvid", width="420", height="315", src="https://www.youtube.com/embed/9Auq9mYxFEE?&autoplay=1")]),
            ], className="cardSize-vid"),    

        dbc.Card([
            dbc.CardHeader("FRANCE 24 (França)"),
            dbc.CardBody([html.Iframe(className="ytvid", width="420", height="315", src="https://www.youtube.com/embed/jNhh-OLzWlE?&autoplay=1")]),
            ], className="cardSize-vid"),

        dbc.Card([
            dbc.CardHeader("DEUTSCH WELLE (Alemanha)"),
            dbc.CardBody([html.Iframe(className="ytvid", width="420", height="315", src="https://www.youtube.com/embed/V9KZGs1MtP4?&autoplay=1")]),
            ], className="cardSize-vid"),

        dbc.Card([
            dbc.CardHeader("RT (Rússia)"),
            dbc.CardBody([html.Iframe(className="ytvid", width="420", height="315", src="https://odysee.com/$/embed/RTlivestream/8c06ebe369b6ecf6ad383e4a32bfca34c0168d79?r=RfLjh5uDhbZHt8SDkQFdZyKTmCbSCpWH&autoplay=1")]),
            ], className="cardSize-vid"),
 
        dbc.Card([
            dbc.CardHeader("CGTN EUROPE (China)"),
            dbc.CardBody([html.Iframe(className="ytvid", width="420", height="315", src="https://www.youtube.com/embed/9Y-LKliWJ4U?&autoplay=1")]),
            ], className="cardSize-vid"),

        dbc.Card([
            dbc.CardHeader("AL JAZEERA (Catar)"),
            dbc.CardBody([html.Iframe(className="ytvid", width="420", height="315", src="https://www.youtube.com/embed/-upyPouRrB8?&autoplay=1")]),
            ], className="cardSize-vid"),
  

        html.Div(dbc.Badge("* Todos os vídeos são transmitidos pelo YouTube, exceto o canal RT, transmitido pelo Odysee."))

    ], fluid=True
)

def get():
    return html.Div(layout)

'''
RT RUSSIA
https://rumble.com/embed/vtp5hp/?pub=4&autoplay=2
https://odysee.com/$/embed/RTlivestream/8c06ebe369b6ecf6ad383e4a32bfca34c0168d79?r=RfLjh5uDhbZHt8SDkQFdZyKTmCbSCpWH
'''