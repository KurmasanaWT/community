import dash_bootstrap_components as dbc
from dash import html

from app import app

layout = dbc.Container(
    children=[ 

        dbc.Card([
            #dbc.CardHeader("EURONEWS (União Européia)"),
            dbc.CardBody([html.Iframe(className="ytvid", width="420", height="315", src="https://www.youtube.com/embed/sPgqEHsONK8?&autoplay=1&mute=1", allow="fullscreen")]),
            ], className="cardSize-vid"),

        dbc.Card([
            #dbc.CardHeader("SKY NEWS (Reino Unido)"),
            dbc.CardBody([html.Iframe(className="ytvid", width="420", height="315", src="https://www.youtube.com/embed/9Auq9mYxFEE?&autoplay=1&mute=1", allow="fullscreen")]),
            ], className="cardSize-vid"),    

        dbc.Card([
            #dbc.CardHeader("FRANCE 24 (França)"),
            dbc.CardBody([html.Iframe(className="ytvid", width="420", height="315", src="https://www.youtube.com/embed/u9foWyMSATM?&autoplay=1&mute=1", allow="fullscreen")]),
            ], className="cardSize-vid"),

        dbc.Card([
            #dbc.CardHeader("DEUTSCH WELLE (Alemanha)"),
            dbc.CardBody([html.Iframe(className="ytvid", width="420", height="315", src="https://www.youtube.com/embed/m01az_TdpQI?&autoplay=1&mute=1", allow="fullscreen")]),
            ], className="cardSize-vid"),

        dbc.Card([
            #dbc.CardHeader("RT (Rússia)"),
            dbc.CardBody([html.Iframe(className="ytvid", width="420", height="315", src="https://odysee.com/$/embed/RTlivestream/8c06ebe369b6ecf6ad383e4a32bfca34c0168d79?r=RfLjh5uDhbZHt8SDkQFdZyKTmCbSCpWH&autoplay=1&mute=1", allow="fullscreen")]),
            ], className="cardSize-vid"),
 
         dbc.Card([
            #dbc.CardHeader("TRT WORLD (Turquia)"),
            dbc.CardBody([html.Iframe(className="ytvid", width="420", height="315", src="https://www.youtube.com/embed/CV5Fooi8YJA?&autoplay=1&mute=1", allow="fullscreen")]),
            ], className="cardSize-vid"),

        dbc.Card([
            #dbc.CardHeader("AL JAZEERA (Catar)"),
            dbc.CardBody([html.Iframe(className="ytvid", width="420", height="315", src="https://www.youtube.com/embed/F-POY4Q0QSI?&autoplay=1&mute=1", allow="fullscreen")]),
            ], className="cardSize-vid"),
  
        dbc.Card([
            #dbc.CardHeader("NDTV (India)"),
            dbc.CardBody([html.Iframe(className="ytvid", width="420", height="315", src="https://www.youtube.com/embed/WB-y7_ymPJ4?&autoplay=1&mute=1", allow="fullscreen")]),
            ], className="cardSize-vid"),

        dbc.Card([
            #dbc.CardHeader("CGTN EUROPE (China)"),
            dbc.CardBody([html.Iframe(className="ytvid", width="420", height="315", src="https://www.youtube.com/embed/FGabkYr-Sfs?&autoplay=1&mute=1", allow="fullscreen")]),
            ], className="cardSize-vid"),     

        dbc.Card([
            #dbc.CardHeader("ANN NEWS (Japão)"),
            dbc.CardBody([html.Iframe(className="ytvid", width="420", height="315", src="https://www.youtube.com/embed/coYw-eVU0Ks?&autoplay=1&mute=1", allow="fullscreen")]),
            ], className="cardSize-vid"),     

        dbc.Card([
            #dbc.CardHeader("NEWS 12 NEW YORK (Estados Unidos)"),
            dbc.CardBody([html.Iframe(className="ytvid", width="420", height="315", src="https://www.youtube.com/embed/RmmRlztXETI?&autoplay=1&mute=1", allow="fullscreen")]),
            ], className="cardSize-vid"),     

        dbc.Card([
            #dbc.CardHeader("WEBCAM UCRÂNIA (Ao Vivo)"),
            dbc.CardBody([html.Iframe(className="ytvid", width="420", height="315", src="https://www.youtube.com/embed/3hiyVq44pK8?&autoplay=1&mute=1", allow="fullscreen")]),
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