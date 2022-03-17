# -*- coding: utf-8 -*-  

import dash
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, suppress_callback_exceptions=False, title='KWT-Community', external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])
server = app.server    
app.config.suppress_callback_exceptions = True