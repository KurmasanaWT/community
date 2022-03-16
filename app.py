# -*- coding: utf-8 -*-

import dash
 
app = dash.Dash(__name__, suppress_callback_exceptions=True, title='KWT-Community')
server = app.server 