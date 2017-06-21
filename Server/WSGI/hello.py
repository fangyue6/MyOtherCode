# -*- coding: utf-8 -*-
"""
Created on Sat May  6 11:24:39 2017

@author: admin
"""

def application(environ, start_response):
    start_response('200 OK', [('Content-Type', 'text/html')])
    body = '<h1>Hello, %s!</h1>' % (environ['PATH_INFO'][1:] or 'web')
    return [body.encode('utf-8')]