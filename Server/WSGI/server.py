# -*- coding: utf-8 -*-
"""
Created on Sat May  6 11:25:04 2017

@author: admin
"""

# server.py
# 从wsgiref模块导入:
from wsgiref.simple_server import make_server
# 导入我们自己编写的application函数:
from hello import application

# 创建一个服务器，IP地址为空，端口是8000，处理函数是application:
httpd = make_server('', 8001, application)
print('Serving HTTP on port 8001...')
# 开始监听HTTP请求:
httpd.serve_forever()