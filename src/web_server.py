"""
web_server.py - A simple web server to view the analysis results.
"""

import http.server
import socketserver
import os

def start_server(path, port=8000):
    """
    Starts a simple web server to serve the files in the given path.
    """
    
    os.chdir(path)
    
    Handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", port), Handler)
    
    print(f"Serving at port {port}")
    httpd.serve_forever()
