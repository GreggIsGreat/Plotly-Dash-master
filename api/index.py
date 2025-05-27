from http.server import BaseHTTPRequestHandler
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from final_dashboard import server as flask_app

# This is the handler that will be used by Vercel
class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        
        # Forward the request to the Flask app
        from wsgiref.handlers import SimpleHandler
        
        environ = {
            'wsgi.input': self.rfile,
            'wsgi.errors': sys.stderr,
            'wsgi.version': (1, 0),
            'wsgi.multithread': False,
            'wsgi.multiprocess': False,
            'wsgi.run_once': False,
            'wsgi.url_scheme': 'http',
            'REQUEST_METHOD': self.command,
            'PATH_INFO': self.path,
            'SCRIPT_NAME': '',
            'SERVER_NAME': self.server.server_name,
            'SERVER_PORT': str(self.server.server_port),
            'REMOTE_ADDR': self.client_address[0],
        }
        
        def start_response(status, headers):
            self.send_response(int(status.split(' ')[0]))
            for header, value in headers:
                self.send_header(header, value)
            self.end_headers()
            return self.wfile.write
        
        result = flask_app(environ, start_response)
        for data in result:
            self.wfile.write(data)
