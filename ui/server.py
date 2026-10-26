"""Local UI and a fixed-target API relay. No additional Python packages required."""
import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MAX_BODY = 12 * 1024 * 1024
ALLOWED = re.compile(r'/api/v1/(?:data/(?:upload|process-and-push)/[1-9][0-9]{0,9}|nlp/index/(?:info|answer|answer-reranked)/[1-9][0-9]{0,9})\Z')


class Handler(BaseHTTPRequestHandler):
    def send(self, status, body, mime='application/json'):
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def relay(self):
        if not ALLOWED.fullmatch(self.path):
            self.send(404, b'{"detail":"Unknown UI request"}'); return
        expected = 'GET' if '/info/' in self.path else 'POST'
        if self.command != expected:
            self.send(405, b'{"detail":"Method not allowed"}'); return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 <= size <= MAX_BODY:
                self.send(413, b'{"detail":"Upload must be smaller than 12 MiB"}'); return
            body = self.rfile.read(size) if size else None
            request = Request(self.server.api + self.path, data=body, method=self.command,
                              headers={'Content-Type': self.headers.get('Content-Type', 'application/json')})
            with urlopen(request, timeout=180) as response:
                result = response.read()
                # Do not expose the legacy endpoint's internal prompt/history in the UI.
                if '/index/answer/' in self.path:
                    value = json.loads(result)
                    result = json.dumps({k: value[k] for k in ('answer', 'signal') if k in value}).encode()
                self.send(response.status, result)
        except HTTPError as exc:
            try:
                value = json.loads(exc.read())
                detail = value.get('detail') or value.get('signal') or 'API request failed'
            except (ValueError, AttributeError):
                detail = 'API request failed. Check application logs if this persists.'
            self.send(exc.code, json.dumps({'detail': detail}).encode())
        except (URLError, TimeoutError, OSError):
            self.send(502, b'{"detail":"Cannot reach the API. Check that Docker services are running."}')
        except ValueError:
            self.send(400, b'{"detail":"Invalid request or API response"}')

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            self.send(200, Path(__file__).with_name('index.html').read_bytes(), 'text/html; charset=utf-8')
        elif self.path == '/favicon.ico':
            self.send(204, b'')
        else:
            self.relay()

    def do_POST(self):
        # A browser form from another site must not drive the local API through this relay.
        origin = self.headers.get('Origin')
        if origin and origin != 'http://' + self.headers.get('Host', ''):
            self.send(403, b'{"detail":"Cross-origin request blocked"}'); return
        self.relay()

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7861)
    parser.add_argument('--api', default='http://127.0.0.1:8000')
    args = parser.parse_args()
    from urllib.parse import urlsplit
    url = urlsplit(args.api)
    if url.scheme != 'http' or url.hostname not in ('localhost', '127.0.0.1') or url.path not in ('', '/') or url.query or url.fragment or url.username:
        parser.error('--api must be a local HTTP API address')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.api = args.api.rstrip('/')
    print(f'RAG workspace: http://127.0.0.1:{args.port}', flush=True)
    print('Keep this terminal open. Ctrl+C closes the UI; Docker services stay running.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
