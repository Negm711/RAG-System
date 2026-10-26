"""Private HTTP adapter for the isolated CPU reranker."""
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from retrieval.reranking import reranker, RerankerUnavailable


class Handler(BaseHTTPRequestHandler):
    def respond(self, code, payload):
        body=json.dumps(payload).encode()
        self.send_response(code)
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != '/ready':
            self.respond(404, {'error':'not_found'}); return
        try:
            reranker.ready()
            self.respond(200, {'ready':True})
        except RerankerUnavailable:
            self.respond(503, {'error':'Model not provisioned or unavailable'})

    def do_POST(self):
        if self.path != '/score':
            self.respond(404, {'error':'not_found'}); return
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0 < size <= 4_000_000: raise ValueError('Request too large or empty')
            payload=json.loads(self.rfile.read(size))
            query=payload['query']; texts=payload['texts']
            if not isinstance(query,str) or not query.strip() or len(query)>1000:
                raise ValueError('Invalid query')
            if not isinstance(texts,list) or not 1<=len(texts)<=100 or any(not isinstance(t,str) for t in texts):
                raise ValueError('Invalid passages')
            scores=reranker.score(query,[SimpleNamespace(text=t) for t in texts])
            self.respond(200, {'scores':scores})
        except (ValueError, KeyError, TypeError):
            self.respond(400, {'error':'Invalid scoring request or pair exceeds token budget'})
        except RerankerUnavailable:
            self.respond(503, {'error':'Model not provisioned or unavailable'})
        except Exception:
            logging.exception('Reranker request failed')
            self.respond(500, {'error':'Scoring failed'})

    def log_message(self, format, *args):
        # Do not log queries, source passages or credentials.
        logging.info(format, *args)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    ThreadingHTTPServer(('0.0.0.0',8080),Handler).serve_forever()
