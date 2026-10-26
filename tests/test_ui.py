"""Exercise the UI relay against a local fake API, without Docker or provider keys."""
import importlib.util
import json
from pathlib import Path
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ui_server',ROOT/'ui/server.py')
ui=importlib.util.module_from_spec(spec);spec.loader.exec_module(ui)


class FakeAPI(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200);self.end_headers()
        self.wfile.write(b'{"collection_info":{"record_count":2}}')
    def do_POST(self):
        body=self.rfile.read(int(self.headers.get('Content-Length','0')))
        self.server.received=(self.path,body,self.headers.get('Content-Type'))
        self.send_response(200);self.end_headers()
        self.wfile.write(b'{"signal":"success","answer":"A safe answer","full_prompt":"private prompt","chat_history":[]}')
    def log_message(self,*args):pass


class UITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api=ThreadingHTTPServer(('127.0.0.1',0),FakeAPI)
        cls.web=ThreadingHTTPServer(('127.0.0.1',0),ui.Handler)
        cls.web.api='http://127.0.0.1:'+str(cls.api.server_port)
        cls.base='http://127.0.0.1:'+str(cls.web.server_port)
        cls.threads=[]
        for server in (cls.api,cls.web):
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();cls.threads.append(thread)
    @classmethod
    def tearDownClass(cls):
        for server in (cls.web,cls.api):server.shutdown();server.server_close()
        for thread in cls.threads:thread.join()
    def test_html_is_local_and_does_not_load_external_assets(self):
        with urlopen(self.base) as r:
            text=r.read().decode();self.assertEqual(r.status,200)
        self.assertIn('NeGm',text);self.assertNotIn('<script src=',text)
        self.assertNotIn('fonts.googleapis',text)
    def test_index_status_is_relayed(self):
        with urlopen(self.base+'/api/v1/nlp/index/info/1063') as r:
            self.assertEqual(json.load(r)['collection_info']['record_count'],2)
    def test_standard_answer_hides_debug_prompt(self):
        request=Request(self.base+'/api/v1/nlp/index/answer/1063',data=b'{"text":"q"}',headers={'Content-Type':'application/json'})
        with urlopen(request) as r:value=json.load(r)
        self.assertEqual(value,{'signal':'success','answer':'A safe answer'})
    def test_upload_passes_multipart_unchanged(self):
        body=b'--boundary\r\nContent-Disposition: form-data; name="file"; filename="a.txt"\r\n\r\ntext\r\n--boundary--\r\n'
        request=Request(self.base+'/api/v1/data/upload/1063',data=body,headers={'Content-Type':'multipart/form-data; boundary=boundary'})
        with urlopen(request) as r:self.assertEqual(r.status,200)
        self.assertEqual(self.api.received[1],body)
    def test_path_traversal_and_arbitrary_api_paths_rejected(self):
        for path in ('/../server.py','/api/v1/data/reset/1063','/api/v1/nlp/index/info/1063?target=elsewhere'):
            with self.assertRaises(HTTPError) as caught:urlopen(self.base+path)
            self.assertEqual(caught.exception.code,404)
    def test_cross_origin_posts_rejected(self):
        request=Request(self.base+'/api/v1/nlp/index/answer/1063',data=b'{}',headers={'Origin':'https://other.example'})
        with self.assertRaises(HTTPError) as caught:urlopen(request)
        self.assertEqual(caught.exception.code,403)
    def test_wrong_methods_rejected(self):
        with self.assertRaises(HTTPError) as caught:urlopen(self.base+'/api/v1/data/upload/1')
        self.assertEqual(caught.exception.code,405)
    def test_unavailable_api_gives_actionable_error(self):
        original=self.web.api;self.web.api='http://127.0.0.1:1'
        try:
            with self.assertRaises(HTTPError) as caught:urlopen(self.base+'/api/v1/nlp/index/info/1')
            self.assertEqual(caught.exception.code,502)
            self.assertIn('Docker',caught.exception.read().decode())
        finally:self.web.api=original

if __name__=='__main__':unittest.main()
