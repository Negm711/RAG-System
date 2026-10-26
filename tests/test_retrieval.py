import ast
import asyncio
import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from retrieval.chunking import split_pages
from retrieval.evaluation import evaluate, score_ranking
from retrieval.reranking import LocalReranker, RerankerUnavailable, rank_documents


class ChunkingTests(unittest.TestCase):
    def test_overlap_offsets_and_complete_coverage(self):
        text = 'abcdefghijklmnopqrstuvwxyz'
        chunks = split_pages([SimpleNamespace(page_content=text, metadata={'page': 7})], 'file.pdf', 10, 3)
        self.assertEqual([(c.metadata['char_start'], c.metadata['char_end']) for c in chunks], [(0,10),(7,17),(14,24),(21,26)])
        for c in chunks:
            self.assertEqual(c.page_content, text[c.metadata['char_start']:c.metadata['char_end']])
            self.assertEqual(c.metadata['page'], 7)
            self.assertEqual(c.metadata['file_id'], 'file.pdf')
        self.assertEqual(chunks[0].page_content[-3:], chunks[1].page_content[:3])

    def test_no_cross_page_chunks_or_metadata_mutation(self):
        metadata = {'page': 0}
        pages = [SimpleNamespace(page_content='first', metadata=metadata), SimpleNamespace(page_content='second', metadata={'page':1})]
        chunks = split_pages(pages, 'a.pdf', 100, 20)
        self.assertEqual([c.page_content for c in chunks], ['first','second'])
        self.assertEqual(metadata, {'page':0})
        self.assertEqual([c.metadata['page_index'] for c in chunks], [0,1])

    def test_invalid_overlap_and_empty_pages(self):
        for size, overlap in [(0,0),(5,5),(5,-1)]:
            with self.assertRaises(ValueError): split_pages([], 'x', size, overlap)
        self.assertEqual(split_pages([SimpleNamespace(page_content=' ',metadata={})], 'x'), [])

    def test_legacy_splitter_is_unchanged(self):
        from types import SimpleNamespace
        source = (ROOT/'src/controllers/ProcessController.py').read_text()
        cls = next(n for n in ast.parse(source).body
                   if isinstance(n, ast.ClassDef) and n.name == 'ProcessController')
        method = next(n for n in cls.body
                      if getattr(n, 'name', '') == 'process_simpler_splitter')
        namespace = {'List': list, 'Document': SimpleNamespace}
        module = ast.Module(body=[method], type_ignores=[])
        exec(compile(module, 'legacy-splitter', 'exec'), namespace)
        split = namespace['process_simpler_splitter']
        chunks = split(None, ['first\nsecond\ntail'], [{'page': 1}], 8)
        self.assertEqual([c.page_content for c in chunks], ['first\nsecond', 'tail'])
        self.assertEqual([c.metadata for c in chunks], [{}, {}])
        self.assertEqual(split(None, [], [], 8), [])


class RerankingTests(unittest.TestCase):
    def test_uses_scores_preserves_sources_and_input(self):
        docs = [SimpleNamespace(text='a', metadata={'page':1}),SimpleNamespace(text='b',metadata={'page':2})]
        ranked = rank_documents('q', docs, 1, lambda q,d: [0.1,2.0])
        self.assertIs(ranked[0][0], docs[1])
        self.assertEqual(docs[0].text, 'a')

    def test_ties_are_stable_and_bad_scores_rejected(self):
        self.assertEqual(rank_documents('q', ['a','b'], 2, lambda q,d:[1,1]), [('a',1.0),('b',1.0)])
        for values in [[1], [1,math.nan], [math.inf,0]]:
            with self.assertRaises(ValueError): rank_documents('q',['a','b'],2,lambda q,d:values)

    def test_unprovisioned_model_does_not_download(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {'RERANK_MODEL_PATH':'/nonexistent/rerank-test'}):
            with self.assertRaises(RerankerUnavailable): LocalReranker().score('q',[])


class EvaluationTests(unittest.TestCase):
    def test_partial_recall_and_rank(self):
        self.assertEqual(score_ranking(['x','a','b'], ['a','b'], 2), {'recall_at_k':0.5,'reciprocal_rank_at_k':0.5})

    def test_metrics_count_misses_and_no_hallucinated_accuracy(self):
        rows = [dict(id='q', relevant_ids=['a'], baseline_ids=['x','a'], reranked_ids=['a','x'],baseline_ms=10,reranked_ms=30)]
        report = evaluate(rows,1)
        self.assertEqual(report['metrics']['baseline']['recall_at_k'],0)
        self.assertEqual(report['metrics']['reranked']['reciprocal_rank_at_k'],1)
        self.assertNotIn('accuracy', report)

    def test_rejects_duplicate_ids_and_missing_labels(self):
        with self.assertRaises(ValueError): score_ranking(['a','a'],['a'],2)
        with self.assertRaises(ValueError): score_ranking([],[],1)


class SchemaTests(unittest.TestCase):
    def load(self, name):
        spec = importlib.util.spec_from_file_location(name,ROOT/f'src/routes/schemes/{name}.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

    def test_legacy_request_defaults_preserved(self):
        request=self.load('data').ProcessRequest()
        self.assertEqual((request.chunk_size,request.overlap_size,request.chunk_strategy),(100,20,'legacy'))
        self.assertEqual(self.load('nlp').SearchRequest(text='q').limit,5)

    def test_new_settings_are_bounded(self):
        cls=self.load('nlp').RerankedSearchRequest
        for kwargs in [dict(text=' '),dict(text='q',limit=10,candidate_limit=5),dict(text='q',candidate_limit=101)]:
            with self.assertRaises(ValueError): cls(**kwargs)
        with self.assertRaises(ValueError): self.load('data').ProcessRequest(chunk_strategy='page_overlap',chunk_size=20,overlap_size=20)



class EndpointTests(unittest.IsolatedAsyncioTestCase):
    def namespace(self, available=True, empty=False):
        from time import perf_counter
        from unittest.mock import Mock
        class HTTPException(Exception):
            def __init__(self,status_code,detail): self.status_code=status_code;self.detail=detail
        class Projects:
            @classmethod
            async def create_instance(cls,db_client): return cls()
            async def get_project_or_create_one(self,project_id): return SimpleNamespace(project_id=project_id)
        docs=[SimpleNamespace(text='irrelevant',chunk_id='1',score=0.9,metadata={'page':0}),
              SimpleNamespace(text='evidence',chunk_id='2',score=0.8,metadata={'page':1})]
        class Controller:
            def __init__(self,**kwargs): pass
            async def search_vector_db_collection(self,**kwargs):
                self.search_limit=kwargs['limit']; return [] if empty else docs
            def answer_from_documents(self,query,documents):
                return 'answer from '+documents[0].text,'prompt',[]
        def ready():
            if not available: raise RerankerUnavailable('missing')
        ranker=SimpleNamespace(ready=ready,score=lambda q,d:[0,1])
        ns=dict(asyncio=asyncio,perf_counter=perf_counter,HTTPException=HTTPException,
                ProjectModel=Projects,NLPController=Controller,reranker=ranker,
                rank_documents=rank_documents,RerankerUnavailable=RerankerUnavailable,
                Request=object,RerankedSearchRequest=object)
        tree=ast.parse((ROOT/'src/routes/nlp.py').read_text())
        funcs=[n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name in ['_reranked_documents','search_reranked','answer_reranked']]
        for fn in funcs: fn.decorator_list=[]
        exec(compile(ast.Module(body=funcs,type_ignores=[]),'route-functions','exec'),ns)
        app=SimpleNamespace(db_client=None,vectordb_client=None,generation_client=None,embedding_client=None,template_parser=None)
        return ns,SimpleNamespace(app=app),SimpleNamespace(text='question',candidate_limit=20,limit=1)

    async def test_expanded_candidates_and_source_preserving_answer(self):
        ns,request,body=self.namespace()
        controller,ranked,result=await ns['_reranked_documents'](request,7,body)
        self.assertEqual(controller.search_limit,20)
        self.assertEqual(result['baseline_ids'],['1','2'])
        self.assertEqual(result['results'][0]['metadata'],{'page':1})
        result=await ns['answer_reranked'](request,7,body)
        self.assertEqual(result['answer'],'answer from evidence')
        self.assertNotIn('full_prompt',result)

    async def test_unavailable_and_empty_index_statuses(self):
        for kwargs,status in [(dict(available=False),503),(dict(empty=True),404)]:
            ns,request,body=self.namespace(**kwargs)
            with self.assertRaises(ns['HTTPException']) as cm:
                await ns['_reranked_documents'](request,7,body)
            self.assertEqual(cm.exception.status_code,status)


if __name__ == '__main__': unittest.main()
