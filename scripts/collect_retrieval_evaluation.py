"""Record paired rankings from the same candidate pool; resume saved questions."""
import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen


def collect(questions, output, base_url):
    ids = [str(q['id']) for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError('Question IDs must be unique')
    records = json.loads(output.read_text()) if output.exists() else []
    saved = {str(row['id']): row for row in records}
    if len(saved) != len(records) or not set(saved).issubset(ids):
        raise ValueError('Output contains duplicate or unrelated questions; use a new file')
    if len({q.get('k', 5) for q in questions}) > 1:
        raise ValueError('Use one k for all questions')
    for question in questions:
        if not question.get('relevant_ids'):
            raise ValueError('Each question needs manually checked relevant chunk IDs')
        if str(question['id']) in saved:
            row = saved[str(question['id'])]
            if row['question'] != question:
                raise ValueError('Saved question changed; use a new output file')
            continue
        body = dict(text=question['text'], limit=question.get('k',5),
                    candidate_limit=question.get('candidate_limit',20))
        request = Request(base_url.rstrip('/') + '/api/v1/nlp/index/search-reranked/' + str(int(question['project_id'])),
                          data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
        with urlopen(request, timeout=180) as response:
            result = json.load(response)
        row = dict(id=question['id'], question=question,
                   relevant_ids=question['relevant_ids'], baseline_ids=result['baseline_ids'],
                   reranked_ids=[r['chunk_id'] for r in result['results']],
                   baseline_ms=result['retrieval_ms'],
                   reranked_ms=result['retrieval_ms']+result['rerank_ms'])
        if any(value is None for value in row['baseline_ids'] + row['reranked_ids']):
            raise ValueError('Missing chunk IDs; update the vector provider before evaluation')
        records.append(row)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + '.tmp')
        temporary.write_text(json.dumps(records, indent=2, ensure_ascii=False)+'\n')
        temporary.replace(output)
        print(f'{len(records)}/{len(questions)} saved',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('questions',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--base-url',default='http://127.0.0.1:8000')
    args=parser.parse_args()
    collect(json.loads(args.questions.read_text()),args.output,args.base_url)
