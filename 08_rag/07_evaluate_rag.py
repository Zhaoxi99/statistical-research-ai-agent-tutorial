"""Stage 7: reproducible retrieval diagnostics; answer grounding still needs review."""
import argparse
import json
from pathlib import Path
from config import ROOT,CACHE
from rag_utils import load_corpus,retrieve,answer_with_context,save_json
ap=argparse.ArgumentParser()
ap.add_argument('--top-k',type=int,default=3)
ap.add_argument('--generate',action='store_true',help='Also produce answers (extra API cost)')
a=ap.parse_args()
chunks,vectors=load_corpus()
cases=json.loads((ROOT/'evaluation_questions.json').read_text(encoding='utf-8'))
records=[]
for case in cases:
    passages=retrieve(case['question'],chunks,vectors,k=a.top_k)
    # Heuristic diagnostic: not a human-labelled recall@k or relevance metric.
    text=' '.join(p['text'].lower() for p in passages)
    terms=case.get('expected_source_terms',[])
    coverage=(sum(term.lower() in text for term in terms)/len(terms)) if terms else None
    record={'id':case['id'],'question':case['question'],'retrieved_ids':[p['chunk_id'] for p in passages],
            'similarities':[p['score'] for p in passages],'keyword_coverage_heuristic':coverage,
            'should_abstain':case.get('should_abstain',False)}
    if a.generate: record['answer']=answer_with_context(case['question'],passages)
    records.append(record)
    print(case['id'],'keyword coverage:',coverage,'top sources:',record['retrieved_ids'])
path=CACHE/'rag_evaluation_results.json'
save_json(path,records)
print('Saved',path)
print('Note: review answer support manually; keyword coverage is NOT recall@k.')
