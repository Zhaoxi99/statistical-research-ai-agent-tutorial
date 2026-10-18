"""Stage 4: fixed RAG pipeline: question -> retrieval -> generation."""
import argparse
from rag_utils import load_corpus,retrieve,answer_with_context
ap=argparse.ArgumentParser()
ap.add_argument('--question',default='What are advantages and limitations of Bayesian adaptive clinical trials?')
ap.add_argument('--top-k',type=int,default=4)
a=ap.parse_args()
chunks,vectors=load_corpus()
passages=retrieve(a.question,chunks,vectors,k=a.top_k)
print('Retrieved evidence:')
for i,p in enumerate(passages,1): print(f"[S{i}] {p['title']} ({p['score']:.3f}) {p['url']}")
print('\nGROUNDED ANSWER:\n'+answer_with_context(a.question,passages))
