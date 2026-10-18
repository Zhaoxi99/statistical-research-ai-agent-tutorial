"""Stage 3: semantic search over abstracts fetched in Stage 1."""
import argparse
from rag_utils import load_corpus,retrieve
ap=argparse.ArgumentParser()
ap.add_argument('--question',default='What are advantages of Bayesian adaptive trial designs?')
ap.add_argument('--top-k',type=int,default=3)
a=ap.parse_args()
chunks,vectors=load_corpus()
for i,p in enumerate(retrieve(a.question,chunks,vectors,k=a.top_k),1):
    print(f"\n[S{i}] cosine={p['score']:.3f} | {p['title']}\n{p['url']}\n{p['text'][:550]}...")
