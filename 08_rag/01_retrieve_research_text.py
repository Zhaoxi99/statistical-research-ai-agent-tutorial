"""Stage 1: download research abstracts from Europe PMC (web access, no API key)."""
import argparse
from config import CACHE
from rag_utils import fetch_abstracts,make_chunks,save_json

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--query',default='(Bayesian adaptive clinical trial) AND OPEN_ACCESS:y')
    ap.add_argument('--limit',type=int,default=12)
    a=ap.parse_args()
    papers=fetch_abstracts(a.query,a.limit)
    if not papers: raise SystemExit('No abstracts found. Try --query "Bayesian clinical trial"')
    chunks=make_chunks(papers)
    save_json(CACHE/'papers.json',papers)
    save_json(CACHE/'chunks.json',chunks)
    print(f'Saved {len(papers)} papers, {len(chunks)} chunks.')
    for p in papers[:5]: print('-',p['title'],p['url'])
if __name__=='__main__': main()
