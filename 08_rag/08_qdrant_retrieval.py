"""Stage 8: upload cached research embeddings to Qdrant Cloud, then search."""
import argparse
from rag_utils import load_corpus, embed_texts
from vector_db import get_client, index_chunks, search_collection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--question', default='What are advantages of Bayesian adaptive clinical trials?')
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--source-id', default=None, help='Exact source_id metadata filter')
    args = parser.parse_args()
    chunks, vectors = load_corpus()
    print(f'Loaded {len(chunks)} chunks; vector shape: {vectors.shape}', flush=True)
    client = get_client()
    print('Connected to Qdrant Cloud.', flush=True)
    name = index_chunks(chunks, vectors, client=client)
    print(f'Collection: {name}; stored points: {client.count(name, exact=True).count}', flush=True)
    print('Existing corpus embeddings reused; query embedding computed once.')
    query_vector = embed_texts([args.question])[0]
    hits = search_collection(client, name, query_vector, k=args.top_k, source_id=args.source_id)
    if not hits:
        print('No passages matched the search/filter.')
    for i, hit in enumerate(hits, 1):
        print(f"\n[S{i}] cosine≈{hit['score']:.3f} | {hit['title']}")
        print(f"source_id: {hit['source_id']} | {hit['url']}")
        text = hit['text']
        print(text[:550] + ('...' if len(text) > 550 else ''))


if __name__ == '__main__':
    main()
