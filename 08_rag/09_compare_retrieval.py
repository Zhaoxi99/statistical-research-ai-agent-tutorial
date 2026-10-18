"""Stage 9: compare local exact NumPy cosine similarity with Qdrant Cloud."""
import argparse
import time
from rag_utils import load_corpus, embed_texts, cosine_search
from vector_db import get_client, index_chunks, search_collection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--question', default='How do Bayesian adaptive trials use interim analyses?')
    parser.add_argument('--top-k', type=int, default=5)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error('--top-k must be positive')
    chunks, vectors = load_corpus()
    client = get_client()
    name = index_chunks(chunks, vectors, client=client)
    query_vector = embed_texts([args.question])[0]

    start = time.perf_counter()
    numpy_hits = cosine_search(query_vector, vectors, k=args.top_k)
    numpy_seconds = time.perf_counter() - start
    start = time.perf_counter()
    qdrant_hits = search_collection(client, name, query_vector, k=args.top_k)
    qdrant_seconds = time.perf_counter() - start
    a = [(chunks[i]['chunk_id'], score) for i, score in numpy_hits]
    b = [(hit['chunk_id'], hit['score']) for hit in qdrant_hits]
    print('NumPy results (exact cosine):')
    for rank, (cid, score) in enumerate(a, 1):
        print(f'  {rank}. {cid} | {score:.4f}')
    print('\nQdrant Cloud results (cosine score):')
    for rank, (cid, score) in enumerate(b, 1):
        print(f'  {rank}. {cid} | {score:.4f}')
    shared = set(cid for cid, _ in a) & set(cid for cid, _ in b)
    print(f'\nTop-k overlap: {len(shared)}/{min(args.top_k, len(chunks))}')
    print(f'NumPy search elapsed: {numpy_seconds:.6f}s')
    print(f'Qdrant Cloud query elapsed (includes network): {qdrant_seconds:.6f}s')
    print('Single-run times are illustrative, not controlled benchmarks.')
    print('A vector database does not inherently make retrieved evidence more relevant.')


if __name__ == '__main__':
    main()
