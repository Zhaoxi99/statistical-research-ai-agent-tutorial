"""Optional Qdrant retrieval of the already-indexed Lesson 08 corpus; web search is a separate tool."""
import os
from dotenv import load_dotenv
from agents import function_tool

@function_tool
def search_indexed_literature(question: str, top_k: int = 3) -> dict:
    """Search Lesson 08's already-indexed Qdrant Cloud paper abstracts. Use for corpus-specific literature questions. If credentials or collection are missing, report the error and use web search instead.

    Args:
        question: A focused scientific or statistical literature search query.
        top_k: Number of passages to return, between 1 and 5.
    """
    load_dotenv()
    if not 1 <= top_k <= 5: return {'error':'top_k must be 1..5'}
    if not os.getenv('QDRANT_URL') or not os.getenv('QDRANT_API_KEY'):
        return {'error':'Qdrant not configured: set QDRANT_URL and QDRANT_API_KEY'}
    try:
        from openai import OpenAI
        from qdrant_client import QdrantClient
        embed = OpenAI().embeddings.create(model='text-embedding-3-small', input=question).data[0].embedding
        client=QdrantClient(url=os.environ['QDRANT_URL'],api_key=os.environ['QDRANT_API_KEY'],timeout=15)
        name=os.getenv('QDRANT_COLLECTION','')
        if not name:
            return {'error':'Set QDRANT_COLLECTION to the actual collection name printed in Lesson 08'}
        hits=client.query_points(collection_name=name,query=embed,limit=top_k,with_payload=True).points
        return {'results':[{'score':float(h.score),'title':(h.payload or {}).get('title',''),
                            'source_id':(h.payload or {}).get('source_id',''),
                            'url':(h.payload or {}).get('url',''),
                            'text':(h.payload or {}).get('text','')[:1500]} for h in hits]}
    except Exception as exc:
        return {'error':f'Qdrant retrieval failed: {type(exc).__name__}: {exc}'}
