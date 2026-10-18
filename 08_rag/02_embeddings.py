"""Stage 2: embed small example texts and inspect cosine similarity."""
import numpy as np
from rag_utils import embed_texts,cosine_search
texts=['Bayesian adaptive clinical trial designs','Trial randomisation and interim analyses','The history of medieval architecture']
vectors=embed_texts(texts)
print('Embedding matrix shape:',vectors.shape)
for idx,score in cosine_search(vectors[0],vectors,3): print(f'{score:.3f}: {texts[idx]}')
