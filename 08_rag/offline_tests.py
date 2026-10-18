"""Offline checks for chunking, cosine ranking, source formatting, and caching."""
import unittest
import numpy as np
from rag_utils import split_text,make_chunks,cosine_search,build_context,cache_key,strip_markup
class RAGTests(unittest.TestCase):
    def test_chunk_overlap(self):
        x=split_text('abcdefghij',size=6,overlap=2)
        self.assertEqual(x,['abcdef','efghij','ij'])
    def test_empty(self): self.assertEqual(split_text(''),[])
    def test_bad_size(self):
        with self.assertRaises(ValueError): split_text('test',size=2,overlap=2)
    def test_cosine(self):
        matches=cosine_search(np.array([1.,0.]),np.array([[0.,1.],[1.,0.],[-1.,0.]]),2)
        self.assertEqual(matches[0][0],1)
        self.assertAlmostEqual(matches[0][1],1)
    def test_zero_vector(self):
        self.assertEqual(cosine_search(np.zeros(2),np.eye(2),1)[0][1],0.)
    def test_chunk_metadata(self):
        data=make_chunks([{'id':'X:1','title':'A','url':'https://example.org','text':'research evidence'}],size=100)
        self.assertEqual(data[0]['source_id'],'X:1')
        self.assertIn('[S1]',build_context(data))
    def test_cache_stable(self):
        self.assertEqual(cache_key([{'chunk_id':'x','text':'y'}]),cache_key([{'chunk_id':'x','text':'y'}]))
    def test_strip_markup(self): self.assertEqual(strip_markup('<b>Text</b>'),'Text')
if __name__=='__main__': unittest.main(verbosity=2)
