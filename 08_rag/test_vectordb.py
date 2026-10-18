"""Offline Qdrant local-mode tests: no cloud account, OpenAI key, or API calls."""
import unittest
import numpy as np
from qdrant_client import QdrantClient
from rag_utils import cosine_search
from vector_db import index_chunks, search_collection


class VectorDBTests(unittest.TestCase):
    def setUp(self):
        self.client = QdrantClient(':memory:')
        self.docs = [
            {'chunk_id': 'a:0', 'source_id': 'a', 'title': 'Bayesian trials', 'url': 'https://example.org/a', 'text': 'Adaptive Bayesian trial designs'},
            {'chunk_id': 'b:0', 'source_id': 'b', 'title': 'Regression', 'url': 'https://example.org/b', 'text': 'Linear regression analysis'},
            {'chunk_id': 'c:0', 'source_id': 'c', 'title': 'Other trials', 'url': 'https://example.org/c', 'text': 'More designs'},
        ]
        self.vectors = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    def test_index_search_repeat_filter(self):
        name = index_chunks(self.docs, self.vectors, client=self.client)
        self.assertEqual(self.client.count(name, exact=True).count, 3)
        index_chunks(self.docs, self.vectors, client=self.client)
        self.assertEqual(self.client.count(name, exact=True).count, 3)
        q = np.array([1.0, 0.0])
        hits = search_collection(self.client, name, q, k=2)
        expected = cosine_search(q, self.vectors, k=2)
        self.assertEqual(hits[0]['chunk_id'], self.docs[expected[0][0]]['chunk_id'])
        self.assertAlmostEqual(hits[0]['score'], 1.0, places=4)
        self.assertEqual(search_collection(self.client, name, q, k=3, source_id='b')[0]['source_id'], 'b')

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            index_chunks([], np.empty((0, 2)), client=self.client)


if __name__ == '__main__':
    unittest.main(verbosity=2)
