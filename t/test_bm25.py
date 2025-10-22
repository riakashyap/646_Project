"""
Copyright:

  Copyright © 2025 bdunahu

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:

  This file tests the efficacy of the BM25 retrieval stage by using mean
  average precision.

Code:
"""

from pyserini.search.lucene import LuceneSearcher
from src.utils import INDEX_DIR


class TestBM25(unittest.TestCase):

    searcher: LuceneSearcher

    @classmethod
    def setUpClass(self):
        super().setUpClass()
        self.searcher = LuceneSearcher(str(INDEX_DIR))
        self.searcher.set_bm25(1.2, 0.75)

    def test_can_retrieve(self):
        query = "serval"
        hits = self.searcher.search(query, k=5)

        self.assertGreater(len(hits), 0, "BM25 search returned no documents.")
