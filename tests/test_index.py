"""
Copyright:

  Copyright © 2025 bdunahu

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:

  This file checks the inverted index's integrity.

Code:
"""

from pyserini.index.lucene import IndexReader
from src.config import INDEX_DIR, PAGES_DIR, WIKI_DIR
from src.index import load_wiki, build_index
import unittest
import itertools
import os

class TestIndex(unittest.TestCase):

    reader: IndexReader

    @classmethod
    def setUpClass(self):
        super().setUpClass()

        if not (PAGES_DIR.exists() and os.listdir(PAGES_DIR)):
            print(f'Downloading the wiki data to {PAGES_DIR}...')
            load_wiki()

        if not (INDEX_DIR.exists() and os.listdir(INDEX_DIR)):
            print(f'Populating the index to {INDEX_DIR}...')
            build_index()

        self.reader = IndexReader(str(INDEX_DIR))

    def test_index_stats(self):
        expected = {
            'total_terms': 322660814,
            'documents': 5396106,
            'non_empty_documents': 5396060,
            'unique_terms': -1,
        }
        actual = self.reader.stats()
        self.assertEqual(expected, actual, "index stats do not match expected.")

    def test_contents_stored(self):
        doc = self.reader.doc("Hesiod_-LRB-name_service-RRB-")
        self.assertIsNotNone(doc.get("contents"), "Index was generated without document contents.")
