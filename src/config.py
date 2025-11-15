"""
Copyright:

  Copyright © 2025 bdunahu
  Copyright © 2025 Eric

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:

  This file contains various configuration variables.

Code:
"""

from pathlib import Path
import os
import logging

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
WIKI_DIR = DATA_DIR / "wiki"
ZIP_PATH = DATA_DIR / "wiki-pages.zip"
INDEX_DIR = WIKI_DIR / "index"
PAGES_DIR = WIKI_DIR / "wiki-pages"

QRELS_PATH = DATA_DIR / "fever-qrel.json"
CLAIMS_PATH = DATA_DIR / "fever-claims.json"
RANKLISTS_PATH = DATA_DIR / "fever-ranklist.json"
TRACE_PATH = SCRIPT_DIR.parent / "logs" / "trace.log"

PROMPTS_DIR = SCRIPT_DIR.parent / "prompts"

# global logger
LOGGER = None

def make_logger():
    if not os.path.exists(TRACE_PATH):
        with open(TRACE_PATH, 'w') as file:
            pass
    global LOGGER
    LOGGER = logging.getLogger('corag_logger')
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.propagate = False
    file_handler = logging.FileHandler(TRACE_PATH)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s --- %(message)s')
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.info("\n" + "═" * 40)
    LOGGER.info("Starting pipeline...")
