"""
Copyright:

  Copyright © 2025 bdunahu
  Copyright © 2025 Eric
  Copyright © 2025 uchuuronin

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
LOGS_DIR = SCRIPT_DIR.parent / "logs"
WIKI_DIR = DATA_DIR / "wiki"
ZIP_PATH = DATA_DIR / "wiki-pages.zip"
INDEX_DIR = WIKI_DIR / "index"
PAGES_DIR = WIKI_DIR / "wiki-pages"
TRACE_PATH = LOGS_DIR / "trace.log"
EVAL_OUT_FNAME_BASE = "metrics"

CLAIMS_PATH = DATA_DIR / "fever-claims.json"
QRELS_PATH = DATA_DIR / "fever-qrel.json"
RANKLISTS_PATH = DATA_DIR / "fever-ranklist.json"
RERANKEDLISTS_PATH = DATA_DIR / "fever-reranked.json"
TOP_CLAIMS_PATH = DATA_DIR / "fever-claims-top.json"
TOP_QRELS_PATH = DATA_DIR / "fever-qrel-top.json"
TOP_RANKLISTS_PATH = DATA_DIR / "fever-ranklist-top.json"

PROMPTS_DIR = SCRIPT_DIR.parent / "prompts"
RAGAR_DIR = PROMPTS_DIR / "ragar"
MADR_DIR = PROMPTS_DIR / "madr"
RAGAR_ORIG_DIR = PROMPTS_DIR / "ragar_orig"
MADR_ORIG_DIR = PROMPTS_DIR / "madr_orig"
SYSTEM_TAG = "<<SYSTEM>>"
USER_TAG = "<<USER>>"

# global logger
LOGGER = logging.getLogger('corag_logger')
LOGGER.setLevel(logging.DEBUG)
LOGGER.propagate = False
LOGGER.disabled = True

def make_log_file():
    if not os.path.exists(TRACE_PATH):
        with open(TRACE_PATH, 'w') as file:
            pass

    file_handler = logging.FileHandler(TRACE_PATH)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s --- %(message)s')
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
