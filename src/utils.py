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


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
WIKI_DIR = DATA_DIR / "wiki"
ZIP_PATH = DATA_DIR / "wiki-pages.zip"
INDEX_DIR = WIKI_DIR / "index"
PAGES_DIR = WIKI_DIR / "wiki-pages"
PROMPTS_DIR = SCRIPT_DIR.parent / "prompts"
