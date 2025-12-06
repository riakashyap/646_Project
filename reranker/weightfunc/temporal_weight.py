"""
Copyright:

  Copyright © 2025 uchuuronin

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:
    Weight function that prioritises more recent documents.
    For FEVER: Explicit dates not available, returns neutral weights.
    For web sources: Extracts publish dates and applies recency bias.
    
Code:
"""

from typing import List, Tuple, Optional
from datetime import datetime
import re
from .weightfunc import BaseWeightFunction

class TemporalWeightFunction(BaseWeightFunction):
  def __init__(
      self, 
      recency_boost: float = 0.2,
      for_fever: bool = True, 
      reference_year: int = 2025
  ):
    super().__init__()
    self.recency_boost = recency_boost
    self.for_fever = for_fever
    self.reference_year = reference_year or datetime.now().year

  def _extract_publication_date(self, text: str, doc_id: str) -> Optional[int]:
    pub_match = re.search(
      r'(?:published|updated|posted|date)[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
      text, re.IGNORECASE
    )
    if pub_match:
      year_match = re.search(r'(\d{4})', pub_match.group(1))
      if year_match:
        return int(year_match.group(1))
    
    copyright_match = re.search(r'©?\s*(?:copyright|©)\s*(\d{4})', text, re.IGNORECASE)
    if copyright_match:
      return int(copyright_match.group(1))
    
    return None

  def _compute_recency_score(self, pub_year: Optional[int]) -> float:
    if pub_year is None:
      return 1.0 
    
    years_old = self.reference_year - pub_year
    if years_old <= 5:
      return 1.0 + self.recency_boost
    elif years_old <= 10:
      return 1.0
    else:
      return 1.0 - (self.recency_boost / 2)

  def compute_weights(
      self, 
      query: str,
      documents: List[Tuple[str, str, float]]
  ) -> List[float]:
    if self.for_fever or self.recency_boost == 0.0:
      return [1.0] * len(documents)
    
    weights = []
    for doc_id, doc_text, _ in documents:
      pub_year = self._extract_publication_date(doc_text, doc_id)
      score = self._compute_recency_score(pub_year)
      weights.append(score)
    return weights