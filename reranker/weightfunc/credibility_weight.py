"""
Copyright:

  Copyright © 2025 uchuuronin

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:
    Weight function that prioritizes credible/trusted sources.
    Can use Stanford Guidelines for Web Credibility: https://credibility.stanford.edu/guidelines/index.html
    Summary of adapted guidelines:
    1. Source Authority (Taking #2, #3, #4): Asses domain reputation and organisation's credibility
    2. Content Quality (#1, #10): Checks citations and avoids docs with typos or other red flags
    3. Recency (#8): SKIPPED HERE- considered seperately in temporal weight func
    4. Objectivity and Bias (#9): Addresses neutrality and commercial bias
    5. Usability (#5, #6, #7): Assesses quality of site and trustworthiness
      
Code:
"""

from typing import List, Tuple, Dict
import re
from .weightfunc import BaseWeightFunction

HIGH_CREDIBILITY_DOMAINS = {
  '.gov', '.edu', '.ac.uk', '.edu.au',
  'snopes.com', 'factcheck.org', 'politifact.com',
  'npr.org', 'pbs.org','nature.com', 'science.org', 'ieee.org', 
  'acm.org', 'un.org', 'who.int', 'worldbank.org',
}

MEDIUM_CREDIBILITY_DOMAINS = {
  'nytimes.com', 'washingtonpost.com', 'wsj.com', 'ft.com',
  'theguardian.com', 'economist.com', 'bloomberg.com',
  'cnbc.com', 'cnn.com', 'foxnews.com', 'nbcnews.com',
  'wikipedia.org', 'wikimedia.org', 'bbc.co.uk', 'bbc.com',
  'techcrunch.com', 'arstechnica.com', 'wired.com',
  '.org', 
}
  
DEFAULT_WEIGHTS = {
  'source_authority': 0.40,   
  'content_quality': 0.30,    
  'objectivity': 0.10,        
  'professionalism': 0.05,  
}
  

class CredibilityWeightFunction(BaseWeightFunction):
  def __init__(self, wikipedia_only: bool = True, rule_weight_map: Dict[str, float] = None):
    super().__init__()
    self.wikipedia_only = wikipedia_only # FEVER eval
    
    self.weights = DEFAULT_WEIGHTS.copy()
    if rule_weight_map is not None:
      self.weights.update(rule_weight_map) # Override defaults with provided values
      
  def _extract_domain(self, doc_id: str) -> str:
    if 'http://' in doc_id or 'https://' in doc_id:
      match = re.search(r'https?://(?:www\.)?([^/]+)', doc_id)
      return match.group(1).lower() if match else ''
    else:
      return 'wikipedia.org'  # Default for FEVER
    
  def _assess_source_authority(self, domain: str) -> float:
    domain_lower = domain.lower()
    for high_domain in HIGH_CREDIBILITY_DOMAINS:
      if domain_lower.endswith(high_domain) or high_domain in domain_lower:
        return 1.5
    
    for med_domain in MEDIUM_CREDIBILITY_DOMAINS:
      if domain_lower.endswith(med_domain) or med_domain in domain_lower:
        return 1.0
    
    return 0.7 # unknown or low credibility domain
  
  def _assess_content_quality(self, text: str) -> float:
    score = 1.0
    
    # Check for citations/references
    has_citations = bool(re.search(r'\[\d+\]|\(\d{4}\)|et al\.', text))
    if has_citations:
      score += 0.2
    
    # Check for specific dates/numbers
    has_specifics = bool(re.search(r'\d{4}|\d+\.\d+%|[A-Z][a-z]+ \d+, \d{4}', text))
    if has_specifics:
      score += 0.1
    
    # Check content length
    word_count = len(text.split())
    if word_count < 50:
      score -= 0.2
    elif word_count > 200:
      score += 0.05
    
    # Check for excessive caps lock
    caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    if caps_ratio > 0.3:
      score -= 0.3
    
    # Check for excessive exclamation
    exclamation_count = text.count('!')
    if exclamation_count > max(1, len(text) / 200):  # More than 1 per 200 chars
      score -= 0.2
    
    return max(0.6, min(score, 1.3))
  
  def _assess_objectivity(self, text: str) -> float:
    score = 1.0
    text_lower = text.lower()
    
    promotional_words = [
      'best', 'amazing', 'revolutionary', 'guaranteed',
      'incredible', 'shocking', 'unbelievable', 'must-see',
      'secret', 'miracle', 'breakthrough'
    ]
    promo_count = sum(1 for word in promotional_words if word in text_lower)
    if promo_count > 2:
      score -= 0.3
    elif promo_count > 0:
      score -= 0.1
    
    balanced_indicators = [
      'however', 'although', 'may', 'suggests', 'could',
      'according to', 'research shows', 'studies indicate'
    ]
    balanced_count = sum(1 for phrase in balanced_indicators if phrase in text_lower)
    if balanced_count > 2:
      score += 0.2
    
    return max(0.5, min(score, 1.2))
    
  def _assess_professionalism(self, text: str, domain: str) -> float:
    score = 1.0
    if 'wikipedia.org' in domain:
      return 1.0
    is_formatted = bool(re.search(r'\n\n|\. [A-Z]', text))  
    if is_formatted:
      score += 0.05
    has_attribution = bool(re.search(r'author|by |editor|published|source|contact', text.lower()))
    if has_attribution:
      score += 0.05
    
    return min(score, 1.1)

  def compute_weights(
    self,
    query: str,
    documents: List[Tuple[str, str, float]]
  ) -> List[float]:
    domains = [self._extract_domain(doc_id) for doc_id, _, _ in documents]
    weights = []
    for (doc_id, doc_text, _), domain in zip(documents, domains):
      source_score = 1.0 if self.wikipedia_only else self._assess_source_authority(domain)
      quality_score = self._assess_content_quality(doc_text)
      objectivity_score = self._assess_objectivity(doc_text)
      professional_score = self._assess_professionalism(doc_text, domain)
      
      final_score = (
        source_score * self.weights['source_authority'] +
        quality_score * self.weights['content_quality'] +
        objectivity_score * self.weights['objectivity'] +
        professional_score * self.weights['professionalism']
      )
      
      normalized = max(0.5, min(final_score, 2.0))
      weights.append(normalized)

    return weights