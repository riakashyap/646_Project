"""
Copyright:

  Copyright © 2025 uchuuronin

  You should have received a copy of the MIT license along with this file.
  If not, see https://mit-license.org/

Commentary:
    Abstract base class for defining weighting functions.
    
Code:
"""
from abc import ABC, abstractmethod
import torch

class BaseWeightFunction(ABC)