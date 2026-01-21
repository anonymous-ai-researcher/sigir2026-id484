"""
OntoEL: Neuro-Symbolic Biomedical Entity Linking with Differentiable Fuzzy EL++ Reasoning

This package implements the OntoEL framework for biomedical entity linking,
combining neural bi-encoder retrieval with differentiable fuzzy Description Logic reasoning.
"""

from .model import OntoEL
from .fuzzy_logic import FuzzyLogicLayer, sigmoidal_reichenbach, product_tnorm
from .type_inference import TypeInferenceModule
from .dataset import BioELDataset, BioELCollator
from .retrieval import CandidateRetriever
from .trainer import OntoELTrainer
from .utils import load_config, set_seed

__version__ = "1.0.0"
__author__ = "Anonymous"

__all__ = [
    "OntoEL",
    "FuzzyLogicLayer",
    "sigmoidal_reichenbach",
    "product_tnorm",
    "TypeInferenceModule",
    "BioELDataset",
    "BioELCollator",
    "CandidateRetriever",
    "OntoELTrainer",
    "load_config",
    "set_seed",
]
