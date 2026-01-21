"""
Fuzzy Logic Module for OntoEL

Implements differentiable fuzzy EL++ operators:
- Product T-Norm for conjunction
- Probabilistic Sum for existential quantification  
- Sigmoidal Reichenbach Implication for subsumption

Key insight from paper: The Sigmoidal Reichenbach implication resolves
the "implication bias" problem that causes vanishing gradients in other
fuzzy logic formulations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal


def product_tnorm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Product T-Norm for fuzzy conjunction.
    
    T_P(a, b) = a * b
    
    Gradient properties:
    - ∂T_P/∂a = b
    - ∂T_P/∂b = a
    
    This provides adaptive gradient scaling where each input receives
    signal proportional to its partner's value, avoiding the sparse
    gradients of Gödel T-norm or dead zones of Łukasiewicz.
    
    Args:
        a: First fuzzy truth value, shape [..., N]
        b: Second fuzzy truth value, shape [..., N]
        
    Returns:
        Product conjunction, shape [..., N]
    """
    return a * b


def godel_tnorm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Gödel T-Norm for fuzzy conjunction (minimum).
    
    T_G(a, b) = min(a, b)
    
    Note: Non-differentiable at a = b, creates sparse gradients.
    Included for comparison but not recommended for training.
    """
    return torch.minimum(a, b)


def lukasiewicz_tnorm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Łukasiewicz T-Norm for fuzzy conjunction.
    
    T_L(a, b) = max(0, a + b - 1)
    
    Note: Suffers from gradient vanishing when a + b <= 1.
    Included for comparison but not recommended for training.
    """
    return torch.clamp(a + b - 1, min=0)


def probabilistic_sum(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Probabilistic Sum (Dual of Product T-Norm) for fuzzy disjunction/existential.
    
    S_P(z_1, ..., z_n) = 1 - Π_i(1 - z_i)
    
    Interpretation: Probability that at least one value holds.
    Used for existential restriction: (∃r.C)^I(d)
    
    Args:
        values: Fuzzy truth values, shape [..., N]
        dim: Dimension to aggregate over
        
    Returns:
        Probabilistic sum, shape [...]
    """
    # Compute in log-space for numerical stability
    # 1 - Π(1-z_i) = 1 - exp(Σ log(1-z_i))
    log_complement = torch.log(1 - values + 1e-10)
    return 1 - torch.exp(torch.sum(log_complement, dim=dim))


def reichenbach_implication(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Linear Reichenbach Implication (S-implication from Probabilistic Sum).
    
    I_R(a, b) = 1 - a + a*b
    
    Derived via: a → b ≡ ¬a ∨ b using Probabilistic Sum
    
    Properties:
    - No implication bias (doesn't saturate when a <= b)
    - Symmetric gradients: ∂I_R/∂a = b-1, ∂I_R/∂b = a
    - Numerically stable (no division)
    
    Args:
        a: Antecedent fuzzy truth value
        b: Consequent fuzzy truth value
        
    Returns:
        Implication value in [0, 1]
    """
    return 1 - a + a * b


def sigmoidal_reichenbach(
    a: torch.Tensor, 
    b: torch.Tensor, 
    sharpness: float = 10.0
) -> torch.Tensor:
    """
    Sigmoidal Reichenbach Implication (Eq. 2 in paper).
    
    I_σ(a, b) = σ(s · (1 - a + ab - 0.5))
    
    Key innovation: The sigmoid concentrates gradient mass near the
    decision boundary (I_R ≈ 0.5), amplifying penalties for ambiguous
    violations while dampening trivial cases.
    
    Theorem 1 (Gradient Non-Degeneracy):
    - Maintains ||∇I_σ(a,b)|| > 0 for all (a,b) ∈ (0,1)²
    - Discrimination ratio grows as ~e^(0.31s) for hard cases
    
    Args:
        a: Antecedent (context-inferred type membership), shape [batch, num_types]
        b: Consequent (candidate type membership), shape [batch, num_types]
        sharpness: Controls sigmoid steepness (paper uses s=10)
        
    Returns:
        Implication value in [0, 1], shape [batch, num_types]
    """
    linear = reichenbach_implication(a, b)
    return torch.sigmoid(sharpness * (linear - 0.5))


def goguen_implication(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Goguen Implication (R-implication of Product T-Norm).
    
    I_GG(a, b) = 1 if a <= b, else b/a
    
    WARNING: Suffers from implication bias - returns 1 whenever a <= b
    regardless of actual values, providing no gradient signal.
    Included for comparison but NOT recommended.
    
    Args:
        a: Antecedent fuzzy truth value
        b: Consequent fuzzy truth value
        
    Returns:
        Implication value in [0, 1]
    """
    # Differentiable approximation using soft minimum
    ratio = b / (a + eps)
    return torch.minimum(ratio, torch.ones_like(ratio))


class FuzzyLogicLayer(nn.Module):
    """
    Differentiable Fuzzy EL++ Logic Layer.
    
    Computes ontological consistency scores between mention type distributions
    and candidate entity types using fuzzy Description Logic operators.
    
    Architecture:
    1. For each semantic type τ:
       cons_τ(m, e) = I_σ(τ^I(m), τ^I(e))
    2. Aggregate across types using Product T-Norm:
       s_onto(m, e) = Π_τ cons_τ(m, e)
       
    The layer is fully differentiable, allowing TBox violations to 
    backpropagate through the type inference module to the encoder.
    """
    
    def __init__(
        self,
        num_types: int,
        sharpness: float = 10.0,
        tnorm: Literal["product", "godel", "lukasiewicz"] = "product",
        implication: Literal["sigmoidal_reichenbach", "reichenbach", "goguen"] = "sigmoidal_reichenbach",
        use_log_space: bool = True,
    ):
        """
        Initialize the fuzzy logic layer.
        
        Args:
            num_types: Number of semantic types |Γ|
            sharpness: Sigmoid sharpness parameter s (default: 10)
            tnorm: T-norm to use for aggregation
            implication: Implication operator to use
            use_log_space: Whether to compute in log-space for stability
        """
        super().__init__()
        self.num_types = num_types
        self.sharpness = sharpness
        self.tnorm = tnorm
        self.implication = implication
        self.use_log_space = use_log_space
        
        # Select T-norm function
        self.tnorm_fn = {
            "product": product_tnorm,
            "godel": godel_tnorm,
            "lukasiewicz": lukasiewicz_tnorm,
        }[tnorm]
        
        # Select implication function
        self.impl_fn = {
            "sigmoidal_reichenbach": lambda a, b: sigmoidal_reichenbach(a, b, sharpness),
            "reichenbach": reichenbach_implication,
            "goguen": goguen_implication,
        }[implication]
    
    def forward(
        self,
        mention_type_probs: torch.Tensor,
        candidate_type_memberships: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute fuzzy consistency scores.
        
        Args:
            mention_type_probs: Context-inferred type probabilities τ^I(m)
                Shape: [batch_size, num_types]
            candidate_type_memberships: Candidate type memberships τ^I(e)
                Shape: [batch_size, num_candidates, num_types]
                Binary (0/1) from TBox entailment
                
        Returns:
            Consistency scores s_onto(m, e)
                Shape: [batch_size, num_candidates]
        """
        batch_size = mention_type_probs.shape[0]
        num_candidates = candidate_type_memberships.shape[1]
        
        # Expand mention probs for broadcasting
        # [batch, num_types] -> [batch, 1, num_types]
        mention_probs = mention_type_probs.unsqueeze(1)
        
        # Compute per-type consistency: I_σ(τ^I(m), τ^I(e))
        # Shape: [batch, num_candidates, num_types]
        type_consistency = self.impl_fn(mention_probs, candidate_type_memberships)
        
        if self.use_log_space:
            # Compute in log-space for numerical stability (Eq. 9 in paper)
            # log s_onto = Σ_τ log I_σ(τ^I(m), τ^I(e))
            log_consistency = torch.log(type_consistency + 1e-10)
            log_onto_score = torch.sum(log_consistency, dim=-1)
            onto_score = torch.exp(log_onto_score)
        else:
            # Direct computation: s_onto = Π_τ cons_τ
            onto_score = torch.prod(type_consistency, dim=-1)
        
        return onto_score
    
    def compute_disjointness_penalty(
        self,
        mention_type_probs: torch.Tensor,
        candidate_type_memberships: torch.Tensor,
        disjointness_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute additional penalty for disjointness violations.
        
        For axioms like Disease ⊓ Finding ⊑ ⊥, penalize candidates
        where both types are inferred from context but candidate
        belongs to one while context suggests the other.
        
        Args:
            mention_type_probs: [batch, num_types]
            candidate_type_memberships: [batch, num_candidates, num_types]
            disjointness_matrix: [num_types, num_types] binary matrix
                disjointness_matrix[i,j] = 1 if types i and j are disjoint
                
        Returns:
            Disjointness penalty, shape [batch, num_candidates]
        """
        # For each pair of disjoint types (i, j):
        # If context suggests type i and candidate is type j, penalize
        batch_size = mention_type_probs.shape[0]
        num_candidates = candidate_type_memberships.shape[1]
        
        penalty = torch.zeros(batch_size, num_candidates, device=mention_type_probs.device)
        
        # Get indices of disjoint pairs
        disjoint_i, disjoint_j = torch.where(disjointness_matrix > 0)
        
        for i, j in zip(disjoint_i.tolist(), disjoint_j.tolist()):
            # Context suggests type i, candidate is type j
            context_i = mention_type_probs[:, i].unsqueeze(1)  # [batch, 1]
            cand_j = candidate_type_memberships[:, :, j]  # [batch, num_cand]
            
            # Penalty: high if context says i AND candidate is j
            # Using product to compute "both hold"
            violation = product_tnorm(context_i, cand_j)
            penalty = penalty + violation
        
        return penalty


class TypeConsistencyScorer(nn.Module):
    """
    Complete type consistency scoring module.
    
    Combines:
    1. Basic fuzzy consistency from FuzzyLogicLayer
    2. Disjointness penalties for hard constraints
    3. Optional hierarchy-aware scoring
    """
    
    def __init__(
        self,
        num_types: int,
        sharpness: float = 10.0,
        use_disjointness: bool = True,
        disjointness_weight: float = 0.5,
    ):
        super().__init__()
        self.fuzzy_layer = FuzzyLogicLayer(
            num_types=num_types,
            sharpness=sharpness,
        )
        self.use_disjointness = use_disjointness
        self.disjointness_weight = disjointness_weight
        
        # Will be set during training
        self.register_buffer("disjointness_matrix", None)
    
    def set_disjointness_matrix(self, matrix: torch.Tensor):
        """Set the disjointness matrix from TBox."""
        self.disjointness_matrix = matrix
    
    def forward(
        self,
        mention_type_probs: torch.Tensor,
        candidate_type_memberships: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute complete consistency score.
        
        Args:
            mention_type_probs: [batch, num_types]
            candidate_type_memberships: [batch, num_candidates, num_types]
            
        Returns:
            Consistency scores, [batch, num_candidates]
        """
        # Basic fuzzy consistency
        base_score = self.fuzzy_layer(mention_type_probs, candidate_type_memberships)
        
        # Add disjointness penalty if enabled
        if self.use_disjointness and self.disjointness_matrix is not None:
            penalty = self.fuzzy_layer.compute_disjointness_penalty(
                mention_type_probs,
                candidate_type_memberships,
                self.disjointness_matrix,
            )
            # Reduce score by penalty
            base_score = base_score * (1 - self.disjointness_weight * penalty)
        
        return base_score
