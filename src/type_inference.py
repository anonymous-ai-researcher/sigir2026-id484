"""
Type Inference Module for OntoEL

Implements context-aware type inference using dual projection (Section 4.1.2).

Key components:
1. Type Name Encoding: Encode semantic type names using the same encoder
2. Dual Projection: Project mentions and type names into shared inference space
3. Fuzzy Membership Computation: Compute τ^I(m) via sigmoid-scaled similarity

Design rationale:
- Zero-Shot Generalization: Encoding type names enables inference for unseen types
- Task-Specific Adaptation: Dual projections learn type-inference space distinct from retrieval
- Semantic-Logic Alignment: Projections enable TBox to reshape encoder representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class TypeInferenceModule(nn.Module):
    """
    Context-Aware Type Inference Module (Section 4.1.2).
    
    Given a mention embedding, infers fuzzy membership degrees for each
    semantic type based on contextual evidence.
    
    Architecture:
        m' = W_m @ m          (project mention to inference space)
        a'_τ = W_t @ a_τ      (project type name embedding)
        τ^I(m) = σ(m' · a'_τ / θ)  (fuzzy membership via scaled dot product)
    
    The projection matrices W_m and W_t are learnable, allowing the model
    to adapt the type inference space during training while keeping the
    base encoder fixed.
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        projection_dim: int = 768,
        num_types: int = 21,
        temperature_init: float = 3.32,  # log(sqrt(768))
        learnable_temperature: bool = True,
    ):
        """
        Initialize type inference module.
        
        Args:
            hidden_dim: Dimension of encoder hidden states (d)
            projection_dim: Dimension of projection space (d')
            num_types: Number of semantic types |Γ|
            temperature_init: Initial value for log(θ), default log(sqrt(d'))
            learnable_temperature: Whether to learn temperature during training
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.num_types = num_types
        
        # Dual projection matrices (Eq. 6-7 in paper)
        # Initialize as identity (scaled) to preserve pre-trained semantics
        self.mention_projection = nn.Linear(hidden_dim, projection_dim, bias=False)
        self.type_projection = nn.Linear(hidden_dim, projection_dim, bias=False)
        
        # Initialize projections
        self._init_projections()
        
        # Learnable temperature parameter θ = exp(θ_hat)
        # Ensures θ > 0 and allows adaptive sharpness (Eq. 8)
        self.log_temperature = nn.Parameter(
            torch.tensor(temperature_init),
            requires_grad=learnable_temperature,
        )
        
        # Buffer for pre-computed projected type embeddings
        self.register_buffer("type_embeddings", None)
        self.register_buffer("projected_type_embeddings", None)
        
    def _init_projections(self):
        """Initialize projection matrices."""
        # Identity-like initialization to preserve pre-trained semantics
        # Paper mentions this helps avoid semantic drift in early training
        if self.hidden_dim == self.projection_dim:
            nn.init.eye_(self.mention_projection.weight)
            nn.init.eye_(self.type_projection.weight)
        else:
            # Orthogonal initialization for dimension mismatch
            nn.init.orthogonal_(self.mention_projection.weight)
            nn.init.orthogonal_(self.type_projection.weight)
    
    @property
    def temperature(self) -> torch.Tensor:
        """Get the temperature value θ = exp(θ_hat)."""
        return torch.exp(self.log_temperature)
    
    def set_type_embeddings(self, type_embeddings: torch.Tensor):
        """
        Set pre-computed type name embeddings.
        
        Args:
            type_embeddings: Tensor of shape [num_types, hidden_dim]
                Embeddings of type names (e.g., "Disease", "Finding")
                obtained from the encoder: a_τ = Enc(name(τ))
        """
        self.type_embeddings = type_embeddings
        # Pre-compute projected type embeddings
        self.projected_type_embeddings = self.type_projection(type_embeddings)
    
    def forward(
        self,
        mention_embeddings: torch.Tensor,
        type_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute fuzzy type memberships for mentions.
        
        Args:
            mention_embeddings: Mention embeddings from encoder
                Shape: [batch_size, hidden_dim]
            type_embeddings: Optional type name embeddings
                Shape: [num_types, hidden_dim]
                If None, uses pre-set embeddings
                
        Returns:
            Fuzzy type memberships τ^I(m) for each mention
                Shape: [batch_size, num_types]
                Values in [0, 1] via sigmoid
        """
        # Project mention embeddings: m' = W_m @ m
        projected_mentions = self.mention_projection(mention_embeddings)
        # Shape: [batch_size, projection_dim]
        
        # Get projected type embeddings
        if type_embeddings is not None:
            projected_types = self.type_projection(type_embeddings)
        elif self.projected_type_embeddings is not None:
            projected_types = self.projected_type_embeddings
        else:
            raise ValueError("No type embeddings available. Call set_type_embeddings first.")
        # Shape: [num_types, projection_dim]
        
        # Compute scaled dot product similarity: m' · a'_τ / θ
        # [batch, proj_dim] @ [proj_dim, num_types] -> [batch, num_types]
        logits = torch.matmul(projected_mentions, projected_types.T) / self.temperature
        
        # Apply sigmoid for fuzzy membership (Eq. 8)
        type_memberships = torch.sigmoid(logits)
        
        return type_memberships
    
    def get_type_distributions(
        self,
        mention_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get both fuzzy memberships and softmax distributions.
        
        Useful for visualization and debugging.
        
        Args:
            mention_embeddings: [batch_size, hidden_dim]
            
        Returns:
            Tuple of:
            - Fuzzy memberships (sigmoid): [batch_size, num_types]
            - Probability distribution (softmax): [batch_size, num_types]
        """
        projected_mentions = self.mention_projection(mention_embeddings)
        projected_types = self.projected_type_embeddings
        
        logits = torch.matmul(projected_mentions, projected_types.T) / self.temperature
        
        fuzzy_memberships = torch.sigmoid(logits)
        prob_distribution = F.softmax(logits, dim=-1)
        
        return fuzzy_memberships, prob_distribution


class CandidateTypeMembership(nn.Module):
    """
    Candidate Type Membership Lookup (Section 4.1.3).
    
    For candidates, type membership is determined by the TBox rather than
    inferred. This module manages the lookup of pre-computed memberships.
    
    τ^I(e) = 1 if T ⊨ e ⊑ τ, else 0
    
    Design: Crisp (binary) assignments ensure TBox entailments are strictly
    enforced, including transitivity. All uncertainty is localized to
    context-based type inference.
    """
    
    def __init__(self, num_entities: int, num_types: int):
        """
        Initialize candidate type membership module.
        
        Args:
            num_entities: Number of entities in target set |E|
            num_types: Number of semantic types |Γ|
        """
        super().__init__()
        self.num_entities = num_entities
        self.num_types = num_types
        
        # Pre-computed type memberships from TBox
        # Binary matrix: membership[e, τ] = 1 iff T ⊨ e ⊑ τ
        self.register_buffer(
            "type_memberships",
            torch.zeros(num_entities, num_types),
        )
    
    def set_memberships(self, memberships: torch.Tensor):
        """
        Set pre-computed type memberships from TBox.
        
        Args:
            memberships: Binary tensor [num_entities, num_types]
                Computed offline using DL reasoner (e.g., ELK)
        """
        self.type_memberships = memberships.float()
    
    def forward(self, entity_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up type memberships for candidate entities.
        
        Args:
            entity_ids: Indices of candidate entities
                Shape: [batch_size, num_candidates]
                
        Returns:
            Type memberships τ^I(e) for each candidate
                Shape: [batch_size, num_candidates, num_types]
        """
        # Gather memberships for specified entities
        # entity_ids: [batch, num_cand] -> memberships: [batch, num_cand, num_types]
        batch_size, num_candidates = entity_ids.shape
        
        # Flatten, gather, reshape
        flat_ids = entity_ids.reshape(-1)
        flat_memberships = self.type_memberships[flat_ids]
        memberships = flat_memberships.reshape(batch_size, num_candidates, self.num_types)
        
        return memberships


class TypeHierarchy:
    """
    Manages the semantic type hierarchy and disjointness constraints.
    
    Handles:
    1. Subsumption hierarchy (τ₁ ⊑ τ₂)
    2. Disjointness axioms (τ₁ ⊓ τ₂ ⊑ ⊥)
    3. Transitive closure computation
    """
    
    def __init__(self, type_names: List[str]):
        """
        Initialize type hierarchy.
        
        Args:
            type_names: List of semantic type names
        """
        self.type_names = type_names
        self.num_types = len(type_names)
        self.type_to_idx = {name: i for i, name in enumerate(type_names)}
        
        # Subsumption matrix: subsumes[i, j] = 1 iff τ_i ⊑ τ_j
        self.subsumption_matrix = torch.eye(self.num_types)
        
        # Disjointness matrix: disjoint[i, j] = 1 iff τ_i ⊓ τ_j ⊑ ⊥
        self.disjointness_matrix = torch.zeros(self.num_types, self.num_types)
    
    def add_subsumption(self, sub_type: str, super_type: str):
        """Add subsumption axiom: sub_type ⊑ super_type."""
        i = self.type_to_idx[sub_type]
        j = self.type_to_idx[super_type]
        self.subsumption_matrix[i, j] = 1
    
    def add_disjointness(self, type1: str, type2: str):
        """Add disjointness axiom: type1 ⊓ type2 ⊑ ⊥."""
        i = self.type_to_idx[type1]
        j = self.type_to_idx[type2]
        self.disjointness_matrix[i, j] = 1
        self.disjointness_matrix[j, i] = 1
    
    def compute_transitive_closure(self):
        """Compute transitive closure of subsumption hierarchy."""
        # Floyd-Warshall for transitive closure
        n = self.num_types
        closure = self.subsumption_matrix.clone()
        
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    closure[i, j] = max(closure[i, j], closure[i, k] * closure[k, j])
        
        self.subsumption_matrix = closure
    
    def get_entity_types(self, direct_types: List[str]) -> torch.Tensor:
        """
        Get all types for an entity including inherited types.
        
        Args:
            direct_types: Direct type assignments for entity
            
        Returns:
            Binary vector of all type memberships (including inherited)
        """
        membership = torch.zeros(self.num_types)
        
        for type_name in direct_types:
            if type_name in self.type_to_idx:
                idx = self.type_to_idx[type_name]
                # Include this type and all supertypes
                membership = torch.maximum(membership, self.subsumption_matrix[idx])
        
        return membership
    
    def to_device(self, device: torch.device):
        """Move tensors to specified device."""
        self.subsumption_matrix = self.subsumption_matrix.to(device)
        self.disjointness_matrix = self.disjointness_matrix.to(device)


# Standard UMLS Semantic Types for MedMentions ST21pv
MEDMENTIONS_ST21PV_TYPES = [
    "T005",  # Virus
    "T007",  # Bacterium
    "T017",  # Anatomical Structure
    "T022",  # Body System
    "T031",  # Body Substance
    "T033",  # Finding
    "T037",  # Injury or Poisoning
    "T038",  # Biologic Function
    "T058",  # Health Care Activity
    "T062",  # Research Activity
    "T074",  # Medical Device
    "T082",  # Spatial Concept
    "T091",  # Biomedical Occupation or Discipline
    "T092",  # Organization
    "T097",  # Professional or Occupational Group
    "T098",  # Population Group
    "T103",  # Chemical
    "T168",  # Food
    "T170",  # Intellectual Product
    "T201",  # Clinical Attribute
    "T204",  # Eukaryote
]

# Human-readable names for semantic types
SEMANTIC_TYPE_NAMES = {
    "T005": "Virus",
    "T007": "Bacterium", 
    "T017": "Anatomical Structure",
    "T022": "Body System",
    "T031": "Body Substance",
    "T033": "Finding",
    "T037": "Injury or Poisoning",
    "T038": "Biologic Function",
    "T058": "Health Care Activity",
    "T062": "Research Activity",
    "T074": "Medical Device",
    "T082": "Spatial Concept",
    "T091": "Biomedical Occupation or Discipline",
    "T092": "Organization",
    "T097": "Professional or Occupational Group",
    "T098": "Population Group",
    "T103": "Chemical",
    "T168": "Food",
    "T170": "Intellectual Product",
    "T201": "Clinical Attribute",
    "T204": "Eukaryote",
}

# Semantic Groups (coarser grouping)
SEMANTIC_GROUPS = {
    "Anatomy": ["T017", "T022", "T031"],
    "Chemicals & Drugs": ["T103"],
    "Disorders": ["T037"],
    "Genes & Molecular Sequences": [],
    "Living Beings": ["T005", "T007", "T204"],
    "Objects": ["T074"],
    "Occupations": ["T091", "T097"],
    "Organizations": ["T092"],
    "Phenomena": ["T038"],
    "Physiology": ["T201"],
    "Procedures": ["T058", "T062"],
    "Concepts & Ideas": ["T082", "T170", "T168"],
    "Activities & Behaviors": [],
    "Geographic Areas": [],
    "Devices": [],
}

# Disjointness between semantic groups
GROUP_DISJOINTNESS = [
    ("Anatomy", "Disorders"),
    ("Anatomy", "Chemicals & Drugs"),
    ("Disorders", "Living Beings"),
    ("Chemicals & Drugs", "Living Beings"),
    ("Procedures", "Disorders"),
    ("Organizations", "Living Beings"),
]


def create_medmentions_type_hierarchy() -> TypeHierarchy:
    """
    Create type hierarchy for MedMentions ST21pv.
    
    Uses the 21 semantic types with standard UMLS relationships.
    """
    type_names = list(SEMANTIC_TYPE_NAMES.values())
    hierarchy = TypeHierarchy(type_names)
    
    # Add disjointness constraints based on semantic groups
    for group1, group2 in GROUP_DISJOINTNESS:
        types1 = SEMANTIC_GROUPS.get(group1, [])
        types2 = SEMANTIC_GROUPS.get(group2, [])
        
        for t1 in types1:
            for t2 in types2:
                name1 = SEMANTIC_TYPE_NAMES.get(t1)
                name2 = SEMANTIC_TYPE_NAMES.get(t2)
                if name1 and name2:
                    hierarchy.add_disjointness(name1, name2)
    
    hierarchy.compute_transitive_closure()
    return hierarchy
