"""
OntoEL Model Architecture

Main model class that combines:
1. Neural bi-encoder for semantic similarity (SapBERT)
2. Type inference module for context-aware type prediction
3. Fuzzy logic layer for ontological consistency scoring
4. Score fusion for final re-ranking

Reference: Section 4 of the paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Optional, Tuple, Any
import math

from .fuzzy_logic import FuzzyLogicLayer, TypeConsistencyScorer
from .type_inference import TypeInferenceModule, CandidateTypeMembership


class OntoEL(nn.Module):
    """
    OntoEL: Neuro-Symbolic Biomedical Entity Linking Model.
    
    Architecture overview:
    
    1. Encoding Stage:
       - Encode mention with context: m = Enc([c_left; m; c_right])
       - Encode candidate names: e = Enc(name(e))
       
    2. Neural Retrieval (Eq. 3):
       - s_neural(m, e) = cos(m, e)
       
    3. Type Inference (Eq. 8):
       - Project: m' = W_m @ m, a'_τ = W_t @ a_τ
       - τ^I(m) = σ(m' · a'_τ / θ)
       
    4. Fuzzy Consistency (Eq. 7-9):
       - cons_τ(m, e) = I_σ(τ^I(m), τ^I(e))
       - s_onto(m, e) = Π_τ cons_τ(m, e)
       
    5. Score Fusion (Eq. 11-12):
       - s_final = α · norm(s_neural) + (1-α) · s_onto
    """
    
    def __init__(
        self,
        encoder_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        hidden_dim: int = 768,
        projection_dim: int = 768,
        num_types: int = 21,
        sigmoid_sharpness: float = 10.0,
        fusion_alpha: float = 0.8,
        temperature_init: float = 3.32,
        learnable_temperature: bool = True,
        learnable_alpha: bool = False,
        freeze_encoder: bool = False,
        use_disjointness: bool = True,
    ):
        """
        Initialize OntoEL model.
        
        Args:
            encoder_name: HuggingFace model name for encoder
            hidden_dim: Encoder hidden dimension (d)
            projection_dim: Type projection dimension (d')
            num_types: Number of semantic types |Γ|
            sigmoid_sharpness: Sharpness (s) for Sigmoidal Reichenbach
            fusion_alpha: Balance between neural and logic scores (α)
            temperature_init: Initial log-temperature for type inference
            learnable_temperature: Whether to learn temperature
            learnable_alpha: Whether to learn fusion weight
            freeze_encoder: Whether to freeze encoder weights
            use_disjointness: Whether to use disjointness penalties
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.num_types = num_types
        self.sigmoid_sharpness = sigmoid_sharpness
        self.use_disjointness = use_disjointness
        
        # Initialize encoder (SapBERT)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Type inference module (Section 4.1.2)
        self.type_inference = TypeInferenceModule(
            hidden_dim=hidden_dim,
            projection_dim=projection_dim,
            num_types=num_types,
            temperature_init=temperature_init,
            learnable_temperature=learnable_temperature,
        )
        
        # Fuzzy logic layer (Section 3)
        self.fuzzy_logic = FuzzyLogicLayer(
            num_types=num_types,
            sharpness=sigmoid_sharpness,
            tnorm="product",
            implication="sigmoidal_reichenbach",
            use_log_space=True,
        )
        
        # Full consistency scorer with disjointness
        self.consistency_scorer = TypeConsistencyScorer(
            num_types=num_types,
            sharpness=sigmoid_sharpness,
            use_disjointness=use_disjointness,
        )
        
        # Fusion weight (α)
        if learnable_alpha:
            # Parameterize via sigmoid to ensure α ∈ [0, 1]
            self._alpha_logit = nn.Parameter(torch.tensor(
                math.log(fusion_alpha / (1 - fusion_alpha))
            ))
        else:
            self.register_buffer("_alpha", torch.tensor(fusion_alpha))
            self._alpha_logit = None
        
        # Buffers for pre-computed embeddings
        self.register_buffer("entity_embeddings", None)
        self.register_buffer("entity_type_memberships", None)
        self.register_buffer("type_name_embeddings", None)
    
    @property
    def alpha(self) -> torch.Tensor:
        """Get fusion weight α."""
        if self._alpha_logit is not None:
            return torch.sigmoid(self._alpha_logit)
        return self._alpha
    
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Encode text using the backbone encoder.
        
        Uses [CLS] token representation as the embedding.
        
        Args:
            input_ids: Token IDs, shape [batch, seq_len]
            attention_mask: Attention mask, shape [batch, seq_len]
            
        Returns:
            Text embeddings, shape [batch, hidden_dim]
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        
        # Use [CLS] token embedding
        embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings
    
    def encode_mentions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode mentions with context.
        
        Args:
            input_ids: Tokenized [context_left; mention; context_right]
            attention_mask: Attention mask
            
        Returns:
            Mention embeddings m, shape [batch, hidden_dim]
        """
        return self.encode_text(input_ids, attention_mask)
    
    def encode_entities(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode entity names.
        
        Args:
            input_ids: Tokenized entity names
            attention_mask: Attention mask
            
        Returns:
            Entity embeddings e, shape [batch, hidden_dim]
        """
        return self.encode_text(input_ids, attention_mask)
    
    def compute_neural_scores(
        self,
        mention_embeddings: torch.Tensor,
        entity_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute neural similarity scores (Eq. 3).
        
        s_neural(m, e) = cos(m, e) = (m · e) / (||m|| ||e||)
        
        Args:
            mention_embeddings: [batch, hidden_dim]
            entity_embeddings: [batch, num_candidates, hidden_dim]
                              or [num_entities, hidden_dim] for all entities
            
        Returns:
            Cosine similarity scores, shape [batch, num_candidates]
        """
        # Normalize embeddings
        mention_norm = F.normalize(mention_embeddings, p=2, dim=-1)
        entity_norm = F.normalize(entity_embeddings, p=2, dim=-1)
        
        if entity_embeddings.dim() == 2:
            # Computing against all entities: [batch, hidden] @ [hidden, entities]
            scores = torch.matmul(mention_norm, entity_norm.T)
        else:
            # Computing against candidates: [batch, 1, hidden] @ [batch, hidden, cand]
            scores = torch.bmm(
                mention_norm.unsqueeze(1),
                entity_norm.transpose(1, 2),
            ).squeeze(1)
        
        return scores
    
    def compute_type_memberships(
        self,
        mention_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute context-inferred type memberships τ^I(m) (Eq. 8).
        
        Args:
            mention_embeddings: [batch, hidden_dim]
            
        Returns:
            Type memberships, shape [batch, num_types]
        """
        return self.type_inference(mention_embeddings)
    
    def compute_onto_scores(
        self,
        mention_type_probs: torch.Tensor,
        candidate_type_memberships: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ontological consistency scores (Eq. 9).
        
        s_onto(m, e) = Π_τ I_σ(τ^I(m), τ^I(e))
        
        Args:
            mention_type_probs: [batch, num_types]
            candidate_type_memberships: [batch, num_candidates, num_types]
            
        Returns:
            Consistency scores, shape [batch, num_candidates]
        """
        return self.fuzzy_logic(mention_type_probs, candidate_type_memberships)
    
    def normalize_neural_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Normalize neural scores to [0, 1] (Eq. 10).
        
        s̃_neural = (s_neural + 1) / 2
        
        Since cosine similarity ∈ [-1, 1], this maps to [0, 1].
        
        Args:
            scores: Cosine similarity scores
            
        Returns:
            Normalized scores in [0, 1]
        """
        return (scores + 1) / 2
    
    def fuse_scores(
        self,
        neural_scores: torch.Tensor,
        onto_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse neural and ontological scores (Eq. 11).
        
        s_final = α · s̃_neural + (1-α) · s_onto
        
        Args:
            neural_scores: Normalized neural scores [batch, num_candidates]
            onto_scores: Ontological scores [batch, num_candidates]
            
        Returns:
            Fused scores, shape [batch, num_candidates]
        """
        alpha = self.alpha
        return alpha * neural_scores + (1 - alpha) * onto_scores
    
    def forward(
        self,
        mention_input_ids: torch.Tensor,
        mention_attention_mask: torch.Tensor,
        candidate_ids: Optional[torch.Tensor] = None,
        candidate_input_ids: Optional[torch.Tensor] = None,
        candidate_attention_mask: Optional[torch.Tensor] = None,
        candidate_type_memberships: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for OntoEL.
        
        Args:
            mention_input_ids: Tokenized mentions [batch, seq_len]
            mention_attention_mask: Mention attention mask [batch, seq_len]
            candidate_ids: Entity IDs for candidates [batch, num_candidates]
                          Used to look up pre-computed embeddings
            candidate_input_ids: Tokenized candidate names [batch, num_cand, seq_len]
                                Used if embeddings not pre-computed
            candidate_attention_mask: Candidate attention mask
            candidate_type_memberships: Pre-computed type memberships
                                       [batch, num_candidates, num_types]
            return_components: Whether to return individual score components
            
        Returns:
            Dictionary containing:
            - "scores": Final fused scores [batch, num_candidates]
            - "neural_scores": Neural similarity scores (if return_components)
            - "onto_scores": Ontological scores (if return_components)
            - "type_probs": Inferred type probabilities (if return_components)
        """
        batch_size = mention_input_ids.shape[0]
        
        # 1. Encode mentions
        mention_embeddings = self.encode_mentions(
            mention_input_ids, mention_attention_mask
        )
        
        # 2. Get candidate embeddings
        if candidate_ids is not None and self.entity_embeddings is not None:
            # Use pre-computed embeddings
            candidate_embeddings = self.entity_embeddings[candidate_ids]
        elif candidate_input_ids is not None:
            # Encode candidates on-the-fly
            batch_size, num_candidates, seq_len = candidate_input_ids.shape
            
            # Flatten for encoding
            flat_input_ids = candidate_input_ids.view(-1, seq_len)
            flat_attention_mask = candidate_attention_mask.view(-1, seq_len)
            
            flat_embeddings = self.encode_entities(
                flat_input_ids, flat_attention_mask
            )
            
            candidate_embeddings = flat_embeddings.view(
                batch_size, num_candidates, -1
            )
        else:
            raise ValueError("Must provide either candidate_ids or candidate_input_ids")
        
        num_candidates = candidate_embeddings.shape[1]
        
        # 3. Compute neural scores
        neural_scores = self.compute_neural_scores(
            mention_embeddings, candidate_embeddings
        )
        normalized_neural = self.normalize_neural_scores(neural_scores)
        
        # 4. Compute type memberships
        mention_type_probs = self.compute_type_memberships(mention_embeddings)
        
        # 5. Get candidate type memberships
        if candidate_type_memberships is not None:
            cand_types = candidate_type_memberships
        elif candidate_ids is not None and self.entity_type_memberships is not None:
            cand_types = self.entity_type_memberships[candidate_ids]
        else:
            # Default to zeros (no type information)
            cand_types = torch.zeros(
                batch_size, num_candidates, self.num_types,
                device=mention_embeddings.device,
            )
        
        # 6. Compute ontological scores
        onto_scores = self.compute_onto_scores(mention_type_probs, cand_types)
        
        # 7. Fuse scores
        final_scores = self.fuse_scores(normalized_neural, onto_scores)
        
        outputs = {"scores": final_scores}
        
        if return_components:
            outputs.update({
                "neural_scores": neural_scores,
                "normalized_neural": normalized_neural,
                "onto_scores": onto_scores,
                "type_probs": mention_type_probs,
                "mention_embeddings": mention_embeddings,
            })
        
        return outputs
    
    def set_entity_embeddings(self, embeddings: torch.Tensor):
        """Set pre-computed entity embeddings."""
        self.entity_embeddings = embeddings
    
    def set_entity_type_memberships(self, memberships: torch.Tensor):
        """Set pre-computed entity type memberships."""
        self.entity_type_memberships = memberships
    
    def set_type_name_embeddings(self, embeddings: torch.Tensor):
        """Set pre-computed type name embeddings."""
        self.type_name_embeddings = embeddings
        self.type_inference.set_type_embeddings(embeddings)
    
    def set_disjointness_matrix(self, matrix: torch.Tensor):
        """Set disjointness matrix for constraint checking."""
        self.consistency_scorer.set_disjointness_matrix(matrix)
    
    @torch.no_grad()
    def precompute_entity_embeddings(
        self,
        entity_input_ids: torch.Tensor,
        entity_attention_mask: torch.Tensor,
        batch_size: int = 256,
    ) -> torch.Tensor:
        """
        Pre-compute embeddings for all entities.
        
        Args:
            entity_input_ids: [num_entities, seq_len]
            entity_attention_mask: [num_entities, seq_len]
            batch_size: Batch size for encoding
            
        Returns:
            Entity embeddings [num_entities, hidden_dim]
        """
        self.eval()
        num_entities = entity_input_ids.shape[0]
        embeddings = []
        
        for i in range(0, num_entities, batch_size):
            batch_ids = entity_input_ids[i:i+batch_size]
            batch_mask = entity_attention_mask[i:i+batch_size]
            
            batch_emb = self.encode_entities(batch_ids, batch_mask)
            embeddings.append(batch_emb.cpu())
        
        all_embeddings = torch.cat(embeddings, dim=0)
        self.set_entity_embeddings(all_embeddings)
        
        return all_embeddings
    
    @torch.no_grad()
    def precompute_type_embeddings(
        self,
        type_names: List[str],
        max_length: int = 32,
    ) -> torch.Tensor:
        """
        Pre-compute embeddings for semantic type names.
        
        Args:
            type_names: List of type names (e.g., ["Disease", "Finding", ...])
            max_length: Maximum token length
            
        Returns:
            Type name embeddings [num_types, hidden_dim]
        """
        self.eval()
        
        # Tokenize type names
        encoded = self.tokenizer(
            type_names,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        
        device = next(self.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        # Encode
        type_embeddings = self.encode_text(input_ids, attention_mask)
        
        self.set_type_name_embeddings(type_embeddings)
        
        return type_embeddings


class OntoELForTraining(OntoEL):
    """
    OntoEL model with training-specific methods.
    
    Adds:
    - Ranking loss computation
    - Type prediction loss computation
    - Hard negative handling
    """
    
    def __init__(
        self,
        margin: float = 0.2,
        type_loss_weight: float = 0.5,
        **kwargs,
    ):
        """
        Initialize training model.
        
        Args:
            margin: Margin (γ) for ranking loss
            type_loss_weight: Weight (λ) for type prediction loss
            **kwargs: Arguments passed to OntoEL
        """
        super().__init__(**kwargs)
        self.margin = margin
        self.type_loss_weight = type_loss_weight
    
    def compute_ranking_loss(
        self,
        scores: torch.Tensor,
        positive_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute margin-based ranking loss (Eq. 13).
        
        L_rank = Σ_{e^- ∈ N(m)} max(0, γ - s(m, e^+) + s(m, e^-))
        
        Args:
            scores: Final scores [batch, num_candidates]
            positive_mask: Binary mask indicating positive [batch, num_candidates]
            
        Returns:
            Ranking loss scalar
        """
        batch_size = scores.shape[0]
        
        # Get positive scores
        positive_scores = (scores * positive_mask).sum(dim=-1) / positive_mask.sum(dim=-1)
        
        # Get negative scores
        negative_mask = 1 - positive_mask
        negative_scores = scores * negative_mask
        
        # Compute margin loss for each negative
        margin_diff = self.margin - positive_scores.unsqueeze(-1) + negative_scores
        margin_loss = F.relu(margin_diff) * negative_mask
        
        # Average over negatives and batch
        loss = margin_loss.sum() / (negative_mask.sum() + 1e-8)
        
        return loss
    
    def compute_type_loss(
        self,
        type_probs: torch.Tensor,
        type_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute type prediction loss (Eq. 14).
        
        L_type = -Σ_τ [y_τ log τ^I(m) + (1-y_τ) log(1 - τ^I(m))]
        
        Args:
            type_probs: Predicted type probabilities [batch, num_types]
            type_labels: Ground truth type labels [batch, num_types]
            
        Returns:
            Type prediction loss scalar
        """
        return F.binary_cross_entropy(type_probs, type_labels)
    
    def training_step(
        self,
        mention_input_ids: torch.Tensor,
        mention_attention_mask: torch.Tensor,
        candidate_ids: torch.Tensor,
        positive_mask: torch.Tensor,
        type_labels: torch.Tensor,
        candidate_type_memberships: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform a training step.
        
        Args:
            mention_input_ids: [batch, seq_len]
            mention_attention_mask: [batch, seq_len]
            candidate_ids: [batch, num_candidates]
            positive_mask: [batch, num_candidates]
            type_labels: [batch, num_types]
            candidate_type_memberships: [batch, num_candidates, num_types]
            
        Returns:
            Dictionary with loss components
        """
        # Forward pass
        outputs = self.forward(
            mention_input_ids=mention_input_ids,
            mention_attention_mask=mention_attention_mask,
            candidate_ids=candidate_ids,
            candidate_type_memberships=candidate_type_memberships,
            return_components=True,
        )
        
        # Compute ranking loss
        ranking_loss = self.compute_ranking_loss(
            outputs["scores"], positive_mask
        )
        
        # Compute type loss
        type_loss = self.compute_type_loss(
            outputs["type_probs"], type_labels
        )
        
        # Combined loss (Eq. 15)
        total_loss = ranking_loss + self.type_loss_weight * type_loss
        
        return {
            "loss": total_loss,
            "ranking_loss": ranking_loss,
            "type_loss": type_loss,
            "scores": outputs["scores"],
            "type_probs": outputs["type_probs"],
        }
