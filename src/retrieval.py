"""
Candidate Retrieval Module for OntoEL

Implements efficient candidate retrieval using FAISS for approximate nearest neighbor search.
This module handles Stage 1 (Candidate Generation) of the OntoEL pipeline.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm
import pickle
from pathlib import Path

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not installed. Using brute-force retrieval.")


class CandidateRetriever:
    """
    Candidate Retrieval using FAISS.
    
    Supports:
    - Flat index (exact search)
    - IVF index (approximate search for large datasets)
    - Pre-computed entity embeddings
    """
    
    def __init__(
        self,
        encoder: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        entity_names: List[str],
        entity_ids: List[str],
        hidden_dim: int = 768,
        index_type: str = "flat",
        nlist: int = 100,
        nprobe: int = 10,
        device: str = "cuda",
    ):
        """
        Initialize retriever.
        
        Args:
            encoder: Pre-trained encoder model
            tokenizer: Tokenizer for encoding
            entity_names: List of entity names
            entity_ids: List of entity IDs (parallel to names)
            hidden_dim: Embedding dimension
            index_type: FAISS index type ("flat", "ivf", "hnsw")
            nlist: Number of clusters for IVF
            nprobe: Number of clusters to search for IVF
            device: Device for encoding
        """
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.entity_names = entity_names
        self.entity_ids = entity_ids
        self.hidden_dim = hidden_dim
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.device = device
        
        self.entity_embeddings: Optional[np.ndarray] = None
        self.index: Optional[Any] = None
        
        # Map from entity ID to index position
        self.id_to_idx = {eid: i for i, eid in enumerate(entity_ids)}
    
    @torch.no_grad()
    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 256,
        max_length: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of texts into embeddings.
        
        Args:
            texts: List of texts to encode
            batch_size: Encoding batch size
            max_length: Maximum sequence length
            show_progress: Whether to show progress bar
            
        Returns:
            Embeddings array [num_texts, hidden_dim]
        """
        self.encoder.eval()
        embeddings = []
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding")
        
        for i in iterator:
            batch_texts = texts[i:i+batch_size]
            
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            # Use [CLS] token
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # Normalize for cosine similarity
            batch_embeddings = batch_embeddings / np.linalg.norm(
                batch_embeddings, axis=1, keepdims=True
            )
            
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings).astype(np.float32)
    
    def build_index(
        self,
        embeddings: Optional[np.ndarray] = None,
        batch_size: int = 256,
    ):
        """
        Build FAISS index from entity embeddings.
        
        Args:
            embeddings: Pre-computed embeddings, or None to compute
            batch_size: Batch size for encoding
        """
        if embeddings is None:
            print("Computing entity embeddings...")
            embeddings = self.encode_texts(
                self.entity_names,
                batch_size=batch_size,
            )
        
        self.entity_embeddings = embeddings
        num_entities = len(embeddings)
        
        if not FAISS_AVAILABLE:
            print("FAISS not available. Using brute-force search.")
            return
        
        print(f"Building FAISS index ({self.index_type})...")
        
        if self.index_type == "flat":
            # Exact search with inner product (cosine for normalized vectors)
            self.index = faiss.IndexFlatIP(self.hidden_dim)
        
        elif self.index_type == "ivf":
            # Approximate search with inverted file index
            quantizer = faiss.IndexFlatIP(self.hidden_dim)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.hidden_dim, self.nlist, faiss.METRIC_INNER_PRODUCT
            )
            # Train the index
            self.index.train(embeddings)
            self.index.nprobe = self.nprobe
        
        elif self.index_type == "hnsw":
            # HNSW index for very fast approximate search
            self.index = faiss.IndexHNSWFlat(self.hidden_dim, 32)
        
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Add vectors to index
        self.index.add(embeddings)
        print(f"Index built with {self.index.ntotal} vectors")
    
    def retrieve(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 64,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve top-k candidates for query embeddings.
        
        Args:
            query_embeddings: Query embeddings [batch_size, hidden_dim]
            top_k: Number of candidates to retrieve
            
        Returns:
            Tuple of (scores, indices) each [batch_size, top_k]
        """
        # Normalize queries for cosine similarity
        query_embeddings = query_embeddings / np.linalg.norm(
            query_embeddings, axis=1, keepdims=True
        )
        query_embeddings = query_embeddings.astype(np.float32)
        
        if FAISS_AVAILABLE and self.index is not None:
            scores, indices = self.index.search(query_embeddings, top_k)
        else:
            # Brute-force fallback
            similarities = np.matmul(query_embeddings, self.entity_embeddings.T)
            indices = np.argsort(-similarities, axis=1)[:, :top_k]
            scores = np.take_along_axis(similarities, indices, axis=1)
        
        return scores, indices
    
    def retrieve_for_mentions(
        self,
        mention_texts: List[str],
        context_lefts: List[str],
        context_rights: List[str],
        top_k: int = 64,
        batch_size: int = 256,
    ) -> List[List[Tuple[str, float]]]:
        """
        Retrieve candidates for a list of mentions.
        
        Args:
            mention_texts: List of mention texts
            context_lefts: Left context for each mention
            context_rights: Right context for each mention
            top_k: Number of candidates per mention
            batch_size: Encoding batch size
            
        Returns:
            List of candidate lists, each containing (entity_id, score) tuples
        """
        # Prepare mention texts with context
        full_texts = [
            f"{left} {mention} {right}"
            for left, mention, right in zip(context_lefts, mention_texts, context_rights)
        ]
        
        # Encode mentions
        query_embeddings = self.encode_texts(
            full_texts,
            batch_size=batch_size,
            show_progress=True,
        )
        
        # Retrieve candidates
        scores, indices = self.retrieve(query_embeddings, top_k)
        
        # Convert to entity IDs
        results = []
        for i in range(len(mention_texts)):
            candidates = [
                (self.entity_ids[idx], float(score))
                for idx, score in zip(indices[i], scores[i])
            ]
            results.append(candidates)
        
        return results
    
    def save(self, path: str):
        """Save retriever state to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        np.save(path / "entity_embeddings.npy", self.entity_embeddings)
        
        # Save FAISS index
        if FAISS_AVAILABLE and self.index is not None:
            faiss.write_index(self.index, str(path / "faiss.index"))
        
        # Save metadata
        metadata = {
            "entity_ids": self.entity_ids,
            "entity_names": self.entity_names,
            "hidden_dim": self.hidden_dim,
            "index_type": self.index_type,
        }
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)
        
        print(f"Retriever saved to {path}")
    
    def load(self, path: str):
        """Load retriever state from disk."""
        path = Path(path)
        
        # Load embeddings
        self.entity_embeddings = np.load(path / "entity_embeddings.npy")
        
        # Load FAISS index
        if FAISS_AVAILABLE and (path / "faiss.index").exists():
            self.index = faiss.read_index(str(path / "faiss.index"))
        
        # Load metadata
        with open(path / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        
        self.entity_ids = metadata["entity_ids"]
        self.entity_names = metadata["entity_names"]
        self.id_to_idx = {eid: i for i, eid in enumerate(self.entity_ids)}
        
        print(f"Retriever loaded from {path}")


class HardNegativeMiner:
    """
    Mine hard negatives from retrieval results.
    
    Hard negatives are high-similarity candidates that are NOT the gold entity.
    These are crucial for training the model to discriminate fine-grained differences.
    """
    
    def __init__(
        self,
        retriever: CandidateRetriever,
        top_k: int = 64,
        num_hard_negatives: int = 15,
    ):
        """
        Initialize hard negative miner.
        
        Args:
            retriever: Candidate retriever
            top_k: Number of candidates to retrieve
            num_hard_negatives: Number of hard negatives to keep per mention
        """
        self.retriever = retriever
        self.top_k = top_k
        self.num_hard_negatives = num_hard_negatives
    
    def mine_hard_negatives(
        self,
        mention_texts: List[str],
        context_lefts: List[str],
        context_rights: List[str],
        gold_entity_ids: List[str],
        batch_size: int = 256,
    ) -> List[List[str]]:
        """
        Mine hard negatives for a batch of mentions.
        
        Args:
            mention_texts: List of mention texts
            context_lefts: Left context
            context_rights: Right context
            gold_entity_ids: Gold entity ID for each mention
            batch_size: Batch size for retrieval
            
        Returns:
            List of hard negative entity ID lists
        """
        # Retrieve candidates
        candidates = self.retriever.retrieve_for_mentions(
            mention_texts=mention_texts,
            context_lefts=context_lefts,
            context_rights=context_rights,
            top_k=self.top_k,
            batch_size=batch_size,
        )
        
        # Filter out gold entities to get hard negatives
        hard_negatives = []
        for gold_id, cand_list in zip(gold_entity_ids, candidates):
            negatives = [
                entity_id for entity_id, _ in cand_list
                if entity_id != gold_id
            ][:self.num_hard_negatives]
            hard_negatives.append(negatives)
        
        return hard_negatives


def precompute_candidates_for_dataset(
    retriever: CandidateRetriever,
    examples: List[Any],
    output_file: str,
    top_k: int = 64,
    batch_size: int = 256,
):
    """
    Pre-compute and cache candidates for all examples in a dataset.
    
    Args:
        retriever: Candidate retriever
        examples: List of MentionExample objects
        output_file: Output JSONL file path
        top_k: Number of candidates per mention
        batch_size: Batch size for retrieval
    """
    import json
    
    # Extract mention info
    mention_texts = [ex.mention_text for ex in examples]
    context_lefts = [ex.context_left for ex in examples]
    context_rights = [ex.context_right for ex in examples]
    
    # Retrieve candidates
    all_candidates = retriever.retrieve_for_mentions(
        mention_texts=mention_texts,
        context_lefts=context_lefts,
        context_rights=context_rights,
        top_k=top_k,
        batch_size=batch_size,
    )
    
    # Save results
    with open(output_file, "w") as f:
        for example, candidates in zip(examples, all_candidates):
            output = {
                "mention_id": example.mention_id,
                "mention_text": example.mention_text,
                "context_left": example.context_left,
                "context_right": example.context_right,
                "gold_entity_id": example.gold_entity_id,
                "gold_entity_name": example.gold_entity_name,
                "gold_type_ids": example.gold_type_ids,
                "candidates": [eid for eid, _ in candidates],
                "candidate_scores": [score for _, score in candidates],
            }
            f.write(json.dumps(output) + "\n")
    
    print(f"Saved candidates to {output_file}")
