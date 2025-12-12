"""
Module de calcul de similarité sémantique.
"""
import torch
from typing import List, Tuple
from embeddings import get_embedder, SBERTEmbeddings
from config import SIMILARITY_THRESHOLD, SBERT_MODEL_TYPE
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticScorer:
    """Calcule les scores de similarité sémantique."""
    
    def __init__(self):
        """Initialise le scorer."""
        self.embedder = get_embedder(SBERT_MODEL_TYPE)
        logger.info("SemanticScorer initialisé")
    
    def score_texts(
        self, 
        queries: List[str], 
        references: List[str]
    ) -> torch.Tensor:
        """
        Calcule les scores de similarité.
        
        Args:
            queries: Textes requêtes
            references: Textes de référence
        
        Returns:
            Matrice de similarité
        """
        query_embeddings = self.embedder.encode(queries)
        ref_embeddings = self.embedder.encode(references)
        
        similarities = self.embedder.cosine_similarity(query_embeddings, ref_embeddings)
        return similarities
    
    def find_best_matches(
        self,
        queries: List[str],
        references: List[str],
        threshold: float = SIMILARITY_THRESHOLD
    ) -> List[List[Tuple[int, float]]]:
        """
        Trouve les meilleures correspondances.
        
        Args:
            queries: Textes requêtes
            references: Textes de référence
            threshold: Seuil minimum
        
        Returns:
            Liste de (index, score) pour chaque requête
        """
        similarities = self.score_texts(queries, references)
        
        matches = []
        for i in range(len(queries)):
            query_matches = []
            for j in range(len(references)):
                score = similarities[i][j].item()
                if score >= threshold:
                    query_matches.append((j, score))
            
            query_matches.sort(key=lambda x: x[1], reverse=True)
            matches.append(query_matches)
        
        return matches


if __name__ == "__main__":
    print("\nTest du semantic scorer\n")
    
    scorer = SemanticScorer()
    
    queries = ["Python programming", "Machine learning"]
    references = [
        "Python for data science",
        "Deep learning with neural networks",
        "Java programming"
    ]
    
    matches = scorer.find_best_matches(queries, references)
    
    for i, query in enumerate(queries):
        print(f"Query: {query}")
        for ref_idx, score in matches[i][:2]:
            print(f"  -> {references[ref_idx]}: {score:.3f}")
        print()
