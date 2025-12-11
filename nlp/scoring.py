"""
Module de calcul des scores de similarité sémantique.
Exigence EF2.3 : Similarité cosinus
"""
import numpy as np
import torch
from typing import List, Dict, Union
from embeddings import EmbeddingManager
from encode_repository import EmbeddingsLoader


class SimilarityScorer:
    """Calcule les scores de similarité entre textes utilisateur et compétences."""
    
    def __init__(self):
        """Initialise le scorer avec le gestionnaire d'embeddings."""
        self.embedding_manager = EmbeddingManager()
        self.embeddings_loader = EmbeddingsLoader()
    
    def calculate_competency_similarity(
        self,
        user_text: str,
        competency_id: str = None,
        competency_embedding: np.ndarray = None
    ) -> float:
        """
        Calcule la similarité entre un texte utilisateur et une compétence.
        Exigence EF2.3
        
        Args:
            user_text: Texte décrivant l'expérience utilisateur
            competency_id: ID de la compétence (si None, utiliser competency_embedding)
            competency_embedding: Embedding direct de la compétence
        
        Returns:
            Score de similarité entre 0 et 1
        
        Example:
            scorer = SimilarityScorer()
            score = scorer.calculate_competency_similarity(
                "I clean data with pandas",
                competency_id="C01"
            )
        """
        # Encoder le texte utilisateur
        user_embedding = self.embedding_manager.encode_texts(user_text)
        
        # Obtenir l'embedding de la compétence
        if competency_embedding is None:
            comp_ids, comp_embeddings = self.embeddings_loader.get_competency_embeddings()
            try:
                comp_idx = comp_ids.index(competency_id)
                competency_embedding = comp_embeddings[comp_idx]
            except ValueError:
                raise ValueError(f"Compétence {competency_id} non trouvée")
        
        # Calculer similarité cosinus
        similarity = self.embedding_manager.calculate_similarity(
            user_embedding,
            torch.tensor(competency_embedding)
        )
        
        # Normaliser entre 0 et 1
        normalized_score = self.normalize_score(similarity.item())
        
        return normalized_score
    
    def calculate_user_vs_all_competencies(
        self,
        user_texts: List[str]
    ) -> Dict[str, float]:
        """
        Calcule la similarité d'un utilisateur avec toutes les compétences.
        
        Args:
            user_texts: Liste des réponses utilisateur
        
        Returns:
            Dict {comp_id: max_similarity_score}
        
        Example:
            scores = scorer.calculate_user_vs_all_competencies([
                "I analyze data with Python",
                "I build ML models"
            ])
        """
        # Encoder toutes les réponses utilisateur
        user_embeddings = self.embedding_manager.encode_texts(user_texts)
        
        # Charger embeddings des compétences
        comp_ids, comp_embeddings = self.embeddings_loader.get_competency_embeddings()
        comp_embeddings_tensor = torch.tensor(comp_embeddings)
        
        # Calculer matrice de similarité
        similarity_matrix = self.embedding_manager.calculate_similarity_matrix(
            user_embeddings,
            comp_embeddings_tensor
        )
        
        # Pour chaque compétence, prendre le score max parmi toutes les réponses
        max_scores = similarity_matrix.max(dim=0).values
        
        # Créer dictionnaire {comp_id: score}
        scores_dict = {}
        for comp_id, score in zip(comp_ids, max_scores):
            scores_dict[comp_id] = self.normalize_score(score.item())
        
        return scores_dict
    
    def normalize_score(self, score: float) -> float:
        """
        Normalise un score de similarité entre 0 et 1.
        
        Args:
            score: Score brut (peut être entre -1 et 1 pour cosinus)
        
        Returns:
            Score normalisé entre 0 et 1
        """
        # Similarité cosinus est déjà entre -1 et 1
        # La normaliser entre 0 et 1: (score + 1) / 2
        # Mais en pratique SBERT donne déjà des scores positifs
        normalized = max(0.0, min(1.0, score))
        return normalized
    
    def get_top_competencies(
        self,
        user_texts: List[str],
        top_k: int = 10,
        threshold: float = 0.3
    ) -> List[tuple]:
        """
        Retourne les top K compétences les plus pertinentes.
        
        Args:
            user_texts: Réponses utilisateur
            top_k: Nombre de compétences à retourner
            threshold: Seuil minimum de similarité
        
        Returns:
            Liste de (comp_id, score, description) triée par score décroissant
        """
        scores = self.calculate_user_vs_all_competencies(user_texts)
        metadata = self.embeddings_loader.get_competency_metadata()
        
        # Filtrer par seuil et trier
        filtered_scores = [
            (comp_id, score, metadata[comp_id]['description'])
            for comp_id, score in scores.items()
            if score >= threshold
        ]
        
        # Trier par score décroissant
        sorted_scores = sorted(filtered_scores, key=lambda x: x[1], reverse=True)
        
        return sorted_scores[:top_k]


if __name__ == "__main__":
    print("\nTest du module scoring.py\n")
    
    scorer = SimilarityScorer()
    
    # Test avec réponses utilisateur
    user_responses = [
        "I have experience cleaning and preprocessing data with pandas",
        "I create visualizations with matplotlib"
    ]
    
    print("Réponses utilisateur:")
    for i, resp in enumerate(user_responses, 1):
        print(f"  {i}. {resp}")
    print()
    
    # Calculer scores
    top_comps = scorer.get_top_competencies(user_responses, top_k=5)
    
    print("Top 5 compétences identifiées:")
    for comp_id, score, description in top_comps:
        print(f"  {comp_id}: {description}")
        print(f"    Score: {score:.3f}")
    
    print("\nTest terminé")
