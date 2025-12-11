"""
Module de récupération (Retriever) pour l'architecture RAG.
Exigence EF3.2 : Top 3 recommandations
"""
import numpy as np
from typing import List, Dict, Tuple
import torch
from embeddings import EmbeddingManager
from encode_repository import EmbeddingsLoader
from nlp.scoring_blocks import BlockScorer
from config import SIMILARITY_THRESHOLD, TOP_N_RECOMMENDATIONS


class CompetencyRetriever:
    """Récupère les compétences et blocs pertinents pour l'utilisateur."""
    
    def __init__(self, threshold: float = SIMILARITY_THRESHOLD):
        """
        Initialise le retriever.
        
        Args:
            threshold: Seuil minimum de similarité (par défaut 0.5)
        """
        self.threshold = threshold
        self.embedding_manager = EmbeddingManager()
        self.embeddings_loader = EmbeddingsLoader()
        self.block_scorer = BlockScorer()
    
    def retrieve_matching_competencies(
        self,
        user_texts: List[str],
        threshold: float = None
    ) -> List[Dict]:
        """
        Récupère les compétences qui correspondent au profil utilisateur.
        
        Args:
            user_texts: Réponses de l'utilisateur
            threshold: Seuil de similarité (si None, utilise self.threshold)
        
        Returns:
            Liste de dicts avec compétences triées par score
        
        Example:
            retriever = CompetencyRetriever()
            matches = retriever.retrieve_matching_competencies([
                "I analyze data with Python"
            ])
        """
        threshold = threshold or self.threshold
        
        # Encoder les textes utilisateur
        user_embeddings = self.embedding_manager.encode_texts(user_texts)
        
        # Charger embeddings des compétences
        comp_ids, comp_embeddings = self.embeddings_loader.get_competency_embeddings()
        comp_metadata = self.embeddings_loader.get_competency_metadata()
        comp_embeddings_tensor = torch.tensor(comp_embeddings)
        
        # Calculer similarités
        similarity_matrix = self.embedding_manager.calculate_similarity_matrix(
            user_embeddings,
            comp_embeddings_tensor
        )
        
        # Pour chaque compétence, prendre le meilleur score
        max_scores = similarity_matrix.max(dim=0).values
        
        # Filtrer et trier
        matching_competencies = []
        for comp_id, score in zip(comp_ids, max_scores):
            score_value = score.item()
            if score_value >= threshold:
                matching_competencies.append({
                    'comp_id': comp_id,
                    'score': score_value,
                    'description': comp_metadata[comp_id]['description'],
                    'block_name': comp_metadata[comp_id]['block_name'],
                    'block_id': comp_metadata[comp_id]['block_id'],
                    'keywords': comp_metadata[comp_id]['keywords']
                })
        
        # Trier par score décroissant
        matching_competencies.sort(key=lambda x: x['score'], reverse=True)
        
        return matching_competencies
    
    def get_top_n_blocks(
        self,
        user_texts: List[str],
        n: int = TOP_N_RECOMMENDATIONS
    ) -> List[Dict]:
        """
        Retourne les top N blocs de compétences recommandés.
        Exigence EF3.2 : Top 3 recommandations
        
        Args:
            user_texts: Réponses utilisateur
            n: Nombre de blocs à retourner (par défaut 3)
        
        Returns:
            Liste des top N blocs avec scores et détails
        
        Example:
            retriever = CompetencyRetriever()
            top_blocks = retriever.get_top_n_blocks(user_texts, n=3)
        """
        # Calculer scores par bloc
        block_scores = self.block_scorer.calculate_block_scores(user_texts)
        
        # Convertir en liste et trier
        blocks_list = []
        for block_name, data in block_scores.items():
            blocks_list.append({
                'block_name': block_name,
                'score': data['score'],
                'num_competencies': data['num_competencies'],
                'num_matched': data['num_matched'],
                'coverage_rate': data['coverage_rate'],
                'top_competencies': data['competencies'][:5]  # Top 5 compétences du bloc
            })
        
        # Trier par score décroissant
        blocks_list.sort(key=lambda x: x['score'], reverse=True)
        
        return blocks_list[:n]
    
    def retrieve_relevant_context(
        self,
        user_texts: List[str]
    ) -> Dict:
        """
        Récupère le contexte complet pertinent pour la génération RAG.
        
        Args:
            user_texts: Réponses utilisateur
        
        Returns:
            Dict contenant tous les éléments de contexte
        """
        # Récupérer compétences correspondantes
        matching_competencies = self.retrieve_matching_competencies(user_texts)
        
        # Récupérer top blocs
        top_blocks = self.get_top_n_blocks(user_texts, n=3)
        
        # Calculer analyse détaillée
        analysis = self.block_scorer.get_detailed_analysis(user_texts)
        
        context = {
            'user_texts': user_texts,
            'matching_competencies': matching_competencies,
            'top_blocks': top_blocks,
            'coverage_score': analysis['coverage_score'],
            'weak_blocks': analysis['weak_blocks'],
            'strong_blocks': analysis['strong_blocks'],
            'interpretation': analysis['interpretation'],
            'num_competencies_matched': len(matching_competencies)
        }
        
        return context
    
    def get_competencies_by_block(
        self,
        user_texts: List[str],
        block_name: str
    ) -> List[Dict]:
        """
        Récupère les compétences d'un bloc spécifique.
        
        Args:
            user_texts: Réponses utilisateur
            block_name: Nom du bloc
        
        Returns:
            Liste des compétences du bloc avec scores
        """
        matching_competencies = self.retrieve_matching_competencies(user_texts)
        
        block_competencies = [
            comp for comp in matching_competencies
            if comp['block_name'] == block_name
        ]
        
        return block_competencies
    
    def get_missing_competencies(
        self,
        user_texts: List[str],
        threshold: float = 0.3
    ) -> List[Dict]:
        """
        Identifie les compétences manquantes ou faiblement couvertes.
        
        Args:
            user_texts: Réponses utilisateur
            threshold: Seuil en dessous duquel une compétence est considérée manquante
        
        Returns:
            Liste des compétences à développer
        """
        # Récupérer toutes les compétences avec scores faibles
        user_embeddings = self.embedding_manager.encode_texts(user_texts)
        comp_ids, comp_embeddings = self.embeddings_loader.get_competency_embeddings()
        comp_metadata = self.embeddings_loader.get_competency_metadata()
        comp_embeddings_tensor = torch.tensor(comp_embeddings)
        
        similarity_matrix = self.embedding_manager.calculate_similarity_matrix(
            user_embeddings,
            comp_embeddings_tensor
        )
        max_scores = similarity_matrix.max(dim=0).values
        
        # Filtrer compétences sous le seuil
        missing = []
        for comp_id, score in zip(comp_ids, max_scores):
            score_value = score.item()
            if score_value < threshold:
                missing.append({
                    'comp_id': comp_id,
                    'score': score_value,
                    'gap': threshold - score_value,
                    'description': comp_metadata[comp_id]['description'],
                    'block_name': comp_metadata[comp_id]['block_name']
                })
        
        # Trier par gap (priorité aux plus grandes lacunes)
        missing.sort(key=lambda x: x['gap'], reverse=True)
        
        return missing


if __name__ == "__main__":
    print("\nTest du module retriever.py\n")
    
    retriever = CompetencyRetriever(threshold=0.3)
    
    user_responses = [
        "I analyze data with Python and pandas",
        "I create charts with matplotlib"
    ]
    
    print("Réponses utilisateur:")
    for resp in user_responses:
        print(f"  - {resp}")
    print()
    
    # Test 1: Compétences correspondantes
    print("1. Compétences correspondantes:")
    matches = retriever.retrieve_matching_competencies(user_responses)
    for comp in matches[:5]:
        print(f"  - {comp['description']} (score: {comp['score']:.3f})")
    print()
    
    # Test 2: Top 3 blocs
    print("2. Top 3 blocs recommandés:")
    top_blocks = retriever.get_top_n_blocks(user_responses, n=3)
    for i, block in enumerate(top_blocks, 1):
        print(f"  {i}. {block['block_name']}")
        print(f"     Score: {block['score']:.2%}")
        print(f"     Couverture: {block['num_matched']}/{block['num_competencies']}")
    print()
    
    # Test 3: Compétences manquantes
    print("3. Compétences à développer (top 5):")
    missing = retriever.get_missing_competencies(user_responses)
    for comp in missing[:5]:
        print(f"  - {comp['description']} ({comp['block_name']})")
        print(f"    Gap: {comp['gap']:.3f}")
    
    print("\nTest terminé")
