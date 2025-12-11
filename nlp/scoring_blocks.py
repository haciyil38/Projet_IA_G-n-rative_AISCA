"""
Module de scoring par blocs de compétences.
Exigence EF3.1 : Formule score pondéré Σ(Wi*Si) / ΣWi
"""
import numpy as np
from typing import Dict, List
from nlp.scoring import SimilarityScorer
from encode_repository import EmbeddingsLoader
from config import BLOCK_WEIGHTS


class BlockScorer:
    """Calcule les scores par blocs de compétences."""
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialise le scorer de blocs.
        
        Args:
            weights: Poids par bloc {block_name: weight}. Si None, utilise config.
        """
        self.similarity_scorer = SimilarityScorer()
        self.embeddings_loader = EmbeddingsLoader()
        self.weights = weights or BLOCK_WEIGHTS
    
    def calculate_block_scores(
        self,
        user_texts: List[str]
    ) -> Dict[str, Dict]:
        """
        Calcule le score pour chaque bloc de compétences.
        
        Args:
            user_texts: Liste des réponses utilisateur
        
        Returns:
            Dict {block_name: {score, num_competencies, matched_competencies}}
        
        Example:
            scorer = BlockScorer()
            block_scores = scorer.calculate_block_scores([
                "I analyze data with Python",
                "I build ML models"
            ])
        """
        # Calculer scores pour toutes les compétences
        comp_scores = self.similarity_scorer.calculate_user_vs_all_competencies(user_texts)
        
        # Grouper par bloc
        comp_metadata = self.embeddings_loader.get_competency_metadata()
        
        blocks = {}
        for comp_id, score in comp_scores.items():
            block_name = comp_metadata[comp_id]['block_name']
            
            if block_name not in blocks:
                blocks[block_name] = {
                    'scores': [],
                    'competencies': []
                }
            
            blocks[block_name]['scores'].append(score)
            blocks[block_name]['competencies'].append({
                'comp_id': comp_id,
                'description': comp_metadata[comp_id]['description'],
                'score': score
            })
        
        # Calculer score moyen par bloc
        block_scores = {}
        for block_name, data in blocks.items():
            scores = data['scores']
            
            # Score du bloc = moyenne des scores des compétences
            block_score = np.mean(scores)
            
            # Nombre de compétences bien couvertes (score > 0.5)
            num_matched = sum(1 for s in scores if s > 0.5)
            
            block_scores[block_name] = {
                'score': float(block_score),
                'num_competencies': len(scores),
                'num_matched': num_matched,
                'coverage_rate': num_matched / len(scores) if scores else 0,
                'competencies': sorted(
                    data['competencies'],
                    key=lambda x: x['score'],
                    reverse=True
                )
            }
        
        return block_scores
    
    def aggregate_coverage_score(
        self,
        block_scores: Dict[str, Dict]
    ) -> float:
        """
        Calcule le score de couverture global pondéré.
        Exigence EF3.1 : Formule Σ(Wi*Si) / ΣWi
        
        Args:
            block_scores: Dict retourné par calculate_block_scores()
        
        Returns:
            Score global de couverture entre 0 et 1
        
        Example:
            coverage = scorer.aggregate_coverage_score(block_scores)
        """
        weighted_sum = 0.0
        total_weight = 0.0
        
        for block_name, data in block_scores.items():
            weight = self.weights.get(block_name, 1.0)
            score = data['score']
            
            weighted_sum += weight * score
            total_weight += weight
        
        # Formule: Σ(Wi*Si) / ΣWi
        coverage_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        return coverage_score
    
    def get_weak_blocks(
        self,
        block_scores: Dict[str, Dict],
        threshold: float = 0.4
    ) -> List[tuple]:
        """
        Identifie les blocs faiblement couverts.
        
        Args:
            block_scores: Scores par bloc
            threshold: Seuil en dessous duquel un bloc est considéré faible
        
        Returns:
            Liste de (block_name, score) pour blocs faibles
        """
        weak_blocks = [
            (block_name, data['score'])
            for block_name, data in block_scores.items()
            if data['score'] < threshold
        ]
        
        return sorted(weak_blocks, key=lambda x: x[1])
    
    def get_strong_blocks(
        self,
        block_scores: Dict[str, Dict],
        threshold: float = 0.7
    ) -> List[tuple]:
        """
        Identifie les blocs fortement couverts.
        
        Args:
            block_scores: Scores par bloc
            threshold: Seuil au-dessus duquel un bloc est considéré fort
        
        Returns:
            Liste de (block_name, score) pour blocs forts
        """
        strong_blocks = [
            (block_name, data['score'])
            for block_name, data in block_scores.items()
            if data['score'] >= threshold
        ]
        
        return sorted(strong_blocks, key=lambda x: x[1], reverse=True)
    
    def get_detailed_analysis(
        self,
        user_texts: List[str]
    ) -> Dict:
        """
        Analyse complète du profil utilisateur.
        
        Args:
            user_texts: Réponses utilisateur
        
        Returns:
            Dict avec analyse détaillée
        """
        block_scores = self.calculate_block_scores(user_texts)
        coverage_score = self.aggregate_coverage_score(block_scores)
        weak_blocks = self.get_weak_blocks(block_scores)
        strong_blocks = self.get_strong_blocks(block_scores)
        
        return {
            'coverage_score': coverage_score,
            'block_scores': block_scores,
            'weak_blocks': weak_blocks,
            'strong_blocks': strong_blocks,
            'interpretation': self._interpret_coverage(coverage_score)
        }
    
    def _interpret_coverage(self, score: float) -> str:
        """Interprète le score de couverture."""
        if score >= 0.8:
            return "Excellent: Profil très complet"
        elif score >= 0.6:
            return "Bon: Profil solide avec quelques axes d'amélioration"
        elif score >= 0.4:
            return "Moyen: Plusieurs compétences à développer"
        else:
            return "Débutant: Nombreuses compétences à acquérir"


if __name__ == "__main__":
    print("\nTest du module scoring_blocks.py\n")
    
    scorer = BlockScorer()
    
    user_responses = [
        "I analyze data with Python and pandas",
        "I create visualizations with matplotlib and seaborn",
        "I know basic statistics"
    ]
    
    print("Analyse du profil utilisateur\n")
    analysis = scorer.get_detailed_analysis(user_responses)
    
    print(f"Score de couverture global: {analysis['coverage_score']:.2%}")
    print(f"Interprétation: {analysis['interpretation']}\n")
    
    print("Scores par bloc:")
    for block_name, data in analysis['block_scores'].items():
        print(f"  {block_name}: {data['score']:.2%}")
        print(f"    Compétences couvertes: {data['num_matched']}/{data['num_competencies']}")
    
    print(f"\nBlocs faibles (à améliorer): {len(analysis['weak_blocks'])}")
    for block, score in analysis['weak_blocks']:
        print(f"  - {block}: {score:.2%}")
    
    print(f"\nBlocs forts: {len(analysis['strong_blocks'])}")
    for block, score in analysis['strong_blocks']:
        print(f"  - {block}: {score:.2%}")
