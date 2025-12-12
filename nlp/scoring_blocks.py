"""
Système de scoring par blocs avec pondération optimisée et stricte.
"""
import torch
from typing import List, Dict
from embeddings import get_embedder
from config import SIMILARITY_THRESHOLD, SBERT_MODEL_TYPE
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BlockScorer:
    """Calcule les scores par blocs de compétences."""
    
    def __init__(self, repository_path: str = "data/repository.json"):
        """
        Initialise le scorer.
        
        Args:
            repository_path: Chemin vers le référentiel
        """
        self.embedder = get_embedder(SBERT_MODEL_TYPE)
        
        with open(repository_path, 'r', encoding='utf-8') as f:
            self.repository = json.load(f)
        
        logger.info("BlockScorer initialisé")
    
    def calculate_block_scores(
        self, 
        user_texts: List[str],
        threshold: float = SIMILARITY_THRESHOLD
    ) -> Dict:
        """
        Calcule les scores par bloc avec seuil strict.
        
        Args:
            user_texts: Textes utilisateur
            threshold: Seuil de similarité (0.45 par défaut)
        
        Returns:
            Dict avec scores par bloc
        """
        # Encoder les textes utilisateur
        user_embeddings = self.embedder.encode(user_texts)
        
        # Agréger tous les blocs de tous les métiers
        all_blocks = {}
        
        for job_title, job_data in self.repository.items():
            for block_name, competencies in job_data.items():
                if block_name not in all_blocks:
                    all_blocks[block_name] = []
                
                for comp in competencies:
                    # Éviter les doublons
                    comp_id = f"{comp['id']}_{comp['description']}"
                    if not any(c.get('_id') == comp_id for c in all_blocks[block_name]):
                        comp['_id'] = comp_id
                        all_blocks[block_name].append(comp)
        
        # Calculer les scores pour chaque bloc
        block_scores = {}
        
        for block_name, competencies in all_blocks.items():
            # Encoder les compétences du bloc
            comp_texts = [c['description'] for c in competencies]
            comp_embeddings = self.embedder.encode(comp_texts)
            
            # Calculer similarités
            similarities = self.embedder.cosine_similarity(
                user_embeddings, 
                comp_embeddings
            )
            
            # Pour chaque compétence, prendre la meilleure similarité
            max_similarities = similarities.max(dim=0)[0]
            
            # Compétences détectées (au-dessus du seuil STRICT)
            matched_indices = (max_similarities >= threshold).nonzero(as_tuple=True)[0]
            matched_competencies = []
            
            total_weight = sum(c['weight'] for c in competencies)
            matched_weight = 0
            
            for idx in matched_indices:
                comp = competencies[idx]
                score = max_similarities[idx].item()
                
                # Appliquer un facteur de pénalité si le score est proche du seuil
                # Plus le score est élevé, plus le poids est conservé
                if score < 0.6:
                    # Pénalité progressive pour scores entre 0.45 et 0.6
                    weight_factor = (score - threshold) / (0.6 - threshold)
                elif score < 0.75:
                    weight_factor = 0.8 + 0.2 * (score - 0.6) / 0.15
                else:
                    weight_factor = 1.0
                
                effective_weight = comp['weight'] * weight_factor
                matched_weight += effective_weight
                
                matched_competencies.append({
                    'description': comp['description'],
                    'score': score,
                    'weight': comp['weight'],
                    'effective_weight': effective_weight
                })
            
            # Score pondéré du bloc
            if total_weight > 0:
                block_score = matched_weight / total_weight
            else:
                block_score = 0.0
            
            # Taux de couverture (nombre de compétences)
            coverage_rate = len(matched_indices) / len(competencies) if competencies else 0
            
            block_scores[block_name] = {
                'score': block_score,
                'coverage_rate': coverage_rate,
                'num_matched': len(matched_indices),
                'num_competencies': len(competencies),
                'matched_weight': matched_weight,
                'total_weight': total_weight,
                'competencies': sorted(
                    matched_competencies, 
                    key=lambda x: x['score'], 
                    reverse=True
                )
            }
        
        return block_scores
    
    def get_detailed_analysis(self, user_texts: List[str]) -> Dict:
        """Analyse détaillée complète."""
        block_scores = self.calculate_block_scores(user_texts)
        
        # Score global (moyenne pondérée de tous les blocs)
        total_weight = sum(bs['total_weight'] for bs in block_scores.values())
        total_matched_weight = sum(bs['matched_weight'] for bs in block_scores.values())
        
        if total_weight > 0:
            coverage_score = total_matched_weight / total_weight
        else:
            coverage_score = 0.0
        
        # Interprétation plus réaliste
        if coverage_score >= 0.8:
            interpretation = "Expert: Excellent niveau de compétences"
        elif coverage_score >= 0.65:
            interpretation = "Avancé: Bonnes compétences, quelques axes d'amélioration"
        elif coverage_score >= 0.45:
            interpretation = "Moyen: Plusieurs compétences à développer"
        elif coverage_score >= 0.25:
            interpretation = "Débutant: Nombreuses compétences à acquérir"
        else:
            interpretation = "Novice: Formation de base nécessaire"
        
        return {
            'coverage_score': coverage_score,
            'interpretation': interpretation,
            'block_scores': block_scores,
            'total_weight': total_weight,
            'matched_weight': total_matched_weight
        }


if __name__ == "__main__":
    print("\nTest du scoring par blocs optimisé\n")
    
    scorer = BlockScorer()
    
    # Test avec profil data scientist
    user_texts = [
        "Python programming for data science level 4 out of 5",
        "data analysis experience with pandas numpy matplotlib seaborn",
        "machine learning level 4 out of 5 with Random Forest XGBoost Neural Networks",
        "database management proficient in MySQL PostgreSQL",
        "statistics level 3 out of 5"
    ]
    
    analysis = scorer.get_detailed_analysis(user_texts)
    
    print(f"Score global: {analysis['coverage_score']:.1%}")
    print(f"Interprétation: {analysis['interpretation']}\n")
    
    print("Top 3 blocs:")
    top_blocks = sorted(
        analysis['block_scores'].items(),
        key=lambda x: x[1]['score'],
        reverse=True
    )[:3]
    
    for block_name, data in top_blocks:
        print(f"  {block_name}: {data['score']:.1%} ({data['num_matched']}/{data['num_competencies']})")
