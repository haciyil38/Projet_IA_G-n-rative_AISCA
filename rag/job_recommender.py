"""
Module de recommandation de métiers.
Exigence EF3.2 : Top 3 métiers recommandés
"""
import numpy as np
import torch
from typing import List, Dict, Tuple
from embeddings import EmbeddingManager
from encode_repository import EmbeddingsLoader
from rag.context_builder import ContextBuilder


class JobRecommender:
    """Recommande les métiers les plus adaptés au profil utilisateur."""
    
    def __init__(self):
        """Initialise le recommandeur de métiers."""
        self.embedding_manager = EmbeddingManager()
        self.embeddings_loader = EmbeddingsLoader()
        self.context_builder = ContextBuilder()
    
    def match_user_to_jobs(
        self,
        user_texts: List[str]
    ) -> Dict[str, Dict]:
        """
        Calcule le score de matching pour chaque métier.
        
        Args:
            user_texts: Réponses utilisateur
        
        Returns:
            Dict {job_id: {score, title, details}}
        
        Example:
            recommender = JobRecommender()
            job_matches = recommender.match_user_to_jobs(user_texts)
        """
        # Récupérer le contexte de matching
        context = self.context_builder.build_job_matching_context(user_texts)
        
        # Charger métadonnées des métiers
        job_ids, job_embeddings = self.embeddings_loader.get_job_embeddings()
        job_metadata = self.embeddings_loader.get_job_metadata()
        comp_metadata = self.embeddings_loader.get_competency_metadata()
        
        # Encoder profil utilisateur
        user_embeddings = self.embedding_manager.encode_texts(user_texts)
        
        # Calculer similarité sémantique globale
        job_embeddings_tensor = torch.tensor(job_embeddings)
        semantic_scores = self.embedding_manager.calculate_similarity_matrix(
            user_embeddings,
            job_embeddings_tensor
        ).max(dim=0).values
        
        # Calculer score basé sur compétences requises
        matched_comp_ids = set(context['matched_competencies_ids'])
        
        job_scores = {}
        for job_id, semantic_score in zip(job_ids, semantic_scores):
            job_info = job_metadata[job_id]
            required_comps = set(job_info['required_competencies'])
            
            # Compétences requises possédées
            matched_required = matched_comp_ids.intersection(required_comps)
            
            # Score de couverture des compétences requises
            competency_coverage = len(matched_required) / len(required_comps) if required_comps else 0
            
            # Score final = moyenne pondérée
            # 60% couverture compétences + 40% similarité sémantique
            final_score = 0.6 * competency_coverage + 0.4 * semantic_score.item()
            
            # Détails des compétences
            matched_comps_details = []
            missing_comps_details = []
            
            for comp_id in required_comps:
                comp_desc = comp_metadata[comp_id]['description']
                if comp_id in matched_comp_ids:
                    matched_comps_details.append(comp_desc)
                else:
                    missing_comps_details.append(comp_desc)
            
            job_scores[job_id] = {
                'score': final_score,
                'title': job_info['title'],
                'semantic_score': semantic_score.item(),
                'competency_coverage': competency_coverage,
                'num_required_competencies': len(required_comps),
                'num_matched_competencies': len(matched_required),
                'matched_competencies': matched_comps_details,
                'missing_competencies': missing_comps_details,
                'readiness': self._calculate_readiness(final_score)
            }
        
        return job_scores
    
    def get_top_recommendations(
        self,
        user_texts: List[str],
        n: int = 3
    ) -> List[Dict]:
        """
        Retourne les top N métiers recommandés.
        Exigence EF3.2
        
        Args:
            user_texts: Réponses utilisateur
            n: Nombre de recommandations (par défaut 3)
        
        Returns:
            Liste des top N métiers triés par score
        
        Example:
            recommender = JobRecommender()
            top_jobs = recommender.get_top_recommendations(user_texts, n=3)
        """
        job_scores = self.match_user_to_jobs(user_texts)
        
        # Convertir en liste et trier
        jobs_list = [
            {
                'job_id': job_id,
                **details
            }
            for job_id, details in job_scores.items()
        ]
        
        jobs_list.sort(key=lambda x: x['score'], reverse=True)
        
        return jobs_list[:n]
    
    def get_detailed_recommendation(
        self,
        user_texts: List[str],
        job_id: str
    ) -> Dict:
        """
        Analyse détaillée pour un métier spécifique.
        
        Args:
            user_texts: Réponses utilisateur
            job_id: ID du métier
        
        Returns:
            Analyse détaillée du matching
        """
        job_scores = self.match_user_to_jobs(user_texts)
        
        if job_id not in job_scores:
            raise ValueError(f"Métier {job_id} non trouvé")
        
        job_details = job_scores[job_id]
        
        # Ajouter recommandations pour combler les lacunes
        missing_count = len(job_details['missing_competencies'])
        
        recommendations = []
        if missing_count > 0:
            recommendations.append(
                f"Développer {missing_count} compétences manquantes"
            )
            recommendations.append(
                "Focus sur: " + ", ".join(job_details['missing_competencies'][:3])
            )
        
        detailed = {
            **job_details,
            'recommendations': recommendations,
            'next_steps': self._generate_next_steps(job_details)
        }
        
        return detailed
    
    def _calculate_readiness(self, score: float) -> str:
        """Calcule le niveau de préparation pour un métier."""
        if score >= 0.8:
            return "Excellent match - Prêt à postuler"
        elif score >= 0.6:
            return "Bon match - Quelques compétences à renforcer"
        elif score >= 0.4:
            return "Match partiel - Formation recommandée"
        else:
            return "Match faible - Développement important nécessaire"
    
    def _generate_next_steps(self, job_details: Dict) -> List[str]:
        """Génère les prochaines étapes recommandées."""
        steps = []
        
        if job_details['competency_coverage'] >= 0.8:
            steps.append("Mettre à jour votre CV avec vos compétences")
            steps.append("Préparer des exemples de projets concrets")
            steps.append("Commencer à postuler")
        elif job_details['competency_coverage'] >= 0.5:
            steps.append("Renforcer les compétences manquantes")
            steps.append("Réaliser des projets pratiques")
            steps.append("Suivre des formations ciblées")
        else:
            steps.append("Acquérir les compétences fondamentales")
            steps.append("Commencer par des projets d'apprentissage")
            steps.append("Envisager une formation complète")
        
        return steps
    
    def compare_jobs(
        self,
        user_texts: List[str],
        job_ids: List[str]
    ) -> Dict:
        """
        Compare plusieurs métiers pour l'utilisateur.
        
        Args:
            user_texts: Réponses utilisateur
            job_ids: Liste des IDs de métiers à comparer
        
        Returns:
            Comparaison détaillée
        """
        job_scores = self.match_user_to_jobs(user_texts)
        
        comparison = {
            'jobs': [],
            'best_match': None,
            'recommendation': ""
        }
        
        for job_id in job_ids:
            if job_id in job_scores:
                comparison['jobs'].append({
                    'job_id': job_id,
                    **job_scores[job_id]
                })
        
        # Trier par score
        comparison['jobs'].sort(key=lambda x: x['score'], reverse=True)
        
        if comparison['jobs']:
            comparison['best_match'] = comparison['jobs'][0]['job_id']
            best_score = comparison['jobs'][0]['score']
            comparison['recommendation'] = f"Meilleur match: {comparison['jobs'][0]['title']} (score: {best_score:.2%})"
        
        return comparison


if __name__ == "__main__":
    print("\nTest du module job_recommender.py\n")
    
    recommender = JobRecommender()
    
    user_responses = [
        "I have 3 years experience in data analysis with Python",
        "I work with pandas, numpy and create visualizations",
        "I know basic statistics and hypothesis testing"
    ]
    
    print("Réponses utilisateur:")
    for resp in user_responses:
        print(f"  - {resp}")
    print()
    
    # Top 3 recommandations
    print("Top 3 métiers recommandés:")
    top_jobs = recommender.get_top_recommendations(user_responses, n=3)
    
    for i, job in enumerate(top_jobs, 1):
        print(f"\n{i}. {job['title']}")
        print(f"   Score global: {job['score']:.2%}")
        print(f"   Préparation: {job['readiness']}")
        print(f"   Compétences couvertes: {job['num_matched_competencies']}/{job['num_required_competencies']}")
        
        if job['matched_competencies']:
            print(f"   Points forts:")
            for comp in job['matched_competencies'][:3]:
                print(f"     - {comp}")
        
        if job['missing_competencies']:
            print(f"   À développer:")
            for comp in job['missing_competencies'][:3]:
                print(f"     - {comp}")
    
    print("\nTest terminé")
