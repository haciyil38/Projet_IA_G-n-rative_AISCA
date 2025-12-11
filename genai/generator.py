"""
Générateur RAG (Retrieval-Augmented Generation).
Combine retrieval avec génération IA.
Exigences EF4.2 et EF4.3 : Plans de progression et bio
"""
import logging
from typing import Dict, List, Optional
from genai.client import GeminiClient
from genai.cache_manager import CacheManager
from rag.context_builder import ContextBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationGenerator:
    """Génère des recommandations personnalisées avec RAG."""
    
    def __init__(
        self, 
        use_cache: bool = True,
        api_key: Optional[str] = None
    ):
        """
        Initialise le générateur RAG.
        
        Args:
            use_cache: Activer le cache (recommandé)
            api_key: Clé API Gemini optionnelle
        """
        self.client = GeminiClient(api_key)
        self.context_builder = ContextBuilder()
        self.use_cache = use_cache
        
        if use_cache:
            self.cache_manager = CacheManager()
            logger.info("Cache activé pour le générateur")
        else:
            self.cache_manager = None
            logger.warning("Cache désactivé - tous les appels API seront facturés")
    
    def generate_progression_plan(
        self,
        user_texts: List[str],
        target_job: Optional[str] = None,
        force_refresh: bool = False
    ) -> str:
        """
        Génère un plan de progression personnalisé.
        Exigence EF4.2 : 1 seul appel API avec cache
        
        Args:
            user_texts: Réponses utilisateur
            target_job: Métier cible optionnel
            force_refresh: Forcer nouveau call API (ignorer cache)
        
        Returns:
            Plan de progression structuré
        """
        logger.info("Génération du plan de progression")
        
        # Construire le contexte enrichi (Retrieval)
        context = self.context_builder.build_progression_context(
            user_texts,
            target_job
        )
        
        # Formater en prompt
        prompt = self.context_builder.format_context_for_prompt(context)
        
        # Vérifier le cache
        if self.use_cache and not force_refresh:
            cached_response = self.cache_manager.get_cached_response(
                prompt,
                task='progression_plan'
            )
            if cached_response:
                logger.info("Plan de progression récupéré du cache")
                return cached_response
        
        # Générer avec l'API (Augmented Generation)
        try:
            plan = self.client.generate_progression_plan(context)
            
            # Sauvegarder dans le cache
            if self.use_cache:
                self.cache_manager.save_to_cache(
                    prompt,
                    plan,
                    metadata={
                        'task': 'progression_plan',
                        'target_job': target_job,
                        'coverage_score': context.get('user_profile', {}).get('coverage_score', 0)
                    },
                    task='progression_plan'
                )
            
            logger.info("Plan de progression généré et mis en cache")
            return plan
        
        except Exception as e:
            logger.error(f"Erreur génération plan: {e}")
            return self._generate_fallback_progression_plan(context)
    
    def generate_professional_bio(
        self,
        user_texts: List[str],
        recommended_jobs: Optional[List[str]] = None,
        force_refresh: bool = False
    ) -> str:
        """
        Génère une bio professionnelle.
        Exigence EF4.3 : 1 seul appel API avec cache
        
        Args:
            user_texts: Réponses utilisateur
            recommended_jobs: Liste des métiers recommandés
            force_refresh: Forcer nouveau call API
        
        Returns:
            Bio professionnelle
        """
        logger.info("Génération de la bio professionnelle")
        
        # Construire le contexte enrichi (Retrieval)
        context = self.context_builder.build_bio_context(
            user_texts,
            recommended_jobs
        )
        
        # Créer prompt
        prompt_key = f"bio_{context.get('coverage_score', 0):.2f}_{len(user_texts)}"
        
        # Vérifier le cache
        if self.use_cache and not force_refresh:
            cached_response = self.cache_manager.get_cached_response(
                prompt_key,
                task='professional_bio'
            )
            if cached_response:
                logger.info("Bio professionnelle récupérée du cache")
                return cached_response
        
        # Générer avec l'API (Augmented Generation)
        try:
            bio = self.client.generate_professional_bio(context)
            
            # Sauvegarder dans le cache
            if self.use_cache:
                self.cache_manager.save_to_cache(
                    prompt_key,
                    bio,
                    metadata={
                        'task': 'professional_bio',
                        'coverage_score': context.get('coverage_score', 0),
                        'profile_level': context.get('profile_level', 'N/A')
                    },
                    task='professional_bio'
                )
            
            logger.info("Bio professionnelle générée et mise en cache")
            return bio
        
        except Exception as e:
            logger.error(f"Erreur génération bio: {e}")
            return self._generate_fallback_bio(context)
    
    def _generate_fallback_progression_plan(self, context: Dict) -> str:
        """Génère un plan de progression basique en cas d'erreur API."""
        logger.warning("Génération du plan de secours (sans API)")
        
        plan_parts = [
            "PLAN DE PROGRESSION DES COMPÉTENCES\n",
            f"Niveau actuel: {context.get('user_profile', {}).get('interpretation', 'N/A')}\n"
        ]
        
        # Objectifs basés sur compétences à développer
        if context.get('competencies_to_develop'):
            plan_parts.append("\nOBJECTIFS PRIORITAIRES:")
            for block, comps in list(context['competencies_to_develop'].items())[:2]:
                plan_parts.append(f"\n{block}:")
                for comp in comps[:3]:
                    plan_parts.append(f"  - Développer: {comp['description']}")
        
        plan_parts.append("\n\nRECOMMANDATIONS:")
        plan_parts.append("1. Concentrez-vous sur les blocs prioritaires identifiés")
        plan_parts.append("2. Pratiquez avec des projets concrets")
        plan_parts.append("3. Suivez des formations en ligne (Coursera, edX, Udemy)")
        
        return "\n".join(plan_parts)
    
    def _generate_fallback_bio(self, context: Dict) -> str:
        """Génère une bio basique en cas d'erreur API."""
        logger.warning("Génération de la bio de secours (sans API)")
        
        bio_parts = []
        
        # Niveau
        level = context.get('profile_level', 'Professionnel')
        bio_parts.append(f"{level} avec")
        
        # Domaines
        if context.get('top_blocks'):
            domains = ", ".join(context['top_blocks'][:2])
            bio_parts.append(f"une expertise en {domains}.")
        
        # Compétences
        if context.get('key_strengths'):
            bio_parts.append(f"Compétences clés: {', '.join(context['key_strengths'][:3])}.")
        
        # Métiers
        if context.get('recommended_jobs'):
            jobs = ", ".join(context['recommended_jobs'][:2])
            bio_parts.append(f"Profil adapté aux rôles de: {jobs}.")
        
        return " ".join(bio_parts)
    
    def get_cache_stats(self) -> Dict:
        """Retourne les statistiques du cache."""
        if self.cache_manager:
            return self.cache_manager.get_cache_stats()
        return {'cache_enabled': False}


if __name__ == "__main__":
    print("\nTest du module generator.py\n")
    
    # Note: Ce test nécessite une clé API valide
    print("Configuration requise:")
    print("- Clé API Gemini dans .env")
    print("- Référentiel encodé (repo_embeddings.npz)")
    print()
    
    try:
        generator = RecommendationGenerator(use_cache=True)
        
        # Test avec données fictives
        user_responses = [
            "I have experience in data analysis with Python",
            "I create visualizations with matplotlib",
            "I know basic statistics"
        ]
        
        print("1. Génération plan de progression:")
        print("   (en cache si déjà généré, sinon appel API)")
        plan = generator.generate_progression_plan(user_responses)
        print(f"   Généré: {len(plan)} caractères\n")
        print(plan[:200] + "...\n")
        
        print("2. Génération bio professionnelle:")
        bio = generator.generate_professional_bio(
            user_responses,
            recommended_jobs=["Data Analyst", "Data Scientist"]
        )
        print(f"   Généré: {len(bio)} caractères\n")
        print(bio + "\n")
        
        print("3. Statistiques du cache:")
        stats = generator.get_cache_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print("\nTests terminés avec succès")
        
    except ValueError as e:
        print(f"\nERREUR: {e}")
        print("\nConfigurez la clé API Gemini dans .env")
    except FileNotFoundError as e:
        print(f"\nERREUR: {e}")
        print("\nExécutez d'abord: python encode_repository.py")
    except Exception as e:
        print(f"\nERREUR inattendue: {e}")
