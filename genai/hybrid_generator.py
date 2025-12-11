"""
Générateur hybride avec support multi-provider.
Bascule automatiquement entre Ollama (local) et Gemini (cloud).
"""
import logging
from typing import Optional, List, Dict
from genai.ollama_client import OllamaClient
from genai.client import GeminiClient
from genai.cache_manager import CacheManager
from rag.context_builder import ContextBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridGenerator:
    """
    Générateur hybride avec fallback automatique.
    Ordre de priorité: Ollama -> Gemini -> Fallback
    """
    
    def __init__(
        self,
        use_cache: bool = True,
        prefer_local: bool = True,
        gemini_api_key: Optional[str] = None
    ):
        """
        Initialise le générateur hybride.
        
        Args:
            use_cache: Activer le cache
            prefer_local: Préférer Ollama (local) si disponible
            gemini_api_key: Clé API Gemini optionnelle
        """
        self.context_builder = ContextBuilder()
        self.prefer_local = prefer_local
        self.providers = []
        
        # Configurer le cache
        if use_cache:
            self.cache_manager = CacheManager()
            logger.info("Cache activé")
        else:
            self.cache_manager = None
        
        # Initialiser les providers disponibles
        self._init_providers(gemini_api_key)
        
        if not self.providers:
            logger.warning("Aucun provider disponible, fallback uniquement")
    
    def _init_providers(self, gemini_api_key: Optional[str]):
        """Initialise les providers dans l'ordre de priorité."""
        
        # 1. Ollama (priorité si prefer_local=True)
        if self.prefer_local:
            try:
                ollama = OllamaClient()
                if ollama.test_connection():
                    self.providers.append(('ollama', ollama))
                    logger.info("✓ Ollama disponible (LOCAL)")
            except Exception as e:
                logger.warning(f"Ollama non disponible: {e}")
        
        # 2. Gemini
        try:
            gemini = GeminiClient(api_key=gemini_api_key)
            if gemini.test_connection():
                self.providers.append(('gemini', gemini))
                logger.info("✓ Gemini disponible (CLOUD)")
        except Exception as e:
            logger.warning(f"Gemini non disponible: {e}")
        
        # 3. Si prefer_local=False, essayer Ollama en dernier recours
        if not self.prefer_local and len(self.providers) == 0:
            try:
                ollama = OllamaClient()
                if ollama.test_connection():
                    self.providers.append(('ollama', ollama))
                    logger.info("✓ Ollama disponible (FALLBACK)")
            except:
                pass
        
        logger.info(f"Providers actifs: {[p[0] for p in self.providers]}")
    
    def _generate_with_fallback(
        self,
        generation_func: str,
        *args,
        **kwargs
    ) -> str:
        """
        Essaie de générer avec les providers disponibles.
        
        Args:
            generation_func: Nom de la méthode à appeler
            *args, **kwargs: Arguments pour la méthode
        
        Returns:
            Texte généré
        """
        # Essayer chaque provider
        for provider_name, provider in self.providers:
            try:
                logger.info(f"Tentative avec {provider_name}...")
                method = getattr(provider, generation_func)
                result = method(*args, **kwargs)
                logger.info(f"✓ Génération réussie avec {provider_name}")
                return result
            except Exception as e:
                logger.warning(f"✗ Échec avec {provider_name}: {e}")
                continue
        
        # Si tous échouent, utiliser le fallback
        logger.warning("Tous les providers ont échoué, utilisation du fallback")
        return self._fallback_generation(generation_func, *args, **kwargs)
    
    def generate_progression_plan(
        self,
        user_texts: List[str],
        target_job: Optional[str] = None,
        force_refresh: bool = False
    ) -> str:
        """
        Génère un plan de progression personnalisé.
        
        Args:
            user_texts: Réponses utilisateur
            target_job: Métier cible optionnel
            force_refresh: Forcer nouveau call (ignorer cache)
        
        Returns:
            Plan de progression
        """
        logger.info("Génération du plan de progression")
        
        # Construire le contexte
        context = self.context_builder.build_progression_context(
            user_texts,
            target_job
        )
        
        # Vérifier le cache
        cache_key = f"progression_{hash(str(user_texts))}"
        if self.cache_manager and not force_refresh:
            cached = self.cache_manager.get_cached_response(cache_key)
            if cached:
                logger.info("Plan récupéré du cache")
                return cached
        
        # Générer avec fallback
        plan = self._generate_with_fallback(
            'generate_progression_plan',
            context
        )
        
        # Sauvegarder en cache
        if self.cache_manager:
            self.cache_manager.save_to_cache(cache_key, plan)
        
        return plan
    
    def generate_professional_bio(
        self,
        user_texts: List[str],
        recommended_jobs: Optional[List[str]] = None,
        force_refresh: bool = False
    ) -> str:
        """
        Génère une bio professionnelle.
        
        Args:
            user_texts: Réponses utilisateur
            recommended_jobs: Métiers recommandés
            force_refresh: Forcer nouveau call
        
        Returns:
            Bio professionnelle
        """
        logger.info("Génération de la bio professionnelle")
        
        # Construire le contexte
        context = self.context_builder.build_bio_context(
            user_texts,
            recommended_jobs
        )
        
        # Vérifier le cache
        cache_key = f"bio_{hash(str(user_texts))}"
        if self.cache_manager and not force_refresh:
            cached = self.cache_manager.get_cached_response(cache_key)
            if cached:
                logger.info("Bio récupérée du cache")
                return cached
        
        # Générer avec fallback
        bio = self._generate_with_fallback(
            'generate_professional_bio',
            context
        )
        
        # Sauvegarder en cache
        if self.cache_manager:
            self.cache_manager.save_to_cache(cache_key, bio)
        
        return bio
    
    def _fallback_generation(self, func_name: str, *args, **kwargs) -> str:
        """Génération de secours si tous les providers échouent."""
        if func_name == 'generate_progression_plan':
            context = args[0]
            return self._fallback_progression_plan(context)
        elif func_name == 'generate_professional_bio':
            context = args[0]
            return self._fallback_bio(context)
        else:
            return "Génération non disponible actuellement."
    
    def _fallback_progression_plan(self, context: Dict) -> str:
        """Plan de progression de secours."""
        plan_parts = [
            "PLAN DE PROGRESSION DES COMPÉTENCES\n",
            f"Niveau actuel: {context.get('user_profile', {}).get('interpretation', 'N/A')}\n"
        ]
        
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
    
    def _fallback_bio(self, context: Dict) -> str:
        """Bio de secours."""
        bio_parts = []
        
        level = context.get('profile_level', 'Professionnel')
        bio_parts.append(f"{level} avec")
        
        if context.get('top_blocks'):
            domains = ", ".join(context['top_blocks'][:2])
            bio_parts.append(f"une expertise en {domains}.")
        
        if context.get('key_strengths'):
            bio_parts.append(f"Compétences clés: {', '.join(context['key_strengths'][:3])}.")
        
        if context.get('recommended_jobs'):
            jobs = ", ".join(context['recommended_jobs'][:2])
            bio_parts.append(f"Profil adapté aux rôles de: {jobs}.")
        
        return " ".join(bio_parts)
    
    def get_cache_stats(self) -> Dict:
        """Retourne les statistiques du cache."""
        if self.cache_manager:
            return self.cache_manager.get_cache_stats()
        return {'cache_enabled': False}
    
    def get_active_provider(self) -> Optional[str]:
        """Retourne le provider actuellement actif."""
        if self.providers:
            return self.providers[0][0]
        return None


if __name__ == "__main__":
    print("\nTest du module hybrid_generator.py\n")
    
    try:
        # Test avec préférence local
        print("=== Test 1: Préférence LOCAL (Ollama) ===")
        generator_local = HybridGenerator(prefer_local=True)
        print(f"Provider actif: {generator_local.get_active_provider()}\n")
        
        # Test génération
        user_texts = [
            "I have experience in data analysis with Python",
            "I create visualizations"
        ]
        
        print("Génération plan de progression...")
        plan = generator_local.generate_progression_plan(user_texts)
        print(f"✓ Plan généré ({len(plan)} caractères)")
        print(f"\nAperçu:\n{plan[:300]}...\n")
        
        print("Génération bio professionnelle...")
        bio = generator_local.generate_professional_bio(
            user_texts,
            recommended_jobs=["Data Analyst", "Data Scientist"]
        )
        print(f"✓ Bio générée ({len(bio)} caractères)")
        print(f"\nRésultat:\n{bio}\n")
        
        # Stats
        stats = generator_local.get_cache_stats()
        print(f"Stats cache: {stats.get('total_entries', 0)} entrées")
        
        print("\n✓ Tests terminés avec succès!")
        
    except Exception as e:
        print(f"\n✗ ERREUR: {e}")
        print("\nAssurez-vous que Ollama est démarré:")
        print("  ollama serve")
