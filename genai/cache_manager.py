"""
Gestionnaire de cache pour les appels API GenAI.
Exigence EF4 : Système de cache obligatoire
"""
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from config import CACHE_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CacheManager:
    """Gère le cache local des réponses API pour optimiser les coûts."""
    
    def __init__(self, cache_dir: str = CACHE_DIR):
        """
        Initialise le gestionnaire de cache.
        
        Args:
            cache_dir: Répertoire pour stocker le cache
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "api_cache.json"
        self.cache_data = self._load_cache()
        logger.info(f"Cache manager initialisé: {self.cache_file}")
    
    def _load_cache(self) -> Dict:
        """Charge le cache depuis le fichier JSON."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"Cache chargé: {len(data)} entrées")
                return data
            except Exception as e:
                logger.error(f"Erreur chargement cache: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Sauvegarde le cache dans le fichier JSON."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data, f, ensure_ascii=False, indent=2)
            logger.info("Cache sauvegardé")
        except Exception as e:
            logger.error(f"Erreur sauvegarde cache: {e}")
    
    def hash_prompt(self, prompt: str, **kwargs) -> str:
        """
        Génère un hash unique pour un prompt.
        
        Args:
            prompt: Le prompt à hasher
            **kwargs: Paramètres additionnels (température, max_tokens, etc.)
        
        Returns:
            Hash MD5 du prompt et paramètres
        """
        # Combiner prompt et paramètres
        cache_key = {
            'prompt': prompt,
            'params': kwargs
        }
        
        # Créer hash
        cache_str = json.dumps(cache_key, sort_keys=True)
        hash_obj = hashlib.md5(cache_str.encode('utf-8'))
        return hash_obj.hexdigest()
    
    def get_cached_response(
        self, 
        prompt: str,
        ttl_hours: int = 24,
        **kwargs
    ) -> Optional[str]:
        """
        Récupère une réponse depuis le cache si elle existe.
        
        Args:
            prompt: Le prompt
            ttl_hours: Durée de vie du cache en heures (par défaut 24h)
            **kwargs: Paramètres additionnels
        
        Returns:
            Réponse cachée ou None si pas trouvée/expirée
        """
        prompt_hash = self.hash_prompt(prompt, **kwargs)
        
        if prompt_hash in self.cache_data:
            cache_entry = self.cache_data[prompt_hash]
            
            # Vérifier expiration
            cached_time = datetime.fromisoformat(cache_entry['timestamp'])
            if datetime.now() - cached_time < timedelta(hours=ttl_hours):
                logger.info(f"Cache HIT: {prompt_hash[:8]}...")
                return cache_entry['response']
            else:
                logger.info(f"Cache EXPIRED: {prompt_hash[:8]}...")
                del self.cache_data[prompt_hash]
                self._save_cache()
        
        logger.info(f"Cache MISS: {prompt_hash[:8]}...")
        return None
    
    def save_to_cache(
        self, 
        prompt: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Sauvegarde une réponse dans le cache.
        
        Args:
            prompt: Le prompt utilisé
            response: La réponse de l'API
            metadata: Métadonnées additionnelles
            **kwargs: Paramètres utilisés
        """
        prompt_hash = self.hash_prompt(prompt, **kwargs)
        
        cache_entry = {
            'prompt': prompt[:200],  # Sauvegarder début du prompt
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.cache_data[prompt_hash] = cache_entry
        self._save_cache()
        
        logger.info(f"Cache SAVED: {prompt_hash[:8]}...")
    
    def clear_cache(self):
        """Vide complètement le cache."""
        self.cache_data = {}
        self._save_cache()
        logger.info("Cache vidé")
    
    def get_cache_stats(self) -> Dict:
        """
        Retourne les statistiques du cache.
        
        Returns:
            Dict avec statistiques
        """
        total_entries = len(self.cache_data)
        
        # Compter entrées expirées
        expired = 0
        for entry in self.cache_data.values():
            cached_time = datetime.fromisoformat(entry['timestamp'])
            if datetime.now() - cached_time > timedelta(hours=24):
                expired += 1
        
        # Taille du cache
        cache_size_bytes = self.cache_file.stat().st_size if self.cache_file.exists() else 0
        cache_size_kb = cache_size_bytes / 1024
        
        return {
            'total_entries': total_entries,
            'active_entries': total_entries - expired,
            'expired_entries': expired,
            'cache_size_kb': round(cache_size_kb, 2),
            'cache_file': str(self.cache_file)
        }
    
    def clean_expired(self, ttl_hours: int = 24):
        """
        Nettoie les entrées expirées du cache.
        
        Args:
            ttl_hours: Durée de vie en heures
        """
        initial_count = len(self.cache_data)
        
        expired_keys = []
        for key, entry in self.cache_data.items():
            cached_time = datetime.fromisoformat(entry['timestamp'])
            if datetime.now() - cached_time > timedelta(hours=ttl_hours):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache_data[key]
        
        if expired_keys:
            self._save_cache()
            logger.info(f"Nettoyage: {len(expired_keys)} entrées expirées supprimées")
        
        return len(expired_keys)


if __name__ == "__main__":
    print("\nTest du module cache_manager.py\n")
    
    cache = CacheManager()
    
    # Test 1: Sauvegarder dans le cache
    print("1. Test sauvegarde dans le cache:")
    prompt = "Generate a skills development plan for data science"
    response = "This is a mock response from the API"
    cache.save_to_cache(prompt, response, metadata={'model': 'gemini-1.5-flash'})
    print("   Réponse sauvegardée\n")
    
    # Test 2: Récupérer depuis le cache
    print("2. Test récupération depuis le cache:")
    cached = cache.get_cached_response(prompt)
    if cached:
        print(f"   Récupéré: '{cached[:50]}...'\n")
    
    # Test 3: Cache miss
    print("3. Test cache miss:")
    missed = cache.get_cached_response("A completely different prompt")
    print(f"   Résultat: {'Trouvé' if missed else 'Non trouvé (normal)'}\n")
    
    # Test 4: Statistiques
    print("4. Statistiques du cache:")
    stats = cache.get_cache_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\nTest terminé")
