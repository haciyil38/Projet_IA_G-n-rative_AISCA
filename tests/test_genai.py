"""
Tests pour les modules GenAI (cache et generator).
Note: Tests du client nécessitent une clé API valide.
"""
import pytest
import os
from genai.cache_manager import CacheManager
from genai.generator import RecommendationGenerator


class TestCacheManager:
    """Tests pour CacheManager."""
    
    @pytest.fixture
    def cache(self, tmp_path):
        """Fixture pour créer un cache temporaire."""
        cache_dir = tmp_path / "test_cache"
        return CacheManager(str(cache_dir))
    
    def test_save_and_retrieve(self, cache):
        """Test sauvegarde et récupération."""
        prompt = "Test prompt"
        response = "Test response"
        
        # Sauvegarder
        cache.save_to_cache(prompt, response)
        
        # Récupérer
        cached = cache.get_cached_response(prompt)
        
        assert cached == response
        print("Sauvegarde et récupération: OK")
    
    def test_cache_miss(self, cache):
        """Test cache miss."""
        result = cache.get_cached_response("Non-existent prompt")
        assert result is None
        print("Cache miss: OK")
    
    def test_hash_prompt(self, cache):
        """Test génération de hash."""
        prompt1 = "Test prompt"
        prompt2 = "Test prompt"
        prompt3 = "Different prompt"
        
        hash1 = cache.hash_prompt(prompt1)
        hash2 = cache.hash_prompt(prompt2)
        hash3 = cache.hash_prompt(prompt3)
        
        assert hash1 == hash2  # Même prompt = même hash
        assert hash1 != hash3  # Prompt différent = hash différent
        print(f"Hash generation: {hash1[:8]}...")
    
    def test_get_stats(self, cache):
        """Test statistiques du cache."""
        cache.save_to_cache("prompt1", "response1")
        cache.save_to_cache("prompt2", "response2")
        
        stats = cache.get_cache_stats()
        
        assert 'total_entries' in stats
        assert stats['total_entries'] >= 2
        print(f"Stats: {stats['total_entries']} entrées")
    
    def test_clear_cache(self, cache):
        """Test vidage du cache."""
        cache.save_to_cache("prompt", "response")
        cache.clear_cache()
        
        stats = cache.get_cache_stats()
        assert stats['total_entries'] == 0
        print("Clear cache: OK")


class TestRecommendationGenerator:
    """Tests pour RecommendationGenerator."""
    
    @pytest.fixture
    def user_texts(self):
        return [
            "I analyze data with Python",
            "I create visualizations"
        ]
    
    def test_generator_init_without_api_key(self):
        """Test initialisation sans clé API."""
        # Temporairement supprimer la clé
        original_key = os.environ.get('GEMINI_API_KEY')
        if 'GEMINI_API_KEY' in os.environ:
            del os.environ['GEMINI_API_KEY']
        
        with pytest.raises(ValueError):
            RecommendationGenerator()
        
        # Restaurer
        if original_key:
            os.environ['GEMINI_API_KEY'] = original_key
        
        print("Validation clé API: OK")
    
    def test_cache_stats(self):
        """Test récupération stats cache."""
        try:
            generator = RecommendationGenerator(use_cache=True)
            stats = generator.get_cache_stats()
            
            assert isinstance(stats, dict)
            print(f"Stats disponibles: {list(stats.keys())}")
        except ValueError:
            pytest.skip("Clé API non configurée")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
