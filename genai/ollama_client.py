"""
Client pour Ollama (LLM local).
Alternative gratuite et illimitée à Gemini.
"""
import requests
import logging
from typing import Optional, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaClient:
    """Client pour interagir avec Ollama en local."""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2"
    ):
        """
        Initialise le client Ollama.
        
        Args:
            base_url: URL du serveur Ollama
            model: Nom du modèle à utiliser
        """
        self.base_url = base_url
        self.model = model
        self.api_generate = f"{base_url}/api/generate"
        self.api_chat = f"{base_url}/api/chat"
        
        # Vérifier que Ollama est disponible
        if not self._check_connection():
            raise ConnectionError(
                "Ollama n'est pas accessible. "
                "Assurez-vous qu'Ollama est démarré avec: ollama serve"
            )
        
        logger.info(f"Client Ollama initialisé: {model}")
    
    def _check_connection(self) -> bool:
        """Vérifie que Ollama est accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate_text(
        self, 
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """
        Génère du texte avec Ollama.
        
        Args:
            prompt: Prompt pour la génération
            system_instruction: Instructions système optionnelles
            temperature: Température de génération
            max_tokens: Nombre max de tokens
        
        Returns:
            Texte généré
        """
        try:
            # Construire le prompt complet
            full_prompt = prompt
            if system_instruction:
                full_prompt = f"{system_instruction}\n\n{prompt}"
            
            # Appel à l'API Ollama
            response = requests.post(
                self.api_generate,
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                },
                timeout=60
            )
            
            response.raise_for_status()
            result = response.json()
            
            generated_text = result.get('response', '').strip()
            logger.info(f"Texte généré avec Ollama ({len(generated_text)} chars)")
            
            return generated_text
        
        except Exception as e:
            logger.error(f"Erreur génération Ollama: {e}")
            raise
    
    def enrich_short_text(self, text: str, min_length: int = 5) -> str:
        """
        Enrichit un texte court.
        
        Args:
            text: Texte à enrichir
            min_length: Longueur minimale en mots
        
        Returns:
            Texte enrichi
        """
        word_count = len(text.split())
        
        if word_count >= min_length:
            logger.info(f"Texte déjà suffisant ({word_count} mots)")
            return text
        
        logger.info(f"Enrichissement du texte: '{text}'")
        
        prompt = f"""You are a technical skill descriptor.
Expand this short skill description into a more detailed one (10-15 words maximum).
Keep it professional and technical.

Short description: {text}

Expanded description (10-15 words only):"""
        
        try:
            enriched = self.generate_text(prompt, max_tokens=50)
            # Nettoyer la réponse (enlever labels possibles)
            enriched = enriched.replace("Expanded description:", "").strip()
            logger.info(f"Texte enrichi: '{enriched}'")
            return enriched
        except:
            return text
    
    def generate_progression_plan(self, context: Dict) -> str:
        """
        Génère un plan de progression personnalisé.
        
        Args:
            context: Contexte avec compétences fortes/faibles
        
        Returns:
            Plan de progression structuré
        """
        logger.info("Génération du plan de progression avec Ollama")
        
        prompt = self._build_progression_prompt(context)
        
        system_instruction = """You are a career coach and skills development expert.
Generate a practical, actionable skills development plan in French.
Be specific and prioritize the most important skills.
Structure your response with clear sections."""
        
        return self.generate_text(prompt, system_instruction, temperature=0.7)
    
    def generate_professional_bio(self, context: Dict) -> str:
        """
        Génère une bio professionnelle.
        
        Args:
            context: Contexte avec compétences et métiers
        
        Returns:
            Bio professionnelle
        """
        logger.info("Génération de la bio professionnelle avec Ollama")
        
        prompt = self._build_bio_prompt(context)
        
        system_instruction = """You are a professional profile writer.
Write a compelling professional bio in French (3-4 sentences maximum).
Highlight key strengths and recommended career paths.
Be concise and impactful."""
        
        return self.generate_text(prompt, system_instruction, temperature=0.8, max_tokens=300)
    
    def _build_progression_prompt(self, context: Dict) -> str:
        """Construit le prompt pour le plan de progression."""
        prompt_parts = [
            "Génère un plan de progression des compétences personnalisé.\n",
            f"\nNiveau actuel: {context.get('user_profile', {}).get('interpretation', 'Non défini')}",
            f"Score de couverture: {context.get('user_profile', {}).get('coverage_score', 0):.1%}\n"
        ]
        
        if context.get('strong_competencies'):
            prompt_parts.append("\nCOMPÉTENCES MAÎTRISÉES:")
            for comp in context['strong_competencies'][:3]:
                prompt_parts.append(f"- {comp['description']}")
        
        if context.get('competencies_to_develop'):
            prompt_parts.append("\n\nCOMPÉTENCES À DÉVELOPPER (par priorité):")
            for block, comps in list(context['competencies_to_develop'].items())[:2]:
                prompt_parts.append(f"\n{block}:")
                for comp in comps[:3]:
                    prompt_parts.append(f"  - {comp['description']}")
        
        prompt_parts.append("\n\nGénère un plan en 3 étapes concrètes:")
        prompt_parts.append("1. Objectifs à court terme (1-3 mois)")
        prompt_parts.append("2. Objectifs à moyen terme (3-6 mois)")
        prompt_parts.append("3. Ressources recommandées (cours, projets, certifications)")
        
        return "\n".join(prompt_parts)
    
    def _build_bio_prompt(self, context: Dict) -> str:
        """Construit le prompt pour la bio professionnelle."""
        prompt_parts = [
            "Rédige une bio professionnelle accrocheuse (style Executive Summary).\n",
            f"\nNiveau: {context.get('profile_level', 'Professionnel')}",
            f"Score de compétences: {context.get('coverage_score', 0):.1%}\n"
        ]
        
        if context.get('top_blocks'):
            prompt_parts.append("\nDomaines d'expertise:")
            for block in context['top_blocks'][:3]:
                prompt_parts.append(f"- {block}")
        
        if context.get('key_strengths'):
            prompt_parts.append("\n\nCompétences principales:")
            for strength in context['key_strengths'][:5]:
                prompt_parts.append(f"- {strength}")
        
        if context.get('recommended_jobs'):
            prompt_parts.append(f"\n\nMétiers recommandés: {', '.join(context['recommended_jobs'][:3])}")
        
        prompt_parts.append("\n\nRédige une bio en 3-4 phrases maximum, professionnelle et impactante.")
        
        return "\n".join(prompt_parts)
    
    def test_connection(self) -> bool:
        """Teste la connexion à Ollama."""
        try:
            response = self.generate_text("Test connection. Reply with 'OK'.", max_tokens=10)
            logger.info("Connexion Ollama: OK")
            return True
        except Exception as e:
            logger.error(f"Erreur connexion Ollama: {e}")
            return False


if __name__ == "__main__":
    print("\nTest du module ollama_client.py\n")
    
    try:
        # Initialiser client
        client = OllamaClient()
        
        # Test 1: Connexion
        print("1. Test de connexion:")
        if client.test_connection():
            print("   ✓ Connexion réussie\n")
        
        # Test 2: Enrichissement
        print("2. Test enrichissement texte court:")
        short_text = "Python"
        enriched = client.enrich_short_text(short_text)
        print(f"   Original: '{short_text}'")
        print(f"   Enrichi: '{enriched}'\n")
        
        # Test 3: Génération simple
        print("3. Test génération simple:")
        response = client.generate_text(
            "Write a one-sentence description of a data analyst.",
            max_tokens=100
        )
        print(f"   Réponse: {response}\n")
        
        print("✓ Tests terminés avec succès")
        
    except ConnectionError as e:
        print(f"\n✗ ERREUR: {e}")
        print("\nPour démarrer Ollama:")
        print("1. Dans un terminal: ollama serve")
        print("2. Dans un autre: ollama pull llama3.2")
    except Exception as e:
        print(f"\n✗ ERREUR inattendue: {e}")
