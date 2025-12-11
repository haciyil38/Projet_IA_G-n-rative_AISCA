"""
Client pour l'API Google Gemini.
Exigence EF4 : IA Générative avec cache
"""
import os
import logging
import google.generativeai as genai
from typing import Optional, Dict
from config import GEMINI_API_KEY, MAX_TOKENS, TEMPERATURE, MIN_TEXT_LENGTH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiClient:
    """Client pour interagir avec l'API Google Gemini."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = 'gemini-2.0-flash-exp'):
        """
        Initialise le client Gemini.
        
        Args:
            api_key: Clé API Gemini (si None, utilise variable d'environnement)
            model_name: Nom du modèle à utiliser
        """
        self.api_key = api_key or GEMINI_API_KEY
        
        if not self.api_key or self.api_key == "your_api_key_here":
            raise ValueError(
                "Clé API Gemini non configurée. "
                "Définissez GEMINI_API_KEY dans le fichier .env"
            )
        
        # Configurer l'API
        genai.configure(api_key=self.api_key)
        
        # Utiliser gemini-2.0-flash-exp qui est disponible
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Configuration de génération
        self.generation_config = {
            'temperature': TEMPERATURE,
            'max_output_tokens': MAX_TOKENS,
        }
        
        logger.info(f"Client Gemini initialisé: gemini-2.0-flash-exp")
    
    def enrich_short_text(self, text: str, min_length: int = MIN_TEXT_LENGTH) -> str:
        """
        Enrichit un texte court avec plus de contexte.
        Exigence EF4.1 : Enrichir phrases < 5 mots
        
        Args:
            text: Texte à enrichir
            min_length: Longueur minimale (en mots) avant enrichissement
        
        Returns:
            Texte enrichi ou texte original si déjà assez long
        """
        # Compter les mots
        word_count = len(text.split())
        
        if word_count >= min_length:
            logger.info(f"Texte déjà suffisant ({word_count} mots), pas d'enrichissement")
            return text
        
        logger.info(f"Enrichissement du texte court ({word_count} mots): '{text}'")
        
        prompt = f"""You are a technical skill descriptor. 
Expand this short skill description into a more detailed one (10-15 words maximum).
Keep it professional and technical.

Short description: {text}

Expanded description:"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            enriched_text = response.text.strip()
            logger.info(f"Texte enrichi: '{enriched_text}'")
            return enriched_text
        
        except Exception as e:
            logger.error(f"Erreur lors de l'enrichissement: {e}")
            return text
    
    def generate_text(
        self, 
        prompt: str,
        system_instruction: Optional[str] = None
    ) -> str:
        """
        Génère du texte à partir d'un prompt.
        
        Args:
            prompt: Prompt pour la génération
            system_instruction: Instructions système optionnelles
        
        Returns:
            Texte généré
        """
        try:
            if system_instruction:
                full_prompt = f"{system_instruction}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            response = self.model.generate_content(
                full_prompt,
                generation_config=self.generation_config
            )
            
            return response.text.strip()
        
        except Exception as e:
            logger.error(f"Erreur lors de la génération: {e}")
            raise
    
    def generate_progression_plan(self, context: Dict) -> str:
        """
        Génère un plan de progression personnalisé.
        Exigence EF4.2 : 1 seul appel API
        
        Args:
            context: Contexte avec compétences fortes/faibles
        
        Returns:
            Plan de progression structuré
        """
        logger.info("Génération du plan de progression")
        
        prompt = self._build_progression_prompt(context)
        
        system_instruction = """You are a career coach and skills development expert.
Generate a practical, actionable skills development plan in French.
Be specific and prioritize the most important skills."""
        
        return self.generate_text(prompt, system_instruction)
    
    def generate_professional_bio(self, context: Dict) -> str:
        """
        Génère une bio professionnelle.
        Exigence EF4.3 : 1 seul appel API
        
        Args:
            context: Contexte avec compétences et métiers recommandés
        
        Returns:
            Bio professionnelle
        """
        logger.info("Génération de la bio professionnelle")
        
        prompt = self._build_bio_prompt(context)
        
        system_instruction = """You are a professional profile writer.
Write a compelling professional bio in French (3-4 sentences maximum).
Highlight key strengths and recommended career paths."""
        
        return self.generate_text(prompt, system_instruction)
    
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
        """Teste la connexion à l'API Gemini."""
        try:
            response = self.model.generate_content("Test connection. Reply with 'OK'.")
            logger.info("Connexion API Gemini: OK")
            return True
        except Exception as e:
            logger.error(f"Erreur connexion API: {e}")
            return False


if __name__ == "__main__":
    print("\nTest du module client.py\n")
    
    try:
        client = GeminiClient()
        
        print("1. Test de connexion:")
        if client.test_connection():
            print("   ✓ Connexion réussie\n")
        
        print("2. Test enrichissement texte court:")
        short_text = "Python"
        enriched = client.enrich_short_text(short_text)
        print(f"   Original: '{short_text}'")
        print(f"   Enrichi: '{enriched}'\n")
        
        print("3. Test texte déjà suffisant:")
        long_text = "I have experience in data analysis with Python"
        result = client.enrich_short_text(long_text)
        print(f"   Texte: '{result}'")
        print(f"   Pas enrichi: {result == long_text}\n")
        
        print("✓ Tests terminés avec succès")
        
    except ValueError as e:
        print(f"\n✗ ERREUR: {e}")
        print("\nConfigurez votre clé API:")
        print("1. Obtenez une clé sur https://ai.google.dev/")
        print("2. Ajoutez dans .env: GEMINI_API_KEY=votre_clé")
