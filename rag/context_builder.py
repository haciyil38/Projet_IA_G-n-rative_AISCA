"""
Module de construction de contexte pour la génération RAG.
Prépare le contexte structuré à envoyer à l'IA générative.
"""
from typing import List, Dict
from rag.retriever import CompetencyRetriever


class ContextBuilder:
    """Construit le contexte enrichi pour la génération."""
    
    def __init__(self):
        """Initialise le constructeur de contexte."""
        self.retriever = CompetencyRetriever()
    
    def identify_weak_competencies(
        self,
        user_texts: List[str],
        threshold: float = 0.4
    ) -> List[Dict]:
        """
        Identifie les compétences faiblement couvertes.
        
        Args:
            user_texts: Réponses utilisateur
            threshold: Score en dessous duquel une compétence est faible
        
        Returns:
            Liste des compétences faibles
        """
        missing = self.retriever.get_missing_competencies(user_texts, threshold)
        return missing
    
    def identify_strong_competencies(
        self,
        user_texts: List[str],
        threshold: float = 0.7
    ) -> List[Dict]:
        """
        Identifie les compétences fortement couvertes.
        
        Args:
            user_texts: Réponses utilisateur
            threshold: Score au-dessus duquel une compétence est forte
        
        Returns:
            Liste des compétences fortes
        """
        matches = self.retriever.retrieve_matching_competencies(user_texts, threshold)
        return matches
    
    def build_progression_context(
        self,
        user_texts: List[str],
        target_job: str = None
    ) -> Dict:
        """
        Construit le contexte pour générer un plan de progression.
        Exigence EF4.2
        
        Args:
            user_texts: Réponses utilisateur
            target_job: Métier cible (optionnel)
        
        Returns:
            Contexte structuré pour la génération du plan
        
        Example:
            builder = ContextBuilder()
            context = builder.build_progression_context(user_texts)
        """
        # Récupérer analyse complète
        full_context = self.retriever.retrieve_relevant_context(user_texts)
        
        # Identifier compétences fortes et faibles
        strong = self.identify_strong_competencies(user_texts)
        weak = self.identify_weak_competencies(user_texts)
        
        # Grouper compétences faibles par bloc
        weak_by_block = {}
        for comp in weak[:10]:  # Top 10 prioritaires
            block = comp['block_name']
            if block not in weak_by_block:
                weak_by_block[block] = []
            weak_by_block[block].append(comp)
        
        context = {
            'user_profile': {
                'coverage_score': full_context['coverage_score'],
                'interpretation': full_context['interpretation'],
                'strong_blocks': [b[0] for b in full_context['strong_blocks']],
                'weak_blocks': [b[0] for b in full_context['weak_blocks']]
            },
            'strong_competencies': [
                {
                    'description': c['description'],
                    'block': c['block_name'],
                    'score': c['score']
                }
                for c in strong[:5]
            ],
            'competencies_to_develop': weak_by_block,
            'priority_blocks': [
                block[0] for block in full_context['weak_blocks'][:3]
            ],
            'target_job': target_job
        }
        
        return context
    
    def build_bio_context(
        self,
        user_texts: List[str],
        recommended_jobs: List[str] = None
    ) -> Dict:
        """
        Construit le contexte pour générer une bio professionnelle.
        Exigence EF4.3
        
        Args:
            user_texts: Réponses utilisateur
            recommended_jobs: Liste des métiers recommandés
        
        Returns:
            Contexte pour génération de bio
        """
        full_context = self.retriever.retrieve_relevant_context(user_texts)
        strong = self.identify_strong_competencies(user_texts)
        
        # Regrouper compétences fortes par bloc
        strong_by_block = {}
        for comp in strong:
            block = comp['block_name']
            if block not in strong_by_block:
                strong_by_block[block] = []
            strong_by_block[block].append(comp['description'])
        
        context = {
            'coverage_score': full_context['coverage_score'],
            'top_blocks': [b['block_name'] for b in full_context['top_blocks']],
            'strong_competencies_by_block': strong_by_block,
            'key_strengths': [c['description'] for c in strong[:5]],
            'recommended_jobs': recommended_jobs or [],
            'profile_level': self._determine_profile_level(full_context['coverage_score'])
        }
        
        return context
    
    def build_job_matching_context(
        self,
        user_texts: List[str]
    ) -> Dict:
        """
        Construit le contexte pour le matching avec les métiers.
        
        Args:
            user_texts: Réponses utilisateur
        
        Returns:
            Contexte pour recommandation métiers
        """
        full_context = self.retriever.retrieve_relevant_context(user_texts)
        matches = self.retriever.retrieve_matching_competencies(user_texts)
        
        # Grouper compétences par bloc
        competencies_by_block = {}
        for comp in matches:
            block = comp['block_name']
            if block not in competencies_by_block:
                competencies_by_block[block] = []
            competencies_by_block[block].append({
                'comp_id': comp['comp_id'],
                'description': comp['description'],
                'score': comp['score']
            })
        
        context = {
            'coverage_score': full_context['coverage_score'],
            'competencies_by_block': competencies_by_block,
            'block_scores': {
                b['block_name']: b['score']
                for b in full_context['top_blocks']
            },
            'matched_competencies_ids': [c['comp_id'] for c in matches]
        }
        
        return context
    
    def _determine_profile_level(self, coverage_score: float) -> str:
        """Détermine le niveau du profil."""
        if coverage_score >= 0.8:
            return "Expert"
        elif coverage_score >= 0.6:
            return "Confirmé"
        elif coverage_score >= 0.4:
            return "Intermédiaire"
        else:
            return "Débutant"
    
    def format_context_for_prompt(self, context: Dict) -> str:
        """
        Formate le contexte en texte structuré pour un prompt.
        
        Args:
            context: Contexte dict
        
        Returns:
            Texte formaté
        """
        lines = []
        
        if 'user_profile' in context:
            lines.append("PROFIL UTILISATEUR:")
            lines.append(f"- Score de couverture: {context['user_profile']['coverage_score']:.2%}")
            lines.append(f"- Niveau: {context['user_profile']['interpretation']}")
            lines.append(f"- Points forts: {', '.join(context['user_profile']['strong_blocks'])}")
            lines.append(f"- À développer: {', '.join(context['user_profile']['weak_blocks'])}")
            lines.append("")
        
        if 'strong_competencies' in context:
            lines.append("COMPÉTENCES MAÎTRISÉES:")
            for comp in context['strong_competencies']:
                lines.append(f"- {comp['description']} ({comp['block']}, score: {comp['score']:.2f})")
            lines.append("")
        
        if 'competencies_to_develop' in context:
            lines.append("COMPÉTENCES À DÉVELOPPER PAR PRIORITÉ:")
            for block, comps in context['competencies_to_develop'].items():
                lines.append(f"\n{block}:")
                for comp in comps[:3]:
                    lines.append(f"  - {comp['description']} (gap: {comp['gap']:.2f})")
            lines.append("")
        
        return "\n".join(lines)


if __name__ == "__main__":
    print("\nTest du module context_builder.py\n")
    
    builder = ContextBuilder()
    
    user_responses = [
        "I analyze data with Python and pandas",
        "I create visualizations with matplotlib"
    ]
    
    print("1. Contexte pour plan de progression:")
    prog_context = builder.build_progression_context(user_responses)
    print(f"   Score: {prog_context['user_profile']['coverage_score']:.2%}")
    print(f"   Blocs prioritaires: {prog_context['priority_blocks']}")
    print()
    
    print("2. Contexte pour bio professionnelle:")
    bio_context = builder.build_bio_context(user_responses)
    print(f"   Niveau: {bio_context['profile_level']}")
    print(f"   Top blocs: {bio_context['top_blocks']}")
    print()
    
    print("3. Contexte formaté pour prompt:")
    formatted = builder.format_context_for_prompt(prog_context)
    print(formatted)
    
    print("Test terminé")
