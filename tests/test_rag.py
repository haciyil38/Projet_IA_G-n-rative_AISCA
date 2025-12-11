"""
Tests pour les modules RAG (retriever, context_builder, job_recommender).
"""
import pytest
from rag.retriever import CompetencyRetriever
from rag.context_builder import ContextBuilder
from rag.job_recommender import JobRecommender


class TestCompetencyRetriever:
    """Tests pour CompetencyRetriever."""
    
    @pytest.fixture
    def retriever(self):
        return CompetencyRetriever(threshold=0.3)
    
    @pytest.fixture
    def user_texts(self):
        return [
            "I analyze data with Python and pandas",
            "I create visualizations with matplotlib"
        ]
    
    def test_retrieve_matching_competencies(self, retriever, user_texts):
        """Test récupération compétences correspondantes."""
        matches = retriever.retrieve_matching_competencies(user_texts)
        
        assert len(matches) > 0
        assert all('comp_id' in m for m in matches)
        assert all('score' in m for m in matches)
        assert all('description' in m for m in matches)
        
        # Vérifier tri par score décroissant
        scores = [m['score'] for m in matches]
        assert scores == sorted(scores, reverse=True)
        
        print(f"Trouvé {len(matches)} compétences correspondantes")
    
    def test_get_top_n_blocks(self, retriever, user_texts):
        """Test récupération top N blocs."""
        top_blocks = retriever.get_top_n_blocks(user_texts, n=3)
        
        assert len(top_blocks) <= 3
        assert all('block_name' in b for b in top_blocks)
        assert all('score' in b for b in top_blocks)
        
        # Vérifier tri par score
        scores = [b['score'] for b in top_blocks]
        assert scores == sorted(scores, reverse=True)
        
        print(f"Top 3 blocs: {[b['block_name'] for b in top_blocks]}")
    
    def test_get_missing_competencies(self, retriever, user_texts):
        """Test identification compétences manquantes."""
        missing = retriever.get_missing_competencies(user_texts, threshold=0.3)
        
        assert isinstance(missing, list)
        assert all('comp_id' in m for m in missing)
        assert all('gap' in m for m in missing)
        
        print(f"Identifié {len(missing)} compétences à développer")


class TestContextBuilder:
    """Tests pour ContextBuilder."""
    
    @pytest.fixture
    def builder(self):
        return ContextBuilder()
    
    @pytest.fixture
    def user_texts(self):
        return [
            "I analyze data with Python",
            "I create charts"
        ]
    
    def test_identify_weak_competencies(self, builder, user_texts):
        """Test identification compétences faibles."""
        weak = builder.identify_weak_competencies(user_texts, threshold=0.4)
        
        assert isinstance(weak, list)
        assert all('description' in w for w in weak)
        
        print(f"Compétences faibles: {len(weak)}")
    
    def test_identify_strong_competencies(self, builder, user_texts):
        """Test identification compétences fortes."""
        strong = builder.identify_strong_competencies(user_texts, threshold=0.5)
        
        assert isinstance(strong, list)
        assert all('description' in s for s in strong)
        
        print(f"Compétences fortes: {len(strong)}")
    
    def test_build_progression_context(self, builder, user_texts):
        """Test construction contexte progression."""
        context = builder.build_progression_context(user_texts)
        
        assert 'user_profile' in context
        assert 'strong_competencies' in context
        assert 'competencies_to_develop' in context
        assert 'priority_blocks' in context
        
        assert 'coverage_score' in context['user_profile']
        
        print(f"Score couverture: {context['user_profile']['coverage_score']:.2%}")
    
    def test_build_bio_context(self, builder, user_texts):
        """Test construction contexte bio."""
        context = builder.build_bio_context(user_texts)
        
        assert 'coverage_score' in context
        assert 'top_blocks' in context
        assert 'key_strengths' in context
        assert 'profile_level' in context
        
        print(f"Niveau profil: {context['profile_level']}")
    
    def test_format_context_for_prompt(self, builder, user_texts):
        """Test formatage contexte pour prompt."""
        context = builder.build_progression_context(user_texts)
        formatted = builder.format_context_for_prompt(context)
        
        assert isinstance(formatted, str)
        assert len(formatted) > 0
        assert 'PROFIL UTILISATEUR' in formatted
        
        print(f"Contexte formaté ({len(formatted)} caractères)")


class TestJobRecommender:
    """Tests pour JobRecommender."""
    
    @pytest.fixture
    def recommender(self):
        return JobRecommender()
    
    @pytest.fixture
    def user_texts(self):
        return [
            "I analyze data with Python and pandas",
            "I create visualizations",
            "I know statistics"
        ]
    
    def test_match_user_to_jobs(self, recommender, user_texts):
        """Test matching utilisateur vers métiers."""
        job_matches = recommender.match_user_to_jobs(user_texts)
        
        assert isinstance(job_matches, dict)
        assert len(job_matches) > 0
        
        for job_id, details in job_matches.items():
            assert 'score' in details
            assert 'title' in details
            assert 'readiness' in details
            assert 0 <= details['score'] <= 1
        
        print(f"Analysé {len(job_matches)} métiers")
    
    def test_get_top_recommendations(self, recommender, user_texts):
        """Test récupération top 3 métiers. Exigence EF3.2"""
        top_jobs = recommender.get_top_recommendations(user_texts, n=3)
        
        assert len(top_jobs) <= 3
        assert all('job_id' in j for j in top_jobs)
        assert all('title' in j for j in top_jobs)
        assert all('score' in j for j in top_jobs)
        
        # Vérifier tri par score
        scores = [j['score'] for j in top_jobs]
        assert scores == sorted(scores, reverse=True)
        
        print("Top 3 métiers:")
        for i, job in enumerate(top_jobs, 1):
            print(f"  {i}. {job['title']} (score: {job['score']:.2%})")
    
    def test_readiness_calculation(self, recommender, user_texts):
        """Test calcul préparation métier."""
        top_jobs = recommender.get_top_recommendations(user_texts, n=1)
        
        if top_jobs:
            job = top_jobs[0]
            assert 'readiness' in job
            assert isinstance(job['readiness'], str)
            
            print(f"Préparation: {job['readiness']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
