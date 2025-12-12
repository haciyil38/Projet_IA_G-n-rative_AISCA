"""
Module de récupération des compétences pertinentes (RAG).
"""
from typing import List, Dict, Tuple
import numpy as np
import json
import logging
from embeddings import get_embedder, SBERTEmbeddings
from config import EMBEDDINGS_PATH, SIMILARITY_THRESHOLD, SBERT_MODEL_TYPE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompetencyRetriever:
    """Récupère les compétences pertinentes depuis le référentiel."""
    
    def __init__(self, embeddings_path: str = EMBEDDINGS_PATH):
        """
        Initialise le retriever.
        
        Args:
            embeddings_path: Chemin vers les embeddings pré-calculés
        """
        logger.info(f"Chargement des embeddings depuis: {embeddings_path}")
        
        data = np.load(embeddings_path, allow_pickle=True)
        self.embeddings = data['embeddings']
        self.metadata = json.loads(str(data['metadata']))
        
        logger.info(f"Embeddings chargés: {self.embeddings.shape}")
        
        self.embedder = get_embedder(SBERT_MODEL_TYPE)
    
    def retrieve(
        self, 
        query_texts: List[str], 
        top_k: int = 10,
        threshold: float = SIMILARITY_THRESHOLD
    ) -> List[Dict]:
        """
        Récupère les compétences les plus similaires.
        
        Args:
            query_texts: Textes de requête
            top_k: Nombre de résultats par requête
            threshold: Seuil de similarité minimum
        
        Returns:
            Liste de compétences avec scores
        """
        import torch
        
        query_embeddings = self.embedder.encode(query_texts)
        repo_embeddings = torch.from_numpy(self.embeddings).float()
        
        similarities = self.embedder.cosine_similarity(query_embeddings, repo_embeddings)
        
        all_results = []
        seen_ids = set()
        
        for i, query in enumerate(query_texts):
            query_sims = similarities[i]
            top_indices = torch.argsort(query_sims, descending=True)[:top_k]
            
            for idx in top_indices:
                score = query_sims[idx].item()
                if score >= threshold:
                    comp = self.metadata[idx]
                    comp_id = comp['competency_id']
                    
                    if comp_id not in seen_ids:
                        seen_ids.add(comp_id)
                        all_results.append({
                            'competency_id': comp_id,
                            'description': comp['description'],
                            'job_title': comp['job_title'],
                            'block': comp['block'],
                            'weight': comp['weight'],
                            'similarity_score': score,
                            'query': query
                        })
        
        all_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return all_results


if __name__ == "__main__":
    print("\nTest du retriever\n")
    
    retriever = CompetencyRetriever()
    
    queries = [
        "Python programming for data analysis",
        "Machine learning with scikit-learn"
    ]
    
    results = retriever.retrieve(queries, top_k=5)
    
    print(f"Trouvé {len(results)} compétences pertinentes:\n")
    for r in results[:5]:
        print(f"  - {r['description']}")
        print(f"    Score: {r['similarity_score']:.3f}")
        print(f"    Métier: {r['job_title']} / {r['block']}\n")
