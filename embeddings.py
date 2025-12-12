"""
Module de gestion des embeddings avec SBERT.
Support de plusieurs modèles spécialisés.
"""
import torch
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Union
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modèles disponibles par domaine
AVAILABLE_MODELS = {
    'general': 'all-MiniLM-L6-v2',
    'multilingual': 'paraphrase-multilingual-MiniLM-L12-v2',
    'technical': 'allenai/scibert_scivocab_uncased',
    'business': 'sentence-transformers/all-mpnet-base-v2',
    'paraphrase': 'paraphrase-MiniLM-L6-v2',
}


class SBERTEmbeddings:
    """Gestionnaire d'embeddings SBERT avec support multi-modèles."""
    
    def __init__(self, model_name: str = 'business'):
        """
        Initialise le modèle SBERT.
        
        Args:
            model_name: Nom du modèle ou path complet
        """
        if model_name in AVAILABLE_MODELS:
            self.model_path = AVAILABLE_MODELS[model_name]
            self.model_type = model_name
        else:
            self.model_path = model_name
            self.model_type = 'custom'
        
        logger.info(f"Chargement du modèle SBERT: {self.model_path}")
        
        try:
            self.model = SentenceTransformer(self.model_path)
            logger.info(f"Modèle SBERT chargé avec succès ({self.model_type})")
        except Exception as e:
            logger.warning(f"Échec chargement {self.model_path}, fallback vers modèle général")
            self.model = SentenceTransformer(AVAILABLE_MODELS['general'])
            self.model_type = 'general'
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize: bool = True
    ) -> torch.Tensor:
        """Encode des textes en embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        
        logger.info(f"Encodage de {len(texts)} texte(s)...")
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_tensor=True,
                normalize_embeddings=normalize
            )
            
            # S'assurer que les embeddings sont sur CPU pour éviter les problèmes de device
            if embeddings.device.type != 'cpu':
                embeddings = embeddings.cpu()
            
            logger.info(f"Encodage réussi. Shape: {embeddings.shape}")
            return embeddings
        
        except Exception as e:
            logger.error(f"Erreur lors de l'encodage: {e}")
            raise
    
    def cosine_similarity(
        self, 
        embeddings1: torch.Tensor, 
        embeddings2: torch.Tensor
    ) -> torch.Tensor:
        """Calcule la similarité cosinus."""
        # S'assurer que les deux tensors sont sur le même device (CPU)
        if embeddings1.device.type != 'cpu':
            embeddings1 = embeddings1.cpu()
        if embeddings2.device.type != 'cpu':
            embeddings2 = embeddings2.cpu()
        
        # Normaliser si nécessaire
        norm1 = torch.norm(embeddings1, dim=1, keepdim=True)
        norm2 = torch.norm(embeddings2, dim=1, keepdim=True)
        
        if not torch.allclose(norm1, torch.ones_like(norm1), atol=1e-3):
            embeddings1 = torch.nn.functional.normalize(embeddings1, p=2, dim=1)
        if not torch.allclose(norm2, torch.ones_like(norm2), atol=1e-3):
            embeddings2 = torch.nn.functional.normalize(embeddings2, p=2, dim=1)
        
        similarity = torch.mm(embeddings1, embeddings2.transpose(0, 1))
        return similarity
    
    def get_model_info(self) -> dict:
        """Retourne les informations sur le modèle."""
        return {
            'model_path': self.model_path,
            'model_type': self.model_type,
            'embedding_dimension': self.model.get_sentence_embedding_dimension(),
            'max_seq_length': self.model.max_seq_length
        }


# Instance globale
_default_embedder = None


def get_embedder(model_name: str = 'business') -> SBERTEmbeddings:
    """Récupère ou crée l'embedder par défaut."""
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = SBERTEmbeddings(model_name)
    return _default_embedder


if __name__ == "__main__":
    print("\nTest des modèles SBERT disponibles\n")
    
    test_texts = [
        "Python programming for data analysis",
        "Machine learning with scikit-learn",
        "Deep learning neural networks"
    ]
    
    for model_type in ['general', 'business']:
        try:
            print(f"=== Test modèle: {model_type} ===")
            embedder = SBERTEmbeddings(model_type)
            
            embeddings = embedder.encode(test_texts)
            
            info = embedder.get_model_info()
            print(f"Dimension: {info['embedding_dimension']}")
            print(f"Max length: {info['max_seq_length']}")
            
            sim = embedder.cosine_similarity(embeddings[0:1], embeddings[1:2])
            print(f"Similarité (Python/ML): {sim[0][0]:.3f}\n")
            
        except Exception as e:
            print(f"Erreur avec {model_type}: {e}\n")
    
    print("✓ Tests terminés")
