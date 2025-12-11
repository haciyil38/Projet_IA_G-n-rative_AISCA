"""
Module de gestion des embeddings SBERT pour l'analyse sémantique.
Exigence EF2.2 : Modèle all-MiniLM-L6-v2
"""
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
from typing import List, Union
import logging

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Gestionnaire des embeddings sémantiques avec SBERT.
    Utilisé pour transformer textes en vecteurs et calculer similarités.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialise le gestionnaire avec le modèle SBERT.
        
        Args:
            model_name (str): Nom du modèle SBERT à utiliser (EF2.2)
        """
        logger.info(f"Chargement du modèle SBERT: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            logger.info("Modèle SBERT chargé avec succès")
        except Exception as e:
            logger.error(f"Erreur chargement modèle: {e}")
            raise
    
    def encode_texts(
        self, 
        texts: Union[str, List[str]], 
        convert_to_tensor: bool = True,
        show_progress_bar: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Encode une liste de textes en embeddings vectoriels.
        
        Args:
            texts: Texte unique ou liste de textes à encoder
            convert_to_tensor: Si True, retourne un tensor PyTorch
            show_progress_bar: Afficher la barre de progression
        
        Returns:
            Embeddings sous forme de tensor ou array numpy
        
        Example:
            >>> manager = EmbeddingManager()
            >>> texts = ["Python programming", "Data analysis"]
            >>> embeddings = manager.encode_texts(texts)
            >>> embeddings.shape
            torch.Size([2, 384])
        """
        # Gérer le cas d'un texte unique
        if isinstance(texts, str):
            texts = [texts]
        
        # Validation
        if not texts or len(texts) == 0:
            raise ValueError("La liste de textes ne peut pas être vide")
        
        logger.info(f"Encodage de {len(texts)} texte(s)...")
        
        try:
            embeddings = self.model.encode(
                texts,
                convert_to_tensor=convert_to_tensor,
                show_progress_bar=show_progress_bar
            )
            logger.info(f"Encodage réussi. Shape: {embeddings.shape}")
            return embeddings
        
        except Exception as e:
            logger.error(f"Erreur lors de l'encodage: {e}")
            raise
    
    def calculate_similarity(
        self, 
        embedding1: Union[torch.Tensor, np.ndarray],
        embedding2: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, float]:
        """
        Calcule la similarité cosinus entre deux embeddings.
        Exigence EF2.3 : Similarité cosinus
        
        Args:
            embedding1: Premier embedding
            embedding2: Deuxième embedding
        
        Returns:
            Score de similarité entre -1 et 1 (plus proche de 1 = plus similaire)
        
        Example:
            >>> manager = EmbeddingManager()
            >>> emb1 = manager.encode_texts("Python programming")
            >>> emb2 = manager.encode_texts("Python coding")
            >>> similarity = manager.calculate_similarity(emb1, emb2)
            >>> print(f"Similarité: {similarity:.3f}")
            Similarité: 0.856
        """
        try:
            similarity = util.cos_sim(embedding1, embedding2)
            return similarity
        
        except Exception as e:
            logger.error(f"Erreur calcul similarité: {e}")
            raise
    
    def calculate_similarity_matrix(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcule une matrice de similarité entre deux ensembles d'embeddings.
        Utile pour comparer plusieurs textes utilisateur avec plusieurs compétences.
        
        Args:
            embeddings1: Embeddings du premier ensemble (shape: [n, dim])
            embeddings2: Embeddings du deuxième ensemble (shape: [m, dim])
        
        Returns:
            Matrice de similarité (shape: [n, m])
        
        Example:
            >>> manager = EmbeddingManager()
            >>> user_texts = ["Python", "Machine Learning"]
            >>> competencies = ["Programming", "Data Science", "Web Dev"]
            >>> emb_user = manager.encode_texts(user_texts)
            >>> emb_comp = manager.encode_texts(competencies)
            >>> matrix = manager.calculate_similarity_matrix(emb_user, emb_comp)
            >>> matrix.shape
            torch.Size([2, 3])
        """
        # Convertir numpy arrays en tensors si nécessaire
        if isinstance(embeddings1, np.ndarray):
            embeddings1 = torch.tensor(embeddings1)
        if isinstance(embeddings2, np.ndarray):
            embeddings2 = torch.tensor(embeddings2)
        
        # S'assurer que les deux sont sur le même device (CPU par défaut)
        if embeddings1.device != embeddings2.device:
            embeddings1 = embeddings1.cpu()
            embeddings2 = embeddings2.cpu()
        
        return util.cos_sim(embeddings1, embeddings2)
    
    def find_most_similar(
        self,
        query_embedding: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        top_k: int = 3
    ) -> List[tuple]:
        """
        Trouve les k embeddings les plus similaires à une requête.
        
        Args:
            query_embedding: Embedding de la requête (shape: [1, dim])
            candidate_embeddings: Embeddings candidats (shape: [n, dim])
            top_k: Nombre de résultats à retourner
        
        Returns:
            Liste de tuples (index, score) triés par similarité décroissante
        
        Example:
            >>> manager = EmbeddingManager()
            >>> query = manager.encode_texts("Data analysis with Python")
            >>> candidates = manager.encode_texts([
            ...     "Python programming",
            ...     "Web development",
            ...     "Data visualization"
            ... ])
            >>> top_matches = manager.find_most_similar(query, candidates, top_k=2)
            >>> for idx, score in top_matches:
            ...     print(f"Index {idx}: {score:.3f}")
            Index 0: 0.823
            Index 2: 0.756
        """
        # Calculer similarités
        similarities = util.cos_sim(query_embedding, candidate_embeddings)[0]
        
        # Trier par ordre décroissant
        top_results = torch.topk(similarities, k=min(top_k, len(similarities)))
        
        # Retourner (index, score)
        results = [
            (idx.item(), score.item()) 
            for idx, score in zip(top_results.indices, top_results.values)
        ]
        
        return results
    
    def get_model_info(self) -> dict:
        """
        Retourne les informations sur le modèle chargé.
        
        Returns:
            Dictionnaire avec nom, dimension des embeddings
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "max_seq_length": self.model.max_seq_length
        }


# Fonction utilitaire pour usage rapide
def quick_similarity(text1: str, text2: str) -> float:
    """
    Fonction utilitaire pour calculer rapidement la similarité entre 2 textes.
    
    Args:
        text1: Premier texte
        text2: Deuxième texte
    
    Returns:
        Score de similarité (0 à 1)
    
    Example:
        >>> score = quick_similarity("Python programming", "Python coding")
        >>> print(f"{score:.3f}")
        0.856
    """
    manager = EmbeddingManager()
    emb1 = manager.encode_texts(text1)
    emb2 = manager.encode_texts(text2)
    similarity = manager.calculate_similarity(emb1, emb2)
    return similarity.item()


if __name__ == "__main__":
    # Test rapide du module
    print("Test du module embeddings.py\n")
    
    # Initialiser
    manager = EmbeddingManager()
    print(f"Modèle: {manager.get_model_info()}\n")
    
    # Test 1: Encoder des textes
    print("Test 1: Encodage de textes")
    texts = [
        "I have experience in Python programming",
        "I worked on machine learning projects",
        "I know web development with React"
    ]
    embeddings = manager.encode_texts(texts)
    print(f"{len(texts)} textes encodés. Shape: {embeddings.shape}\n")
    
    # Test 2: Similarité entre 2 textes
    print("Test 2: Similarité entre textes")
    emb1 = manager.encode_texts("Python programming")
    emb2 = manager.encode_texts("Python coding")
    emb3 = manager.encode_texts("Web design")
    
    sim_python = manager.calculate_similarity(emb1, emb2).item()
    sim_different = manager.calculate_similarity(emb1, emb3).item()
    
    print(f"Similarité 'Python programming' vs 'Python coding': {sim_python:.3f}")
    print(f"Similarité 'Python programming' vs 'Web design': {sim_different:.3f}\n")
    
    # Test 3: Trouver les plus similaires
    print("Test 3: Recherche des compétences les plus similaires")
    query = "I want to learn data analysis"
    competencies = [
        "Data visualization with Python",
        "Web development with JavaScript",
        "Machine learning algorithms",
        "Database management",
        "Statistical analysis"
    ]
    
    query_emb = manager.encode_texts(query)
    comp_emb = manager.encode_texts(competencies)
    
    top_matches = manager.find_most_similar(query_emb, comp_emb, top_k=3)
    
    print(f"Requête: '{query}'")
    print("Top 3 compétences similaires:")
    for idx, score in top_matches:
        print(f"  - {competencies[idx]} (score: {score:.3f})")
    
    print("\nTous les tests réussis!")

    def ensure_same_device(self, tensor1, tensor2):
        """
        S'assure que deux tensors sont sur le même device.
        
        Args:
            tensor1: Premier tensor
            tensor2: Deuxième tensor
        
        Returns:
            (tensor1, tensor2) sur le même device
        """
        if isinstance(tensor1, np.ndarray):
            tensor1 = torch.tensor(tensor1)
        if isinstance(tensor2, np.ndarray):
            tensor2 = torch.tensor(tensor2)
        
        # Mettre les deux sur CPU par défaut
        if tensor1.device != tensor2.device:
            tensor1 = tensor1.cpu()
            tensor2 = tensor2.cpu()
        
        return tensor1, tensor2

