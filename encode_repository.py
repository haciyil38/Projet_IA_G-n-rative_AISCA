"""
Script pour pré-calculer les embeddings du référentiel.
Utilise le modèle spécialisé 'business' pour meilleurs scores.
"""
import json
import numpy as np
from embeddings import SBERTEmbeddings
from config import REPOSITORY_PATH, EMBEDDINGS_PATH, SBERT_MODEL_TYPE
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def encode_repository():
    """Encode toutes les compétences du référentiel."""
    
    logger.info(f"Chargement du référentiel: {REPOSITORY_PATH}")
    with open(REPOSITORY_PATH, 'r', encoding='utf-8') as f:
        repository = json.load(f)
    
    # Initialiser l'embedder avec le modèle spécialisé
    logger.info(f"Utilisation du modèle: {SBERT_MODEL_TYPE}")
    embedder = SBERTEmbeddings(model_name=SBERT_MODEL_TYPE)
    
    # Préparer les données
    all_texts = []
    metadata = []
    
    for job_title, job_data in repository.items():
        for block_name, competencies in job_data.items():
            for comp in competencies:
                # Créer un texte enrichi pour meilleur matching
                text = f"{comp['description']}. {comp.get('details', '')}"
                all_texts.append(text)
                
                metadata.append({
                    'job_title': job_title,
                    'block': block_name,
                    'competency_id': comp['id'],
                    'description': comp['description'],
                    'weight': comp['weight']
                })
    
    logger.info(f"Encodage de {len(all_texts)} compétences...")
    
    # Encoder avec barre de progression
    embeddings = embedder.encode(all_texts, show_progress_bar=True)
    
    # Convertir en numpy pour sauvegarde
    embeddings_np = embeddings.cpu().numpy()
    
    # Sauvegarder
    logger.info(f"Sauvegarde dans: {EMBEDDINGS_PATH}")
    np.savez_compressed(
        EMBEDDINGS_PATH,
        embeddings=embeddings_np,
        metadata=json.dumps(metadata),
        model_info=json.dumps(embedder.get_model_info())
    )
    
    logger.info(f"✓ Encodage terminé!")
    logger.info(f"  - {len(all_texts)} compétences encodées")
    logger.info(f"  - Shape: {embeddings_np.shape}")
    logger.info(f"  - Modèle: {embedder.model_type}")
    
    return embeddings_np, metadata


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ENCODAGE DU RÉFÉRENTIEL AVEC MODÈLE OPTIMISÉ")
    print("="*60 + "\n")
    
    embeddings, metadata = encode_repository()
    
    print("\n" + "="*60)
    print(f"✓ Fichier créé: {EMBEDDINGS_PATH}")
    print(f"  Taille: {embeddings.shape}")
    print(f"  Métiers: {len(set(m['job_title'] for m in metadata))}")
    print(f"  Blocs: {len(set(m['block'] for m in metadata))}")
    print("="*60 + "\n")
