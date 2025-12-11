"""
Script de pré-calcul des embeddings du référentiel de compétences.
Permet d'optimiser les performances en calculant une seule fois les embeddings.
Exigence EF2.2 : Utilisation de SBERT
"""
import json
import numpy as np
from pathlib import Path
import logging
import torch
from embeddings import EmbeddingManager
from config import REPOSITORY_PATH, EMBEDDINGS_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RepositoryEncoder:
    """Encode le référentiel de compétences en embeddings."""
    
    def __init__(self, repository_path: str = REPOSITORY_PATH):
        """
        Initialise l'encodeur.
        
        Args:
            repository_path: Chemin vers repository.json
        """
        self.repository_path = Path(repository_path)
        # Forcer CPU pour éviter problèmes de device
        self.embedding_manager = EmbeddingManager()
        # Force le modèle sur CPU
        if hasattr(self.embedding_manager.model, 'to'):
            self.embedding_manager.model = self.embedding_manager.model.to('cpu')
        self.repository_data = None
        self.embeddings_data = {}
    
    def load_repository(self) -> dict:
        """
        Charge le référentiel JSON.
        
        Returns:
            Dictionnaire contenant les données du référentiel
        """
        logger.info(f"Chargement du référentiel: {self.repository_path}")
        
        if not self.repository_path.exists():
            raise FileNotFoundError(f"Référentiel non trouvé: {self.repository_path}")
        
        with open(self.repository_path, 'r', encoding='utf-8') as f:
            self.repository_data = json.load(f)
        
        logger.info(f"Référentiel chargé avec succès")
        return self.repository_data
    
    def extract_competencies(self) -> dict:
        """
        Extrait toutes les compétences du référentiel.
        
        Returns:
            Dict avec structure: {comp_id: {description, block_name, keywords}}
        """
        if self.repository_data is None:
            self.load_repository()
        
        competencies = {}
        
        for block in self.repository_data.get('competency_blocks', []):
            block_name = block['block_name']
            block_id = block['block_id']
            
            for comp in block.get('competencies', []):
                comp_id = comp['comp_id']
                competencies[comp_id] = {
                    'description': comp['description'],
                    'keywords': comp.get('keywords', []),
                    'block_name': block_name,
                    'block_id': block_id
                }
        
        logger.info(f"Extraction de {len(competencies)} compétences")
        return competencies
    
    def encode_competencies(self) -> dict:
        """
        Encode toutes les compétences en embeddings.
        
        Returns:
            Dict avec embeddings par compétence
        """
        competencies = self.extract_competencies()
        
        # Préparer les textes à encoder
        comp_ids = list(competencies.keys())
        texts = [competencies[cid]['description'] for cid in comp_ids]
        
        logger.info(f"Encodage de {len(texts)} compétences avec SBERT...")
        
        # Encoder tous les textes d'un coup (plus efficace)
        # Forcer convert_to_tensor=False pour avoir numpy array
        embeddings = self.embedding_manager.model.encode(
            texts, 
            convert_to_tensor=False,
            show_progress_bar=True,
            device='cpu'  # Forcer CPU
        )
        
        # Stocker avec métadonnées
        self.embeddings_data = {
            'comp_ids': comp_ids,
            'embeddings': embeddings,
            'competencies_metadata': competencies,
            'model_name': self.embedding_manager.model_name
        }
        
        logger.info("Encodage terminé")
        return self.embeddings_data
    
    def encode_blocks(self) -> dict:
        """
        Encode les descriptions de blocs de compétences.
        
        Returns:
            Dict avec embeddings par bloc
        """
        if self.repository_data is None:
            self.load_repository()
        
        blocks = {}
        block_ids = []
        block_names = []
        
        for block in self.repository_data.get('competency_blocks', []):
            block_id = block['block_id']
            block_name = block['block_name']
            
            # Créer description du bloc à partir des compétences
            comp_descriptions = [c['description'] for c in block.get('competencies', [])]
            block_description = f"{block_name}: " + ", ".join(comp_descriptions)
            
            blocks[block_id] = {
                'name': block_name,
                'description': block_description,
                'num_competencies': len(block.get('competencies', []))
            }
            
            block_ids.append(block_id)
            block_names.append(block_description)
        
        logger.info(f"Encodage de {len(blocks)} blocs...")
        
        # Encoder
        block_embeddings = self.embedding_manager.model.encode(
            block_names,
            convert_to_tensor=False,
            device='cpu'
        )
        
        return {
            'block_ids': block_ids,
            'embeddings': block_embeddings,
            'blocks_metadata': blocks
        }
    
    def encode_jobs(self) -> dict:
        """
        Encode les profils métiers.
        
        Returns:
            Dict avec embeddings par métier
        """
        if self.repository_data is None:
            self.load_repository()
        
        jobs = {}
        job_ids = []
        job_descriptions = []
        
        competencies = self.extract_competencies()
        
        for job in self.repository_data.get('job_profiles', []):
            job_id = job['job_id']
            job_title = job['job_title']
            required_comps = job.get('required_competencies', [])
            
            # Créer description du métier à partir des compétences requises
            comp_descs = [
                competencies[comp_id]['description'] 
                for comp_id in required_comps 
                if comp_id in competencies
            ]
            job_description = f"{job_title}: requires " + ", ".join(comp_descs)
            
            jobs[job_id] = {
                'title': job_title,
                'required_competencies': required_comps,
                'description': job_description
            }
            
            job_ids.append(job_id)
            job_descriptions.append(job_description)
        
        logger.info(f"Encodage de {len(jobs)} métiers...")
        
        # Encoder
        job_embeddings = self.embedding_manager.model.encode(
            job_descriptions,
            convert_to_tensor=False,
            device='cpu'
        )
        
        return {
            'job_ids': job_ids,
            'embeddings': job_embeddings,
            'jobs_metadata': jobs
        }
    
    def save_embeddings(self, output_path: str = EMBEDDINGS_PATH):
        """
        Sauvegarde tous les embeddings dans un fichier .npz.
        
        Args:
            output_path: Chemin de sortie pour le fichier .npz
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Sauvegarde des embeddings dans: {output_path}")
        
        # Encoder tout
        comp_data = self.encode_competencies()
        block_data = self.encode_blocks()
        job_data = self.encode_jobs()
        
        # Sauvegarder au format .npz (compressé)
        np.savez_compressed(
            output_path,
            # Compétences
            competency_ids=comp_data['comp_ids'],
            competency_embeddings=comp_data['embeddings'],
            competency_metadata=json.dumps(comp_data['competencies_metadata']),
            # Blocs
            block_ids=block_data['block_ids'],
            block_embeddings=block_data['embeddings'],
            block_metadata=json.dumps(block_data['blocks_metadata']),
            # Métiers
            job_ids=job_data['job_ids'],
            job_embeddings=job_data['embeddings'],
            job_metadata=json.dumps(job_data['jobs_metadata']),
            # Métadonnées
            model_name=comp_data['model_name']
        )
        
        logger.info(f"Embeddings sauvegardés avec succès")
        logger.info(f"  - {len(comp_data['comp_ids'])} compétences")
        logger.info(f"  - {len(block_data['block_ids'])} blocs")
        logger.info(f"  - {len(job_data['job_ids'])} métiers")
    
    def get_stats(self) -> dict:
        """
        Retourne les statistiques du référentiel.
        
        Returns:
            Dict avec statistiques
        """
        if self.repository_data is None:
            self.load_repository()
        
        num_blocks = len(self.repository_data.get('competency_blocks', []))
        num_comps = sum(
            len(block.get('competencies', []))
            for block in self.repository_data.get('competency_blocks', [])
        )
        num_jobs = len(self.repository_data.get('job_profiles', []))
        
        return {
            'num_blocks': num_blocks,
            'num_competencies': num_comps,
            'num_jobs': num_jobs
        }


class EmbeddingsLoader:
    """Charge les embeddings pré-calculés (lazy loading)."""
    
    def __init__(self, embeddings_path: str = EMBEDDINGS_PATH):
        """
        Initialise le chargeur.
        
        Args:
            embeddings_path: Chemin vers le fichier .npz
        """
        self.embeddings_path = Path(embeddings_path)
        self._data = None
        self._competency_metadata = None
        self._block_metadata = None
        self._job_metadata = None
    
    def _load(self):
        """Charge les données (lazy loading)."""
        if self._data is None:
            logger.info(f"Chargement des embeddings depuis: {self.embeddings_path}")
            
            if not self.embeddings_path.exists():
                raise FileNotFoundError(
                    f"Embeddings non trouvés: {self.embeddings_path}\n"
                    f"Exécutez d'abord: python encode_repository.py"
                )
            
            self._data = np.load(self.embeddings_path, allow_pickle=True)
            logger.info("Embeddings chargés")
    
    def get_competency_embeddings(self) -> tuple:
        """
        Retourne les embeddings des compétences.
        
        Returns:
            (comp_ids, embeddings)
        """
        self._load()
        return (
            self._data['competency_ids'].tolist(),
            self._data['competency_embeddings']
        )
    
    def get_block_embeddings(self) -> tuple:
        """
        Retourne les embeddings des blocs.
        
        Returns:
            (block_ids, embeddings)
        """
        self._load()
        return (
            self._data['block_ids'].tolist(),
            self._data['block_embeddings']
        )
    
    def get_job_embeddings(self) -> tuple:
        """
        Retourne les embeddings des métiers.
        
        Returns:
            (job_ids, embeddings)
        """
        self._load()
        return (
            self._data['job_ids'].tolist(),
            self._data['job_embeddings']
        )
    
    def get_competency_metadata(self) -> dict:
        """Retourne les métadonnées des compétences."""
        if self._competency_metadata is None:
            self._load()
            self._competency_metadata = json.loads(
                str(self._data['competency_metadata'])
            )
        return self._competency_metadata
    
    def get_block_metadata(self) -> dict:
        """Retourne les métadonnées des blocs."""
        if self._block_metadata is None:
            self._load()
            self._block_metadata = json.loads(
                str(self._data['block_metadata'])
            )
        return self._block_metadata
    
    def get_job_metadata(self) -> dict:
        """Retourne les métadonnées des métiers."""
        if self._job_metadata is None:
            self._load()
            self._job_metadata = json.loads(
                str(self._data['job_metadata'])
            )
        return self._job_metadata


def main():
    """Script principal pour encoder le référentiel."""
    print("\n" + "="*70)
    print("Encodage du référentiel de compétences AISCA")
    print("="*70 + "\n")
    
    try:
        # Créer l'encodeur
        encoder = RepositoryEncoder()
        
        # Afficher les stats
        stats = encoder.get_stats()
        print(f"Statistiques du référentiel:")
        print(f"  - Blocs de compétences: {stats['num_blocks']}")
        print(f"  - Compétences totales: {stats['num_competencies']}")
        print(f"  - Profils métiers: {stats['num_jobs']}")
        print()
        
        # Encoder et sauvegarder
        encoder.save_embeddings()
        
        print("\n" + "="*70)
        print("Encodage terminé avec succès")
        print("="*70 + "\n")
        
        # Test du chargement
        print("Test du chargement...")
        loader = EmbeddingsLoader()
        comp_ids, comp_emb = loader.get_competency_embeddings()
        print(f"  - {len(comp_ids)} compétences chargées")
        print(f"  - Shape embeddings: {comp_emb.shape}")
        
        print("\nVous pouvez maintenant utiliser les embeddings dans votre application.")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'encodage: {e}")
        raise


if __name__ == "__main__":
    main()
