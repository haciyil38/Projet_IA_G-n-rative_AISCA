"""
Configuration centralisée du projet AISCA
"""
import os
from dotenv import load_dotenv

# Charger variables d'environnement depuis .env
load_dotenv()

# API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# SBERT Configuration (EF2.2)
MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))

# Paths
DATA_DIR = "data"
REPOSITORY_PATH = os.path.join(DATA_DIR, "repository.json")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "repo_embeddings.npz")
USER_RESPONSES_DIR = os.path.join(DATA_DIR, "user_responses")
CACHE_DIR = "genai/models_cache"

# Scoring Configuration (EF3.1)
BLOCK_WEIGHTS = {
    "Data Analysis": 1.0,
    "Machine Learning": 1.0,
    "NLP": 1.0
}

# GenAI Configuration (EF4)
MAX_TOKENS = 500
TEMPERATURE = 0.7
MIN_TEXT_LENGTH = 5  # EF4.1: Enrichir si < 5 mots

# Recommandation Configuration (EF3.2)
TOP_N_RECOMMENDATIONS = 3
