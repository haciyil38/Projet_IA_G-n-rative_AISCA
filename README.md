# AISCA - Agent Intelligent Sémantique et Génératif

Application d'analyse sémantique pour la cartographie des compétences et la recommandation de métiers.

## 🚀 Installation

### Prérequis
- Python 3.8+
- pip

### Configuration

1. Cloner le repository
```bash
git clone https://github.com/haciyil38/projet-ia-generative.git
cd projet-ia-generative
```

2. Créer environnement virtuel
```bash
python3 -m venv venv
source venv/bin/activate # macOS/Linux
```

3. Installer dépendances
```bash
pip install -r requirements.txt
```

4. Configurer clés API
```bash
cp .env.example .env
```

Éditer `.env` et ajouter votre clé API Gemini

## 🎯 Utilisation

Lancer l'application :
```bash
streamlit run app.py
```

## 📁 Structure du projet

```
aisca/
├── app.py              # Interface Streamlit
├── nlp/                # Moteur NLP SBERT
├── rag/                # Architecture RAG
├── genai/              # IA Générative Gemini
├── visualization/      # Graphiques radar
└── data/               # Référentiel compétences
```

## 👥 Auteurs

- Haci
- Neïl

## 📄 Licence

Projet académique EFREI 2025-26
