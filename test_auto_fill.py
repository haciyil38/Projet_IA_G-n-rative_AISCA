"""
Script pour pré-remplir automatiquement le questionnaire.
Utilisez ce profil pour tester l'application rapidement.
"""

# Profil de test : Data Scientist Avancé
TEST_PROFILE = {
    1: "4 - Avancé",
    2: "J'ai 3 ans d'expérience en analyse de données avec pandas, numpy pour la manipulation de données, et scipy pour les calculs scientifiques. J'ai travaillé sur des projets d'analyse de ventes, de segmentation client avec K-means, et de prédiction de churn avec des modèles de classification. J'utilise quotidiennement Jupyter Notebook pour l'analyse exploratoire et la création de rapports interactifs.",
    3: ["Matplotlib", "Seaborn", "Plotly", "Power BI"],
    4: "4 - Avancé",
    5: {
        'main': "Oui, occasionnellement",
        'followup': "J'ai utilisé NLTK et spaCy pour la tokenization dans des projets d'analyse de sentiments sur des avis clients et de classification de textes. J'ai aussi expérimenté avec des tokenizers de transformers comme BERT pour des tâches de NLP avancées."
    },
    6: "MySQL/PostgreSQL",
    7: ["Régression linéaire/logistique", "Random Forest", "XGBoost/LightGBM", "Neural Networks"],
    8: "3 - Intermédiaire",
    9: "J'ai développé un modèle de prédiction de churn pour une entreprise de télécommunications avec XGBoost. Après avoir testé plusieurs algorithmes (Random Forest, Logistic Regression, SVM), le modèle final atteint 87% de précision et 82% de recall. J'ai créé un dashboard Power BI pour suivre les prédictions en temps réel et identifier les clients à risque. Ce projet a permis de réduire le taux de churn de 15% en 6 mois.",
    10: ["AWS", "Google Cloud Platform"]
}

# Profil alternatif : Débutant en Data
TEST_PROFILE_BEGINNER = {
    1: "2 - Élémentaire",
    2: "Je commence à apprendre l'analyse de données. J'ai suivi des tutoriels en ligne sur pandas et j'ai fait quelques exercices sur Kaggle. Je sais lire des fichiers CSV et faire des statistiques descriptives basiques.",
    3: ["Matplotlib"],
    4: "1 - Débutant",
    5: {
        'main': "Je ne sais pas ce que c'est",
        'followup': ""
    },
    6: "Aucune",
    7: ["Régression linéaire/logistique"],
    8: "2 - Élémentaire",
    9: "J'ai fait un petit projet d'analyse des ventes d'un magasin fictif avec pandas. J'ai calculé des moyennes et créé quelques graphiques simples avec matplotlib.",
    10: ["Aucun"]
}

# Profil Expert : Machine Learning Engineer
TEST_PROFILE_EXPERT = {
    1: "5 - Expert",
    2: "J'ai 5+ ans d'expérience en data science et machine learning. Expert en pandas, numpy, scikit-learn, et frameworks de deep learning (TensorFlow, PyTorch). J'ai déployé plus de 20 modèles en production sur AWS et GCP. Spécialisé dans les pipelines MLOps avec Airflow, MLflow et Kubeflow. J'ai également contribué à des projets open-source dans l'écosystème Python data science.",
    3: ["Matplotlib", "Seaborn", "Plotly", "Tableau", "Power BI", "D3.js"],
    4: "5 - Expert",
    5: {
        'main': "Oui, régulièrement",
        'followup': "Expert en NLP avec transformers (BERT, GPT, T5). J'ai développé des systèmes de tokenization personnalisés pour des langues peu dotées. Maîtrise de Hugging Face, spaCy avancé, et création de modèles de langue from scratch."
    },
    6: "MySQL/PostgreSQL",
    7: ["Régression linéaire/logistique", "Arbres de décision", "Random Forest", "XGBoost/LightGBM", "Neural Networks", "Deep Learning"],
    8: "5 - Expert",
    9: "J'ai architecturé et déployé un système de recommandation temps réel pour un e-commerce (10M+ utilisateurs) utilisant des embeddings neuronaux et du collaborative filtering. Le système traite 5000 req/s avec une latence <50ms sur Kubernetes. J'ai aussi développé un modèle de détection de fraude avec deep learning atteignant 99.2% de précision, réduisant les pertes de 3M€/an. Publication de 2 papers en conférence ML.",
    10: ["AWS", "Azure", "Google Cloud Platform"]
}

def print_profile(profile_name, profile):
    """Affiche un profil de test."""
    print(f"\n{'='*60}")
    print(f"PROFIL DE TEST : {profile_name}")
    print('='*60)
    
    questions = [
        "Niveau Python",
        "Expérience analyse de données",
        "Outils de visualisation",
        "Niveau Machine Learning",
        "Tokenization NLP",
        "Base de données principale",
        "Compétences ML",
        "Niveau statistiques",
        "Projet data science",
        "Services cloud"
    ]
    
    for i, (q_id, response) in enumerate(profile.items(), 1):
        print(f"\nQuestion {i}: {questions[i-1]}")
        if isinstance(response, dict):
            print(f"  → {response['main']}")
            if response['followup']:
                print(f"    Détail: {response['followup'][:80]}...")
        elif isinstance(response, list):
            print(f"  → {', '.join(response)}")
        elif isinstance(response, str) and len(response) > 100:
            print(f"  → {response[:100]}...")
        else:
            print(f"  → {response}")


if __name__ == "__main__":
    print("\n" + "🎓 PROFILS DE TEST AISCA ".center(60, "="))
    
    print_profile("DATA SCIENTIST AVANCÉ (RECOMMANDÉ)", TEST_PROFILE)
    print_profile("DÉBUTANT EN DATA", TEST_PROFILE_BEGINNER)
    print_profile("MACHINE LEARNING ENGINEER EXPERT", TEST_PROFILE_EXPERT)
    
    print("\n" + "="*60)
    print("\n💡 Pour utiliser ces profils :")
    print("   1. Copiez-collez les réponses manuellement dans Streamlit")
    print("   2. Ou utilisez le profil dans un test automatisé")
    print("\n" + "="*60 + "\n")
