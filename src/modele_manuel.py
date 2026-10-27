import time 
from pathlib import Path 
import joblib 
import pandas as pd 
import mlflow 
import mlflow.sklearn 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score 

# --- 1. CHARGEMENT ET PRÉPARATION DES DONNÉES ---
# Chargement du jeu de données de transactions
df = pd.read_csv('data/transactions.csv') 

# Séparation des features (X) et de la variable cible (y, ici 'fraude')
X, y = df.drop(columns='fraude'), df['fraude'] 

# Découpage des données : 75% pour l'entraînement, 25% pour le test
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42) 


# --- 2. ENTRAÎNEMENT DU MODÈLE MANUEL (RANDOM FOREST) ---
t0 = time.time() 

# Initialisation et entraînement d'une forêt aléatoire avec des hyperparamètres fixes (TP2)
modele = RandomForestClassifier(
    n_estimators=500, 
    max_depth=16,   
    random_state=42
).fit(Xtr, ytr) 

duree = round(time.time() - t0, 1) 

# Évaluation de la performance du modèle avec le score F1 sur les données de test
f1 = round(f1_score(yte, modele.predict(Xte)), 4) 

# Création du dossier de sauvegarde s'il n'existe pas et export local du modèle
Path('models').mkdir(exist_ok=True) 
joblib.dump(modele, 'models/model_manuel.pkl') 

print('modele manuel : F1 =', f1, '| entrainement', duree, 's') 


# --- 3. SUIVI ET ENREGISTREMENT DANS MLFLOW ---
# Connexion au serveur local MLflow (déjà démarré sur le port 5000)
mlflow.set_tracking_uri('http://127.0.0.1:5000') 
mlflow.set_experiment('tp5-automl-deploiement') 

# Démarrage d'un run MLflow spécifique pour ce modèle manuel
with mlflow.start_run(run_name='manuel-rf-500-16'): 
    
    # Enregistrement des hyperparamètres utilisés
    mlflow.log_params({
        'outil': 'scikit-learn', 
        'n_estimators': 500, 
        'max_depth': 16
    }) 
    
    # Enregistrement des métriques (score F1 et temps d'entraînement)
    mlflow.log_metrics({
        'f1': f1, 
        'duree_entrainement_s': duree
    }) 
    
    # Enregistrement du modèle dans le Model Registry de MLflow
    info = mlflow.sklearn.log_model( 
        modele, 
        name='model', 
        input_example=Xte.head(3), 
        registered_model_name='fraude-manuel', 
        skops_trusted_types=['sklearn.tree._tree.Tree']
    )   # Paramètre de sécurité spécifique (comme au TP2)

print('enregistre : fraude-manuel, version', info.registered_model_version) 