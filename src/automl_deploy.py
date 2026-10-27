import time
from pathlib import Path
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score  

# --- 1. CONFIGURATION ET CHARGEMENT DES DONNÉES ---
BUDGET = 90  # Temps maximal alloué à la recherche AutoML (en secondes)

# Chargement du jeu de données des transactions
df = pd.read_csv('data/transactions.csv')

# Séparation des caractéristiques (X) et de la cible/label à prédire (y, ici 'fraude')
X, y = df.drop(columns='fraude'), df['fraude']

# Découpage des données en ensembles d'entraînement (75%) et de test (25%)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)  

# Création des dossiers de sortie s'ils n'existent pas déjà
Path('reports').mkdir(exist_ok=True)
Path('models').mkdir(exist_ok=True)


# --- 2. ENTRAÎNEMENT AUTOMATISÉ (AUTOML AVEC FLAML) ---
automl = AutoML()
t0 = time.time()

# Lancement de la recherche du meilleur modèle
automl.fit(
    Xtr, ytr, 
    task='classification', 
    metric='f1', 
    time_budget=BUDGET,
    seed=42, 
    verbose=0, 
    log_file_name='reports/flaml_tp5.log'
)

duree = round(time.time() - t0, 1)  

# Récupération du meilleur modèle sous-jacent (ex: LightGBM, XGBoost, etc.)
modele = automl.model.estimator

# Évaluation et calcul du score F1 sur l'ensemble de test
f1_flaml = round(f1_score(yte, automl.predict(Xte)), 4)
f1_modele = round(f1_score(yte, modele.predict(Xte)), 4)

# Affichage des résultats comparatifs dans la console
print('famille retenue    :', automl.best_estimator, '| classe :', type(modele).__name__)
print('F1 (objet FLAML)   :', f1_flaml)
print('F1 (modele exporte):', f1_modele, ' <- doit etre identique')

# Sauvegarde locale du modèle sous format .pkl
joblib.dump(modele, 'models/model_automl.pkl')  


# --- 3. SUIVI DES EXPÉRIENCES ET TRACKING AVEC MLFLOW ---
# Connexion au serveur local MLflow
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('tp5-automl-deploiement')

# Démarrage d'un nouveau run MLflow
with mlflow.start_run(run_name=f'automl-{BUDGET}s'):
     
     # Enregistrement des hyperparamètres généraux de l'expérience
     mlflow.log_params({
         'outil': 'flaml', 
         'budget_s': BUDGET,
         'famille': automl.best_estimator
     })
     
     # Enregistrement dynamique de la configuration optimale trouvée par FLAML
     mlflow.log_params({f'cfg_{k}': v for k, v in automl.best_config.items()})
     
     # Enregistrement des métriques de performance et de durée
     mlflow.log_metrics({
         'f1': f1_modele, 
         'duree_recherche_s': duree
     })
     
     # Sauvegarde des logs d'entraînement FLAML en tant qu'artefact
     mlflow.log_artifact('reports/flaml_tp5.log')
     
     # Enregistrement du modèle entraîné dans le Model Registry de MLflow
     info = mlflow.sklearn.log_model(
         modele, 
         name='model', 
         input_example=Xte.head(3),
         registered_model_name='fraude-automl',
         serialization_format='cloudpickle'
     )
     
     # Confirmation de l'enregistrement dans la console
     print('enregistre : fraude-automl, version', info.registered_model_version)