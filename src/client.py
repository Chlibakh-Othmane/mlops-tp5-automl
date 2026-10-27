import sys 
import time 
import statistics 
import requests 

# --- 1. CONFIGURATION DU PORT ET DE L'URL ---
# Récupère le port passé en argument dans la ligne de commande (ex: python script.py 8100).
# Si aucun argument n'est fourni, il utilise '8100' par défaut.
port = sys.argv[1] if len(sys.argv) > 1 else '8100' 

# Construit l'URL dynamique de l'endpoint d'inférence de votre modèle MLflow en local.
URL = f'http://127.0.0.1:{port}/invocations'   # le service local de MLflow 

# Définit un exemple de transaction frauduleuse (payload) au format JSON 
# attendu par MLflow (structure 'dataframe_records').
CORPS = {'dataframe_records': [{'montant': 8500.0, 'anciennete': 3, 
                                'freq_24h': 9, 'nuit': 1}]} 

# --- 2. PREMIER TEST DE VÉRIFICATION ---
# Envoie une requête HTTP POST au serveur pour vérifier qu'il répond correctement.
r = requests.post(URL, json=CORPS, timeout=10) 
print('reponse :', r.status_code, r.json()) 


# --- 3. ÉCHAUFFEMENT DU SERVEUR (WARM-UP) ---
# Effectue 10 requêtes "à blanc" (les résultats ne sont pas mesurés). 
# Cela permet au serveur de charger le modèle en mémoire, d'initialiser les 
# connexions et d'éviter que le premier vrai test ne soit faussé par le temps de démarrage.
for _ in range(10):                               # echauffement, non mesure 
    requests.post(URL, json=CORPS, timeout=10) 


# --- 4. BENCHMARK / MESURE DES PERFORMANCES ---
lat = [] 
# Boucle pour réaliser 200 appels successifs et mesurer leur temps de réponse.
for _ in range(200):                              # 200 appels mesures 
    t = time.perf_counter() # Enregistre le temps de départ précis
    requests.post(URL, json=CORPS, timeout=10) # Envoie la requête de prédiction
    # Calcule le temps écoulé en millisecondes et l'ajoute à la liste 'lat' (latence)
    lat.append((time.perf_counter() - t) * 1000) 


# --- 5. CALCUL ET AFFICHAGE DES STATISTIQUES ---
# Calcule la latence médiane et le percentile 95 (p95 : 95% des requêtes sont plus rapides que cette valeur).
print(f'port {port} | mediane : {statistics.median(lat):.1f} ms' 
      f' | p95 : {statistics.quantiles(lat, n=20)[18]:.1f} ms')