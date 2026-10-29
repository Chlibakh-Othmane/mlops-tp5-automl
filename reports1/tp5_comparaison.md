# Modele manuel ou modele AutoML ? (TP05)

| Critere                    | Manuel (rf 500/16)                                          | AutoML (FLAML 90 s)                                            |
|----------------------------|-------------------------------------------------------------|----------------------------------------------------------------|
| Famille de modele          | foret aleatoire (RandomForestClassifier)                    | xgboost (XGBClassifier)                                        |
| F1 sur le jeu de test      | 0.8669                                                      | 0.8561                                                         |
| Temps de recherche machine | 1.3 s (un seul entrainement, pas de recherche)              | 90.3 s                                                         |
| Temps humain (estimation)  | ~1 h (choix des hyperparametres au TP2)                     | ~5 min (lancer le script)                                      |
| Latence mediane (ms)       | 54.9 (mesure 1) / 64.2 (mesure 2)                           | <A MESURER>                                                    |
| Latence p95 (ms)           | 77.3 (mesure 1) / 86.5 (mesure 2)                           | <A MESURER>                                                    |
| Taille du fichier (Mo)     | 13.37                                                       | 0.28                                                           |
| Explicabilite              | importance des variables, 500 arbres, profondeur 16         | importance des variables, hyperparametres cfg_ (nb d'arbres, profondeur, learning rate) |

## Decision (5 lignes) : quel modele mettre en production, et pourquoi ?

Le modele manuel a le meilleur F1 (0.8669 contre 0.8561), mais l'ecart est faible (~0.011).
Le modele AutoML (XGBoost) est environ 48 fois plus leger (0.28 Mo contre 13.37 Mo), donc plus simple a embarquer (Docker, mobile).
AutoML a trouve ce modele en 90 s sans effort de reglage, alors que le manuel a demande environ 1 h de travail humain.
Latence : <completer apres mesure du port AutoML : plus rapide / comparable / plus lente que le manuel (~55-64 ms)>.
Decision : <AutoML si la latence est meilleure ou comparable, car plus leger et presque aussi performant ; manuel si la detection de fraude exige le meilleur F1>.