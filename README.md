# Notice 

Installation de poetry:
```
curl -sSL https://install.python-poetry.org | python3 -
```

Installation des dépendances dans l'environnement virtuel.
```
poetry install
```
Lancement du shell poetry
```
poetry shell
```
Lancement du script
```
python3 main.py
```
# Remarques
# Prédiction de pannes de machines - Rapport
Suite à mon exploration de la prédiction de pannes de machines avec le dataset "ai4i2020.csv", j'ai rencontré et résolu plusieurs défis :

1. Déséquilibre des classes
   Mon dataset était très déséquilibré (97% sans panne, 3% avec panne). Ma première approche utilisait des poids de classes, mais cela générait trop de faux positifs.
2. Optimisation de la prédiction
   Je suis passé à l'utilisation du F1-score comme métrique principale au lieu de l'accuracy, ce qui m'a permis de mieux gérer le compromis entre la détection des vraies pannes et la limitation des fausses alertes.
3. Amélioration par régularisation
   L'ajout d'une régularisation a significativement amélioré mes résultats comme le montrent les graphiques :

Les courbes de perte sont plus lisses et montrent une meilleure convergence
Le F1-score s'est amélioré progressivement jusqu'à environ 0.35-0.40
La matrice de confusion avec un seuil de 0.8 montre moins de faux positifs (60 au lieu de 279 précédemment)
L'écart entre les performances en train et en validation s'est réduit, suggérant moins de surapprentissage

# Conclusion
Mon modèle final offre un meilleur compromis entre la détection des pannes et les fausses alertes, même si la précision reste un point d'amélioration possible. La régularisation a clairement aidé à obtenir un modèle plus robuste et plus fiable.
