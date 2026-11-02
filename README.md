# Test de day trading (papier)

Un robot GitHub Actions teste des stratégies de day trading sur 15 titres américains très liquides, **sans argent réel**. Le site se met à jour tout seul toutes les 15 minutes pendant la séance.

- **Site** : https://moringo22.github.io/day-trading-test/
- **Test en direct** : à partir du 3 novembre 2026, bilan après un mois.
- **Backtest** : les 60 jours de bougies 5 min disponibles avant le début, pour donner un point de départ au modèle.

## Ce qui tourne tout seul

| Heure (Montréal, hiver) | Action |
|---|---|
| Toutes les 15 min, 9 h à 16 h 45, lundi au vendredi | Télécharge les bougies 5 min (Yahoo Finance), cherche les signaux, simule les trades, met à jour le site |

Lancer à la main : onglet **Actions** → **Test day trading** → **Run workflow**.

## Stratégies (règles fixes, rien d'optimisé)

| Code | Idée | Cible |
|---|---|---|
| ORB | Cassure du range 9 h 30–9 h 45 avec volume ≥ 1,5× et du bon côté du VWAP | 2 R |
| VWAP | Repli sur le VWAP / EMA 21 dans une tendance (EMA 9 > 21, ADX ≥ 20) | 2 R |
| REVERS | Chandelle de retournement (marteau, avalement, étoile du matin/soir) + RSI extrême + Bollinger | 1,5 R |

Stop selon la structure, borné entre 0,5 et 2 ATR. Entrée à l'ouverture de la bougie suivante, stop compté en premier si les deux sont touchés dans la même bougie, sortie à 15 h 55. Frais IBKR et glissement de 0,02 % inclus.

## Le modèle

Score de confluence (0 à 5) puis probabilité de gain et EV estimées par statistique bayésienne à partir des trades déjà fermés (jamais de données du futur). Le portefeuille « modèle » ne prend que les signaux +EV (EV ≥ 0,05 R), au plus 4 positions à la fois, arrêt après −3 R dans la journée.

## Fichiers

- `config.json` : titres, risque, frais, seuils
- `dt/engine.py` : indicateurs, chandelles, stratégies, simulation
- `dt/model.py` : probabilités, EV, statistiques
- `run.py` : exécution complète
- `data/bars/` : bougies 5 min, un fichier par jour
- `trades.csv` : tous les trades simulés
- `docs/` : le site (GitHub Pages) et `data.json`

## Critère de verdict

Avantage probable si les trades choisis par le modèle font **plus de +0,10 R en moyenne** et que **l'intervalle de confiance à 95 % reste au-dessus de 0**, sur **100 trades ou plus**.

Règles inspirées des skills Claude de la communauté : [tradermonty/claude-trading-skills](https://github.com/tradermonty/claude-trading-skills) et [agiprolabs/claude-trading-skills](https://github.com/agiprolabs/claude-trading-skills). Graphiques : TradingView Lightweight Charts™ (Apache 2.0).

Ce n'est pas un conseil financier.
