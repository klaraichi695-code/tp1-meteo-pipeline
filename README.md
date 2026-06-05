# tp1-meteo-pipeline
## Objectif

Construire un pipeline ETL complet en Python pour nettoyer des données météo brutes, générer un rapport de qualité, créer des features et exporter les résultats.

---

## Prérequis

- Python 3.11+
- pip

---

## Installation

```bash
cd TPs/TP1_Meteo_Pipeline
pip install -r requirements.txt
Exécution
bash
python src/pipeline.py
Structure
text
TP1_Meteo_Pipeline/
├── src/
│   └── pipeline.py          # Code source
├── data/raw/
│   └── meteo_brut.csv       # Données d'entrée
├── outputs/                 # Résultats générés
├── logs/                    # Journaux
└── requirements.txt
Résultats
Indicateur	Valeur
Lignes entrée	16
Lignes sortie	14
Taux conservation	87.5%
Features créées	5
Fichiers générés :

outputs/meteo_clean.csv – données nettoyées

outputs/meteo_features.csv – données enrichies

outputs/quality_report.txt – rapport qualité

logs/pipeline.log – journal

Anomalies corrigées
Problème	Correction
Température texte ("vingt")	→ NaN → médiane par station
Humidité négative (-5)	→ NaN → médiane globale
Température 999°C	→ NaN → médiane par station
Date "15/06/2025"	Parsing automatique
Date "2025-02-30"	Ligne supprimée
irrigation "OUI"/"off"	Normalisé → ON/OFF
Doublons	Supprimés
Features créées
Feature	Règle
jour_semaine	0=lundi → 6=dimanche
temp_classe	froid(<15) / tempéré(15-25) / chaud(>25)
irrigation_bin	1=ON, 0=OFF
heat_index	température + 0.5 × humidité
besoin_arrosage	1 si (rain_mm=0 ET température>25)
Captures
Données nettoyées (extrait)
text
id,station,date,temperature,humidity,rain_mm,wind_kmh,irrigation
1,ST-01,2025-06-01,22.5,65.0,0.0,12,OFF
2,ST-01,2025-06-02,24.1,58.0,0.0,15,ON
Sortie console
text
EXTRACT : 16 lignes chargées
TRANSFORM : 14 lignes conservées (sur 16)
BUILD FEATURES : 5 features créées
✅ Toutes les assertions passent.
PIPELINE TERMINÉ en 0.15s
Difficultés rencontrées
Déduplication : la colonne id empêchait la détection des vrais doublons → suppression avant drop_duplicates()

Dates dd/mm/yyyy : non reconnues automatiquement par pandas → utilisation de format='mixed'

Imputation par station : syntaxe complexe → df.groupby('station')['temperature'].transform('median')
