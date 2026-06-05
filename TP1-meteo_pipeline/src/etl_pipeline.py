"""
ETL Pipeline - Traitement des données météo et irrigation
"""

import pandas as pd
import numpy as np
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "paths": {
        "input": "data/raw/meteo_brut.csv",
        "output_clean": "outputs/data_clean.csv",
        "output_features": "outputs/data_features.csv",
        "output_report": "outputs/quality_report.txt",
        "log": "logs/execution.log"
    },
    "bounds": {
        "temp_min": -40,
        "temp_max": 60,
        "humidity_min": 0,
        "humidity_max": 100
    },
    "mappings": {
        "irrigation": {"ON": "ON", "OFF": "OFF", "OUI": "ON", "NON": "OFF"}
    }
}

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_path):
    """Configure le système de logging"""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger("ETL")

# ============================================================================
# STEP 1: EXTRACTION
# ============================================================================

def extract_data(filepath, logger):
    """
    Extrait les données depuis le fichier CSV
    
    Args:
        filepath: Chemin vers le fichier source
        logger: Instance du logger
    
    Returns:
        DataFrame contenant les données brutes
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: EXTRACTION")
    logger.info("=" * 60)
    
    if not Path(filepath).exists():
        logger.error(f"Fichier introuvable: {filepath}")
        raise FileNotFoundError(f"Le fichier {filepath} n'existe pas")
    
    df = pd.read_csv(filepath)
    
    logger.info(f"✓ Extraction réussie: {len(df)} lignes, {len(df.columns)} colonnes")
    logger.debug(f"Colonnes: {list(df.columns)}")
    logger.debug(f"Types: \n{df.dtypes}")
    
    return df

# ============================================================================
# STEP 2: TRANSFORMATION
# ============================================================================

def standardize_text_columns(df, logger):
    """Standardise les colonnes textuelles"""
    # Station: majuscules et suppression espaces
    df['station'] = df['station'].str.strip().str.upper()
    logger.debug("  → station: standardisé en majuscules")
    
    # Irrigation: mapping des valeurs
    n_before = df['irrigation'].notna().sum()
    df['irrigation'] = (
        df['irrigation']
        .astype(str)
        .str.strip()
        .str.upper()
        .map(CONFIG["mappings"]["irrigation"])
    )
    n_mapped = df['irrigation'].notna().sum()
    logger.debug(f"  → irrigation: {n_mapped}/{n_before} valeurs mappées")
    
    return df

def convert_numeric_columns(df, logger):
    """Convertit les colonnes en types numériques"""
    numeric_cols = ['temperature', 'humidity', 'rain_mm', 'wind_kmh']
    
    for col in numeric_cols:
        before = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors='coerce')
        after = df[col].isna().sum()
        converted = after - before
        if converted > 0:
            logger.warning(f"  → {col}: {converted} valeur(s) non-numérique(s) converties en NaN")
    
    return df

def parse_dates(df, logger):
    """Parse et valide les dates"""
    before = df['date'].isna().sum()
    df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
    after = df['date'].isna().sum()
    invalid = after - before
    if invalid > 0:
        logger.warning(f"  → date: {invalid} date(s) invalide(s) → NaT")
    return df

def remove_duplicates(df, logger):
    """Supprime les doublons exacts"""
    n_dups = df.duplicated().sum()
    if n_dups > 0:
        df = df.drop_duplicates()
        logger.warning(f"  → {n_dups} doublon(s) supprimé(s)")
    else:
        logger.debug("  → aucun doublon détecté")
    return df

def validate_ranges(df, logger):
    """Valide les valeurs hors bornes"""
    bounds = CONFIG["bounds"]
    
    # Température
    mask = (df['temperature'] < bounds["temp_min"]) | (df['temperature'] > bounds["temp_max"])
    n_out = mask.sum()
    if n_out > 0:
        df.loc[mask, 'temperature'] = np.nan
        logger.warning(f"  → temperature: {n_out} valeur(s) hors [{bounds['temp_min']}, {bounds['temp_max']}] → NaN")
    
    # Humidité
    mask = (df['humidity'] < bounds["humidity_min"]) | (df['humidity'] > bounds["humidity_max"])
    n_out = mask.sum()
    if n_out > 0:
        df.loc[mask, 'humidity'] = np.nan
        logger.warning(f"  → humidity: {n_out} valeur(s) hors [{bounds['humidity_min']}, {bounds['humidity_max']}] → NaN")
    
    return df

def remove_invalid_dates(df, logger):
    """Supprime les lignes sans date valide"""
    n_before = len(df)
    df = df.dropna(subset=['date'])
    n_removed = n_before - len(df)
    if n_removed > 0:
        logger.warning(f"  → {n_removed} ligne(s) supprimée(s) (date invalide)")
    return df

def impute_missing_values(df, logger):
    """Impute les valeurs manquantes"""
    
    # Température: médiane par station
    n_missing = df['temperature'].isna().sum()
    if n_missing > 0:
        df['temperature'] = df.groupby('station')['temperature'].transform(
            lambda x: x.fillna(x.median())
        )
        logger.info(f"  → temperature: {n_missing} NaN imputés (médiane par station)")
    
    # Humidité: médiane globale
    n_missing = df['humidity'].isna().sum()
    if n_missing > 0:
        median_hum = df['humidity'].median()
        df['humidity'] = df['humidity'].fillna(median_hum)
        logger.info(f"  → humidity: {n_missing} NaN imputés (médiane globale = {median_hum:.1f})")
    
    # Pluie: 0.0 par défaut
    n_missing = df['rain_mm'].isna().sum()
    if n_missing > 0:
        df['rain_mm'] = df['rain_mm'].fillna(0.0)
        logger.info(f"  → rain_mm: {n_missing} NaN imputés (0.0)")
    
    return df

def detect_outliers(df, logger):
    """Détecte les outliers (sans suppression)"""
    def iqr_outliers(series):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        return (series < q1 - 1.5*iqr) | (series > q3 + 1.5*iqr)
    
    for col in ['temperature', 'humidity', 'rain_mm', 'wind_kmh']:
        outliers = iqr_outliers(df[col].dropna())
        n_out = outliers.sum()
        if n_out > 0:
            logger.warning(f"  → {col}: {n_out} outlier(s) IQR détecté(s) (conservés)")
    
    return df

def transform_data(df, logger):
    """
    Pipeline complet de transformation
    
    Args:
        df: DataFrame brut
        logger: Instance du logger
    
    Returns:
        DataFrame nettoyé
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: TRANSFORMATION")
    logger.info("=" * 60)
    
    n_initial = len(df)
    logger.info(f"Début: {n_initial} lignes")
    
    # Application des transformations
    df = standardize_text_columns(df, logger)
    df = convert_numeric_columns(df, logger)
    df = parse_dates(df, logger)
    df = remove_duplicates(df, logger)
    df = validate_ranges(df, logger)
    df = detect_outliers(df, logger)
    df = remove_invalid_dates(df, logger)
    df = impute_missing_values(df, logger)
    
    logger.info(f"✓ Fin transformation: {len(df)} lignes conservées (sur {n_initial})")
    
    return df

# ============================================================================
# STEP 3: RAPPORT QUALITE
# ============================================================================

def generate_quality_report(df_raw, df_clean, output_path, logger):
    """
    Génère un rapport détaillé de la qualité des données
    
    Args:
        df_raw: DataFrame brut
        df_clean: DataFrame nettoyé
        output_path: Chemin de sortie du rapport
        logger: Instance du logger
    """
    logger.info("=" * 60)
    logger.info("PHASE 3: RAPPORT QUALITE")
    logger.info("=" * 60)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    lines = []
    lines.append("=" * 70)
    lines.append("RAPPORT DE QUALITE DES DONNEES METEO")
    lines.append(f"Date de génération: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")
    
    # 1. Volumétrie
    lines.append("1. VOLUMETRIE")
    lines.append("-" * 50)
    lines.append(f"  Lignes en entrée      : {len(df_raw):>6}")
    lines.append(f"  Lignes en sortie      : {len(df_clean):>6}")
    lines.append(f"  Lignes supprimées     : {len(df_raw) - len(df_clean):>6}")
    lines.append(f"  Taux de conservation  : {100 * len(df_clean) / len(df_raw):>5.1f}%")
    lines.append("")
    
    # 2. Qualité par colonne (brut)
    lines.append("2. VALEURS MANQUANTES (données brutes)")
    lines.append("-" * 50)
    lines.append(f"  {'Colonne':<15} {'Manquants':>10} {'Taux':>8}")
    lines.append(f"  {'-'*15} {'-'*10} {'-'*8}")
    for col in df_raw.columns:
        n_miss = df_raw[col].isna().sum()
        pct = 100 * n_miss / len(df_raw)
        lines.append(f"  {col:<15} {n_miss:>10} {pct:>7.1f}%")
    lines.append("")
    
    # 3. Doublons
    lines.append("3. DOUBLONS")
    lines.append("-" * 50)
    lines.append(f"  Doublons exacts : {df_raw.duplicated().sum()}")
    lines.append("")
    
    # 4. Statistiques (nettoyé)
    lines.append("4. STATISTIQUES DESCRIPTIVES (données nettoyées)")
    lines.append("-" * 50)
    
    numeric_cols = ['temperature', 'humidity', 'rain_mm', 'wind_kmh']
    lines.append(f"  {'Variable':<12} {'Min':>8} {'Max':>8} {'Moyenne':>10} {'Médiane':>10} {'Ecart-type':>12}")
    lines.append(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*12}")
    
    for col in numeric_cols:
        if col in df_clean.columns:
            s = df_clean[col].dropna()
            lines.append(f"  {col:<12} {s.min():>8.1f} {s.max():>8.1f} {s.mean():>10.1f} {s.median():>10.1f} {s.std():>12.1f}")
    lines.append("")
    
    # 5. Détection d'outliers
    lines.append("5. OUTLIERS DETECTES (méthode IQR, données brutes)")
    lines.append("-" * 50)
    
    def count_outliers(series):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        outliers = (series < q1 - 1.5*iqr) | (series > q3 + 1.5*iqr)
        return outliers.sum()
    
    for col in numeric_cols:
        if col in df_raw.columns:
            series = pd.to_numeric(df_raw[col], errors='coerce').dropna()
            if len(series) > 0:
                n_out = count_outliers(series)
                lines.append(f"  {col:<15} : {n_out:>2} outlier(s)")
    
    lines.append("")
    lines.append("=" * 70)
    lines.append("FIN DU RAPPORT")
    lines.append("=" * 70)
    
    # Écriture du fichier
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"✓ Rapport qualité généré: {output_path}")

# ============================================================================
# STEP 4: FEATURE ENGINEERING
# ============================================================================

def build_features(df, logger):
    """
    Crée des caractéristiques dérivées
    
    Args:
        df: DataFrame nettoyé
        logger: Instance du logger
    
    Returns:
        DataFrame enrichi
    """
    logger.info("=" * 60)
    logger.info("PHASE 4: FEATURE ENGINEERING")
    logger.info("=" * 60)
    
    df = df.copy()
    
    # Feature 1: Jour de la semaine (0=lundi, 6=dimanche)
    df['day_of_week'] = df['date'].dt.dayofweek
    logger.debug("  → day_of_week: extrait")
    
    # Feature 2: Catégorie de température
    df['temp_category'] = pd.cut(
        df['temperature'],
        bins=[-float('inf'), 15, 25, float('inf')],
        labels=['cold', 'moderate', 'hot']
    )
    logger.debug("  → temp_category: discrétisée")
    
    # Feature 3: Indicateur d'irrigation binaire
    df['irrigation_flag'] = (df['irrigation'] == 'ON').astype(int)
    logger.debug("  → irrigation_flag: binarisé")
    
    # Feature 4: Indice de chaleur (température + 0.5 * humidité)
    df['heat_index'] = df['temperature'] + 0.5 * df['humidity']
    logger.debug("  → heat_index: calculé")
    
    # Feature 5: Besoin d'arrosage (règle métier)
    df['water_need'] = ((df['rain_mm'] == 0) & (df['temperature'] > 25)).astype(int)
    logger.debug("  → water_need: règle métier appliquée")
    
    # Feature 6: Saison (approximative par mois)
    df['season'] = df['date'].dt.month.map({
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'autumn', 10: 'autumn', 11: 'autumn'
    })
    logger.debug("  → season: déduite du mois")
    
    features = ['day_of_week', 'temp_category', 'irrigation_flag', 'heat_index', 'water_need', 'season']
    logger.info(f"✓ {len(features)} caractéristiques créées: {features}")
    
    return df

# ============================================================================
# STEP 5: EXPORT
# ============================================================================

def export_data(df_clean, df_features, logger):
    """
    Exporte les données vers des fichiers CSV
    
    Args:
        df_clean: DataFrame nettoyé
        df_features: DataFrame avec caractéristiques
        logger: Instance du logger
    """
    logger.info("=" * 60)
    logger.info("PHASE 5: EXPORT")
    logger.info("=" * 60)
    
    # Création du dossier outputs
    Path("outputs").mkdir(parents=True, exist_ok=True)
    
    # Export données nettoyées
    df_clean.to_csv(CONFIG["paths"]["output_clean"], index=False)
    logger.info(f"✓ Données nettoyées: {CONFIG['paths']['output_clean']} ({len(df_clean)} lignes)")
    
    # Export données enrichies
    df_features.to_csv(CONFIG["paths"]["output_features"], index=False)
    logger.info(f"✓ Données enrichies: {CONFIG['paths']['output_features']} ({len(df_features)} lignes, {len(df_features.columns)} colonnes)")
    
    # Aperçu des données
    logger.debug(f"\nAperçu des données nettoyées:\n{df_clean.head(3)}")
    logger.debug(f"\nAperçu des caractéristiques:\n{df_features[['station', 'temperature', 'water_need', 'heat_index']].head(3)}")

# ============================================================================
# VALIDATIONS FINALES
# ============================================================================

def run_validations(df_clean, df_features, logger):
    """
    Exécute les validations post-traitement
    
    Args:
        df_clean: DataFrame nettoyé
        df_features: DataFrame enrichi
        logger: Instance du logger
    """
    logger.info("=" * 60)
    logger.info("VALIDATIONS FINALES")
    logger.info("=" * 60)
    
    validations_passed = 0
    total_validations = 7
    
    # Validation 1: Données non vides
    assert len(df_clean) > 0, "ERREUR: Dataset clean vide"
    validations_passed += 1
    logger.debug("  ✓ Validation 1/7: Dataset non vide")
    
    # Validation 2: Pas de dates NaT
    assert df_clean['date'].isna().sum() == 0, "ERREUR: Dates manquantes"
    validations_passed += 1
    logger.debug("  ✓ Validation 2/7: Toutes les dates sont valides")
    
    # Validation 3: Pas de température NaN
    assert df_clean['temperature'].isna().sum() == 0, "ERREUR: Températures manquantes"
    validations_passed += 1
    logger.debug("  ✓ Validation 3/7: Aucune température manquante")
    
    # Validation 4: Irrigation valide
    valid_irr = {'ON', 'OFF'}
    assert set(df_clean['irrigation'].unique()).issubset(valid_irr), \
        f"ERREUR: Valeurs irrigation invalides: {df_clean['irrigation'].unique()}"
    validations_passed += 1
    logger.debug("  ✓ Validation 4/7: Irrigation correctement encodée")
    
    # Validation 5: Humidité dans [0, 100]
    assert (df_clean['humidity'] >= 0).all() and (df_clean['humidity'] <= 100).all(), \
        "ERREUR: Humidité hors bornes"
    validations_passed += 1
    logger.debug("  ✓ Validation 5/7: Humidité dans les bornes [0,100]")
    
    # Validation 6: Pas de doublons
    assert df_clean.duplicated().sum() == 0, "ERREUR: Doublons résiduels"
    validations_passed += 1
    logger.debug("  ✓ Validation 6/7: Aucun doublon résiduel")
    
    # Validation 7: Cohérence des tailles
    assert len(df_clean) == len(df_features), "ERREUR: Tailles incohérentes"
    validations_passed += 1
    logger.debug("  ✓ Validation 7/7: Cohérence des tailles")
    
    logger.info(f"✅ Toutes les {validations_passed}/{total_validations} validations sont passées")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Point d'entrée principal du pipeline ETL"""
    
    # Initialisation
    logger = setup_logging(CONFIG["paths"]["log"])
    start_time = datetime.now()
    
    logger.info("=" * 60)
    logger.info("DÉMARRAGE DU PIPELINE ETL MÉTÉO")
    logger.info("=" * 60)
    logger.info(f"Début: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Étape 1: Extraction
        df_raw = extract_data(CONFIG["paths"]["input"], logger)
        
        # Étape 2: Transformation
        df_clean = transform_data(df_raw, logger)
        
        # Étape 3: Rapport qualité
        generate_quality_report(df_raw, df_clean, CONFIG["paths"]["output_report"], logger)
        
        # Étape 4: Feature engineering
        df_features = build_features(df_clean, logger)
        
        # Étape 5: Export
        export_data(df_clean, df_features, logger)
        
        # Validations finales
        run_validations(df_clean, df_features, logger)
        
        # Fin du pipeline
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("=" * 60)
        logger.info("PIPELINE TERMINÉ AVEC SUCCÈS")
        logger.info(f"Fin: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Durée: {duration:.2f} secondes")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"ERREUR FATALE: {str(e)}")
        raise

if __name__ == "__main__":
    main()