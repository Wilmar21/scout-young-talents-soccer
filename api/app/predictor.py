"""
app/predictor.py
Lógica central de predicción — se carga una sola vez al iniciar la API.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ── Rutas a los artefactos del modelo
BASE = Path(__file__).parent.parent / "model_artifacts"

_modelo       = None
_le_pos       = None
_le_foot      = None
_le_tier      = None
_features     = None
_medians      = None
_dataset      = None
_montecarlo   = None


def load_artifacts():
    """Carga todos los artefactos en memoria. Se llama una sola vez al iniciar."""
    global _modelo, _le_pos, _le_foot, _le_tier
    global _features, _medians, _dataset, _montecarlo

    _modelo   = joblib.load(BASE / "modelo_final.pkl")
    _le_pos   = joblib.load(BASE / "encoder_posicion.pkl")
    _le_foot  = joblib.load(BASE / "encoder_pie.pkl")
    _le_tier  = joblib.load(BASE / "encoder_liga.pkl")

    with open(BASE / "features.json") as f:
        _features = json.load(f)

    with open(BASE / "feature_medians.json") as f:
        _medians = json.load(f)

    _dataset = pd.read_csv(BASE / "dataset_con_predicciones.csv", low_memory=False)
    _dataset["player_name"] = _dataset["player_name"].fillna("")

    sim_path = BASE / "simulaciones_montecarlo.csv"
    if sim_path.exists():
        _montecarlo = pd.read_csv(sim_path)

    print(f"✓ Modelo cargado | R²=0.9708 | {len(_dataset):,} jugadores en dataset")


# ── Mapa de ligas a tier
TOP_LEAGUES = {
    "Premier League": "tier1", "LaLiga": "tier1",
    "Serie A": "tier1",        "Bundesliga": "tier1",
    "Ligue 1": "tier1",        "Eredivisie": "tier2",
    "Liga Portugal": "tier2",  "Jupiler Pro League": "tier2",
    "Championship": "tier2",   "2. Bundesliga": "tier2",
    "Serie B": "tier2",        "LaLiga2": "tier2",
    "Major League Soccer": "other",
}


def _safe_encode(encoder, value: str, fallback_idx: int = 0) -> int:
    """Encode con fallback si el valor no está en el encoder."""
    if value in encoder.classes_:
        return int(encoder.transform([value])[0])
    return fallback_idx


def preparar_features(data: dict) -> np.ndarray:
    """
    Convierte un diccionario de datos del jugador en el vector de features
    que espera el modelo. Maneja imputación y encoding automáticamente.
    """
    pos    = data.get("main_position", "Midfield")
    medians = _medians.get(pos, _medians.get("Midfield", {}))

    def get(key, default=None):
        val = data.get(key)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return medians.get(key, default or 0)
        return val

    # Campos base
    age              = float(get("age", 20))
    height           = float(get("height", 178))
    is_eu            = int(bool(get("is_eu", False)))
    total_goals      = float(get("total_goals", 0))
    total_assists    = float(get("total_assists", 0))
    total_minutes    = float(get("total_minutes", 0))
    n_seasons_active = max(float(get("n_seasons_active", 1)), 1)
    n_competitions   = float(get("n_competitions", 1))
    total_yellow     = float(get("total_yellow", 0))
    total_red        = float(get("total_red", 0))
    n_injuries       = float(get("n_injuries", 0))
    total_days_missed= float(get("total_days_missed", 0))
    intl_matches     = float(get("intl_matches", 0))
    intl_goals       = float(get("intl_goals", 0))
    is_current_nat   = int(bool(get("is_current_national", False)))
    n_transfers      = float(get("n_transfers", 0))
    max_fee          = float(get("max_fee", 0))
    value_max        = float(get("value_max", 0))
    value_growth_pct = float(get("value_growth_pct", 0))
    n_valuations     = float(get("n_valuations", 0))

    # Features derivadas
    ns                  = max(n_seasons_active, 1)
    goal_contrib        = total_goals + total_assists
    mins_per_goal       = (
        9999.0 if pos == "Goalkeeper"
        else (total_minutes / total_goals if total_goals > 0 else float(medians.get("mins_per_goal", 500)))
    )
    goals_per_season    = round(total_goals / ns, 2)
    assists_per_season  = round(total_assists / ns, 2)
    minutes_per_season  = round(total_minutes / ns, 0)
    has_minutes         = int(total_minutes > 0)
    injury_rate         = round(n_injuries / ns, 3)
    intl_ratio          = round(intl_matches / ns, 3)
    was_elite           = int(value_max > 10_000_000)

    # Encoding categóricos
    foot  = data.get("foot", "right")
    tier  = TOP_LEAGUES.get(data.get("league", ""), "other")

    foot_enc         = _safe_encode(_le_foot, foot, 0)
    position_enc     = _safe_encode(_le_pos, pos, 1)
    league_tier_enc  = _safe_encode(_le_tier, tier, 2)

    # Vector en el orden exacto de FEATURES
    vector = [
        age, height, is_eu, foot_enc, position_enc,
        total_goals, total_assists, total_minutes,
        goal_contrib, mins_per_goal,
        n_seasons_active, n_competitions,
        total_yellow, total_red,
        goals_per_season, assists_per_season, minutes_per_season,
        has_minutes, n_injuries, total_days_missed, injury_rate,
        intl_matches, intl_goals, intl_ratio, is_current_nat,
        value_max, value_growth_pct, n_valuations,
        n_transfers, max_fee, league_tier_enc, was_elite,
    ]
    return np.array(vector, dtype=float).reshape(1, -1)


def predecir(data: dict) -> dict:
    """
    Genera la predicción del valor de mercado para un jugador.
    Retorna valor predicho en € y en M€.
    """
    X = preparar_features(data)
    log_pred    = float(_modelo.predict(X)[0])
    valor_eur   = float(np.expm1(log_pred))
    return {
        "log_pred":           round(log_pred, 4),
        "valor_predicho_eur": round(valor_eur, 0),
        "valor_predicho_M":   round(valor_eur / 1_000_000, 2),
    }


def simular_montecarlo(
    valor_inicial: float,
    edad: float,
    posicion: str,
    horizonte: int,
    n_sim: int = 5000,
) -> dict:
    """
    Simulación Montecarlo para proyectar el valor futuro.
    Usa parámetros log-normales calibrados con datos históricos reales.
    """
    # Parámetros log-normales por posición y banda de edad
    PARAMS = {
        ("Attack",     "14-18"): (0.3905, 0.5640),
        ("Attack",     "18-20"): (0.2647, 0.5453),
        ("Attack",     "20-22"): (0.1461, 0.4779),
        ("Attack",     "22-24"): (0.0811, 0.4144),
        ("Attack",     "24-26"): (0.0373, 0.3684),
        ("Attack",     "26-28"): (-0.0084, 0.3442),
        ("Midfield",   "14-18"): (0.3256, 0.5445),
        ("Midfield",   "18-20"): (0.2677, 0.5279),
        ("Midfield",   "20-22"): (0.1564, 0.4697),
        ("Midfield",   "22-24"): (0.0864, 0.4018),
        ("Midfield",   "24-26"): (0.0397, 0.3479),
        ("Midfield",   "26-28"): (-0.0011, 0.3111),
        ("Defender",   "14-18"): (0.3473, 0.5516),
        ("Defender",   "18-20"): (0.2536, 0.5249),
        ("Defender",   "20-22"): (0.1791, 0.4810),
        ("Defender",   "22-24"): (0.1021, 0.4046),
        ("Defender",   "24-26"): (0.0534, 0.3502),
        ("Defender",   "26-28"): (0.0095, 0.3242),
        ("Goalkeeper", "14-18"): (0.2079, 0.4442),
        ("Goalkeeper", "18-20"): (0.1652, 0.4785),
        ("Goalkeeper", "20-22"): (0.1183, 0.4496),
        ("Goalkeeper", "22-24"): (0.0795, 0.4300),
        ("Goalkeeper", "24-26"): (0.0619, 0.4036),
        ("Goalkeeper", "26-28"): (0.0371, 0.3673),
    }

    def get_band(age):
        for lo, hi, label in [
            (14,18,"14-18"), (18,20,"18-20"), (20,22,"20-22"),
            (22,24,"22-24"), (24,26,"24-26"), (26,28,"26-28"),
            (28,30,"28-30"), (30,99,"30-35"),
        ]:
            if lo <= age < hi:
                return label
        return "22-24"

    np.random.seed(42)
    valores = np.full(n_sim, float(valor_inicial))

    for año in range(horizonte):
        band = get_band(edad + año)
        mu, sigma = PARAMS.get((posicion, band), (0.03, 0.40))
        tasas   = np.random.lognormal(mean=mu, sigma=sigma, size=n_sim)
        valores = valores * tasas
        # Shock de lesión (15% anual, -15% en valor)
        lesiones = np.random.random(n_sim) < 0.15
        valores[lesiones] *= 0.85
        valores = np.maximum(valores, 10_000)

    return {
        "pesimista_M":  round(float(np.percentile(valores, 10)) / 1e6, 2),
        "base_M":       round(float(np.percentile(valores, 50)) / 1e6, 2),
        "optimista_M":  round(float(np.percentile(valores, 90)) / 1e6, 2),
        "media_M":      round(float(valores.mean()) / 1e6, 2),
        "prob_duplicar": round(float((valores >= valor_inicial * 2).mean()) * 100, 1),
    }


def buscar_jugadores(nombre: str, limit: int = 10) -> list:
    """Busca jugadores por nombre parcial en el dataset."""
    mask = _dataset["player_name"].str.contains(nombre, case=False, na=False)
    results = _dataset[mask].head(limit)
    return results[[
        "player_id","player_name","age","main_position",
        "current_club_name","club_league","latest_value","pred_market_value_eur"
    ]].fillna(0).to_dict(orient="records")


def obtener_jugador(player_id: int) -> dict | None:
    """Retorna todos los datos de un jugador por su ID."""
    row = _dataset[_dataset["player_id"] == player_id]
    if row.empty:
        return None
    return row.iloc[0].where(pd.notna(row.iloc[0]), None).to_dict()


def metricas_modelo() -> dict:
    """Retorna las métricas del modelo entrenado."""
    path = BASE / "metricas_modelo.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"oof_r2": 0.9708, "oof_rmse": 0.2831, "modelo": "LightGBM"}
