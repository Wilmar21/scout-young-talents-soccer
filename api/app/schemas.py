"""
app/schemas.py
Modelos de request y response con Pydantic.
FastAPI los usa para validar inputs y documentar la API automáticamente.
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal


# ── REQUESTS ────────────────────────────────────────────────

class JugadorNuevoRequest(BaseModel):
    """
    Datos de un jugador nuevo para predecir su valor de mercado.
    Los campos marcados como Optional tienen valores por defecto razonables.
    """
    # Obligatorios
    nombre:          str   = Field(..., example="Carlos Rodríguez")
    edad:            float = Field(..., ge=14, le=40, example=21.0)
    main_position:   Literal["Attack","Midfield","Defender","Goalkeeper"] = Field(..., example="Attack")

    # Recomendados
    height:          Optional[float] = Field(None, ge=140, le=220, example=178.0)
    foot:            Optional[Literal["right","left","both"]] = Field("right", example="right")
    is_eu:           Optional[bool]  = Field(False, example=True)
    league:          Optional[str]   = Field("", example="Bundesliga")

    # Rendimiento
    total_goals:     Optional[float] = Field(0,  ge=0, example=15)
    total_assists:   Optional[float] = Field(0,  ge=0, example=8)
    total_minutes:   Optional[float] = Field(0,  ge=0, example=3200)
    n_seasons_active:Optional[float] = Field(1,  ge=1, example=3)
    n_competitions:  Optional[float] = Field(1,  ge=1, example=4)
    total_yellow:    Optional[float] = Field(0,  ge=0, example=5)
    total_red:       Optional[float] = Field(0,  ge=0, example=0)

    # Lesiones
    n_injuries:      Optional[float] = Field(0,  ge=0, example=1)
    total_days_missed:Optional[float]= Field(0,  ge=0, example=14)

    # Selección nacional
    intl_matches:    Optional[float] = Field(0,  ge=0, example=5)
    intl_goals:      Optional[float] = Field(0,  ge=0, example=1)
    is_current_national: Optional[bool] = Field(False, example=False)

    # Historial de mercado (si existe)
    value_max:       Optional[float] = Field(0,  ge=0, example=1_500_000)
    value_growth_pct:Optional[float] = Field(0,  example=2.5)
    n_valuations:    Optional[float] = Field(0,  ge=0, example=6)
    n_transfers:     Optional[float] = Field(0,  ge=0, example=1)
    max_fee:         Optional[float] = Field(0,  ge=0, example=500_000)

    class Config:
        json_schema_extra = {
            "example": {
                "nombre":          "Carlos Rodríguez",
                "edad":            21.0,
                "main_position":   "Attack",
                "height":          178.0,
                "foot":            "right",
                "is_eu":           False,
                "league":          "Bundesliga",
                "total_goals":     15,
                "total_assists":   8,
                "total_minutes":   3200,
                "n_seasons_active":3,
                "n_competitions":  4,
                "total_yellow":    5,
                "total_red":       0,
                "n_injuries":      1,
                "total_days_missed":14,
                "intl_matches":    5,
                "intl_goals":      1,
                "is_current_national": False,
                "value_max":       1_500_000,
                "value_growth_pct":2.5,
                "n_valuations":    6,
                "n_transfers":     1,
                "max_fee":         500_000,
            }
        }


class ProyeccionRequest(BaseModel):
    """Parámetros para la simulación Montecarlo."""
    valor_inicial_eur: float = Field(..., gt=0, example=3_000_000)
    edad:              float = Field(..., ge=14, le=35, example=21.0)
    main_position:     Literal["Attack","Midfield","Defender","Goalkeeper"] = Field(..., example="Attack")
    horizonte_anos:    Literal[1, 3, 5] = Field(3, example=3)
    n_simulaciones:    Optional[int]    = Field(5000, ge=1000, le=10000, example=5000)


# ── RESPONSES ───────────────────────────────────────────────

class PrediccionResponse(BaseModel):
    nombre:               str
    edad:                 float
    posicion:             str
    valor_predicho_eur:   float
    valor_predicho_M:     float
    mensaje:              str


class JugadorExistenteResponse(BaseModel):
    player_id:            int
    nombre:               str
    edad:                 Optional[float]
    posicion:             Optional[str]
    club:                 Optional[str]
    liga:                 Optional[str]
    valor_real_eur:       Optional[float]
    valor_predicho_eur:   Optional[float]
    valor_predicho_M:     Optional[float]
    diferencia_pct:       Optional[float]


class ProyeccionResponse(BaseModel):
    horizonte_anos:   int
    valor_actual_M:   float
    pesimista_M:      float
    base_M:           float
    optimista_M:      float
    media_M:          float
    prob_duplicar_pct:float
    interpretacion:   str


class BusquedaResponse(BaseModel):
    total_encontrados: int
    jugadores:         list


class ModeloInfoResponse(BaseModel):
    modelo:    str
    oof_r2:    float
    oof_rmse:  float
    n_train:   int
    n_features:int
    target:    str
