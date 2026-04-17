"""
app/main.py
API principal — Scout de Jóvenes Promesas de Fútbol
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import numpy as np

from app.schemas import (
    JugadorNuevoRequest, ProyeccionRequest,
    PrediccionResponse, JugadorExistenteResponse,
    ProyeccionResponse, BusquedaResponse, ModeloInfoResponse,
)
from app import predictor


# ── Lifespan: carga los artefactos una sola vez al iniciar
@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load_artifacts()
    yield


app = FastAPI(
    title="Scout de Jóvenes Promesas — API",
    description="""
API para predicción de valor de mercado de jugadores de fútbol jóvenes.

## Endpoints disponibles

### Jugadores existentes
- **GET /jugadores/buscar** — Busca jugadores por nombre en el dataset
- **GET /jugadores/{player_id}** — Obtiene el perfil completo y valor predicho de un jugador

### Predicción para jugadores nuevos
- **POST /predecir** — Calcula el valor de mercado para un jugador con sus stats

### Proyección futura
- **POST /proyectar** — Simulación Montecarlo del valor futuro (1, 3 o 5 años)

### Info del modelo
- **GET /modelo/info** — Métricas y detalles del modelo entrenado
    """,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────
# HEALTH CHECK
# ──────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "api":    "Scout de Jóvenes Promesas",
        "version":"1.0.0",
        "docs":   "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}


# ──────────────────────────────────────────────────────────────
# JUGADORES EXISTENTES
# ──────────────────────────────────────────────────────────────

@app.get(
    "/jugadores/buscar",
    response_model=BusquedaResponse,
    tags=["Jugadores existentes"],
    summary="Buscar jugadores por nombre",
)
def buscar_jugadores(
    nombre: str = Query(..., min_length=2, example="Bellingham"),
    limit:  int = Query(10, ge=1, le=50),
):
    """
    Busca jugadores en el dataset por nombre parcial.
    Retorna hasta `limit` resultados con su valor de mercado real y predicho.
    """
    resultados = predictor.buscar_jugadores(nombre, limit)
    if not resultados:
        raise HTTPException(
            status_code=404,
            detail=f"No se encontraron jugadores con el nombre '{nombre}'"
        )

    jugadores_fmt = []
    for r in resultados:
        val_real = r.get("latest_value") or 0
        val_pred = r.get("pred_market_value_eur") or 0
        jugadores_fmt.append({
            "player_id":          int(r.get("player_id", 0)),
            "nombre":             str(r.get("player_name", "")),
            "edad":               round(float(r.get("age") or 0), 1),
            "posicion":           str(r.get("main_position", "")),
            "club":               str(r.get("current_club_name", "")),
            "liga":               str(r.get("club_league", "")),
            "valor_real_M":       round(val_real / 1e6, 2) if val_real else None,
            "valor_predicho_M":   round(val_pred / 1e6, 2) if val_pred else None,
        })

    return {
        "total_encontrados": len(jugadores_fmt),
        "jugadores":         jugadores_fmt,
    }


@app.get(
    "/jugadores/{player_id}",
    response_model=JugadorExistenteResponse,
    tags=["Jugadores existentes"],
    summary="Perfil completo y predicción de un jugador",
)
def obtener_jugador(player_id: int):
    """
    Retorna el perfil completo de un jugador del dataset junto con
    su valor de mercado real (si existe) y el valor predicho por el modelo.
    También incluye la diferencia porcentual entre ambos.
    """
    jugador = predictor.obtener_jugador(player_id)
    if not jugador:
        raise HTTPException(
            status_code=404,
            detail=f"Jugador con player_id={player_id} no encontrado"
        )

    val_real = jugador.get("latest_value")
    val_pred = jugador.get("pred_market_value_eur")

    dif_pct = None
    if val_real and val_pred and val_real > 0:
        dif_pct = round(((val_pred - val_real) / val_real) * 100, 1)

    return {
        "player_id":           int(jugador.get("player_id", 0)),
        "nombre":              str(jugador.get("player_name", "")),
        "edad":                float(jugador.get("age") or 0),
        "posicion":            str(jugador.get("main_position", "")),
        "club":                str(jugador.get("current_club_name", "")),
        "liga":                str(jugador.get("club_league", "")),
        "valor_real_eur":      float(val_real) if val_real else None,
        "valor_predicho_eur":  float(val_pred) if val_pred else None,
        "valor_predicho_M":    round(float(val_pred) / 1e6, 2) if val_pred else None,
        "diferencia_pct":      dif_pct,
    }


# ──────────────────────────────────────────────────────────────
# PREDICCIÓN PARA JUGADOR NUEVO
# ──────────────────────────────────────────────────────────────

@app.post(
    "/predecir",
    response_model=PrediccionResponse,
    tags=["Predicción"],
    summary="Predecir valor de mercado de un jugador nuevo",
)
def predecir_jugador(body: JugadorNuevoRequest):
    """
    Calcula el valor de mercado actual estimado para un jugador
    a partir de sus estadísticas.

    No es necesario proporcionar todos los campos — los que falten
    se imputarán con la mediana de jugadores de la misma posición.

    El valor predicho está en euros y en millones de euros (M€).
    """
    data = body.model_dump()
    data["main_position"] = data.pop("main_position")
    data["age"]           = data.pop("edad")
    data["height"]        = data.pop("height")

    resultado = predictor.predecir(data)

    val_M = resultado["valor_predicho_M"]

    # Mensaje interpretativo
    if val_M >= 50:
        nivel = "jugador de élite mundial"
    elif val_M >= 20:
        nivel = "jugador de primer nivel europeo"
    elif val_M >= 5:
        nivel = "jugador de alto potencial"
    elif val_M >= 1:
        nivel = "jugador con proyección interesante"
    else:
        nivel = "jugador en desarrollo"

    mensaje = (
        f"{body.nombre} ({body.main_position}, {body.edad}a) — "
        f"valor estimado: {val_M}M€ ({nivel})"
    )

    return {
        "nombre":             body.nombre,
        "edad":               body.edad,
        "posicion":           body.main_position,
        "valor_predicho_eur": resultado["valor_predicho_eur"],
        "valor_predicho_M":   val_M,
        "mensaje":            mensaje,
    }


# ──────────────────────────────────────────────────────────────
# PROYECCIÓN FUTURA (MONTECARLO)
# ──────────────────────────────────────────────────────────────

@app.post(
    "/proyectar",
    response_model=ProyeccionResponse,
    tags=["Proyección futura"],
    summary="Proyectar valor de mercado futuro con simulación Montecarlo",
)
def proyectar_valor(body: ProyeccionRequest):
    """
    Proyecta el valor de mercado futuro usando simulación Montecarlo
    con parámetros calibrados sobre datos históricos reales (2003–2025).

    Retorna tres escenarios:
    - **Pesimista (P10):** solo el 10% de los futuros posibles son peores
    - **Base (P50):** el valor más probable (mediana de simulaciones)
    - **Optimista (P90):** solo el 10% de los futuros posibles son mejores

    También incluye la probabilidad de que el jugador duplique su valor.
    """
    resultado = predictor.simular_montecarlo(
        valor_inicial=body.valor_inicial_eur,
        edad=body.edad,
        posicion=body.main_position,
        horizonte=body.horizonte_anos,
        n_sim=body.n_simulaciones,
    )

    val_act_M = round(body.valor_inicial_eur / 1e6, 2)
    base_M    = resultado["base_M"]
    upside    = round((base_M - val_act_M) / val_act_M * 100, 1) if val_act_M > 0 else 0

    # Interpretación automática
    if upside > 100:
        interp = f"Alto potencial de crecimiento (+{upside}% en escenario base). Ventana de oportunidad clara."
    elif upside > 30:
        interp = f"Crecimiento moderado esperado (+{upside}%). Perfil con recorrido de valorización."
    elif upside > 0:
        interp = f"Crecimiento conservador (+{upside}%). Jugador cerca de su pico de valor."
    else:
        interp = f"Posible declive en el valor ({upside}%). Evaluar si el momento de compra ya pasó."

    return {
        "horizonte_anos":    body.horizonte_anos,
        "valor_actual_M":    val_act_M,
        "pesimista_M":       resultado["pesimista_M"],
        "base_M":            base_M,
        "optimista_M":       resultado["optimista_M"],
        "media_M":           resultado["media_M"],
        "prob_duplicar_pct": resultado["prob_duplicar"],
        "interpretacion":    interp,
    }


# ──────────────────────────────────────────────────────────────
# INFO DEL MODELO
# ──────────────────────────────────────────────────────────────

@app.get(
    "/modelo/info",
    response_model=ModeloInfoResponse,
    tags=["Modelo"],
    summary="Información y métricas del modelo entrenado",
)
def info_modelo():
    """
    Retorna los detalles del modelo predictivo:
    métricas de validación cruzada OOF, número de features y target.
    """
    m = predictor.metricas_modelo()
    return {
        "modelo":     m.get("modelo", "LightGBM"),
        "oof_r2":     m.get("oof_r2", 0.9708),
        "oof_rmse":   m.get("oof_rmse", 0.2831),
        "n_train":    m.get("n_train", 17035),
        "n_features": m.get("n_features", 32),
        "target":     m.get("target", "log_market_value"),
    }
