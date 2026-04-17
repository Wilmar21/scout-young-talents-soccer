# ⚽ Scout de Jóvenes Promesas de Fútbol
### Predicción de valor de mercado futuro mediante Machine Learning

---

## Descripción

Sistema de scouting basado en datos históricos de Transfermarkt que identifica
jugadores jóvenes (≤26 años) con mayor potencial de convertirse en estrellas,
usando el **valor de mercado futuro** como proxy de ese potencial.

El proyecto integra extracción y exploración de datos, modelado predictivo,
simulación de escenarios futuros y recomendaciones automatizadas mediante LLM.

---

## Estructura del proyecto

```
scout_promesas/
│
├── data/                              ← CSV fuente (no incluidos en el repo)
│   ├── player_profiles.csv
│   ├── player_market_value.csv
│   ├── player_latest_market_value.csv
│   ├── player_performances.csv
│   ├── player_injuries.csv
│   ├── player_national_performances.csv
│   ├── player_teammates_played_with.csv
│   ├── transfer_history.csv
│   ├── team_details.csv
│   ├── team_children.csv
│   └── team_competitions_seasons.csv
│
├── notebooks/
│   ├── 01_extraccion_exploracion.ipynb
│   ├── 02_modelo_predictivo.ipynb
│   ├── 03_simulacion_montecarlo.ipynb
│   └── 04_integracion_llm.ipynb
│
├── outputs/
│   ├── paso1/                         ← plots EDA + dataset curado
│   ├── paso2/                         ← modelo, ranking, predicciones
│   ├── paso3/                         ← simulaciones Montecarlo
│   └── paso4/                         ← reportes de scouting (.md)
│
├── mlruns/                            ← experimentos MLflow (local)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

**Fuente:** [Transfermarkt via Kaggle](https://www.kaggle.com/datasets/xfkzujqjvx97n/football-datasets)

| Archivo | Filas | Descripción |
|---|---|---|
| `player_profiles.csv` | 92,671 | Perfil demográfico y contractual |
| `player_market_value.csv` | 901,429 | Histórico de valoraciones (2003–2025) |
| `player_latest_market_value.csv` | 69,441 | Último valor registrado por jugador |
| `player_performances.csv` | 1,878,719 | Stats por temporada y competición |
| `player_injuries.csv` | 143,195 | Historial de lesiones |
| `player_national_performances.csv` | 92,701 | Participación en selecciones |
| `transfer_history.csv` | 1,101,440 | Historial de transferencias |
| `team_details.csv` | 2,176 | Info de clubes y ligas |

> Los archivos CSV no se incluyen en el repositorio por su tamaño.
> Descárgalos desde el enlace de Kaggle y colócalos en la carpeta `data/`.

---

## Pipeline del proyecto

### ✅ Paso 1 — Extracción, Exploración y Visualización
`notebooks/01_extraccion_exploracion.ipynb`

- Carga y exploración de los 11 archivos fuente
- Construcción del dataset maestro con 7 merges encadenados
- 10 visualizaciones EDA (calidad de datos, distribuciones, correlaciones, top 25 promesas)
- Dataset curado: **31,405 jugadores × 48 variables**

**Outputs:**
- `outputs/paso1/scouting_dataset_v1.csv`
- `outputs/paso1/*.png` (10 plots)

---

### ✅ Paso 2 — Modelo Predictivo + MLflow
`notebooks/02_modelo_predictivo.ipynb`

- Feature engineering: 31 features finales, imputación, encoding, variables derivadas
- Entrenamiento LightGBM y XGBoost con validación cruzada 5-Fold
- Métricas OOF (Out-Of-Fold): cada jugador predicho por un modelo que nunca lo vio
- Análisis de residuos y comparación de modelos
- Interpretabilidad con SHAP values
- Ranking completo de 31,405 jugadores por valor predicho
- Tracking de experimentos con MLflow

**Variable objetivo:** `log(1 + market_value_eur)`

**Outputs:**
- `outputs/paso2/ranking_promesas.csv`
- `outputs/paso2/scouting_dataset_v2_con_predicciones.csv`
- `outputs/paso2/*.png` (8 plots)
- `mlruns/` (experimentos MLflow — ver con `mlflow ui`)

---

### ✅ Paso 3 — Simulación Montecarlo
`notebooks/03_simulacion_montecarlo.ipynb`

- Calibración de distribución log-normal con 394,243 transiciones reales de valor (2003–2025)
- Parámetros por posición × banda de edad (14-18, 18-20, 20-22, 22-24, 24-26, 26-28, 28-30)
- 10,000 simulaciones por jugador a 1, 3 y 5 años
- Escenarios pesimista (P10), base (P50) y optimista (P90)
- Shock estocástico de lesión: 15% de probabilidad anual
- Ranking de "joyas ocultas": alto upside, valor actual moderado

**Outputs:**
- `outputs/paso3/simulaciones_montecarlo.csv` (proyecciones top 100)
- `outputs/paso3/joyas_ocultas_top50.csv`
- `outputs/paso3/*.png` (5 plots)

---

### ✅ Paso 4 — Integración LLM
`notebooks/04_integracion_llm.ipynb`

- LLM: **Llama 3.3 70B** vía **Groq API** (gratuito, sin tarjeta de crédito)
- Tres tipos de análisis: reporte individual, comparativa por posición, joyas ocultas
- Modo interactivo: `analizar_jugador("nombre")` para consultas en tiempo real
- Informe consolidado en Markdown con todos los reportes

**Outputs:**
- `outputs/paso4/reporte_[jugador].md`
- `outputs/paso4/comparativa_[posicion].md`
- `outputs/paso4/joyas_ocultas_analisis.md`
- `outputs/paso4/informe_completo_scouting.md`

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/scout-promesas.git
cd scout-promesas

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Colocar los 11 CSV en la carpeta data/

# 5. Ejecutar los notebooks en orden (01 → 02 → 03 → 04)
```

---

## Configuración del LLM (Paso 4)

El Paso 4 usa la API gratuita de Groq. Para obtener la key:

1. Ve a https://console.groq.com/keys
2. Crea una cuenta gratuita (sin tarjeta)
3. Genera una API key
4. Configúrala como variable de entorno:

```bash
# Windows PowerShell
$env:GROQ_API_KEY = "gsk_..."

# Mac/Linux
export GROQ_API_KEY="gsk_..."

# PyCharm: Run → Edit Configurations → Environment variables
```

---

## Visualizar experimentos MLflow

```bash
# Desde la carpeta raíz del proyecto
mlflow ui
# Abrir: http://127.0.0.1:5000
```

Registra automáticamente parámetros, métricas OOF por fold y artefactos
de LightGBM y XGBoost para comparación de experimentos.

---

## Dependencias principales

| Librería | Uso |
|---|---|
| `pandas` / `numpy` | Manipulación de datos |
| `matplotlib` / `seaborn` | Visualización |
| `lightgbm` / `xgboost` | Modelos predictivos |
| `scikit-learn` | Validación cruzada y métricas |
| `shap` | Interpretabilidad del modelo |
| `mlflow` | Tracking de experimentos |
| `scipy` | Simulación Montecarlo |
| `groq` | Integración LLM (Llama 3.3 70B) |

---

## Estado del proyecto

| Paso | Estado | Descripción |
|---|---|---|
| Extracción y exploración | ✅ Completado | 31,405 jugadores × 48 variables |
| Modelo predictivo + MLflow | ✅ Completado | LightGBM/XGBoost + SHAP + OOF CV |
| Simulación Montecarlo | ✅ Completado | Proyección 1/3/5 años · 10K sims/jugador |
| Integración LLM | ✅ Completado | Llama 3.3 70B vía Groq (gratuito) |



   
# Scout de Jóvenes Promesas — API
 
API REST para predicción de valor de mercado de jugadores de fútbol jóvenes.
Construida con **FastAPI** · Modelo **LightGBM** · R² OOF = 0.97
 
---
 
## Estructura
 
```
api/
├── app/
│   ├── main.py          ← endpoints FastAPI
│   ├── predictor.py     ← lógica de predicción y simulación
│   └── schemas.py       ← modelos Pydantic (request/response)
├── model_artifacts/
│   ├── modelo_final.pkl
│   ├── encoder_posicion.pkl
│   ├── encoder_pie.pkl
│   ├── encoder_liga.pkl
│   ├── features.json
│   ├── feature_medians.json
│   ├── metricas_modelo.json
│   └── dataset_con_predicciones.csv
└── requirements.txt
```
 
---
 
## Instalación y ejecución local
 
```bash
cd api
pip install -r requirements.txt
uvicorn app.main:app --reload
```
 
Documentación interactiva disponible en: **http://localhost:8000/docs**
 
---
 
## Endpoints
 
### `GET /jugadores/buscar?nombre={nombre}`
Busca jugadores por nombre parcial en el dataset.
 
**Parámetros:**
- `nombre` (str, requerido) — fragmento del nombre del jugador
- `limit` (int, opcional, default=10) — máximo de resultados
**Ejemplo:**
```bash
curl "http://localhost:8000/jugadores/buscar?nombre=Bellingham"
```
```json
{
  "total_encontrados": 2,
  "jugadores": [
    {
      "player_id": 581678,
      "nombre": "Jude Bellingham",
      "edad": 21.5,
      "posicion": "Midfield",
      "club": "Real Madrid",
      "liga": "LaLiga",
      "valor_real_M": 180.0,
      "valor_predicho_M": 140.3
    }
  ]
}
```
 
---
 
### `GET /jugadores/{player_id}`
Perfil completo y valor predicho de un jugador por su ID.
 
**Ejemplo:**
```bash
curl "http://localhost:8000/jugadores/581678"
```
```json
{
  "player_id": 581678,
  "nombre": "Jude Bellingham",
  "edad": 21.5,
  "posicion": "Midfield",
  "club": "Real Madrid",
  "liga": "LaLiga",
  "valor_real_eur": 180000000.0,
  "valor_predicho_eur": 140300000.0,
  "valor_predicho_M": 140.3,
  "diferencia_pct": -22.1
}
```
 
---
 
### `POST /predecir`
Calcula el valor de mercado estimado para un jugador nuevo.
Los campos no proporcionados se imputan con la mediana de su posición.
 
**Body (JSON):**
 
| Campo | Tipo | Requerido | Descripción |
|---|---|---|---|
| `nombre` | string | ✓ | Nombre del jugador |
| `edad` | float | ✓ | Edad en años (ej: 21.5) |
| `main_position` | string | ✓ | Attack / Midfield / Defender / Goalkeeper |
| `height` | float | — | Altura en cm |
| `foot` | string | — | right / left / both |
| `is_eu` | bool | — | Ciudadano de la UE |
| `league` | string | — | Liga actual (ej: "Bundesliga") |
| `total_goals` | float | — | Goles en toda la carrera |
| `total_assists` | float | — | Asistencias en toda la carrera |
| `total_minutes` | float | — | Minutos jugados totales |
| `n_seasons_active` | float | — | Temporadas activas |
| `n_injuries` | float | — | Número de lesiones |
| `total_days_missed` | float | — | Días de baja por lesiones |
| `intl_matches` | float | — | Partidos con selección nacional |
| `value_max` | float | — | Valor de mercado máximo histórico (€) |
| `value_growth_pct` | float | — | Multiplicador de crecimiento histórico |
| `n_valuations` | float | — | Número de valoraciones registradas |
| `n_transfers` | float | — | Número de transferencias |
| `max_fee` | float | — | Fee máximo pagado por el jugador (€) |
 
**Ejemplo:**
```bash
curl -X POST http://localhost:8000/predecir \
  -H "Content-Type: application/json" \
  -d '{
    "nombre": "Carlos Rodriguez",
    "edad": 21,
    "main_position": "Attack",
    "height": 178,
    "foot": "right",
    "is_eu": false,
    "league": "Bundesliga",
    "total_goals": 15,
    "total_assists": 8,
    "total_minutes": 3200,
    "n_seasons_active": 3,
    "n_injuries": 1,
    "intl_matches": 5,
    "value_max": 1500000,
    "n_transfers": 1,
    "max_fee": 500000
  }'
```
```json
{
  "nombre": "Carlos Rodriguez",
  "edad": 21.0,
  "posicion": "Attack",
  "valor_predicho_eur": 1280000.0,
  "valor_predicho_M": 1.28,
  "mensaje": "Carlos Rodriguez (Attack, 21.0a) — valor estimado: 1.28M€ (jugador con proyección interesante)"
}
```
 
---
 
### `POST /proyectar`
Simulación Montecarlo del valor futuro a 1, 3 o 5 años.
Calibrada con 394,243 transiciones reales de valor (2003–2025).
 
**Body (JSON):**
 
| Campo | Tipo | Requerido | Descripción |
|---|---|---|---|
| `valor_inicial_eur` | float | ✓ | Valor de mercado actual en € |
| `edad` | float | ✓ | Edad actual |
| `main_position` | string | ✓ | Attack / Midfield / Defender / Goalkeeper |
| `horizonte_anos` | int | ✓ | 1, 3 o 5 |
| `n_simulaciones` | int | — | Entre 1000 y 10000 (default: 5000) |
 
**Ejemplo:**
```bash
curl -X POST http://localhost:8000/proyectar \
  -H "Content-Type: application/json" \
  -d '{
    "valor_inicial_eur": 1280000,
    "edad": 21,
    "main_position": "Attack",
    "horizonte_anos": 3
  }'
```
```json
{
  "horizonte_anos": 3,
  "valor_actual_M": 1.28,
  "pesimista_M": 0.62,
  "base_M": 1.62,
  "optimista_M": 4.35,
  "media_M": 2.1,
  "prob_duplicar_pct": 27.3,
  "interpretacion": "Crecimiento moderado esperado (+26.6%). Perfil con recorrido de valorización."
}
```
 
---
 
### `GET /modelo/info`
Métricas del modelo entrenado.
 
```bash
curl http://localhost:8000/modelo/info
```
```json
{
  "modelo": "LightGBM",
  "oof_r2": 0.9708,
  "oof_rmse": 0.2831,
  "n_train": 17035,
  "n_features": 32,
  "target": "log_market_value"
}
```
 
---
 
## Deploy gratuito en Render
 
1. Sube la carpeta `api/` a un repositorio de GitHub
2. Ve a https://render.com → New Web Service
3. Conecta el repo → selecciona la carpeta `api/`
4. Configura:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Deploy → URL pública disponible en ~2 minutos
> **Nota:** el plan gratuito de Render hiberna el servicio tras 15 minutos
> de inactividad. La primera request después de la hibernación tarda ~30
> segundos en responder (cold start).