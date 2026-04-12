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
│   └── 03_simulacion_montecarlo.ipynb  ← en desarrollo
│
├── src/                               ← pipelines de producción
│   ├── pipeline_modelo.py             ← en desarrollo
│   └── integracion_llm.py             ← en desarrollo
│
├── outputs/
│   ├── paso1/                         ← plots EDA + dataset curado
│   └── paso2/                         ← modelo, ranking, predicciones
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
- Construcción del dataset maestro (7 merges encadenados)
- 10 visualizaciones EDA
- Dataset curado: **31,405 jugadores × 48 variables**

**Outputs:**
- `outputs/paso1/scouting_dataset_v1.csv`
- `outputs/paso1/*.png` (10 plots)

---

### ✅ Paso 2 — Modelo Predictivo + MLflow
`notebooks/02_modelo_predictivo.ipynb`

- Feature engineering (31 features finales)
- Entrenamiento LightGBM y XGBoost con 5-Fold CV
- Comparación de modelos por métricas OOF
- Análisis de residuos
- Interpretabilidad con SHAP
- Ranking de 31,405 jugadores por valor predicho
- Tracking de experimentos con MLflow

**Variable objetivo:** `log(1 + market_value_eur)`

**Outputs:**
- `outputs/paso2/ranking_promesas.csv`
- `outputs/paso2/scouting_dataset_v2_con_predicciones.csv`
- `outputs/paso2/*.png` (8 plots)
- `mlruns/` (experimentos MLflow)

---

### 🔜 Paso 3 — Simulación Montecarlo
`notebooks/03_simulacion_montecarlo.ipynb`

- Proyección de valor de mercado a 3 y 5 años
- 10,000 simulaciones por jugador
- Escenarios optimista, base y pesimista
- Intervalos de confianza por perfil

---

### 🔜 Paso 4 — Integración LLM
`src/integracion_llm.py`

- Recomendaciones automáticas basadas en el ranking
- Justificación en lenguaje natural de cada candidato
- Generación de reportes de scouting

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

# 4. Colocar los CSV en la carpeta data/

# 5. Ejecutar los notebooks en orden
```

---

## Visualizar experimentos MLflow

```bash
mlflow ui
# Abrir en el navegador: http://127.0.0.1:5000
```

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
| `scipy` | Simulación Montecarlo (Paso 3) |
| `anthropic` | Integración LLM (Paso 4) |

---

## Estado del proyecto

| Paso | Estado |
|---|---|
| Extracción y exploración | ✅ Completado |
| Modelo predictivo + MLflow | ✅ Completado |
| Simulación Montecarlo | 🔄 En desarrollo |
| Integración LLM | 🔄 Pendiente |
