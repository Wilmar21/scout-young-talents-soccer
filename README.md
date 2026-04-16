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
