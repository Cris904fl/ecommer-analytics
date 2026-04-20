# 📦 E-Commerce Revenue Analytics

Pipeline completo de análisis de ventas y KPIs de negocio para e-commerce. Incluye ETL, detección de anomalías con ML, forecasting de revenue y una API REST lista para producción.

---

## 🏗️ Arquitectura

```
data/raw/          →  [Airflow DAG]  →  data/processed/
                          │
                    ┌─────┴──────┐
                    │            │
              Cleaner.py    IsolationForest
              KPIs.py       + Revenue Forecaster
                    │            │
                    └─────┬──────┘
                          │
                    FastAPI REST API
                    /kpis/*  /ml/*
```

## 📁 Estructura del Proyecto

```
ecommerce-analytics/
├── dags/
│   └── ecommerce_pipeline_dag.py     # Airflow DAG (orquestación diaria)
├── src/
│   ├── pipeline/
│   │   ├── cleaner.py                # ETL: dedup, imputación, métricas derivadas
│   │   └── kpis.py                   # KPIs: revenue, crecimiento MoM/QoQ, churn
│   ├── ml/
│   │   └── models.py                 # IsolationForest + Revenue Forecaster
│   └── api/
│       └── main.py                   # FastAPI: endpoints REST
├── data/
│   ├── raw/orders_raw.csv            # Datos crudos de ejemplo (5,050 órdenes)
│   └── processed/                    # Output del pipeline de limpieza
├── docker/
│   ├── docker-compose.yml            # Airflow + PostgreSQL + API
│   └── Dockerfile.api                # Imagen Docker de la API
├── tests/
│   └── test_pipeline.py              # Tests unitarios (cleaning, KPIs, ML)
├── scripts/
│   └── generate_sample_data.py       # Generador de datos de prueba
├── .github/workflows/
│   └── ci.yml                        # CI/CD: lint → tests → docker build
└── requirements.txt
```

## ⚙️ Tech Stack

| Capa | Tecnología |
|---|---|
| Orquestación | Apache Airflow 2.8 |
| ETL & Análisis | Python 3.11, Pandas, NumPy |
| Machine Learning | scikit-learn (IsolationForest, LinearRegression) |
| API REST | FastAPI + Uvicorn |
| Contenedores | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Tests | Pytest + pytest-cov |

## 🚀 Inicio Rápido

### Opción A — Local (sin Docker)

```bash
git clone https://github.com/tu-usuario/ecommerce-analytics.git
cd ecommerce-analytics

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. Generar datos de ejemplo
python scripts/generate_sample_data.py

# 2. Limpiar y transformar
python -m src.pipeline.cleaner

# 3. Entrenar modelos ML
python -m src.ml.models

# 4. Levantar la API
uvicorn src.api.main:app --reload
# → http://localhost:8000/docs
```

### Opción B — Docker Compose (Airflow + API)

```bash
docker-compose -f docker/docker-compose.yml up -d
```

| Servicio | URL | Credenciales |
|---|---|---|
| Airflow UI | http://localhost:8080 | admin / admin |
| API Docs | http://localhost:8000/docs | — |

## 📊 KPIs del Pipeline

El módulo `src/pipeline/kpis.py` calcula:

- **Revenue total y neto** (bruto - descuentos)
- **Crecimiento MoM** (Month-over-Month) y **QoQ** (Quarter-over-Quarter)
- **AOV** — Average Order Value
- **Revenue por categoría, canal y región**
- **Repeat Purchase Rate** — tasa de clientes recurrentes

## 🤖 Modelos ML

### 1. Detección de Anomalías — `IsolationForest`
Detecta órdenes inusuales en base a precio, cantidad y descuento. Útil para identificar fraude o errores de carga.

```bash
GET /ml/anomalies?top_n=20
```

### 2. Forecasting de Revenue — `LinearRegression + Fourier`
Predice el revenue neto semanal para las próximas N semanas usando features de tendencia y estacionalidad.

```bash
GET /ml/forecast?weeks=12
```

## 🔌 Endpoints Principales

```
GET  /                    → health check
GET  /kpis/summary        → resumen ejecutivo
GET  /kpis/monthly        → revenue mensual + MoM growth
GET  /kpis/quarterly      → revenue trimestral + QoQ growth
GET  /kpis/by-category    → desglose por categoría
GET  /kpis/by-channel     → desglose por canal de venta
GET  /kpis/by-region      → desglose por región
GET  /ml/forecast         → forecast de revenue (N semanas)
GET  /ml/anomalies        → órdenes anómalas detectadas
POST /pipeline/run        → ejecuta el pipeline completo
```

## 🧪 Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

## 🔄 CI/CD

El workflow `.github/workflows/ci.yml` ejecuta en cada push:

1. **Lint** — Black, isort, Flake8
2. **Tests** — pytest con reporte de cobertura
3. **Integration** — corre el pipeline completo de punta a punta
4. **Docker Build** — construye y publica imagen (solo rama `main`)

---

> Proyecto desarrollado como demostración de un pipeline de Data Analytics end-to-end para e-commerce, con foco en calidad de datos, ML aplicado y buenas prácticas de ingeniería.
