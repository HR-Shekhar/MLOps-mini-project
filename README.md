# SCSE3040 MLOps Mini Project --- Group G109

## Project: MP137 --- Village Milk Collection (Patna)

**Client:** Patna District Dairy Cooperative  
**Target Variable:** `litres_per_day` (Daily milk collection in litres)  
**Evaluation Metric:** MAE (Mean Absolute Error, in litres)  
**Faculty:** Pulkit Dwivedi | **Batch:** 11  

---

### Group Members

| Enrolment No. | Name | GitHub Username | Role |
|---|---|---|---|
| S24CSEU1007 | Yogya Mittal | yoge2006 | Data Lead
| **S24CSEU1016** | Himanshu Shekhar | | Model Lead |
| **S24CSEU1025** | Aarav Bansal | | Service Lead |
| **S24CSEU1040** | Shivangi Duggal | | Ops Lead |

---

### Project Overview

A dairy cooperative near Patna collects milk daily from village collection centres. In order to optimize logistics, prevent milk spoilage, and reduce fuel waste, truck capacity and chilling infrastructure must be booked 24 hours in advance. This project implements a complete, reproducible production MLOps pipeline to forecast daily collection volume.

### Pipeline Architecture & Modules
- **Data Engineering & DVC:** Tracks data versions and audits data quality issues in `train.csv`.
- **Model Development & Tracking:** Tracks experiments in MLflow using SQLite backend (`sqlite:///mlflow.db`).
- **Inference Microservice:** High-performance FastAPI service containerized with Docker.
- **Continuous Integration:** GitHub Actions verifying test suites on every push.
- **Kubernetes:** Manifests for zero-downtime deployment, service routing, and health probes.
- **Drift Monitoring:** PSI computation and automated retraining pipeline for continuous delivery.

### Repository Layout
```
├── README.md                           # Group and project overview
├── requirements.txt                    # Pinned project dependencies
├── params.yaml                         # Central configuration parameters
├── src/                                # Source code package
│   └── milk_collection/
│       ├── __init__.py
│       ├── data_audit.py               # Audits the 7 data issues
│       ├── clean.py                    # Data cleaning pipeline
│       └── utils.py                    # Config loading and SHA-256 helpers
├── tests/                              # Automated test suites
│   ├── test_audit.py
│   └── test_clean.py
├── data/
│   └── raw/
│       ├── train.csv.dvc               # DVC tracking pointer
│       ├── later.csv.dvc               # DVC tracking pointer
│       └── holdout_features.csv        # Holdout inference features
├── reports/
│   └── findings.yaml                   # Official findings report
├── .dvc/                               # DVC internal configuration
└── .gitignore                          # Standard git ignore definitions
```
## Running the Data Audit

The data audit module checks seven data quality
issues present in the training dataset.

### Install dependencies

```bash
pip install -r requirements.txt