"""Caminhos, ficheiros e parâmetros globais do projeto."""
from pathlib import Path

# Raiz do repositório (src/telco_churn/config.py -> sobe 2 níveis)
ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIG_DIR = OUTPUTS / "figures"
TAB_DIR = OUTPUTS / "tables"
ARTIFACTS_DIR = OUTPUTS / "artifacts"

FILES = {
    "demographics": "Telco_customer_churn_demographics.xlsx",
    "location": "Telco_customer_churn_location.xlsx",
    "population": "Telco_customer_churn_population.xlsx",
    "services": "Telco_customer_churn_services.xlsx",
    "status": "Telco_customer_churn_status.xlsx",
}

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "churn"


def ensure_dirs() -> None:
    """Cria as pastas de saída, caso não existam."""
    for d in (DATA_PROCESSED, FIG_DIR, TAB_DIR, ARTIFACTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
