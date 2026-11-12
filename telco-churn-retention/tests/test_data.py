"""Contratos de dados: os ficheiros reais cumprem-nos e um erro deliberado pára a pipeline."""
import shutil

import pandas as pd
import pytest

from telco_churn import data, schemas
from telco_churn.config import DATA_RAW, FILES


def _failed_errors(report: pd.DataFrame) -> list[str]:
    bad = report[(report["severidade"] == schemas.ERRO) & (report["resultado"] == "falha")]
    return [f"[{r.camada}] {r.regra}: {r.linhas_a_falhar}" for r in bad.itertuples()]


def test_ficheiros_brutos_cumprem_os_contratos(raw):
    assert _failed_errors(schemas.validate_raw(raw)) == []


def test_base_unida_cumpre_os_contratos(merged):
    report = schemas.validate_merged(merged)
    assert _failed_errors(report) == []
    # Nos dados de referência, também nenhum aviso deve disparar
    assert (report["resultado"] == "passa").all(), report[report["resultado"] == "falha"]


def test_derivacoes_a_100_por_cento(merged):
    checks = data.derivation_checks(merged)
    assert (checks == 1.0).all(), checks[checks < 1.0]


def test_merge_um_para_um_sem_perder_clientes(raw, merged):
    assert len(merged) == len(raw["status"])
    assert merged["Customer ID"].is_unique
    assert set(merged["Customer ID"]) == set(raw["status"]["Customer ID"])
    assert merged["Population"].notna().all()   # todos os Zip Code encontraram população


@pytest.mark.parametrize("table, column, value, rule", [
    ("status", "Satisfaction Score", 9, "Satisfaction Score ∈ {1, 2, 3, 4, 5}"),
    ("status", "Churn Score", 150, "Churn Score ∈ [0, 100]"),
    ("services", "Contract", "Monthly", "Contract ∈ {Month-to-Month, One Year, Two Year}"),
    ("demographics", "Age", -3, "Age ∈ [18, 100]"),
])
def test_valor_fora_do_dominio_e_erro(raw, report_tmp, table, column, value, rule):
    bad = {name: df.copy() for name, df in raw.items()}
    bad[table].loc[0, column] = value
    report = schemas.validate_raw(bad)
    assert any(rule in e for e in _failed_errors(report))
    with pytest.raises(schemas.DataValidationError, match="Satisfaction|Churn Score|Contract|Age"):
        schemas.enforce(report, fresh=True)
    assert report_tmp.exists()   # o relatório é escrito mesmo quando a validação falha


def test_id_duplicado_e_erro(raw):
    bad = {name: df.copy() for name, df in raw.items()}
    bad["demographics"].loc[1, "Customer ID"] = bad["demographics"].loc[0, "Customer ID"]
    errors = _failed_errors(schemas.validate_raw(bad))
    assert any("Customer ID único" in e for e in errors)
    assert any("mesmos Customer ID" in e for e in errors)


def test_regra_de_negocio_violada_e_erro(merged):
    bad = merged.copy()
    stayed = bad.index[bad["Customer Status"] == "Stayed"][0]
    bad.loc[stayed, "Churn Label"] = "Yes"
    assert any("Churn Label = Yes" in e for e in _failed_errors(schemas.validate_merged(bad)))


def test_estatistica_fora_do_esperado_e_so_aviso(merged):
    half = merged.sample(frac=0.5, random_state=0)
    report = schemas.validate_merged(half)
    assert _failed_errors(report) == []
    warnings = report[(report["severidade"] == schemas.AVISO) & (report["resultado"] == "falha")]
    assert warnings["regra"].str.contains("nº de clientes").any()


def test_load_raw_para_com_ficheiro_errado(tmp_path, report_tmp):
    """Ponta a ponta: um Excel com Satisfaction Score = 9 faz load_raw parar."""
    for f in FILES.values():
        shutil.copy(DATA_RAW / f, tmp_path / f)
    status = pd.read_excel(tmp_path / FILES["status"], keep_default_na=False)
    status.loc[0, "Satisfaction Score"] = 9
    status.to_excel(tmp_path / FILES["status"], index=False)
    with pytest.raises(schemas.DataValidationError, match="Satisfaction Score"):
        data.load_raw(tmp_path)
