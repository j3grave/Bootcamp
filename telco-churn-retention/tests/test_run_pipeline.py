"""Ponto de entrada (scripts/run_pipeline.py): ordem das fases e testes antes de publicar."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "run_pipeline", Path(__file__).resolve().parents[1] / "scripts" / "run_pipeline.py")
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)

NADA = dict(preprocessing=False, profiling=False, modelling=False, rapido=False, tests=False,
            publish=False, dry_run=True, model_repo="conta/m", space="conta/s")


def labels(**escolhas):
    return [label for label, _ in rp.plan({**NADA, **escolhas})]


def test_fases_por_ordem():
    assert labels(preprocessing=True, profiling=True, modelling=True, tests=True) == [
        "Fase 1 · pré-processamento e EDA", "Fase 1b · profiling", "Fase 2 · modelação", "Testes"]


def test_publicar_obriga_a_testar():
    assert labels(publish=True) == ["Testes", "Publicação no Hugging Face"]


def test_dry_run_passa_ao_publish():
    cmd = dict(rp.plan({**NADA, "publish": True}))["Publicação no Hugging Face"]
    assert cmd[-1] == "--dry-run"
