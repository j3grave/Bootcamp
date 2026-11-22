"""Corre o projeto por ordem: pré-processamento, profiling e (se existir) modelação.

Uso, a partir da raiz do projeto:
    python scripts/run_all.py [--sem-profiling]
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    ("Pré-processamento e EDA", "scripts/run_pipeline.py"),
    ("Relatórios ydata-profiling", "scripts/run_profiling.py"),
    ("Modelação", "scripts/run_modelling.py"),        # corre só quando existir
]


def main() -> None:
    skip_profiling = "--sem-profiling" in sys.argv
    t0 = time.time()

    for label, script in STEPS:
        path = ROOT / script
        if skip_profiling and "profiling" in script:
            print(f"\n— {label}: ignorado (--sem-profiling)")
            continue
        if not path.exists():
            print(f"\n— {label}: {script} ainda não existe, passo ignorado")
            continue
        print(f"\n=== {label} · {script} ===")
        t = time.time()
        subprocess.run([sys.executable, str(path)], check=True, cwd=ROOT)
        print(f"✔ {label} concluído em {time.time() - t:.0f}s")

    print(f"\nProjeto concluído em {time.time() - t0:.0f}s. Ver outputs/ e data/processed/.")


if __name__ == "__main__":
    main()
