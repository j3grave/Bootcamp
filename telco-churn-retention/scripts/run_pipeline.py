"""Ponto de entrada do projeto: pergunta que fases correr e corre-as por ordem.

Uso, a partir da raiz do projeto:
    python scripts/run_pipeline.py                  # pergunta no terminal
    python scripts/run_pipeline.py --sem-profiling  # sem perguntas: fases 1 e 2 (usado no deploy)
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
MODEL_REPO = os.environ.get("HF_MODEL_REPO", "tvf007/EDSBG1TP2-model")
SPACE = os.environ.get("HF_SPACE", "tvf007/EDSBG1TP2")


def ask(question: str, default: bool) -> bool:
    """Pergunta sim/não no terminal; Enter aceita o valor por omissão."""
    answer = input(f"{question} {'[S/n]' if default else '[s/N]'} ").strip().lower()
    return default if not answer else answer.startswith("s")


def plan(c: dict) -> list[tuple[str, list[str]]]:
    """Comandos a correr, por ordem. Publicar obriga a correr os testes antes."""
    steps = []
    if c["preprocessing"]:
        steps.append(("Fase 1 · pré-processamento e EDA", [PY, "scripts/run_preprocessing.py"]))
    if c["profiling"]:
        steps.append(("Fase 1b · profiling", [PY, "scripts/run_profiling.py"]))
    if c["modelling"]:
        steps.append(("Fase 2 · modelação",
                      [PY, "scripts/run_modelling.py"] + (["--rapido"] if c["rapido"] else [])))
    if c["tests"] or c["publish"]:
        steps.append(("Testes", [PY, "-m", "pytest", "tests", "-q"]))
    if c["publish"]:
        steps.append(("Publicação no Hugging Face",
                      [PY, "scripts/publish_hf.py", "--model-repo", c["model_repo"],
                       "--space", c["space"]] + (["--dry-run"] if c["dry_run"] else [])))
    return steps


def interactive() -> dict:
    """Pergunta no terminal que fases correr e devolve as escolhas."""
    c = {"preprocessing": ask("Fase 1 · pré-processamento e EDA?", True),
         "profiling": ask("Fase 1b · relatórios ydata-profiling (lento)?", False),
         "modelling": ask("Fase 2 · modelação?", True)}
    c["rapido"] = c["modelling"] and ask("  Modo rápido (sem a variante com Satisfaction Score)?", False)
    c["tests"] = ask("Testes (pytest)?", True)
    c["publish"] = ask("Publicação no Hugging Face (corre os testes antes)?", False)
    c.update(dry_run=True, model_repo=MODEL_REPO, space=SPACE)
    if c["publish"]:
        c["model_repo"] = input(f"  Model repo [{MODEL_REPO}] ").strip() or MODEL_REPO
        c["space"] = input(f"  Space [{SPACE}] ").strip() or SPACE
        if c["rapido"]:
            print("  Modo rápido: só se preparam os ficheiros, sem publicar.")
        else:
            c["dry_run"] = not ask("  Publicar mesmo (e não só preparar os ficheiros)?", False)
        if not c["dry_run"] and not os.environ.get("HF_TOKEN"):
            sys.exit("Falta a variável de ambiente HF_TOKEN para publicar.")
    return c


def main() -> None:
    """Corre as fases escolhidas, por ordem; pára na primeira que falhar."""
    if len(sys.argv) > 1 or not sys.stdin.isatty():
        c = dict(preprocessing=True, profiling="--sem-profiling" not in sys.argv,
                 modelling=True, rapido=False, tests=False, publish=False)
    else:
        c = interactive()
    t0 = time.time()
    for label, cmd in plan(c):
        print(f"\n=== {label} ===")
        t = time.time()
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:
            sys.exit(f"✘ {label} falhou: os passos seguintes não correm.")
        print(f"✔ {label} em {time.time() - t:.0f}s")
    print(f"\nConcluído em {time.time() - t0:.0f}s. Ver outputs/ e data/processed/.")


if __name__ == "__main__":
    main()
