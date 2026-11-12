"""Publica o modelo (Model repo) e a app (Space) no Hugging Face.

Chamado pelo GitHub Actions (deploy-hf.yml) só depois de a pipeline e os testes passarem.

Uso, a partir da raiz do projeto (telco-churn-retention):
    HF_TOKEN=... python scripts/publish_hf.py --model-repo conta/telco-churn-model --space conta/telco-churn-app
    python scripts/publish_hf.py --model-repo conta/m --space conta/s --dry-run   # só prepara

O Model repo recebe final_model.joblib (modelo core) e priority_model.joblib (clientes
estabelecidos). O Space recebe a pasta app/, uma cópia do pacote src/telco_churn (para add_features e
scoring) e um CSV de exemplo para o separador "Lote". A variável MODEL_REPO do Space é
definida aqui, para a app saber de onde descarregar o modelo.
"""
import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telco_churn import scoring  # noqa: E402
from telco_churn.config import ARTIFACTS_DIR, DATA_PROCESSED  # noqa: E402

MODEL = ARTIFACTS_DIR / "final_model.joblib"
PRIORITY_MODEL = ARTIFACTS_DIR / "priority_model.joblib"


def metrics_section(bundle: dict, priority: dict) -> str:
    """Tabela de métricas desta versão, para o fim do model card."""
    rows = [("Modelo core: " + bundle["name"], bundle["cv"], bundle["holdout"]),
            ("Churn Score IBM (benchmark)", bundle["benchmark_cv"], bundle["benchmark_holdout"]),
            ("Modelo de priorização (tenure > 3): " + priority["name"], priority["cv"],
             priority["holdout"]),
            ("Churn Score IBM nos mesmos clientes (tenure > 3)", priority["benchmark_cv"],
             priority["benchmark_holdout"])]
    lines = ["## Métricas desta versão", "",
             f"Seleção: {bundle['selection']}", "",
             "| | AUC CV (± dp) | AUC holdout | PR AUC holdout | Brier holdout |",
             "|---|---|---|---|---|"]
    for name, cv, hold in rows:
        h = f"{hold['auc']:.3f} | {hold['pr_auc']:.3f} | {hold['brier']:.3f}"
        lines.append(f"| {name} | {cv['auc']:.3f} ± {cv['auc_std']:.3f} | {h} |")
    lv, cap = priority["risk_levels"], priority["capacity_cuts"]
    b = priority["bootstrap_vs_ibm"]
    lines += ["", "O modelo de priorização e a sua linha IBM são avaliados no holdout com "
                  "tenure > 3. Nesses clientes a IBM ordena melhor: diferença de AUC "
                  f"{b['diferenca']:+.3f}, IC 95% [{b['ic_inferior']:+.3f}; {b['ic_superior']:+.3f}] "
                  "(bootstrap emparelhado). Não é gate do deploy.", "",
              f"Nível de risco (absoluto): baixo < {lv['médio']:.0%}, médio "
              f"{lv['médio']:.0%}–{lv['alto']:.0%}, alto ≥ {lv['alto']:.0%}. "
              f"Prioridade de contacto (capacidade, só clientes estabelecidos, modelo de "
              f"priorização): top 10% se P ≥ {cap['top 10%']:.3f}, top 20% se P ≥ "
              f"{cap['top 20%']:.3f}."]
    return "\n".join(lines) + "\n"


def stage_space(staging: Path, bundle: dict) -> None:
    """Pasta do Space: app/ + pacote telco_churn + CSV de exemplo."""
    shutil.copytree(ROOT / "app", staging, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "telco_churn"))
    shutil.copytree(ROOT / "src" / "telco_churn", staging / "telco_churn",
                    ignore=shutil.ignore_patterns("__pycache__"))
    test = pd.read_csv(DATA_PROCESSED / "test.csv", keep_default_na=False, na_values=[""])
    cols = [scoring.ID_COL, scoring.CLTV_COL] + scoring.input_columns(bundle)
    test[cols].head(30).to_csv(staging / "exemplo_lote.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-repo", default=os.environ.get("HF_MODEL_REPO"), required=False)
    parser.add_argument("--space", default=os.environ.get("HF_SPACE"), required=False)
    parser.add_argument("--dry-run", action="store_true", help="prepara os ficheiros sem publicar")
    args = parser.parse_args()
    missing = [n for n, v in (("HF_MODEL_REPO / --model-repo", args.model_repo),
                              ("HF_SPACE / --space", args.space)) if not v]
    if missing:
        sys.exit("Falta: " + ", ".join(missing) + ". No GitHub, criar como *Repository "
                 "variables* (Settings > Secrets and variables > Actions > Variables), não "
                 "dentro de um Environment.")
    bundle, priority = joblib.load(MODEL), joblib.load(PRIORITY_MODEL)

    staging = Path(tempfile.mkdtemp(prefix="hf_"))
    model_dir, space_dir = staging / "model", staging / "space"
    model_dir.mkdir()
    shutil.copy(MODEL, model_dir / "final_model.joblib")
    shutil.copy(PRIORITY_MODEL, model_dir / "priority_model.joblib")
    card = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
    (model_dir / "README.md").write_text(card + "\n" + metrics_section(bundle, priority), encoding="utf-8")
    stage_space(space_dir, bundle)
    print(f"Ficheiros preparados em {staging}")
    if args.dry_run:
        return

    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("Falta o secret HF_TOKEN.")
    api = HfApi(token=token)

    def step(msg, fn, *a, **kw):
        """Corre um passo da publicação; se falhar, diz qual e o que verificar."""
        print(f"→ {msg}", flush=True)
        try:
            return fn(*a, **kw)
        except HfHubHTTPError as e:
            code = e.response.status_code if e.response is not None else "?"
            hint = {401: "token inválido ou expirado: gerar outro e atualizar o secret HF_TOKEN.",
                    403: "o token não tem permissão. Usar um token do tipo *Write* ou, se for "
                         "fine-grained, dar-lhe 'write' e 'create repos' no namespace "
                         f"'{args.model_repo.split('/')[0]}'.",
                    404: "repositório ou conta não encontrados: confirmar HF_MODEL_REPO e HF_SPACE."
                    }.get(code, "ver a mensagem acima.")
            sys.exit(f"FALHOU em '{msg}' (HTTP {code}): {hint}\n{e}")

    who = step("Validar o token", api.whoami)
    auth = who.get("auth", {}).get("accessToken", {})
    print(f"  conta: {who.get('name')} · tipo de token: {auth.get('role', '?')}")
    # Público, para o Space descarregar o modelo sem token
    step(f"Criar/confirmar o Model repo {args.model_repo} (público)", api.create_repo,
         args.model_repo, repo_type="model", private=False, exist_ok=True)
    step("Enviar os modelos e o model card", api.upload_folder, folder_path=model_dir,
         repo_id=args.model_repo, repo_type="model",
         commit_message=f"Modelo {bundle['name']} (AUC holdout {bundle['holdout']['auc']:.3f})")
    step(f"Criar/confirmar o Space {args.space}", api.create_repo, args.space,
         repo_type="space", space_sdk="gradio", exist_ok=True)
    step("Definir a variável MODEL_REPO no Space", api.add_space_variable, args.space,
         "MODEL_REPO", args.model_repo)
    if step("Ler a visibilidade do Model repo", api.model_info, args.model_repo).private:
        # já existia e é privado: o Space precisa de token para o descarregar
        step("Criar o secret HF_TOKEN no Space", api.add_space_secret, args.space,
             "HF_TOKEN", token)
        print("Aviso: o Model repo é privado; foi criado o secret HF_TOKEN no Space.")
    step("Enviar a app para o Space", api.upload_folder, folder_path=space_dir,
         repo_id=args.space, repo_type="space", commit_message="Atualiza a app")
    print(f"Modelo: https://huggingface.co/{args.model_repo}")
    print(f"App:    https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
