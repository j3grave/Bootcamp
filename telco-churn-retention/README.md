<p align="center">
  <img src="https://img.shields.io/badge/NOVA%20IMS-EDSB%202026-000000?style=for-the-badge&labelColor=000000&color=3777FF" alt="EDSB 2026">
  <img src="https://img.shields.io/badge/Projeto%20D-Telco%20Churn-000000?style=for-the-badge&labelColor=000000&color=0EE071" alt="Projeto D">
  <img src="https://img.shields.io/badge/Python-3.10%2B-000000?style=for-the-badge&logo=python&logoColor=0EE071&labelColor=000000&color=3777FF" alt="Python">
</p>

# Telco Customer Churn — Retention Management

Projeto de grupo da cadeira **Enterprise Data Science Bootcamp (EDSB 2026)** da NOVA IMS.

O objetivo é construir um sistema que ajude o *Head of Customer Retention* a decidir **que clientes devem receber uma intervenção de retenção** e **com que prioridade**. A base é a estimativa do risco de churn, combinada com o valor do cliente (CLTV). Um churn score isolado não é o objetivo.

---

## Dataset

Amostra IBM *Telco Customer Churn*: 7.043 clientes na Califórnia, com um único snapshot do Q3.

| Ficheiro | Grão | Chave de ligação |
|---|---|---|
| `Telco_customer_churn_demographics.xlsx` | 1 linha por cliente | `Customer ID` |
| `Telco_customer_churn_location.xlsx` | 1 linha por cliente | `Customer ID` |
| `Telco_customer_churn_services.xlsx` | 1 linha por cliente | `Customer ID` |
| `Telco_customer_churn_status.xlsx` | 1 linha por cliente | `Customer ID` |
| `Telco_customer_churn_population.xlsx` | 1 linha por ZIP code | `Zip Code` |

- **População de modelação:** clientes `Churned` + `Stayed` (6.589). Os `Joined` (454) ficam fora do modelo core.
- **Target:** `churn` = 1 se *Churned*, 0 se *Stayed* (taxa de 28,4%).
- **Avaliação:** holdout estratificado 80/20 ao nível do cliente.

Fonte: [IBM Cognos Analytics — Telco Customer Churn](https://www.ibm.com/docs/en/cognos-analytics/12.0.x?topic=samples-telco-customer-churn). O guia do projeto está em `reports/project_guide.pdf`.

---

## Estrutura do repositório

```
telco-churn-retention/
├── data/
│   ├── raw/                  # 5 ficheiros Excel originais (não editar)
│   └── processed/            # train/test/joined gerados pelo pipeline (git-ignored)
├── notebooks/
│   ├── 01_preprocessing_eda.ipynb   # EDA e pré-processamento, documentado por secções
│   ├── 01b_profiling_ydata.ipynb    # relatórios ydata-profiling
│   └── 02_modelling_eda.ipynb       # modelos, rede neuronal, ensembles, benchmark IBM e priorização
├── docs/
│   ├── models.md             # como funciona cada modelo, hiperparâmetros e papel no projeto
│   └── modelling.md          # protocolo de avaliação, métricas, gráficos e decisões
├── src/telco_churn/          # código reutilizável
│   ├── config.py             # caminhos, seed, % de teste
│   ├── style.py              # paleta, tema dos gráficos, tabelas formatadas
│   ├── data.py               # carregar, validar chaves, unir, qualidade
│   ├── features.py           # população, papéis das variáveis, feature engineering
│   ├── analysis.py           # perfis, Cramér's V, Mutual Information
│   ├── plots.py              # gráficos da EDA
│   ├── profiling.py          # relatórios ydata-profiling com a paleta do projeto
│   ├── preprocessing.py      # split, ColumnTransformers, exportação
│   ├── modelling.py          # modelos, tuning em CV, ensembles, métricas, holdout, P(churn) × CLTV
│   └── model_plots.py        # gráficos da modelação
├── scripts/
│   ├── run_pipeline.py       # pré-processamento e EDA sem notebook
│   ├── run_profiling.py      # relatórios ydata-profiling
│   ├── run_modelling.py      # modelação completa sem notebook
│   └── run_all.py            # corre os três, por ordem
├── outputs/                  # figuras, tabelas e artefactos (git-ignored)
├── reports/                  # guia do projeto, status reports, apresentação
├── .vscode/                  # definições partilhadas do VS Code
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Como começar

### 0. Criar o repositório no GitHub (apenas um membro do grupo)

1. No GitHub, criar um repositório **vazio** chamado `telco-churn-retention`, sem README nem .gitignore.
2. Na pasta deste projeto:

```bash
git init -b main
git add .
git commit -m "Estrutura inicial: pipeline de pré-processamento e EDA"
git remote add origin https://github.com/<utilizador>/telco-churn-retention.git
git push -u origin main
```

3. Em *Settings → Collaborators*, convidar os restantes membros do grupo.

### 1. Clonar e criar o ambiente

```bash
git clone https://github.com/<organizacao-ou-utilizador>/telco-churn-retention.git
cd telco-churn-retention

python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
pip install -e .          # opcional: torna `telco_churn` importável em qualquer lado
nbstripout --install      # limpa outputs dos notebooks em cada commit
```

### 2. Correr o pipeline

As fases correm por ordem, porque cada uma usa os ficheiros gerados pela anterior:

| Fase | No VS Code (kernel `.venv`, *Run All*) | Pela linha de comandos |
|---|---|---|
| 1. Pré-processamento e EDA | `notebooks/01_preprocessing_eda.ipynb` | `python scripts/run_pipeline.py` |
| 1b. Profiling (opcional) | `notebooks/01b_profiling_ydata.ipynb` | `python scripts/run_profiling.py` |
| 2. Modelação | `notebooks/02_modelling_eda.ipynb` | `python scripts/run_modelling.py` |

Para correr tudo de uma vez: `python scripts/run_all.py` (ou `--sem-profiling`, para saltar os relatórios). Para testar alterações à modelação depressa: `python scripts/run_modelling.py --rapido`.

A fase 1 gera:

- `data/processed/`: `train.csv`, `test.csv`, `joined_para_scoring.csv`, `telco_merged_full.csv`
- `outputs/figures/`: cerca de 40 gráficos da EDA
- `outputs/tables/`: perfis, dicionário de categorias, Cramér's V, Mutual Information
- `outputs/artifacts/preprocessing_config.joblib`: pré-processadores e listas de features

A fase 2 gera:

- `outputs/tables/10_*` a `15_*`: tuning, métricas em CV, variantes, holdout, importância e captura de valor
- `outputs/figures/20_*` a `27_*`: comparação dos modelos, métricas por modelo, curvas, matrizes de confusão, rede neuronal e priorização
- `outputs/artifacts/final_model.joblib`: modelo final treinado, features e métricas
- `data/processed/prioridade_holdout.csv` e `prioridade_joined.csv`: listas de intervenção

---

## Etapas do pipeline

| # | Etapa | O que faz |
|---|---|---|
| 1–2 | Setup e carregamento | Lê os Excel com `keep_default_na=False`, para preservar o literal `None` |
| 3 | Validação | Verifica a unicidade das chaves, a igualdade dos IDs e a cobertura do join por ZIP |
| 4 | União | Faz o merge com `validate="one_to_one"` / `"many_to_one"`, ficando 1 linha por cliente |
| 5 | Qualidade | Analisa missing e strings vazias e confirma as derivações documentadas |
| 6 | População e target | Exclui `Joined` e cria o target binário |
| 7 | Papéis das variáveis | Classifica cada coluna como id, leakage, benchmark, redundante, numérica ou categórica |
| 8 | Numéricas | Perfil (skew, % zeros, % outliers IQR), distribuições por target, Spearman |
| 9 | Categóricas | Dicionário de categorias, composição, Stayed vs Churned, Cramér's V |
| 10 | Associação e leakage | Mutual Information; verificação de Satisfaction Score e Churn Score |
| 11 | Feature engineering | `n_add_on_services`, `has_refund`, `avg_hist_charge`, `charge_delta` |
| 12 | Split e pré-processadores | Split 80/20 estratificado; `ColumnTransformer` *linear* e *trees* |
| 13 | Exportação | CSVs e `joblib` para a fase de modelação |

### Notebook 02 — modelação

Corre depois do notebook 01 (usa `data/processed/` e `preprocessing_config.joblib`). A explicação de cada modelo está em [`docs/models.md`](docs/models.md) e o protocolo de avaliação em [`docs/modelling.md`](docs/modelling.md).

| # | Etapa | O que faz |
|---|---|---|
| 3 | Modelos | Tabela dos modelos (família, pré-processador, funcionamento), a rede neuronal e as nove métricas |
| 4 | Tuning | `RandomizedSearchCV` com `StratifiedKFold`, só no treino, para os 7 modelos base: Logistic Regression, Decision Tree, k-Nearest Neighbors, Neural Network (MLP), Random Forest, Extra Trees e Gradient Boosting |
| 5 | Comparação e ensembles | Soft Voting e Stacking; todos os modelos no mesmo 5-fold com nove métricas; diversidade, painel de métricas, sobreajuste, curvas ROC/PR, matrizes de confusão, limiar e treino da rede neuronal; o modelo final é o de maior AUC em CV |
| 6–7 | Variantes | `Satisfaction Score` com e sem; `City` com `TargetEncoder` dentro do Pipeline |
| 8 | Holdout | Avaliação única no teste, com as nove métricas, lado a lado com o Churn Score IBM |
| 9 | Interpretação | Permutation importance ao nível das variáveis originais |
| 10 | Priorização | Lista ordenada por **P(churn) × CLTV** e curva de captura de valor |
| 11 | Exportação | `final_model.joblib`, `prioridade_holdout.csv`, `prioridade_joined.csv` |

---

## Decisões principais

| Decisão | Justificação |
|---|---|
| Excluir `Customer Status`, `Churn Label/Value/Category/Reason` | Codificam o resultado ou só existem depois do churn (leakage) |
| `Churn Score` como **benchmark**, não como feature | É o output do modelo IBM existente (AUC ≈ 0,94 no holdout) |
| `CLTV` fora das features e mantido nos CSV | É usado na priorização: **P(churn) × CLTV** |
| `Satisfaction Score` fora por defeito | Scores 1–2 dão 100% de churn, o que indica suspeita de leakage. A testar com e sem |
| Remover `Under 30`, `Senior Citizen`, `Dependents`, `Referred a Friend`, `Total Revenue` | São derivações exatas de outras colunas (verificadas a 100%) |
| Remover `Country`, `State`, `Lat Long`, `Zip Code`, `City` | São constantes, duplicadas ou de alta cardinalidade |
| PCA não incluída nesta fase | Os ensembles de árvores não precisam dela e a interpretabilidade é crítica para a retenção |
| Seleção do modelo por ROC AUC em CV (5-fold), holdout usado uma vez | A retenção trabalha com uma lista ordenada; o desvio-padrão entre folds mostra se as diferenças são reais |
| Rede neuronal com `MLPClassifier` (scikit-learn) | Encaixa no `Pipeline`, no tuning e nos ensembles sem novas dependências; equivale a Keras para uma rede deste tamanho |
| Ensembles com LR, MLP, Random Forest, Extra Trees e Gradient Boosting | Membros fortes e de famílias diferentes, fixados antes de ver os resultados |
| Gráficos da modelação em `model_plots.py`, separados de `plots.py` | Uma alteração aos gráficos da EDA não apaga os da modelação, e vice-versa |

---

## Identidade visual

Todos os gráficos e secções do notebook usam a paleta do projeto:

| Cor | HEX | Uso |
|---|---|---|
| ![](https://img.shields.io/badge/-%20%20%20%20-3777FF?style=flat-square) Azul Digital | `#3777FF` | Destaque principal, *Stayed*, títulos de secção |
| ![](https://img.shields.io/badge/-%20%20%20%20-0EE071?style=flat-square) Verde Néon | `#0EE071` | Acento, *Churned*, subtítulos e realces |
| ![](https://img.shields.io/badge/-%20%20%20%20-000000?style=flat-square) Preto Profundo | `#000000` | Fundo |

A paleta está definida em `src/telco_churn/style.py`. As secções do notebook usam HTML com estilos *inline*, que aparecem com as cores no VS Code e no Jupyter. O GitHub remove estes estilos ao pré-visualizar notebooks, mas o conteúdo continua legível.

---

## Fluxo de trabalho em grupo (Git)

```bash
git checkout -b feature/<nome-curto>     # um ramo por tarefa
# ... trabalhar ...
git add .
git commit -m "EDA: adiciona análise de contratos"
git push -u origin feature/<nome-curto>
# abrir Pull Request no GitHub -> revisão por um colega -> merge para main
```

Regras do grupo:

- Não fazer commit de dados processados nem de outputs: cada membro regenera-os localmente.
- Não editar os ficheiros em `data/raw/`.
- Novas funções vão para `src/telco_churn/` e os notebooks servem para explorar e narrar.
- Notebooks numerados por fase: `01_preprocessing_eda`, `02_modelling_eda`, `03_prioritization`.
- Cada notebook tem um script equivalente em `scripts/`; uma alteração a um modelo faz-se em `src/telco_churn/` e vale para os dois.

---

## Equipa

| Nome | Nº aluno | Email |
Francisco Monteiro, 20251694@novaims.unl.pt
José Grave, 20251843@novaims.unl.pt
Tiago Vacas Felgueira, 20242047@novaims.unl.pt

