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
- **Artefacto dos clientes novos:** os Joined têm todos 1–3 meses de casa, por isso dentro de Churned + Stayed todos os clientes com tenure ≤ 3 saíram. Além do modelo core, há um **modelo de priorização** treinado só com tenure > 3, e os clientes novos são tratados por uma **regra de onboarding** (ver `docs/modelling.md`).
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
│   └── 02_modelling_eda.ipynb       # modelos, benchmark IBM, modelo de priorização e decisões de retenção
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
│   ├── retention.py          # P vs P × CLTV, custo-benefício, ações sugeridas, regra de onboarding
│   └── model_plots.py        # gráficos da modelação
├── scripts/
│   ├── run_pipeline.py       # pré-processamento e EDA sem notebook
│   ├── run_profiling.py      # relatórios ydata-profiling
│   ├── run_modelling.py      # modelação completa sem notebook
│   └── run_all.py            # corre os três, por ordem
├── outputs/                  # figuras, tabelas, artefactos e profiling (git-ignored)
├── reports/                  # guia do projeto, status reports (pptx), dicionário de fontes
├── .vscode/                  # definições partilhadas do VS Code
├── .gitattributes            # nbstripout: remove os outputs dos notebooks em cada commit
├── requirements.txt          # dependências (versões mínimas)
├── requirements-lock.txt     # versões exatas usadas para gerar os resultados
├── pyproject.toml
└── README.md
```

---

## Como começar

### 1. Clonar e criar o ambiente

O repositório do grupo é [`j3grave/Bootcamp`](https://github.com/j3grave/Bootcamp); o projeto está na subpasta `telco-churn-retention`.

```bash
git clone https://github.com/j3grave/Bootcamp.git
cd Bootcamp/telco-churn-retention

python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements-lock.txt   # mesmas versões com que os resultados foram gerados
# (ou pip install -r requirements.txt, para as versões mais recentes compatíveis)
pip install -e .          # opcional: torna `telco_churn` importável em qualquer lado
nbstripout --install      # obrigatório uma vez por membro: limpa os outputs dos notebooks em cada commit
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

- `outputs/tables/10_*` a `21_*`: tuning, métricas em CV, variantes, holdout, importância, captura de valor, modelo de priorização, custo-benefício, ações sugeridas e regra de onboarding
- `outputs/figures/20_*` a `30_*`: comparação dos modelos, métricas por modelo, curvas, matrizes de confusão, rede neuronal, priorização e valor líquido por cenário
- `outputs/artifacts/final_model.joblib` (modelo core) e `priority_model.joblib` (modelo de priorização, tenure > 3)
- `data/processed/prioridade_holdout.csv`: lista de intervenção dos clientes estabelecidos do holdout, com a ação sugerida
- `data/processed/onboarding_joined.csv`: lista de onboarding dos Joined, por segmento

---

## Etapas do pipeline

| # | Etapa | O que faz |
|---|---|---|
| 1–2 | Setup e carregamento | Lê os Excel com `keep_default_na=False`, para preservar o literal `None` |
| 3 | Validação | Verifica a unicidade das chaves, a igualdade dos IDs e a cobertura do join por ZIP |
| 4 | União | Faz o merge com `validate="one_to_one"` / `"many_to_one"`, ficando 1 linha por cliente |
| 5 | Qualidade | Analisa missing e strings vazias e confirma as derivações documentadas |
| 6 | População e target | Exclui `Joined`, cria o target binário e documenta o artefacto do tenure ≤ 3 |
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
| 10 | Modelo de priorização | Mesmo protocolo só com tenure > 3; tabela core vs priorização vs IBM no mesmo holdout |
| 11 | Priorização | Lista por **P(churn) × CLTV**, comparação com só P, custo-benefício por cenários e ações sugeridas (hipóteses) |
| 12 | Clientes novos | Regra de onboarding para os Joined, em vez de P(churn) |
| 13 | Exportação | `final_model.joblib`, `priority_model.joblib`, `prioridade_holdout.csv`, `onboarding_joined.csv` |

---

## Decisões principais

| Decisão | Justificação |
|---|---|
| Excluir `Customer Status`, `Churn Label/Value/Category/Reason` | Codificam o resultado ou só existem depois do churn (leakage) |
| `Churn Score` como **benchmark**, não como feature | É o output do modelo IBM existente (AUC ≈ 0,94 no holdout) |
| `CLTV` fora das features e mantido nos CSV | É usado na priorização: **P(churn) × CLTV** |
| Dois modelos: core (todos os tenures) e priorização (tenure > 3) | Dentro de Churned + Stayed, tenure ≤ 3 = churn por construção (os novos que ficam são Joined). O core responde ao enunciado; o de priorização pontua os clientes estabelecidos |
| Joined sem P(churn); regra de onboarding | O modelo core dar-lhes-ia P ≈ 1 (extrapolação). A regra (contrato e tipo de internet) é uma hipótese a validar com grupo de controlo |
| Taxa de sucesso e custo por contacto como pressupostos explícitos | Não existem no dataset; ficam numa célula de parâmetros com três cenários e análise de sensibilidade |
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
- Notebooks numerados por fase: `01_preprocessing_eda`, `01b_profiling_ydata`, `02_modelling_eda` (a priorização está nas secções 10 a 12 do notebook 02).
- Cada notebook tem um script equivalente em `scripts/`; uma alteração a um modelo faz-se em `src/telco_churn/` e vale para os dois.

---

## Equipa

> **Para o grupo confirmar (não alterei nomes nem dados):**
> - a tabela não aparece formatada no GitHub: falta a linha separadora `|---|---|---|` e as linhas abaixo não usam `|`;
> - a coluna "Nº aluno" não tem valores próprios (o número aparece só dentro do email);
> - estão listados três membros: confirmar se o grupo tem mais alguém e se falta uma coluna de papéis.

| Nome | Nº aluno | Email |
Francisco Monteiro, 20251694@novaims.unl.pt
José Grave, 20251843@novaims.unl.pt
Tiago Vacas Felgueira, 20242047@novaims.unl.pt

