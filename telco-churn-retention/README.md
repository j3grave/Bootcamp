<p align="center">
  <img src="https://img.shields.io/badge/NOVA%20IMS-EDSB%202026-000000?style=for-the-badge&labelColor=000000&color=3777FF" alt="EDSB 2026">
  <img src="https://img.shields.io/badge/Projeto%20D-Telco%20Churn-000000?style=for-the-badge&labelColor=000000&color=0EE071" alt="Projeto D">
  <img src="https://img.shields.io/badge/Python-3.10%2B-000000?style=for-the-badge&logo=python&logoColor=0EE071&labelColor=000000&color=3777FF" alt="Python">
  <a href="https://huggingface.co/spaces/tvf007/EDSBG1TP2"><img src="https://img.shields.io/badge/App-Hugging%20Face-000000?style=for-the-badge&labelColor=000000&color=0EE071" alt="App no Hugging Face"></a>
</p>

# Telco Customer Churn — Retention Management

Projeto D da cadeira Enterprise Data Science Bootcamp (EDSB 2026), NOVA IMS.

O sistema ajuda o *Head of Customer Retention* a decidir que clientes contactar e por que ordem. A ordem vem da perda esperada, P(churn) × CLTV, e não do risco de churn isolado.

- **App:** <https://huggingface.co/spaces/tvf007/EDSBG1TP2>
- **Modelos:** <https://huggingface.co/tvf007/EDSBG1TP2-model> (`priority_model.joblib` para clientes estabelecidos, `final_model.joblib` para o modelo core)

---

## Fluxo

```mermaid
flowchart LR
    A[5 ficheiros Excel<br/>data/raw] --> B{Contratos de dados<br/>schemas.py · pandera}
    B -- erro --> X[Pipeline pára<br/>00_validacao_dados.csv]
    B -- ok --> C[Pré-processamento<br/>run_preprocessing.py]
    C --> D[Treino e seleção<br/>run_modelling.py]
    D --> E[Testes<br/>pytest]
    E -- falham --> Y[Não publica]
    E -- passam --> F[Publicação<br/>deploy-hf.yml]
    F --> G[Model repo]
    F --> H[Space<br/>app Gradio]
    G --> H
```

| Passo | Onde | Quando corre |
|---|---|---|
| Validação dos dados | `src/telco_churn/schemas.py`, chamado em `data.load_raw` e `data.merge_datasets` | Sempre que os dados são lidos |
| Pré-processamento e modelação | `scripts/run_pipeline.py` (no deploy, com `--sem-profiling`) | Localmente e no `deploy-hf.yml` |
| Testes | `tests/` (pytest) | Em cada push ou PR (`ci.yml`) e antes de publicar |
| Publicação | `scripts/publish_hf.py` | No `deploy-hf.yml`, manual ou em push para `main`, só se a pipeline e os testes passarem |

---

## Dados

Amostra IBM *Telco Customer Churn*: 7.043 clientes na Califórnia, num único extrato do Q3 ([fonte](https://www.ibm.com/docs/en/cognos-analytics/12.0.x?topic=samples-telco-customer-churn)). O enunciado está em `reports/project_guide.pdf`.

| Ficheiro | Grão | Chave |
|---|---|---|
| `Telco_customer_churn_demographics.xlsx` | cliente | `Customer ID` |
| `Telco_customer_churn_location.xlsx` | cliente | `Customer ID` |
| `Telco_customer_churn_services.xlsx` | cliente | `Customer ID` |
| `Telco_customer_churn_status.xlsx` | cliente | `Customer ID` |
| `Telco_customer_churn_population.xlsx` | ZIP code | `Zip Code` |

- **População de modelação:** clientes `Churned` e `Stayed` (6.589). Os 454 `Joined` ficam fora do treino.
- **Target:** `churn` = 1 se *Churned*, 0 se *Stayed* (28,4%).
- **Avaliação:** holdout estratificado 80/20 ao nível do cliente, `RANDOM_STATE = 42`. A divisão é feita uma vez, no pré-processamento, e todos os modelos são avaliados nos mesmos clientes.

---

## Estrutura

```
telco-churn-retention/
├── data/
│   ├── raw/                  # 5 Excel originais (não editar)
│   └── processed/            # gerado pela pipeline (git-ignored)
├── notebooks/
│   ├── 01_preprocessing_eda.ipynb   # EDA e pré-processamento
│   ├── 01b_profiling_ydata.ipynb    # relatórios ydata-profiling
│   └── 02_modelling_eda.ipynb       # modelos, benchmark IBM, priorização e decisões de retenção
├── src/telco_churn/
│   ├── config.py             # caminhos, seed, % de teste
│   ├── style.py              # paleta e tema dos gráficos
│   ├── schemas.py            # contratos de dados (pandera)
│   ├── data.py               # carregar com validação, unir, qualidade
│   ├── features.py           # população, papéis das variáveis, feature engineering
│   ├── analysis.py           # perfis, Cramér's V, Mutual Information
│   ├── plots.py              # gráficos da EDA
│   ├── profiling.py          # ydata-profiling com a paleta do projeto
│   ├── preprocessing.py      # split, ColumnTransformers, exportação
│   ├── modelling.py          # modelos, tuning, ensembles, métricas, holdout
│   ├── retention.py          # P vs P × CLTV, custo-benefício, ações, regra de onboarding
│   ├── scoring.py            # cliente, lote, fatores e decisão de desconto
│   └── model_plots.py        # gráficos da modelação
├── scripts/
│   ├── run_pipeline.py       # ponto de entrada: pergunta que fases correr
│   ├── run_preprocessing.py  # fase 1 sem notebook
│   ├── run_profiling.py      # fase 1b
│   ├── run_modelling.py      # fase 2 sem notebook
│   └── publish_hf.py         # publica modelo e app no Hugging Face
├── tests/                    # pytest: dados, pré-processamento, modelo, app
├── app/                      # app Gradio do Space
├── MODEL_CARD.md             # publicado com o modelo
├── outputs/                  # figuras, tabelas, artefactos (git-ignored)
├── reports/                  # enunciado, status reports, dicionário de fontes
├── requirements.txt          # versões mínimas
├── requirements-lock.txt     # versões exatas usadas nos resultados
└── pyproject.toml
```

Cada notebook tem um script equivalente, e os dois usam as mesmas funções de `src/telco_churn/`. Uma alteração a um modelo faz-se em `modelling.py` e vale para ambos.

---

## Como correr

### 1. Ambiente

```bash
git clone https://github.com/j3grave/Bootcamp.git
cd Bootcamp/telco-churn-retention

python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements-lock.txt
pip install -e .          # opcional: torna telco_churn importável em qualquer pasta
nbstripout --install      # uma vez por clone: limpa os outputs dos notebooks em cada commit
```

### 2. Pipeline

```bash
python scripts/run_pipeline.py
```

Pergunta no terminal o que correr: fase 1, profiling, fase 2 (normal ou rápida), testes e publicação no Hugging Face. Publicar corre sempre os testes antes; no modo rápido só prepara os ficheiros. Com `--sem-profiling` corre as fases 1 e 2 sem perguntas, como no deploy.

Cada fase também corre sozinha, por ordem, porque cada uma usa os ficheiros da anterior:

| Fase | Notebook (kernel `.venv`, *Run All*) | Linha de comandos |
|---|---|---|
| 1. Pré-processamento e EDA | `01_preprocessing_eda.ipynb` | `python scripts/run_preprocessing.py` |
| 1b. Profiling (opcional) | `01b_profiling_ydata.ipynb` | `python scripts/run_profiling.py` |
| 2. Modelação | `02_modelling_eda.ipynb` | `python scripts/run_modelling.py` |

A fase 2 demora cerca de 5 minutos, porque o tuning corre duas vezes (modelo core e de priorização). Para testar alterações depressa: `python scripts/run_modelling.py --rapido`.

### 3. Testes

```bash
pytest                          # cerca de 5 s; os do modelo precisam da fase 2
pytest tests/test_data.py -v    # só os contratos de dados
```

| Ficheiro | O que testa |
|---|---|
| `test_data.py` | Os ficheiros cumprem os contratos; derivações a 100%; merge 1:1; um valor fora do domínio (ex.: `Satisfaction Score = 9`) pára a pipeline |
| `test_preprocessing.py` | Treino e teste disjuntos e estratificados; sem leakage nas features; sem NaN depois do pré-processador |
| `test_model.py` | Os dois artefactos, sem retreinar: separação de estabelecidos e novos no lote, cortes de capacidade, [critério face à IBM](#critério-de-aceitação-face-ao-churn-score-ibm) e [gate do modelo de priorização](#gate-do-modelo-de-priorização) |
| `test_bootstrap.py` | O método do intervalo de confiança (scores iguais dão 0, resultado reprodutível) |
| `test_run_pipeline.py` | Ordem das fases e testes obrigatórios antes de publicar |
| `test_app.py` | Decisão do desconto nos casos limite (P = 0, P = 1, custo = 0, eficácia = 0, empate, valores inválidos) |

### 4. App local

```bash
pip install gradio==6.31.0
python app/app.py        # http://127.0.0.1:7860, com os modelos de outputs/artifacts/
```

---

## Fase 1 — Pré-processamento e EDA

| # | Etapa | O que faz |
|---|---|---|
| 1–2 | Carregamento | Lê os Excel com `keep_default_na=False`, para não perder o literal `None` |
| 3 | Validação | [Contratos de dados](#contratos-de-dados); pára se uma regra de erro falhar |
| 4 | União | Merge com `validate="one_to_one"` / `"many_to_one"`: uma linha por cliente |
| 5 | Qualidade | Missing, strings vazias, confirmação das derivações documentadas |
| 6 | População e target | Exclui `Joined`, cria o target, documenta o artefacto do tenure ≤ 3 |
| 7 | Papéis das variáveis | id, leakage, benchmark, redundante, numérica ou categórica |
| 8 | Numéricas | Skew, % de zeros, outliers IQR, distribuições por target, Spearman |
| 9 | Categóricas | Dicionário de categorias, Stayed vs Churned, Cramér's V |
| 10 | Associação e leakage | Mutual Information; verificação de Satisfaction Score e Churn Score |
| 11 | Feature engineering | `n_add_on_services`, `has_refund`, `avg_hist_charge`, `charge_delta` |
| 12 | Split e pré-processadores | 80/20 estratificado; `ColumnTransformer` *linear* e *trees* |
| 13 | Exportação | CSV e `joblib` para a fase 2 |

**Gera:** `data/processed/` (`train.csv`, `test.csv`, `joined_para_scoring.csv`, `telco_merged_full.csv`), cerca de 40 gráficos em `outputs/figures/`, tabelas `01_*` a `07_*` e `outputs/artifacts/preprocessing_config.joblib`.

---

## Fase 2 — Modelação

| # | Etapa | O que faz |
|---|---|---|
| 3 | Modelos | Os modelos, a rede neuronal e as nove métricas |
| 4 | Tuning | `RandomizedSearchCV` com `StratifiedKFold`, só no treino |
| 5 | Comparação | Todos no mesmo 5-fold, ao lado de três referências; regra de parcimónia; limiares de negócio |
| 6–7 | Variantes | Com e sem `Satisfaction Score`; `City` com `TargetEncoder` |
| 8 | Holdout | Avaliação única no teste, lado a lado com o Churn Score IBM |
| 9 | Interpretação | Permutation importance |
| 10 | Modelo de priorização | O mesmo protocolo só com tenure > 3 |
| 11 | Priorização | Lista por P(churn) × CLTV, custo-benefício por cenários, ações sugeridas |
| 12 | Clientes novos | Regra de onboarding para os Joined |
| 13 | Exportação | `final_model.joblib`, `priority_model.joblib`, `prioridade_holdout.csv`, `onboarding_joined.csv` |

### Protocolo

1. **Pipelines.** Cada modelo é um `Pipeline(prep, clf)`. O pré-processador é ajustado dentro de cada fold, por isso as médias, escalas e categorias do one-hot saem só do treino desse fold.
2. **Tuning.** `RandomizedSearchCV` com 20 combinações (`N_ITER`), 5 folds estratificados e ROC AUC, só no treino.
3. **Comparação.** Os sete modelos base e os dois ensembles são avaliados no mesmo 5-fold, com nove métricas, ao lado de três referências que nunca são candidatas:
   - **Dummy:** dá a todos a taxa de churn do treino;
   - **Regra de negócio** (`modelling.BusinessRuleScore`): sem ML, 1 ponto por cada fator de risco tirado da EDA (contrato Month-to-Month, tenure ≤ 12 meses, fibra ótica), de 0 a 3;
   - **Churn Score IBM:** o score / 100 nos mesmos folds.

   Guarda-se a média e o desvio-padrão em validação, a média no treino (para medir o sobreajuste) e as previsões *out-of-fold*, que servem para as curvas, matrizes de confusão e limiares sem tocar no holdout.
4. **Regra de parcimónia** (`select_parsimonious`). Parte-se do modelo com maior ROC AUC em CV. Os modelos a menos de um desvio-padrão dele contam como empatados, e entre eles escolhe-se o mais simples segundo `COMPLEXITY_ORDER`: Logistic Regression < Decision Tree < kNN < Random Forest < Extra Trees < Gradient Boosting < MLP < Soft Voting < Stacking. A decisão fica em `11d_selecao_modelo.csv` e na chave `selection` do `final_model.joblib`.
5. **Variantes.** O modelo escolhido é testado com `Satisfaction Score` (para medir o leakage) e com `City` via `TargetEncoder`. `City` só entra se ganhar mais do que um desvio-padrão em AUC.
6. **Holdout.** Usado uma vez, depois da escolha. Os outros modelos aparecem só por transparência.
7. **Modelo de priorização.** Os passos 2 a 4 repetem-se só com tenure > 3, no mesmo treino, e o resultado é comparado com o core e a IBM no mesmo holdout.

Usa-se validação cruzada em vez de um conjunto de validação fixo porque o extrato não tem datas e, com 6.589 clientes, 20% de validação seria só cerca de 1.300. Com 5 folds cada cliente do treino é validado uma vez, e o desvio-padrão entre folds mostra se uma diferença é real.

### Porque há dois modelos

Os Joined têm todos 1 a 3 meses de casa. Um cliente novo que fica é rotulado Joined e sai da população; um que sai é rotulado Churned e fica. Resultado: dentro de Churned + Stayed, os 597 clientes com tenure ≤ 3 saíram todos (`02b_tenure_estado.csv`). Não é comportamento, é a forma como os estados foram registados.

- O **modelo core** usa a população do enunciado (todos os tenures). Aprende que "cliente recente = churn", e parte do seu AUC vem desses casos fáceis.
- O **modelo de priorização** é treinado só com tenure > 3 e pontua os clientes estabelecidos. Nos mesmos clientes tem um AUC quase igual ao do core, mas não depende do artefacto: a importância de `Tenure in Months` cai para perto de zero.
- Os **clientes novos** não recebem P(churn): o core dar-lhes-ia P ≈ 1 e o de priorização nunca os viu. Recebem um segmento de onboarding (`retention.onboarding_segment`).

### Modelos

Há dois pré-processadores em `preprocessing.py`: `linear` (imputação, `StandardScaler` e one-hot) para os modelos que somam variáveis ou medem distâncias, e `trees` (imputação e one-hot, sem escala) para as árvores, que comparam cada variável consigo própria.

| Modelo | Como funciona | Hiperparâmetros afinados | Papel |
|---|---|---|---|
| Baseline (Dummy) | Dá a todos a taxa de churn do treino | — | Fasquia mínima: AUC 0,5, accuracy ~72% sem sinalizar ninguém |
| Logistic Regression | Soma ponderada das variáveis passada por uma sigmoide | `C` (0,001 a 100) | Referência interpretável e calibrada; não capta interações |
| Decision Tree | Perguntas sucessivas sobre uma variável de cada vez | `max_depth` (3–10), `min_samples_leaf` (5–50), `criterion` | O mais fácil de explicar; instável |
| k-Nearest Neighbors | Taxa de churn entre os `k` clientes mais parecidos | `n_neighbors` (15–100), `weights` | Comparação; sofre com muitas colunas one-hot |
| Neural Network (MLP) | Uma ou duas camadas escondidas e saída sigmoide, treino com Adam | `hidden_layer_sizes`, `activation`, `alpha`, `learning_rate_init` | Capta não linearidades; diversidade nos ensembles |
| Random Forest | 300 árvores em amostras bootstrap, média das probabilidades (bagging) | `max_depth`, `min_samples_leaf`, `max_features` | Forte e estável com pouca afinação |
| Extra Trees | Como a Random Forest, com cortes sorteados | Os mesmos | Variante com menos variância |
| Gradient Boosting | Árvores pequenas em sequência, cada uma corrige as anteriores (`HistGradientBoostingClassifier`) | `learning_rate`, `max_iter`, `max_leaf_nodes`, `min_samples_leaf`, `l2_regularization` | Normalmente o melhor em dados tabulares |
| Soft Voting | Média das probabilidades dos cinco membros | — | Ensemble simples, difícil de sobreajustar |
| Stacking | Regressão logística sobre as previsões out-of-fold dos membros (`cv=5`) | — | Ensemble com pesos aprendidos |
| Churn Score IBM | Score 0–100 do modelo que a empresa já usa | — | Benchmark; nunca entra como feature |

- **MLP.** Usa-se o `MLPClassifier` do scikit-learn porque encaixa no `Pipeline`, no tuning e nos ensembles sem novas dependências; para uma rede deste tamanho, Keras daria o mesmo. Tem *early stopping*: separa 15% do treino e pára ao fim de 20 épocas sem melhoria (máximo 500). A evolução está em `23e_rede_neuronal_treino.png`.
- **Ensembles.** Os membros (Logistic Regression, MLP, Random Forest, Extra Trees, Gradient Boosting) estão fixados em `ENSEMBLE_MEMBERS` antes de ver os resultados. A árvore isolada e o kNN ficam de fora por serem mais fracos. Um ensemble só ganha se os membros errarem em clientes diferentes: `20_diversidade_modelos.png` mostra a correlação entre eles.

### Métricas

As quatro primeiras usam as probabilidades; as outras cinco usam o limiar 0,5.

| Métrica | O que mede | Leitura neste projeto |
|---|---|---|
| ROC AUC | Probabilidade de um churner ter score mais alto do que um não-churner | **Métrica de seleção.** 0,5 = acaso |
| PR AUC | Área sob a curva precision-recall | Foca os churners; a base é 0,28 |
| Brier | Erro quadrático entre probabilidade e resultado | **Calibração**, precisa para P × CLTV |
| Log loss | Penaliza probabilidades confiantes e erradas | Também calibração |
| Accuracy | % bem classificados | Enganadora: o Dummy tem ~72% |
| Balanced accuracy | Média do recall nas duas classes | 0,5 = acaso |
| Precision | % dos sinalizados que saíram | Contactos bem gastos |
| Recall | % dos churners sinalizados | Cobertura |
| F1 | Média harmónica de precision e recall | Equilíbrio no limiar 0,5 |

A seleção é por ROC AUC porque a equipa de retenção trabalha com uma lista ordenada e contacta os primeiros N que o orçamento permite; o que conta é a ordenação. O Brier é o critério secundário.

**Limiares.** Além de 0,5 e do melhor F1, há dois limiares de capacidade: o P(churn) que sinaliza os top 10% e os top 20% dos clientes (`capacity_thresholds`). São fixados nas previsões out-of-fold do treino e aplicados uma vez ao holdout. A escolha final de quantos contactar vem do custo-benefício, não de uma métrica.

**Nível de risco e prioridade de contacto.** A app mostra as duas leituras de P(churn):

| | Pergunta | Cortes |
|---|---|---|
| Nível de risco (`RISK_LEVELS`) | Quão provável é sair? | baixo < 30% ≤ médio < 60% ≤ alto |
| Prioridade de contacto (`capacity_cuts`) | Cabe na capacidade da equipa? | top 10% dos estabelecidos (P ≥ 0,696), top 20% (P ≥ 0,443) |

Os cortes de capacidade são calculados só com clientes estabelecidos, nas previsões out-of-fold do modelo de priorização (`16d_*` e `17b_*`). No holdout sinalizam 8,4% e 18,0% dos estabelecidos, e o top 20% apanha 61% dos churners estabelecidos.

Os níveis leem-se como percentagens reais porque o modelo de priorização está calibrado. Nos 1.183 clientes estabelecidos do holdout (`17c_*`):

| Nível | % dos estabelecidos | P média | Churn real |
|---|---|---|---|
| baixo (< 30%) | 75% | 7% | 8% |
| médio (30–60%) | 13% | 44% | 42% |
| alto (≥ 60%) | 12% | 77% | 76% |

Na app, os clientes estabelecidos são pontuados pelo modelo de priorização e ordenados por P × CLTV. Os clientes novos ficam numa lista à parte, por segmento de onboarding e CLTV; a P(churn) do core aparece só como valor indicativo, com aviso.

### Decisões de retenção

- **P × CLTV vs só P.** `15b_P_vs_PxCLTV.csv` compara as duas listas: clientes em comum, churners e CLTV cobertos. A comparação com a IBM é sempre entre ordenações do mesmo tipo.
- **Quantos contactar.** Depende de dois pressupostos que o dataset não tem, a taxa de sucesso e o custo por contacto. Estão em `retention.CENARIOS`, com três cenários e uma grelha de sensibilidade. Contacta-se quando P × CLTV × taxa de sucesso > custo.
- **Ações por segmento** (`suggest_action`) e **regra de onboarding** são hipóteses. A importância das variáveis mostra o que o modelo usa, não o efeito de as mudar. Cada ação deve ser testada com um grupo de controlo sorteado (por exemplo, 20% de cada segmento sem contacto).

### Critério de aceitação face ao Churn Score IBM

O teste `test_nao_e_pior_do_que_o_churn_score_ibm` decide se o modelo core pode ser publicado. Exigir AUC ≥ AUC da IBM falharia (0,936 contra 0,941), por isso o critério é de não inferioridade. As três condições têm de passar:

| Condição (holdout) | Modelo | IBM | Limite | Resultado |
|---|---|---|---|---|
| ROC AUC ≥ AUC IBM − desvio-padrão da IBM em CV | 0,9362 | 0,9408 | 0,9331 | passa |
| PR AUC ≥ PR AUC IBM | 0,8826 | 0,8762 | 0,8762 | passa |
| Brier < Brier IBM | 0,0865 | 0,2142 | 0,2142 | passa |

A diferença de AUC (0,0046) é menor do que a variação da IBM entre folds (0,0077). É a mesma regra da seleção: dois modelos estão empatados quando a diferença é menor do que o desvio-padrão do melhor, que aqui é a IBM. O PR AUC e o Brier garantem que o modelo acrescenta algo: ordena pelo menos tão bem os churners e dá probabilidades utilizáveis (o Churn Score / 100 não é uma probabilidade).

O intervalo de confiança a 95% da diferença, por bootstrap emparelhado (`paired_bootstrap_auc_diff`, 2.000 reamostras dos 1.318 clientes do holdout, `random_state = 42`), é [−0,0217; +0,0121]. Inclui zero, o que confirma o empate por outro caminho. É só reporte (`13d_*`) e não decide o deploy.

A leitura certa é "empatado com a IBM em AUC, melhor em PR AUC e calibração", não "bate a IBM".

### Gate do modelo de priorização

A app usa o `priority_model.joblib` nos clientes estabelecidos, por isso este modelo tem um gate próprio. Nos 1.183 estabelecidos do holdout:

| Verificação | Priorização | IBM | Limite | Resultado | Papel |
|---|---|---|---|---|---|
| AUC > Dummy e > regra de negócio | 0,899 | — | 0,5 / 0,80 | passa | gate |
| Sobreajuste (AUC treino − CV) | 0,039 | — | < 0,05 | passa | gate |
| Brier < Brier IBM | 0,097 | 0,234 | 0,234 | passa | gate |
| Níveis de risco calibrados (\|real − P média\| < 0,10) | 0,02 | — | 0,10 | passa | gate |
| Cortes sinalizam ≈ 10% / 20% | 8,4% / 18,0% | — | ± 3 p.p. | passa | gate |
| AUC ≥ AUC IBM − desvio-padrão | 0,899 | 0,939 | 0,927 | não passaria | reporte |
| PR AUC ≥ PR AUC IBM | 0,726 | 0,831 | 0,831 | não passaria | reporte |
| IC 95% da diferença de AUC | −0,040 | | [−0,063; −0,018] | não inclui zero | reporte |

Nos clientes estabelecidos, a IBM ordena melhor, e a diferença é real. O modelo é publicado na mesma porque:

1. dá probabilidades calibradas, de que P × CLTV e a decisão de desconto precisam;
2. usa só variáveis conhecidas antes do churn, enquanto não se sabe com que informação o Churn Score foi calculado;
3. bate a Dummy e a regra de negócio, sem sobreajuste.

Esta inferioridade aparece na app (que diz não substituir o Churn Score e mostra as métricas da IBM), no `MODEL_CARD.md` e nas tabelas `17_*` e `17d_*`. O empate do core com a IBM na base completa vem em parte dos clientes novos, fáceis para ambos; nos estabelecidos o core também fica abaixo (AUC 0,900).

### Limitações

- **Brier e log loss da IBM** são indicativos: o score / 100 não é uma probabilidade. A comparação justa é por ROC AUC, PR AUC e curva de captura.
- **Validação cruzada não aninhada.** O tuning e a comparação usam os mesmos folds, por isso o AUC em CV do vencedor é ligeiramente otimista. A confirmação final é no holdout, que não entrou no tuning.
- **Uso do holdout.** Não serve para escolher modelos nem hiperparâmetros. É usado depois disso para as métricas finais, a variante com Satisfaction Score, a importância das variáveis, a curva de captura e o ótimo observado do custo-benefício.
- **Variáveis geográficas.** `Latitude`, `Longitude` e `Population` entram (AUC univariada 0,52–0,54). `City` só é testada com `TargetEncoder`; `Zip Code`, `Lat Long`, `Country` e `State` saem.

### Outputs da fase 2

| Tabelas (`outputs/tables/`) | Conteúdo |
|---|---|
| `10_tuning_modelos_base` | AUC em CV e melhores hiperparâmetros |
| `11_cv_modelos` | Nove métricas por modelo, CV e treino |
| `11b_limiares_modelo_final` | Precision, recall, F1 por limiar |
| `11c_correlacao_modelos` | Correlação entre previsões out-of-fold |
| `11d_selecao_modelo` | Regra de parcimónia e modelo escolhido |
| `11e_limiares_negocio_cv`, `13b_limiares_negocio_holdout` | Limiares 0,5, melhor F1, top 10% e top 20% |
| `12_variantes_satisfaction_city` | Efeito de Satisfaction Score e City |
| `13_holdout` | Nove métricas no holdout, com as referências |
| `13c_niveis_risco_holdout` | Clientes, P média e churn real por nível |
| `13d_bootstrap_auc_vs_ibm` | IC 95% da diferença de AUC, modelo core |
| `14_importancia_permutacao` | Queda de AUC por variável |
| `15_captura_valor`, `15b_P_vs_PxCLTV` | Captura de valor; P vs P × CLTV |
| `16_*`, `16b_*`, `16c_*` | Tuning, CV e seleção do modelo de priorização |
| `16d_*`, `17b_*` | Cortes de capacidade nos estabelecidos (OOF e holdout) |
| `17_comparacao_core_priorizacao` | Core vs priorização vs IBM |
| `17c_*`, `17d_*` | Níveis de risco e IC 95% do modelo de priorização |
| `18_importancia_priorizacao` | Importância no modelo de priorização |
| `19_regra_onboarding` | Segmentos de onboarding e taxa de saída |
| `20_cenarios_custo_beneficio`, `20b_sensibilidade_pct_contactar` | N a contactar por cenário e sensibilidade |
| `21_acoes_sugeridas` | Top-N do cenário Base por ação |

| Gráficos (`outputs/figures/`) | O que mostra |
|---|---|
| `20_diversidade_modelos` | Correlação entre modelos |
| `21_cv_modelos` | ROC AUC em CV com as referências |
| `22_painel_metricas_cv`, `22b_heatmap_metricas_cv` | Nove métricas por modelo em CV |
| `23_sobreajuste` | AUC treino vs CV |
| `23b_roc_pr_cv`, `23c_matrizes_confusao_cv` | Curvas e matrizes de confusão out-of-fold |
| `23d_limiar_modelo_final` | Métricas por limiar |
| `23e_rede_neuronal_treino` | Treino da MLP e early stopping |
| `24_roc_calibracao`, `24b_painel_metricas_holdout` | ROC, calibração e métricas no holdout |
| `25_captura_valor`, `26_mapa_prioridade` | Captura de CLTV e mapa P × CLTV |
| `27_importancia_permutacao`, `29_importancia_priorizacao` | Importância das variáveis |
| `28_cv_modelos_estabelecidos` | AUC em CV com tenure > 3 |
| `30_valor_liquido` | Valor líquido por nº de contactos e cenário |

Também gera `outputs/artifacts/final_model.joblib`, `priority_model.joblib` e, em `data/processed/`, `prioridade_holdout.csv` e `onboarding_joined.csv`.

---

## Decisões principais

| Decisão | Porquê |
|---|---|
| Excluir `Customer Status` e `Churn Label/Value/Category/Reason` | Codificam o resultado ou só existem depois do churn |
| `Churn Score` como benchmark, não feature | É a saída do modelo IBM existente |
| `CLTV` fora das features | É usado na priorização, P(churn) × CLTV |
| `Satisfaction Score` fora por defeito | Scores 1–2 dão 100% de churn; com ele o AUC sobe para 0,996, sinal de leakage |
| Remover `Under 30`, `Senior Citizen`, `Dependents`, `Referred a Friend`, `Total Revenue` | Derivações exatas de outras colunas (verificadas a 100%) |
| Sem PCA | As árvores não precisam e a interpretabilidade conta na retenção |
| Taxa de sucesso e custo por contacto como pressupostos | Não existem nos dados; três cenários e sensibilidade |
| Contratos com duas severidades | Erro pára a pipeline; aviso regista e continua |
| Modelo num Model repo separado do Space | O Space descarrega-o com `hf_hub_download`; o pacote `telco_churn` é copiado no deploy |
| Gráficos da modelação em `model_plots.py` | Mexer nos gráficos da EDA não apaga os da modelação |

---

## Contratos de dados

As regras estão em `src/telco_churn/schemas.py` e o resultado de cada uma em `outputs/tables/00_validacao_dados.csv`. Uma regra de **erro** pára a pipeline com `DataValidationError`; um **aviso** fica registado e a pipeline continua. Todas as regras de erro passam a 100% nos dados atuais.

| Camada | Tipo | Regras | Severidade |
|---|---|---|---|
| 5 ficheiros | Esquema | Colunas esperadas e tipos | erro |
| demographics, location, services, status | Chaves | `Customer ID` único e igual nos 4 ficheiros | erro |
| population / location | Chaves | `Zip Code` único em population e presente para todos os de location | erro |
| status | Domínios | `Satisfaction Score` ∈ {1..5}; `Churn Score` ∈ [0, 100]; `CLTV` > 0; estados, labels e categorias conhecidos | erro |
| services | Domínios | Contrato, oferta, internet e pagamento nas categorias conhecidas; `Tenure` ∈ [1, 120]; `Monthly Charge` ∈ [0, 300]; `Total Charges` ∈ [0, 50 000]; valores ≥ 0 | erro |
| demographics, location, population | Domínios | `Age` ∈ [18, 100]; dependentes ∈ [0, 15]; Califórnia, EUA; `Population` > 0 | erro |
| base unida | Regras de negócio | `Churn Label = Yes` ⇔ `Churned`; razão e categoria só nos Churned; sem internet ⇒ sem add-ons; sem telefone ⇒ sem linhas múltiplas | erro |
| base unida | Derivações | `Under 30`, `Senior Citizen`, `Dependents`, `Referred a Friend` e `Total Revenue` coerentes com a origem | erro |
| base unida | Estatísticas | ≈ 7.043 clientes (± 5%); churn ≈ 28,4% (± 3 p.p.); ≤ 5% de missing; Joined ⇒ tenure ≤ 3 | aviso |
| pré-processamento | Saída | Treino e teste sem IDs comuns; churn semelhante (± 1 p.p.); sem colunas de leakage; zero NaN | erro |

Os intervalos são plausíveis, não os mínimos e máximos observados (idades observadas 19–80, contrato aceita 18–100). Para ver uma regra a falhar: mudar um `Satisfaction Score` para 9 no Excel de status e correr `python scripts/run_preprocessing.py`.

---

## Identidade visual

| Cor | HEX | Uso |
|---|---|---|
| Azul Digital | `#3777FF` | Destaque principal, *Stayed*, títulos |
| Verde Néon | `#0EE071` | Acento, *Churned*, realces |
| Preto Profundo | `#000000` | Fundo |

A paleta está em `src/telco_churn/style.py`. As secções dos notebooks usam HTML com estilos *inline*, visíveis no VS Code e no Jupyter; o GitHub remove-os na pré-visualização, mas o texto continua legível.

---

## Git

```bash
git checkout -b feature/<nome-curto>
git add .
git commit -m "EDA: adiciona análise de contratos"
git push -u origin feature/<nome-curto>
# Pull Request -> revisão -> merge para main
```

- Não fazer commit de dados processados nem de outputs: regeneram-se localmente.
- Não editar `data/raw/`.
- Funções novas vão para `src/telco_churn/`; os notebooks servem para explorar e narrar.
