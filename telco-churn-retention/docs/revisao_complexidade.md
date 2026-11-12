# Revisão de complexidade do código

Fase 1 do plano em `INSTRUCOES_CLAUDE_CODE.md`. Objetivo: código que um colega e o professor
percebam numa leitura, sem perder funcionalidade e **sem alterar resultados**.

Estado: **concluído**. As 12 propostas foram aprovadas e aplicadas, uma por commit; métricas finais na secção 5.

Base medida: `main` + `Nova-Versao-com-ML-Supervionado` + `fix/auditoria-correcoes`
(branch `feature/validacao-e-app-hf`).

## 1. Métricas antes

Ferramentas: `radon cc -s -a src scripts`, `radon mi -s src scripts`, `vulture src scripts`.

- Complexidade ciclomática (CC): nº de caminhos possíveis numa função. A = 1–5, B = 6–10,
  C = 11–20. Quanto mais baixa, mais fácil de ler e de testar.
- Índice de manutenibilidade (MI): 0–100, maior é melhor. O radon dá nota A acima de 20.

| Módulo | Linhas | MI | CC máxima |
|---|---:|---:|---:|
| `src/telco_churn/analysis.py` | 76 | 40,9 | 4 (A) |
| `src/telco_churn/config.py` | 30 | 71,6 | 2 (A) |
| `src/telco_churn/data.py` | 79 | 52,0 | 6 (B) |
| `src/telco_churn/features.py` | 87 | 67,9 | **13 (C)** |
| `src/telco_churn/model_plots.py` | 361 | 33,8 | **12 (C)** |
| `src/telco_churn/modelling.py` | 382 | 51,3 | 5 (A) |
| `src/telco_churn/plots.py` | 178 | 44,6 | 6 (B) |
| `src/telco_churn/preprocessing.py` | 49 | 55,3 | 2 (A) |
| `src/telco_churn/profiling.py` | 187 | 61,6 | 5 (A) |
| `src/telco_churn/retention.py` | 169 | 65,8 | 3 (A) |
| `src/telco_churn/style.py` | 81 | 73,7 | 3 (A) |
| `scripts/run_all.py` | 45 | 80,1 | 5 (A) |
| `scripts/run_modelling.py` | 294 | 45,3 | **11 (C)** |
| `scripts/run_pipeline.py` | 81 | 59,3 | 3 (A) |
| `scripts/run_profiling.py` | 38 | 77,5 | 2 (A) |

Resumo: 87 funções, CC média 3,0 (A). 78 com nota A, 5 com B, **4 com C**:

| Função | CC |
|---|---:|
| `features.feature_roles` | 13 (C) |
| `model_plots.plot_metrics_panel` | 12 (C) |
| `model_plots.plot_roc_pr` | 12 (C) |
| `run_modelling.main` | 11 (C) |
| `model_plots.plot_confusion_matrices` | 8 (B) |
| `model_plots.plot_threshold` | 6 (B) |
| `plots.plot_categorical_detail` | 6 (B) |
| `data.validate_keys` | 6 (B) |
| `data.merge_datasets` | 6 (B) |

Notebooks (linhas de código nas células):

| Notebook | Células de código | Linhas | Maior célula |
|---|---:|---:|---:|
| `01_preprocessing_eda` | 23 | 222 | 72 |
| `01b_profiling_ydata` | 6 | 42 | 21 |
| `02_modelling_eda` | 34 | 437 | 44 |

Código sem uso (vulture, confirmado à mão contra `src/`, `scripts/` e os três notebooks):

| Símbolo | Situação |
|---|---|
| `style.CMAP_SEQUENTIAL` | Sem uso em lado nenhum |
| `y_tr` em `run_pipeline.py` | Variável nunca lida |
| `analysis.high_correlations` | Só usada no notebook 01 — manter |
| `modelling.MODEL_INFO`, `modelling.metrics_glossary` | Só usadas no notebook 02 — manter |
| `style.style_table` | Usada 23 vezes nos notebooks — manter |
| `profiling.primary_colors` | Falso positivo (atributo de configuração do ydata-profiling) |

## 2. Diagnóstico

O código **não está demasiado complexo no geral**: 90% das funções têm nota A e os módulos
são pequenos. Os problemas estão concentrados em três sítios:

1. **A lógica da modelação existe duas vezes.** `scripts/run_modelling.py` (função `main` com
   245 linhas) e o notebook 02 (437 linhas) repetem os mesmos passos, incluindo decisões:
   a regra para incluir `City`, a escolha do modelo final e a exportação. As duas versões já
   divergiram: o notebook guarda a chave `cv` no `final_model.joblib` e o script não. Qualquer
   mudança (por exemplo, a regra de parcimónia da Fase 3) teria de ser feita nos dois lados.
2. **Duas funções de gráficos com nota C** (mais `feature_roles`) e formatação repetida em `model_plots.py`
   (anotação de células de heatmap em três funções; a grelha de limiares calculada em três
   sítios).
3. **Pequenos restos**: uma função redundante (`cv_compare`), uma constante e uma variável
   sem uso, e o docstring de `plots.py`, que diz "fundo preto" quando o tema ativo é claro.

Dois pontos das instruções já estavam resolvidos pelo branch de auditoria:
`modelling.score_metrics` foi removida e o workflow `python-publish.yml` (template PyPI) foi
apagado.

## 3. Propostas

Cada proposta é aplicada num commit separado e validada com
`python scripts/run_all.py --sem-profiling`, comparando `outputs/tables/` com a cópia de
referência em `outputs/_referencia/`.

### Risco baixo

| # | O quê | Porquê | Ganho |
|---|---|---|---|
| 1 | `features.feature_roles`: construir a tabela de papéis a partir de um dicionário `papel → colunas`, com um ciclo | A nota C vem de sete list comprehensions seguidas | C (13) → A |
| 2 | `model_plots.plot_metrics_panel`: extrair `_label_bars` (rótulos dentro/fora da barra) | A função mistura desenho, escala e rótulos | C (12) → B |
| 3 | `model_plots.plot_roc_pr`: extrair `_curve_style` (cor, espessura e traço de cada modelo) | Quatro condições por curva dentro do ciclo | C (12) → A/B |
| 4 | `model_plots`: helper `_annotate_cells` para escrever valores nas células | O mesmo ciclo duplo está em `plot_metrics_heatmap`, `plot_confusion_matrices` e `plot_diversity` | −15 linhas; B (8) → A |
| 5 | `plot_threshold` e `best_f1_threshold` passam a usar `modelling.threshold_table` | Precision/recall/F1 por limiar são calculados três vezes, uma delas com um import dentro da função | Uma só fonte; B (6) → A |
| 6 | Remover `modelling.cv_compare`; usar `cv_evaluate` com a constante `SUMMARY_COLS` | É um atalho de três linhas e a lista de colunas está repetida no script e no notebook | −1 função pública |
| 7 | `modelling.permutation_ranking(model, X, y)` | O bloco de permutation importance está escrito duas vezes no script e duas no notebook | −20 linhas |
| 8 | Constante `features.BUSINESS_COLS` (`Customer ID`, `CLTV`, `Churn Score`, `Satisfaction Score`) | A lista está repetida em `run_pipeline.py`, `run_profiling.py` e no notebook 01 | Uma só fonte |
| 9 | Remover `style.CMAP_SEQUENTIAL` e `y_tr`; corrigir o docstring de `plots.py` | Código e texto sem correspondência com o que corre | Limpeza |

### Risco médio

| # | O quê | Porquê | Risco | Ganho |
|---|---|---|---|---|
| 10 | Dividir `run_modelling.main` em funções por etapa (`stage_cv`, `stage_variants`, `stage_holdout`, `stage_priority_model`, `stage_retention`, `stage_onboarding`) e passar as **decisões** para `modelling.py` (`include_city`, `fit_predict`, `save_bundle`) e `retention.py` (`CENARIOS`, `onboarding_rule`). *Nomes finais; na proposta original estavam em português — ficaram em inglês, como as restantes funções do projeto.* | 245 linhas numa função; decisões duplicadas e já divergentes entre script e notebook | A ordem de execução pode mudar resultados se alguma etapa partilhar estado; valida-se com a comparação das tabelas | C (11) → A; uma só versão das decisões; a Fase 3 (parcimónia) e a Fase 5 (deploy) ficam com um único sítio para mexer |
| 11 | Notebook 02 passa a chamar as funções da proposta 10 em vez de repetir a lógica | Regra do projeto: a lógica vive em `src/`, o notebook mostra resultados | O notebook não é coberto por `run_all.py`; valida-se executando-o com `nbconvert` e comparando as tabelas | 437 → cerca de 300 linhas; fim da divergência |
| 12 | Notebook 01: substituir a célula de 72 linhas que instala pacotes com `pip` por uma nota a apontar para `requirements.txt` | Instalar pacotes dentro de um notebook é frágil e não é análise | Quem abrir o notebook sem ambiente tem de correr `pip install -r requirements.txt` | −70 linhas; o notebook começa nos dados |

## 4. O que não vale a pena mudar

- **`profiling.py`** (187 linhas): mais de metade é o dicionário de descrições das variáveis.
  É dado, não lógica.
- **`analysis.py`** (MI 40,9): o índice baixa por causa das fórmulas (Cramér's V com correção),
  não por causa da estrutura. Funções curtas e todas com nota A.
- **`retention.py`, `preprocessing.py`, `config.py`, `style.py`**: pequenos, nota A, lineares.
- **`modelling.MODEL_INFO` e `METRICS`**: tabelas descritivas usadas no notebook; ocupam linhas
  mas não acrescentam complexidade.
- **Separação `plots.py` / `model_plots.py`**: juntar daria um módulo de 540 linhas; a divisão
  por fase (EDA vs modelação) é clara.
- **`plot_categorical_detail` (B 6)**: gráfico com dois painéis, lido de cima para baixo;
  partir em funções não ajuda a leitura.
- **`data.validate_keys` e `data.merge_datasets` (B 6)**: são reescritas na Fase 2
  (validação com pandera); não se mexe duas vezes.
- **Uma camada de "pipeline" ou classes de configuração**: o projeto corre com quatro scripts
  simples; acrescentar abstrações iria contra o objetivo.

## 5. Métricas depois

Todas as propostas foram aplicadas. Validação após cada uma: `run_all.py --sem-profiling` e
comparação com `outputs/_referencia/` — as 25 tabelas, os ficheiros de `data/processed/` e as
59 figuras (pixel a pixel) ficaram **iguais**. Para as propostas de gráficos usou-se também um
banco de ensaio que desenha cada função com dados fixos antes e depois. Os notebooks 01 e 02
foram executados de ponta a ponta e as tabelas que escrevem também ficaram iguais.

Única diferença intencional: `final_model.joblib` passa a guardar também a chave `cv`
(métricas de CV do modelo escolhido), como o notebook já fazia.

| Módulo | Linhas antes | Linhas depois | MI antes | MI depois | CC máx. antes | CC máx. depois |
|---|---:|---:|---:|---:|---:|---:|
| `features.py` | 87 | 89 | 67,9 | 69,7 | 13 (C) | 10 (B) |
| `model_plots.py` | 361 | 368 | 33,8 | 35,8 | 12 (C) | 10 (B) |
| `modelling.py` | 382 | 417 | 51,3 | 51,6 | 5 (A) | 5 (A) |
| `retention.py` | 169 | 189 | 65,8 | 65,1 | 3 (A) | 3 (A) |
| `style.py` | 81 | 79 | 73,7 | 73,2 | 3 (A) | 3 (A) |
| `scripts/run_modelling.py` | 294 | 320 | 45,3 | 41,4 | 11 (C) | 4 (A) |
| restantes | — | sem alteração | — | — | — | — |

| | Antes | Depois |
|---|---:|---:|
| Funções analisadas | 87 | 103 |
| Nota A / B / C | 78 / 5 / 4 | 96 / 7 / **0** |
| CC média | 3,0 | 2,5 |
| Notebook 01 (linhas de código / maior célula) | 222 / 72 | 158 / 24 |
| Notebook 02 (linhas de código / maior célula) | 437 / 44 | 402 / 39 |

Leitura honesta dos números:

- **Nenhuma função fica com nota C.** As quatro que tinham C descem para B (10, 10, 7) ou A (4).
  `feature_roles` e `plot_metrics_panel` ficam em B 10: o radon conta cada condição dentro das
  list comprehensions; partir mais estas funções não as tornaria mais legíveis.
- **`run_modelling.py` tem mais linhas e MI ligeiramente mais baixo.** O MI penaliza o volume
  de código, e dividir `main` em 11 funções com docstrings acrescenta linhas. Em troca, a função
  mais complexa passa de C 11 para A 4 e cada etapa lê-se isoladamente.
- **O notebook 02 encolheu menos do que o previsto (≈300).** A maior parte das células já só
  chamava funções e mostrava tabelas, o que é o papel de um notebook. O ganho real está nas
  decisões que deixaram de estar duplicadas: inclusão de City (`include_city`), treino final
  (`fit_predict`), exportação (`save_bundle`), cenários de custo-benefício
  (`retention.CENARIOS`) e regra de onboarding (`retention.onboarding_rule`) passam a existir
  num só sítio, usado pelo script e pelo notebook.
- **`modelling.py` cresceu 35 linhas** com as funções que antes estavam repetidas no script e
  no notebook.
