# Instruções para o Claude Code — Projeto Telco Churn (EDSB 2026, Grupo 1)

> Copiar este ficheiro para a raiz do repositório e pedir ao Claude Code:
> **"Lê o INSTRUCOES_CLAUDE_CODE.md e executa as fases por ordem, parando nos pontos de confirmação."**

---

## Contexto

- Repositório: `https://github.com/j3grave/Bootcamp`, projeto na pasta `telco-churn-retention/`.
- Código em `src/telco_churn/` (data, features, preprocessing, analysis, modelling, plots, model_plots, profiling, style, config), scripts em `scripts/` (`run_pipeline.py`, `run_profiling.py`, `run_modelling.py`, `run_all.py`) e notebooks em `notebooks/` (01, 01b, 02).
- O modelo final é exportado para `outputs/artifacts/final_model.joblib` como dicionário com `model` (Pipeline sklearn com preprocessor + classificador), `name`, `features`, `target`, `best_params` e `holdout`.
- População de modelação: Churned vs Stayed; os "Joined" servem para scoring. Benchmark a bater: AUC do `Churn Score` da IBM.
- O professor pediu três coisas: **(1) testes de validação aos dados na ingestão**, tal como existem testes ao código; **(2) seleção de modelo validada contra baselines**; **(3) publicar o modelo como app no Hugging Face, integrada no pipeline**. O grupo quer ainda **(4) avaliar se o código está demasiado complexo e simplificá-lo**.

## Regras de trabalho

1. Trabalhar num branch novo: `feature/validacao-e-app-hf`. Nunca fazer commit direto no `main`.
2. Commits pequenos, um por fase/subfase, com mensagens em português.
3. Comentários, docstrings, README e textos da app em **português europeu**.
4. Paleta gráfica obrigatória: Azul Digital `#3777FF` (destaque), Verde Néon `#0EE071` (acento), Preto `#000000` (fundo). Reutilizar `style.py`.
5. **Não alterar resultados**: depois de cada refactor, correr `python scripts/run_all.py --sem-profiling` e confirmar que as tabelas em `outputs/tables/` (AUC, PR-AUC, Brier, holdout) ficam iguais às de antes. Guardar uma cópia de referência das tabelas antes de começar.
6. Preferir soluções simples: pandera + pytest (não Great Expectations), Gradio (não Streamlit/Docker), sem novas camadas de abstração desnecessárias.
7. Parar e pedir confirmação nos pontos marcados com ⏸.

---

## Fase 0 — Diagnóstico inicial

1. `git pull`, criar o branch, instalar dependências e correr `run_all.py --sem-profiling` para garantir que tudo funciona.
2. Guardar uma cópia de referência de `outputs/tables/` em `outputs/_referencia/` (não versionar).
3. Verificar `.github/workflows/python-publish.yml` na raiz do repositório: é um template de publicação no PyPI que não se aplica ao projeto — propor removê-lo e substituí-lo na Fase 5.

## Fase 1 — Revisão de complexidade e simplificação

Objetivo: código que um colega e o professor percebam numa leitura, sem perder funcionalidade.

1. Medir com `radon cc -s -a src scripts` e `radon mi -s src scripts`, e contar linhas por módulo. Pontos já identificados como candidatos:
   - funções com complexidade C: `features.feature_roles`, `model_plots.plot_metrics_panel`, `model_plots.plot_roc_pr`;
   - `modelling.score_metrics` ("mantida por compatibilidade") — remover se não tiver uso;
   - `modelling.cv_compare` vs `cv_evaluate` — sobreposição; ver se uma chega;
   - `model_plots.py` (≈340 linhas) e `plots.py` — procurar código repetido de formatação de eixos/títulos/cores que possa ir para um helper;
   - notebooks vs scripts — a lógica deve viver em `src/`; os notebooks só chamam funções e mostram resultados. Identificar lógica duplicada nos notebooks;
   - código morto: funções, imports e constantes sem uso (usar `vulture src scripts`).
2. Produzir `docs/revisao_complexidade.md` com: tabela de métricas antes, lista de propostas (o quê, porquê, risco, ganho) e o que **não** vale a pena mudar.
3. ⏸ **Mostrar o relatório e esperar aprovação antes de refatorar.**
4. Aplicar as propostas aprovadas, uma de cada vez, validando a regra 5 após cada uma. Atualizar o relatório com as métricas depois.

## Fase 2 — Validação de dados (data contracts)

1. Criar `src/telco_churn/schemas.py` com esquemas **pandera** para os 5 ficheiros brutos e para a base unida:
   - **Esquema:** colunas esperadas e tipos.
   - **Chaves:** Customer ID único; mesmos IDs nos 4 ficheiros de cliente; Zip Code único em population (reaproveitar a lógica de `data.validate_keys`).
   - **Domínios:** Satisfaction Score ∈ {1..5}; Churn Score ∈ [0, 100]; Contract ∈ {Month-to-Month, One Year, Two Year}; Customer Status ∈ {Churned, Stayed, Joined}; Churn Label ∈ {Yes, No}; State = California; intervalos de Age, Tenure in Months, Monthly Charge, Total Charges — **confirmar primeiro os mínimos/máximos reais nos dados** e definir intervalos plausíveis.
   - **Regras de negócio:** Churn Label = Yes ⇔ Customer Status = Churned; Churn Reason/Category preenchidos só nos Churned; Internet Service = No ⇒ add-ons de internet = No e Internet Type = None; Phone Service = No ⇒ Multiple Lines = No e Avg Monthly Long Distance Charges = 0; as derivações de `data.derivation_checks` passam a regra com 100% obrigatório. Verificar cada regra nos dados antes de a tornar obrigatória; se alguma não for 100%, documentar e passá-la a aviso.
   - **Estatísticas (aviso, não erro):** nº de linhas ≈ 7 043; taxa de churn perto da de referência; % de missing abaixo de um limiar.
2. Duas severidades: **erro** (a pipeline pára) e **aviso** (regista e continua).
3. Aplicar a validação em `data.load_raw()`/`merge_datasets()` e gerar `outputs/tables/00_validacao_dados.csv` (regra, camada, severidade, resultado, nº de linhas a falhar).
4. Validação de saída do pré-processamento: treino e teste sem Customer IDs em comum; taxas de churn semelhantes (estratificação); nenhuma coluna de `LEAKAGE_COLS`, `MODEL_DERIVED_COLS` ou `SUSPECT_COLS` nas features finais; zero NaN depois do preprocessor.

## Fase 3 — Validação e seleção de modelo

O que já existe e deve manter-se: Dummy baseline, benchmark IBM no holdout, StratifiedKFold no treino, holdout usado uma vez, métricas AUC/PR-AUC/Brier, `tied_with_best`, variantes com/sem Satisfaction e City, curva de captura CLTV.

Acrescentar:

1. **Regra de parcimónia:** entre os modelos empatados com o melhor (`tied_with_best`), escolher o mais simples, segundo uma ordem explícita em `modelling.py` (ex.: Logistic Regression < Random Forest < Boosting < MLP < Stacking). Registar no output qual foi o modelo de maior AUC e qual foi escolhido, e porquê.
2. **Benchmark IBM também na CV:** calcular o AUC do `Churn Score` em cada fold de validação e incluí-lo na tabela de CV como linha de referência.
3. **Baseline de regra de negócio:** uma regra simples sem ML (ex.: risco = contrato Month-to-Month e tenure baixo, ou score por pontos com 2–3 variáveis), avaliada na CV e no holdout ao lado das outras baselines.
4. **Limiar de decisão de negócio:** além de 0,5 e do melhor F1, um limiar definido pela capacidade da equipa de retenção (top 10% / 20% de clientes) e mostrar precisão/recall nesses cortes.
5. Atualizar `docs/modelling.md` com a lógica de seleção.

## Fase 4 — Testes automáticos (pytest)

Criar `tests/` com fixtures que leem os dados uma vez (escopo de sessão):

- `test_data.py` — os esquemas pandera passam; derivações a 100%; merge 1:1 sem perder clientes.
- `test_preprocessing.py` — treino/teste disjuntos; sem leakage nas features; sem NaN após o preprocessor; `add_features` não usa estatísticas do dataset (dá o mesmo resultado linha a linha).
- `test_model.py` (corre sobre `final_model.joblib`) — AUC no holdout > AUC da Dummy e ≥ AUC do Churn Score IBM; gap treino-validação na CV abaixo de um limiar (definir, ex.: 0,05); o modelo aceita uma linha do formulário da app e devolve uma probabilidade em [0, 1].
- `test_app.py` — a função de decisão do desconto devolve resultados coerentes em casos limite (P=0, P=1, custo=0).

Os testes devem correr em menos de ~2 minutos (sem retreinar o modelo completo; usar o artefacto exportado).

## Fase 5 — App no Hugging Face e integração no pipeline

1. Criar a pasta `app/` com:
   - `app.py` em **Gradio**:
     - **Separador "Cliente individual":** formulário com as variáveis de entrada do modelo (apenas as originais; as derivadas são calculadas com `features.add_features`). Valores por omissão = mediana/moda. Sem campos de leakage.
     - **Resultado:** P(churn), segmento de risco (baixo/médio/alto, com cortes definidos na Fase 3) e principais fatores do cliente (SHAP, se não complicar demasiado — caso contrário a permutation importance global).
     - **Decisão de desconto:** inputs CLTV (por omissão a mediana), desconto mensal (€), duração (meses) e eficácia assumida do desconto (slider, por omissão 30%). Regra: dar desconto se `P(churn) × eficácia × CLTV > desconto × meses`. Mostrar o ganho esperado e a recomendação em linguagem simples.
     - **Aviso** quando o tenure for muito baixo (cliente novo = extrapolação face aos dados de treino) e nota fixa de que a eficácia é um pressuposto (o modelo prevê risco, não o efeito causal do desconto).
     - **Separador "Lote":** upload de CSV → devolve a lista ordenada por P(churn) × CLTV (reutilizar `modelling.priority_list`) para download.
     - Tema com a paleta do projeto.
   - `requirements.txt` mínimo (versões fixas, compatíveis com as usadas no treino — sobretudo scikit-learn, para o joblib carregar).
   - `README.md` com o cabeçalho YAML exigido pelos Spaces (`sdk: gradio`, `app_file: app.py`, título, etc.).
2. O modelo vai para um **Model repo** no Hugging Face (não dentro do Space), com um `MODEL_CARD.md`: objetivo, dados, população, features, métricas no holdout vs IBM, limitações (clientes novos, causalidade do desconto, dados sintéticos IBM). A app descarrega o modelo com `huggingface_hub.hf_hub_download`.
3. O pacote `telco_churn` tem de estar disponível na app (para `add_features`): copiar o necessário no deploy ou instalar o pacote — escolher a opção mais simples e documentá-la.
4. Testar a app localmente (`python app/app.py`) antes de qualquer publicação.
5. **GitHub Actions** (na raiz do repositório, `.github/workflows/`), substituindo o template PyPI:
   - `ci.yml` — em cada push/PR: instalar, correr `pytest` (dados + preprocessing + app).
   - `deploy-hf.yml` — manual (`workflow_dispatch`) ou em push para `main`: correr pipeline → testes de modelo → **só se tudo passar**, publicar o modelo no Model repo e a pasta `app/` no Space, usando o secret `HF_TOKEN`.
6. ⏸ **Antes de publicar no Hugging Face, pedir: nome da conta/organização HF, nomes do Model repo e do Space, e confirmação de que o `HF_TOKEN` foi criado nos secrets do GitHub.** Não criar contas nem tokens.

## Fase 6 — Documentação e entrega

1. Atualizar o `README.md` do projeto: diagrama do fluxo (dados → validação → treino → testes → publicação → app), como correr os testes, link da app.
2. Secção "Data Contracts" no README com a tabela de regras.
3. Abrir um Pull Request para `main` com um resumo das alterações por fase e o relatório de complexidade (antes/depois).

## Critérios de aceitação

- [ ] `run_all.py --sem-profiling` corre sem erros e as métricas do modelo são iguais às de referência (ou as diferenças estão explicadas).
- [ ] `pytest` passa localmente e no GitHub Actions.
- [ ] Introduzir um erro deliberado num ficheiro de dados (ex.: Satisfaction Score = 9) faz a pipeline parar com uma mensagem clara.
- [ ] O relatório de complexidade mostra as métricas antes/depois e nenhum módulo fica com funções de complexidade C ou pior sem justificação.
- [ ] A app abre no Space, devolve P(churn) e a recomendação de desconto para um cliente introduzido manualmente e para um CSV.
- [ ] O modelo só é publicado quando os testes passam.
