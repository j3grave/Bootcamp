# Fase de modelação: protocolo, métricas e decisões

Este documento descreve como os modelos são treinados, comparados e escolhidos, e como cada métrica deve ser lida. A descrição de cada modelo está em [`models.md`](models.md).

## Onde está cada coisa

| Ficheiro | Conteúdo |
|---|---|
| `notebooks/02_modelling_eda.ipynb` | Narrativa completa da modelação, com tabelas e gráficos |
| `scripts/run_modelling.py` | Mesmo fluxo sem notebook (também chamado por `scripts/run_all.py`) |
| `src/telco_churn/modelling.py` | Modelos, tuning, ensembles, métricas, avaliação e priorização |
| `src/telco_churn/retention.py` | Decisões de retenção: P vs P × CLTV, custo-benefício, ações sugeridas e regra de onboarding |
| `src/telco_churn/model_plots.py` | Gráficos da modelação |
| `docs/models.md` | Explicação de cada modelo |

O notebook e o script usam as mesmas funções de `src/telco_churn`. Uma alteração num modelo faz-se uma vez em `modelling.py` e passa a valer nos dois.

## Ligação ao pré-processamento

A modelação começa onde o notebook 01 (ou `scripts/run_pipeline.py`) acaba:

```
notebook 01 / run_pipeline.py
    └─> data/processed/train.csv, test.csv, joined_para_scoring.csv, telco_merged_full.csv
    └─> outputs/artifacts/preprocessing_config.joblib  (listas de features e target)
            │
            ▼
notebook 02 / run_modelling.py
    └─> outputs/tables/10_* a 21_*      outputs/figures/20_* a 30_*
    └─> outputs/artifacts/final_model.joblib, priority_model.joblib
    └─> data/processed/prioridade_holdout.csv, onboarding_joined.csv
```

A divisão treino/teste (80/20, estratificada, `RANDOM_STATE = 42`) é feita uma única vez, no pré-processamento. A modelação não volta a dividir os dados, por isso todos os modelos são avaliados exatamente nos mesmos clientes.

## Protocolo

1. **Pipelines.** Cada modelo é um `Pipeline(prep, clf)`. O pré-processador é ajustado dentro de cada fold, por isso nenhuma estatística do fold de validação passa para o treino.
2. **Tuning (secção 3).** Cada modelo base é afinado com `RandomizedSearchCV`: 20 combinações de hiperparâmetros (`N_ITER`), `StratifiedKFold` com 5 folds, métrica ROC AUC, só no treino.
3. **Comparação (secção 4).** Os sete modelos base, os dois ensembles e **três referências** são avaliados no **mesmo** 5-fold estratificado, com as nove métricas abaixo. As referências servem para mostrar que o modelo acrescenta valor, e nunca são candidatas:
   - **Dummy** (`Baseline (taxa de churn)`): dá a todos a taxa de churn do treino;
   - **Regra de negócio** (`Regra de negócio (pontos)`, `modelling.BusinessRuleScore`): sem ML, 1 ponto por cada fator de risco — contrato Month-to-Month, tenure ≤ 12 meses, internet por fibra ótica — de 0 a 3 pontos. Os fatores foram fixados a partir da EDA no treino (Cramér's V e taxa de churn por categoria). A regra só aprende a taxa de churn de cada nº de pontos, para dar uma probabilidade;
   - **Churn Score IBM** (`Churn Score IBM (benchmark)`): o score / 100, avaliado nos mesmos folds de validação (argumento `benchmarks` de `cv_evaluate`), para comparar a variabilidade entre folds com a dos modelos.

   Para cada modelo guardamos:
   - a média e o desvio-padrão de cada métrica nos 5 folds de validação;
   - a média nos folds de treino, para medir o sobreajuste;
   - as previsões *out-of-fold*: cada cliente do treino recebe a probabilidade do modelo que não o viu. É com elas que se desenham as curvas ROC/PR, as matrizes de confusão e a análise de limiar, sem tocar no holdout.
4. **Seleção com regra de parcimónia** (`select_parsimonious`). Parte-se do modelo de maior ROC AUC médio em CV (`select_best`). `tied_with_best` lista os modelos a menos de um desvio-padrão do vencedor, que na prática estão empatados com ele: a diferença está dentro do ruído entre folds. Entre o vencedor e os empatados escolhe-se o **mais simples**, segundo a ordem fixa à partida em `modelling.COMPLEXITY_ORDER`:

   Logistic Regression < Decision Tree < k-Nearest Neighbors < Random Forest < Extra Trees < Gradient Boosting < Neural Network (MLP) < Soft Voting < Stacking

   A ordem segue o custo de explicar e manter o modelo: um modelo linear, depois árvores (uma, depois florestas, depois boosting), depois a rede neuronal e por fim os ensembles, que juntam cinco modelos. A tabela `11d_selecao_modelo.csv` regista o modelo de maior AUC, os empatados e o escolhido, e a justificação fica no `final_model.joblib` (chave `selection`).
5. **Variantes (secções 5 e 6).** O modelo escolhido é testado com `Satisfaction Score`, para medir o efeito do leakage, e com `City` codificada por `TargetEncoder`. `City` só entra se o ganho em AUC for maior do que o desvio-padrão entre folds.
6. **Holdout (secção 7).** Todos os modelos são treinados no treino completo e avaliados **uma única vez** nos 20% de teste, ao lado das três referências. A escolha já foi feita no passo 4. Os outros modelos aparecem só por transparência.
7. **Interpretação (secção 9).** Permutation importance do modelo core no holdout.
8. **Modelo de priorização (secção 10).** Os passos 2 a 4 (incluindo as referências e a regra de parcimónia, `16c_selecao_modelo_estabelecidos.csv`) repetem-se só com os clientes estabelecidos (tenure > 3) do mesmo treino. O modelo escolhido é avaliado uma vez nos clientes estabelecidos do mesmo holdout e comparado, nesse subconjunto, com o modelo core e com o IBM (`17_comparacao_core_priorizacao.csv`).
9. **Priorização dos clientes estabelecidos (secção 11).** Lista ordenada por P(churn) × CLTV do modelo de priorização, comparação com a ordenação só por P, análise custo-benefício por cenários e ação sugerida por segmento.
10. **Clientes novos (secção 12).** Os Joined não recebem P(churn). Recebem um segmento de onboarding definido por uma regra simples tirada da EDA.

### Porque há dois modelos

Os Joined têm todos 1 a 3 meses de casa. Um cliente novo que fica é rotulado Joined e sai da população core, e um cliente novo que sai é rotulado Churned e fica. Por isso, dentro de Churned + Stayed, **todos os 597 clientes com tenure ≤ 3 saíram** (`02b_tenure_estado.csv`). Não é comportamento, é a forma como os estados foram definidos no snapshot.

- O **modelo core** mantém a população do enunciado (Churned + Stayed, todos os tenures). Aprende que "cliente recente = churn", e uma parte do seu AUC no holdout completo vem desses casos fáceis.
- O **modelo de priorização** é treinado só com tenure > 3. É o que pontua os clientes estabelecidos na lista de intervenção. Nos mesmos clientes do holdout, o seu AUC é praticamente igual ao do modelo core; a diferença está em não depender do artefacto (a importância de `Tenure in Months` cai quase para zero) e em não ser usado fora da população em que foi treinado.
- Os **clientes novos** não recebem P(churn) de nenhum dos modelos: o core dar-lhes-ia P ≈ 1, e o de priorização nunca viu clientes novos.

### Porque validação cruzada e não um conjunto de validação fixo

O notebook das aulas 6 e 7 usa um conjunto de validação fixo, separado por data. Neste dataset não há datas (é um único snapshot do Q3), e com 6.589 clientes um conjunto fixo de 20% teria só cerca de 1.300 clientes. Com 5 folds, cada cliente do treino é usado uma vez como validação, e o desvio-padrão entre folds mostra se uma diferença entre modelos é real ou ruído.

## As nove métricas

As quatro primeiras usam as probabilidades. As cinco últimas precisam de uma decisão sim/não e usam o limiar 0,5.

| Métrica | O que mede | Melhor | Como ler neste projeto |
|---|---|---|---|
| ROC AUC | Probabilidade de um churner, escolhido ao acaso, ter score mais alto do que um não-churner | maior | **Métrica de seleção.** 0,5 = ao acaso, 1 = ordenação perfeita |
| PR AUC | Área sob a curva precision-recall (average precision) | maior | Foca-se nos churners. A linha de base vale a taxa de churn (~0,28) |
| Brier | Erro quadrático médio entre a probabilidade e o resultado (0/1) | menor | **Calibração.** Importa porque P(churn) multiplica o CLTV na priorização |
| Log loss | Penaliza sobretudo probabilidades confiantes e erradas | menor | Também calibração; muito sensível a previsões extremas erradas |
| Accuracy | % de clientes bem classificados | maior | Enganadora com classes desequilibradas: a linha de base tem ~72% |
| Balanced accuracy | Média do recall nos churners e nos não-churners | maior | Corrige o desequilíbrio; 0,5 = ao acaso |
| Precision | Dos clientes sinalizados, % que saiu de facto | maior | Eficiência das campanhas: contactos "bem gastos" |
| Recall | Dos clientes que saíram, % que foi sinalizado | maior | Cobertura: churners que a equipa consegue abordar |
| F1 | Média harmónica de precision e recall | maior | Equilíbrio entre as duas, no limiar 0,5 |

### Porque a seleção é feita pela ROC AUC

A equipa de retenção não vai contactar "todos os clientes com P > 0,5". Vai contactar os primeiros N de uma lista ordenada, sendo N o orçamento. O que importa é a **ordenação**, e é isso que a ROC AUC mede, independentemente do limiar. O Brier entra como critério secundário, porque a priorização multiplica a probabilidade pelo CLTV e precisa que a probabilidade seja realista.

### O limiar de decisão

As métricas com limiar usam 0,5 por convenção. Com 28% de churn, esse limiar tende a deixar escapar churners: dá uma precision alta e um recall mais baixo. A secção 4 mostra, para o modelo final, como precision, recall e F1 mudam com o limiar (tabela `11b_limiares_modelo_final.csv` e gráfico `23d_limiar_modelo_final.png`), e indica o limiar que maximiza o F1 como referência. A escolha final deve vir do negócio, comparando o custo de uma ação de retenção com o valor de um cliente perdido, e não da maximização de uma métrica.

**Limiares de negócio (capacidade da equipa).** A equipa de retenção só consegue contactar uma fração da base. Por isso há, além de 0,5 e do melhor F1, dois limiares definidos pela capacidade: o valor de P(churn) que sinaliza exatamente os **top 10%** e os **top 20%** dos clientes (`capacity_thresholds`). Os quatro limiares são fixados nas previsões out-of-fold do treino (`11e_limiares_negocio_cv.csv`) e aplicados uma vez ao holdout (`13b_limiares_negocio_holdout.csv`), com precision, recall e F1 em cada corte.

**Nível de risco e prioridade de contacto: duas leituras de P(churn).** A app e o lote mostram as duas lado a lado, porque respondem a perguntas diferentes:

| | Pergunta | Cortes | Onde |
|---|---|---|---|
| **Nível de risco** (absoluto) | Quão provável é este cliente sair? | baixo < 30% ≤ médio < 60% ≤ alto | `modelling.RISK_LEVELS`, `risk_level` |
| **Prioridade de contacto** (relativa) | Cabe na capacidade da equipa? | top 10% da base (P ≥ 0,925) · top 20% (P ≥ 0,640) · fora | `capacity_cuts` no `final_model.joblib`, `contact_priority` |

Numa primeira versão havia um só "segmento de risco" definido pelos quantis da capacidade: alto = top 10%, médio = top 10–20%. Era pouco intuitivo, porque o top 10% é ocupado quase só por clientes com P ≈ 1 (sobretudo clientes novos, pelo artefacto do tenure ≤ 3), o que punha o corte do "alto" em 0,925. Um cliente com P = 77% aparecia como risco "médio", e no holdout o "médio" tinha 79% de churn real.

Os níveis absolutos leem-se como percentagens reais porque o modelo está calibrado (Brier 0,087). No holdout (`13c_niveis_risco_holdout.csv`):

| Nível | Clientes | % da base | P média | Churn real |
|---|---|---|---|---|
| baixo (< 30%) | 876 | 66% | 7% | 8% |
| médio (30–60%) | 171 | 13% | 44% | 40% |
| alto (≥ 60%) | 271 | 21% | 89% | 88% |

**Ligação à capacidade.** O corte do risco alto (60%) fica perto do limiar do top 20% (0,64). Por isso quase todos os clientes de risco alto cabem numa campanha que contacte 20% da base. A ordem de contacto do lote continua a ser por P(churn) × CLTV (perda esperada).

## Decisões de retenção: o que é dado e o que é pressuposto

- **P × CLTV vs só P.** A tabela `15b_P_vs_PxCLTV.csv` mostra quantos clientes as duas listas têm em comum e que % dos churners e do CLTV perdido cada uma cobre. A comparação com o IBM faz-se sempre entre ordenações do mesmo tipo (P × CLTV com Churn Score × CLTV; só P com só Churn Score).
- **Quantos contactar.** Depende de dois **pressupostos** que o dataset não tem: a taxa de sucesso do contacto e o custo por contacto (incluindo o incentivo). Estão definidos uma só vez em `retention.CENARIOS` (usados pelo notebook 02, secção 11.2, e por `scripts/run_modelling.py`), com três cenários e uma grelha de sensibilidade. O N de cada cenário vem da regra do valor esperado (contactar se P × CLTV × taxa_sucesso > custo) e é verificado com o ótimo observado no holdout.
- **Ações por segmento** (`retention.suggest_action`) e **regra de onboarding** (`retention.onboarding_segment`) são hipóteses de negócio. A importância das variáveis mostra o que o modelo usa para ordenar clientes, não o efeito causal de mudar essas variáveis. Cada ação deve ser testada com um grupo de controlo sorteado (por exemplo 20% de cada segmento sem contacto), e a diferença de churn entre os dois grupos substitui a taxa de sucesso assumida.

## Critério de aceitação face ao Churn Score IBM

O teste `tests/test_model.py::test_nao_e_pior_do_que_o_churn_score_ibm` decide se o modelo pode ser publicado face ao benchmark IBM. As instruções iniciais pediam **AUC do modelo ≥ AUC da IBM** no holdout. Com o modelo atual, essa condição **falha**: 0,936 contra 0,941. O grupo decidiu substituí-la por um critério de **não inferioridade**. O teste passa só se as três condições se verificarem:

| Condição | Valor do modelo | Valor da IBM | Limite | Resultado |
|---|---|---|---|---|
| ROC AUC ≥ AUC IBM − desvio-padrão da IBM em CV | 0,9362 | 0,9408 | 0,9408 − 0,0077 = 0,9331 | passa |
| PR AUC ≥ PR AUC IBM | 0,8826 | 0,8762 | 0,8762 | passa |
| Brier < Brier IBM | 0,0865 | 0,2142 | 0,2142 | passa |

Os valores são do holdout. O desvio-padrão da IBM vem da CV de 5 folds no treino (`benchmark_cv` no `final_model.joblib`).

**Porque se tolera uma diferença em AUC.**

- **A diferença está dentro do ruído entre folds.** A diferença (0,0046) é menor do que a variação da AUC da IBM de fold para fold (0,0077). Com outra amostra de clientes, a ordem entre os dois podia inverter-se.
- **É a mesma regra da seleção.** `tied_with_best` considera empatados dois modelos cuja diferença é menor do que o desvio-padrão do melhor dos dois, e o melhor aqui é a IBM. O teste não inventa uma tolerância nova; aplica à IBM a regra que já se aplica aos modelos.
- **A tolerância usa o desvio-padrão da IBM, não o do modelo.** O do modelo (0,0040) é menor do que a diferença, por isso, medida contra ele, a diferença não seria ruído. A escolha segue a regra de seleção (o desvio-padrão do melhor) e fica declarada aqui para não parecer feita à medida.

**Porque também se exigem PR AUC e Brier.** Um modelo empatado em AUC não chega para justificar trocar o score existente. Exige-se ainda:

- **PR AUC pelo menos igual.** É a ordenação com foco nos churners, que é o que a equipa de retenção usa.
- **Brier melhor.** É a calibração, de que a priorização por P(churn) × CLTV precisa: o Churn Score IBM / 100 não é uma probabilidade (Brier 0,214).

O modelo ganha nas duas, e é aí que acrescenta valor face à IBM.

**Quando o teste falha.**

- **Por perda de AUC:** se um retreino deixar a AUC mais de um desvio-padrão da IBM abaixo dela.
- **Por perda de PR AUC ou de calibração:** se perder em PR AUC, ou se deixar de ser mais bem calibrado do que a IBM.

Em qualquer destes casos, o `deploy-hf.yml` não publica.

**O que o critério não afirma.**

- Não afirma que o modelo é melhor do que a IBM a ordenar clientes.
- Não se sabe com que informação o Churn Score foi calculado. Nos clientes estabelecidos (tenure > 3), a IBM mantém AUC 0,94 e os nossos modelos ficam em ≈ 0,90.

O relatório deve dizer "empatado com a IBM em AUC, melhor em PR AUC e em calibração", e não "bate a IBM".

## Limitações e decisões a declarar no relatório

- **Brier e log loss do IBM.** `Churn Score / 100` não é uma probabilidade calibrada, por isso o Brier e o log loss do IBM só servem como referência. A comparação justa com o IBM é pelas métricas de ordenação (ROC AUC, PR AUC) e pela curva de captura.
- **O modelo não bate o IBM em ROC AUC** (0,936 vs 0,941 no holdout). Ver [Critério de aceitação face ao Churn Score IBM](#critério-de-aceitação-face-ao-churn-score-ibm): o teste exige "não pior do que a IBM", e não "melhor do que a IBM".
- **O IBM nos clientes estabelecidos.** O Churn Score IBM mantém um AUC próximo de 0,94 nos clientes com tenure > 3, acima dos nossos modelos (≈ 0,90). Não sabemos com que informação foi calculado: pode usar variáveis que não temos ou informação posterior ao churn. Fica como benchmark e não como feature.
- **Validação cruzada não aninhada.** O tuning e a comparação dos modelos usam os mesmos folds do treino, por isso o AUC em CV do vencedor é ligeiramente otimista. Optámos por não usar CV aninhada (seria 5× mais lenta) porque a escolha final é confirmada uma única vez no holdout, que não entrou no tuning.
- **Utilização do holdout.** O holdout não é usado para escolher modelos nem hiperparâmetros. É usado, depois das escolhas feitas, para: as métricas finais dos dois modelos e do IBM; a variante com Satisfaction Score (só para documentar o leakage); a permutation importance; a curva de captura, a comparação P vs P × CLTV e o ótimo observado da análise custo-benefício. Os pressupostos de negócio não são afinados no holdout.
- **Variáveis geográficas.** `Latitude`, `Longitude` e `Population` entram como features: são informação conhecida antes do churn e têm pouca associação com o target (AUC univariada entre 0,52 e 0,54). `City` só é testada com `TargetEncoder` (secção 7), e `Zip Code`, `Lat Long`, `Country` e `State` são removidas por serem constantes, duplicadas ou de alta cardinalidade.

## Gráficos da modelação

| Ficheiro | O que mostra |
|---|---|
| `20_diversidade_modelos.png` | Correlação entre as probabilidades dos modelos (base para os ensembles) |
| `21_cv_modelos.png` | ROC AUC em CV ± desvio-padrão, com as referências (regra de negócio a cinzento, IBM a preto) |
| `22_painel_metricas_cv.png` | As nove métricas por modelo, em CV, com o valor de treino como losango |
| `22b_heatmap_metricas_cv.png` | Grelha modelo × métrica, colorida pela posição em cada métrica |
| `23_sobreajuste.png` | ROC AUC no treino vs em CV: a distância mede o sobreajuste |
| `23b_roc_pr_cv.png` | Curvas ROC e precision-recall de todos os modelos (out-of-fold) |
| `23c_matrizes_confusao_cv.png` | Matriz de confusão de cada modelo, no limiar 0,5 |
| `23d_limiar_modelo_final.png` | Precision, recall, F1 e % sinalizados por limiar, para o modelo final |
| `23e_rede_neuronal_treino.png` | Evolução do treino da rede neuronal e ponto de early stopping |
| `24_roc_calibracao.png` | ROC e calibração no holdout, com o IBM |
| `24b_painel_metricas_holdout.png` | As nove métricas por modelo no holdout |
| `25_captura_valor.png` | % do CLTV perdido captado ao contactar os top-k% |
| `26_mapa_prioridade.png` | P(churn) × CLTV, com os top-N destacados |
| `27_importancia_permutacao.png` | Variáveis mais importantes do modelo core |
| `28_cv_modelos_estabelecidos.png` | ROC AUC em CV dos modelos treinados só com tenure > 3 |
| `29_importancia_priorizacao.png` | Variáveis mais importantes do modelo de priorização |
| `30_valor_liquido.png` | Valor líquido da campanha por nº de contactos, em cada cenário |

## Tabelas da modelação

| Ficheiro | Conteúdo |
|---|---|
| `10_tuning_modelos_base.csv` | AUC em CV e melhores hiperparâmetros de cada modelo base |
| `11_cv_modelos.csv` | Nove métricas por modelo: média e desvio em CV, média no treino |
| `11b_limiares_modelo_final.csv` | Precision, recall, F1 e % sinalizados por limiar |
| `11c_correlacao_modelos.csv` | Correlação entre as probabilidades out-of-fold |
| `11d_selecao_modelo.csv` | Regra de parcimónia: AUC, ordem de complexidade, empatados e modelo escolhido |
| `11e_limiares_negocio_cv.csv` | Precision/recall/F1 nos limiares 0,5, melhor F1, top 10% e top 20% (out-of-fold) |
| `12_variantes_satisfaction_city.csv` | Efeito do Satisfaction Score e de City |
| `13_holdout.csv` | Nove métricas no holdout, para todos os modelos e para as referências |
| `13b_limiares_negocio_holdout.csv` | Os mesmos quatro limiares, aplicados ao holdout |
| `13c_niveis_risco_holdout.csv` | Clientes, P média e churn real por nível de risco (baixo/médio/alto) |
| `14_importancia_permutacao.csv` | Queda de AUC por variável |
| `15_captura_valor.csv` | Captura de valor nos top 10/20/30% (clientes estabelecidos) |
| `15b_P_vs_PxCLTV.csv` | Lista por P vs lista por P × CLTV: clientes em comum, churners e CLTV cobertos |
| `16_tuning_estabelecidos.csv`, `16b_cv_modelos_estabelecidos.csv`, `16c_selecao_modelo_estabelecidos.csv` | Tuning, CV e seleção do modelo de priorização |
| `17_comparacao_core_priorizacao.csv` | Modelo core vs modelo de priorização vs IBM, por conjunto do holdout |
| `18_importancia_priorizacao.csv` | Queda de AUC por variável no modelo de priorização |
| `19_regra_onboarding.csv` | Segmentos de onboarding dos clientes novos e taxa de saída |
| `20_cenarios_custo_beneficio.csv`, `20b_sensibilidade_pct_contactar.csv` | N a contactar por cenário e grelha de sensibilidade |
| `21_acoes_sugeridas.csv` | Top-N do cenário Base por ação sugerida |

## Tempo de execução

Com `N_ITER = 20`, a fase completa demora cerca de 5 minutos num portátil recente, porque o tuning e a comparação em CV correm duas vezes (modelo core e modelo de priorização). Os passos mais pesados são o tuning da rede neuronal e o stacking, que treina os cinco membros dentro de cada fold. Para testar alterações depressa, há duas opções: baixar `N_ITER` no início do notebook, ou correr `python scripts/run_modelling.py --rapido`.
