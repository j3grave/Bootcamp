# Fase de modelação: protocolo, métricas e decisões

Este documento descreve como os modelos são treinados, comparados e escolhidos, e como cada métrica deve ser lida. A descrição de cada modelo está em [`models.md`](models.md).

## Onde está cada coisa

| Ficheiro | Conteúdo |
|---|---|
| `notebooks/02_modelling_eda.ipynb` | Narrativa completa da modelação, com tabelas e gráficos |
| `scripts/run_modelling.py` | Mesmo fluxo sem notebook (também chamado por `scripts/run_all.py`) |
| `src/telco_churn/modelling.py` | Modelos, tuning, ensembles, métricas, avaliação e priorização |
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
    └─> outputs/tables/10_* a 15_*      outputs/figures/20_* a 27_*
    └─> outputs/artifacts/final_model.joblib
    └─> data/processed/prioridade_holdout.csv, prioridade_joined.csv
```

A divisão treino/teste (80/20, estratificada, `RANDOM_STATE = 42`) é feita uma única vez, no pré-processamento. A modelação não volta a dividir os dados, por isso todos os modelos são avaliados exatamente nos mesmos clientes.

## Protocolo

1. **Pipelines.** Cada modelo é um `Pipeline(prep, clf)`. O pré-processador é ajustado dentro de cada fold, por isso nenhuma estatística do fold de validação passa para o treino.
2. **Tuning (secção 3).** Cada modelo base é afinado com `RandomizedSearchCV`: 20 combinações de hiperparâmetros (`N_ITER`), `StratifiedKFold` com 5 folds, métrica ROC AUC, só no treino.
3. **Comparação (secção 4).** Os sete modelos base, a linha de base e os dois ensembles são avaliados no **mesmo** 5-fold estratificado, com as nove métricas abaixo. Para cada modelo guardamos:
   - a média e o desvio-padrão de cada métrica nos 5 folds de validação;
   - a média nos folds de treino, para medir o sobreajuste;
   - as previsões *out-of-fold*: cada cliente do treino recebe a probabilidade do modelo que não o viu. É com elas que se desenham as curvas ROC/PR, as matrizes de confusão e a análise de limiar, sem tocar no holdout.
4. **Seleção.** O modelo final é o de maior ROC AUC médio em CV (`select_best`). A linha de base não é candidata. A função `tied_with_best` lista os modelos a menos de um desvio-padrão do vencedor, que na prática estão empatados com ele. Nesse caso, a escolha pode ser revista em favor do modelo mais simples ou mais bem calibrado, e essa decisão deve ficar justificada no relatório.
5. **Variantes (secções 5 e 6).** O modelo escolhido é testado com `Satisfaction Score`, para medir o efeito do leakage, e com `City` codificada por `TargetEncoder`. `City` só entra se o ganho em AUC for maior do que o desvio-padrão entre folds.
6. **Holdout (secção 7).** Todos os modelos são treinados no treino completo e avaliados **uma única vez** nos 20% de teste, ao lado do Churn Score IBM. A escolha já foi feita no passo 4. Os outros modelos aparecem só por transparência.
7. **Interpretação e priorização (secções 8 e 9).** Permutation importance no holdout, depois a lista de intervenção ordenada por P(churn) × CLTV e a curva de captura de valor.

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

## Gráficos da modelação

| Ficheiro | O que mostra |
|---|---|
| `20_diversidade_modelos.png` | Correlação entre as probabilidades dos modelos (base para os ensembles) |
| `21_cv_modelos.png` | ROC AUC em CV ± desvio-padrão, com a fasquia IBM |
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
| `27_importancia_permutacao.png` | Variáveis mais importantes do modelo final |

## Tabelas da modelação

| Ficheiro | Conteúdo |
|---|---|
| `10_tuning_modelos_base.csv` | AUC em CV e melhores hiperparâmetros de cada modelo base |
| `11_cv_modelos.csv` | Nove métricas por modelo: média e desvio em CV, média no treino |
| `11b_limiares_modelo_final.csv` | Precision, recall, F1 e % sinalizados por limiar |
| `11c_correlacao_modelos.csv` | Correlação entre as probabilidades out-of-fold |
| `12_variantes_satisfaction_city.csv` | Efeito do Satisfaction Score e de City |
| `13_holdout.csv` | Nove métricas no holdout, para todos os modelos e para o IBM |
| `14_importancia_permutacao.csv` | Queda de AUC por variável |
| `15_captura_valor.csv` | Captura de valor nos top 10/20/30% |

## Tempo de execução

Com `N_ITER = 20`, a fase completa demora vários minutos num portátil recente. Os passos mais pesados são o tuning da rede neuronal e o stacking, que treina os cinco membros dentro de cada fold. Para testar alterações depressa, há duas opções: baixar `N_ITER` no início do notebook, ou correr `python scripts/run_modelling.py --rapido`.
