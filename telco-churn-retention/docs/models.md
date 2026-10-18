# Modelos em avaliação

Este documento explica cada modelo do notebook `02_modelling_eda.ipynb`: como funciona, porque foi incluído, que hiperparâmetros são afinados e o que esperar dele. O protocolo de avaliação (validação cruzada, holdout, métricas) está em [`modelling.md`](modelling.md).

Todos os modelos são construídos como um `Pipeline` do scikit-learn com dois passos, o pré-processador (`prep`) e o classificador (`clf`). O pré-processador é ajustado dentro de cada fold da validação cruzada. Por isso as médias, os desvios-padrão e as categorias do one-hot são aprendidos só com os dados de treino desse fold, e não há fuga de informação.

Há dois pré-processadores, definidos em `src/telco_churn/preprocessing.py`:

| Pré-processador | O que faz | Usado por |
|---|---|---|
| `linear` | Imputação (mediana / moda), `StandardScaler` nas numéricas e one-hot nas categóricas | Logistic Regression, k-Nearest Neighbors, Neural Network (MLP) |
| `trees` | Imputação e one-hot, sem escala | Decision Tree, Random Forest, Extra Trees, Gradient Boosting |

A escala é necessária nos modelos que somam variáveis ou medem distâncias, porque sem ela uma variável medida em centenas (por exemplo `Total Charges`) dominaria uma medida em unidades (`Number of Referrals`). As árvores comparam cada variável consigo própria ("`Tenure in Months` < 6?"), por isso a escala não lhes altera nada.

---

## Resumo

| Modelo | Família | Papel no projeto |
|---|---|---|
| Baseline (taxa de churn) | referência | Fasquia mínima: qualquer modelo útil tem de a bater |
| Logistic Regression | linear | Referência interpretável; membro dos ensembles |
| Decision Tree | árvore | Modelo mais fácil de explicar; mostra o limite de uma árvore isolada |
| k-Nearest Neighbors | distância | Abordagem diferente (semelhança entre clientes); comparação |
| Neural Network (MLP) | rede neuronal | Capta relações não lineares; membro dos ensembles |
| Random Forest | ensemble: bagging | Modelo forte e robusto; membro dos ensembles |
| Extra Trees | ensemble: bagging | Variante mais aleatória do Random Forest; membro dos ensembles |
| Gradient Boosting | ensemble: boosting | Normalmente o mais forte em dados tabulares; membro dos ensembles |
| Soft Voting | ensemble: voting | Combina os cinco membros pela média das probabilidades |
| Stacking | ensemble: stacking | Combina os cinco membros com pesos aprendidos |
| Churn Score IBM | benchmark externo | Modelo que a empresa já tem; não é treinado nem candidato |

---

## Baseline (taxa de churn)

**Como funciona.** `DummyClassifier(strategy="prior")` dá a todos os clientes a mesma probabilidade, a taxa de churn do treino (cerca de 28%).

**Porque está incluído.** Dá o ponto de partida de cada métrica. Tem ROC AUC de 0,5 (não ordena nada) e PR AUC igual à taxa de churn. A accuracy é enganadora: como nunca sinaliza ninguém, acerta em todos os clientes que ficam e chega a cerca de 72%. É por isso que a accuracy não serve para escolher o modelo.

---

## Logistic Regression

**Como funciona.** Calcula uma soma ponderada das variáveis, `z = b0 + b1·x1 + b2·x2 + ...`, e passa-a por uma função sigmoide, `P(churn) = 1 / (1 + e^(-z))`, que transforma qualquer número numa probabilidade entre 0 e 1. Os pesos são aprendidos minimizando o log loss. Cada peso tem uma leitura direta: um peso positivo aumenta o risco de churn.

**Hiperparâmetro afinado.** `C`, o inverso da força da regularização (procura entre 0,001 e 100). Um `C` pequeno encolhe os pesos para zero, o que dá um modelo mais simples e menos sobreajustado.

**O que esperar.** Desempenho sólido e probabilidades bem calibradas. Não capta interações (por exemplo, "contrato mensal *e* fibra ótica") a não ser que sejam criadas à mão.

---

## Decision Tree

**Como funciona.** Divide os clientes com perguntas sucessivas sobre uma variável de cada vez ("`Contract` = Month-to-Month?", "`Tenure in Months` < 6?"). Em cada nó escolhe a pergunta que mais separa churners de não-churners, medida pelo índice de Gini ou pela entropia. A probabilidade final é a taxa de churn da folha onde o cliente cai.

**Hiperparâmetros afinados.** `max_depth` (profundidade máxima, entre 3 e 10), `min_samples_leaf` (mínimo de clientes por folha, entre 5 e 50) e `criterion` (gini ou entropia). Sem estes limites, a árvore cresce até decorar o treino (sobreajuste).

**O que esperar.** É o modelo mais fácil de explicar a um gestor, porque se desenha como um fluxograma. É instável: pequenas mudanças nos dados mudam a árvore. Fica abaixo das florestas e do boosting, que existem precisamente para corrigir esta fraqueza.

---

## k-Nearest Neighbors (kNN)

**Como funciona.** Não aprende pesos. Para um cliente novo, encontra os `k` clientes de treino mais parecidos (distância euclidiana nas variáveis escaladas) e usa a taxa de churn entre eles como probabilidade.

**Hiperparâmetros afinados.** `n_neighbors` (k, entre 15 e 100) e `weights` (todos os vizinhos contam o mesmo, ou os mais próximos contam mais).

**O que esperar.** É intuitivo ("clientes parecidos comportam-se de forma parecida"), mas sofre com muitas variáveis one-hot, que tornam as distâncias menos informativas. Serve sobretudo como comparação.

---

## Neural Network (MLP)

**Como funciona.** Um *multilayer perceptron* é uma rede de neurónios organizados em camadas:

1. A **camada de entrada** recebe as variáveis já escaladas e codificadas (uma entrada por coluna).
2. Uma ou duas **camadas escondidas** de neurónios. Cada neurónio calcula uma soma ponderada das entradas e aplica uma função de ativação não linear (ReLU, `max(0, z)`, ou tangente hiperbólica).
3. A **camada de saída** tem um neurónio com ativação sigmoide, que devolve `P(churn)`.

Os pesos são aprendidos por *backpropagation* com o otimizador Adam, minimizando o log loss.

**Relação com a regressão logística.** Uma regressão logística é exatamente uma rede neuronal sem camadas escondidas: um único neurónio sigmoide ligado às entradas. As camadas escondidas permitem à rede criar as suas próprias combinações das variáveis, e assim captar interações e relações não lineares sem que tenhamos de as definir à mão.

**Porque não uma rede de regressão.** Uma rede de regressão tem saída contínua e perda quadrática (prevê um número, como um valor em euros). Aqui o alvo é binário (saiu ou ficou), por isso a configuração correta é a de classificação: saída sigmoide e perda log loss. O resto da arquitetura é igual.

**Hiperparâmetros afinados.**

| Hiperparâmetro | Valores testados | Significado |
|---|---|---|
| `hidden_layer_sizes` | (16,), (32,), (64,), (32, 16), (64, 32) | Nº de camadas escondidas e de neurónios em cada |
| `activation` | relu, tanh | Função de ativação das camadas escondidas |
| `alpha` | 1e-5 a 1e-1 | Regularização L2: penaliza pesos grandes, reduz sobreajuste |
| `learning_rate_init` | 1e-4 a 1e-2 | Tamanho do passo do otimizador Adam |

**Early stopping.** A rede separa internamente 15% do seu treino como validação e pára quando a accuracy nessa validação não melhora durante 20 épocas (máximo de 500). Isto evita treinar demasiado tempo e decorar o treino. O gráfico `23e_rede_neuronal_treino.png` mostra a evolução da perda e o ponto de paragem.

**Implementação.** Usamos o `MLPClassifier` do scikit-learn, e não Keras ou PyTorch, porque encaixa diretamente no `Pipeline`, na validação cruzada, no `RandomizedSearchCV` e nos ensembles, sem novas dependências. Para uma rede deste tamanho em dados tabulares, o resultado seria equivalente.

**O que esperar.** Com cerca de 5.300 clientes de treino e dados tabulares, as redes neuronais raramente superam o gradient boosting. Mesmo assim, como "pensam" de forma diferente das árvores, podem acrescentar diversidade aos ensembles. Ambas as coisas ficam medidas e documentadas no notebook.

---

## Random Forest

**Como funciona.** Treina centenas de árvores (aqui 300). Cada uma vê uma amostra diferente dos clientes, tirada com reposição (*bootstrap*), e em cada nó só pode escolher entre um subconjunto aleatório das variáveis. A probabilidade final é a média das 300 árvores. Esta técnica chama-se **bagging** (*bootstrap aggregating*): árvores diferentes erram em clientes diferentes, e a média reduz a variância.

**Hiperparâmetros afinados.** `max_depth`, `min_samples_leaf` e `max_features` (fração das variáveis disponível em cada nó).

**O que esperar.** É robusto, precisa de pouca afinação e raramente fica mal. As probabilidades tendem a ficar comprimidas longe de 0 e de 1.

---

## Extra Trees

**Como funciona.** É igual ao Random Forest, com uma diferença: o ponto de corte de cada pergunta é sorteado em vez de otimizado. As árvores ficam mais diferentes entre si, o que reduz ainda mais a variância, à custa de cada árvore ser individualmente mais fraca.

**Hiperparâmetros afinados.** Os mesmos do Random Forest.

---

## Gradient Boosting

**Como funciona.** Constrói árvores pequenas **em sequência**. Cada árvore nova é treinada para corrigir os erros que as anteriores ainda cometem (tecnicamente, segue o gradiente do log loss). A previsão final é a soma de todas, cada uma multiplicada por uma taxa de aprendizagem. Esta técnica chama-se **boosting**. Usamos o `HistGradientBoostingClassifier`, que agrupa os valores numéricos em intervalos para treinar mais depressa e segue a mesma abordagem do LightGBM.

**Hiperparâmetros afinados.**

| Hiperparâmetro | Significado |
|---|---|
| `learning_rate` | Peso de cada árvore nova: mais baixo é mais lento mas mais estável |
| `max_iter` | Número de árvores |
| `max_leaf_nodes` | Tamanho de cada árvore |
| `min_samples_leaf` | Mínimo de clientes por folha |
| `l2_regularization` | Penalização que suaviza as previsões |

**O que esperar.** Em dados tabulares é, normalmente, o modelo individual mais forte.

---

## Ensembles: Soft Voting e Stacking

O Random Forest, o Extra Trees e o Gradient Boosting já são ensembles de árvores. Os dois ensembles abaixo vão mais longe e combinam **modelos de famílias diferentes**.

**Membros.** Logistic Regression, Neural Network (MLP), Random Forest, Extra Trees e Gradient Boosting, já com os hiperparâmetros afinados. A árvore isolada e o kNN ficam de fora porque são mais fracos e estão representados, com vantagem, pelas florestas. Os membros são fixados no código (`ENSEMBLE_MEMBERS`) antes de olhar para os resultados, para que a comparação em validação cruzada continue justa.

**Soft Voting.** A probabilidade final é a média simples das cinco probabilidades. Não aprende nada novo e é difícil de sobreajustar.

**Stacking.** Um segundo modelo, uma regressão logística chamada *meta-modelo*, recebe as cinco probabilidades como entradas e aprende a melhor forma de as combinar (os pesos). Para não aprender com previsões feitas sobre os mesmos dados em que os membros foram treinados, as entradas do meta-modelo são previsões *out-of-fold* (`cv=5`): cada cliente recebe a previsão de membros que não o viram.

**Quando ganham.** Um ensemble só melhora se os membros errarem em clientes diferentes. O gráfico de diversidade (`20_diversidade_modelos.png`) mede isso: correlações muito altas entre as probabilidades significam que há pouco a ganhar em juntar os modelos.

---

## Benchmark externo: Churn Score IBM

O `Churn Score` (0 a 100) é o resultado de um modelo IBM SPSS que a empresa já usa. Não entra como variável, porque seria usar a previsão de outro modelo como entrada. É avaliado no mesmo holdout, como `Churn Score / 100`, e serve para responder à pergunta "o nosso modelo é melhor do que o que já existe?". O Brier e o log loss do IBM são apenas indicativos, porque um score de 0 a 100 dividido por 100 não é uma probabilidade calibrada.
