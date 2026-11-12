---
language: pt
library_name: sklearn
tags:
  - tabular-classification
  - churn
  - scikit-learn
---

# Telco Churn — modelo de risco de saída (EDSB 2026, Grupo 1)

## Objetivo

Estimar a probabilidade de um cliente de telecomunicações sair, P(churn), para que a equipa de
retenção contacte primeiro os clientes com maior **perda esperada = P(churn) × CLTV**.
O repositório tem dois modelos, ambos usados pela app Gradio do projeto:

- `priority_model.joblib` — **modelo de priorização**, treinado só com clientes estabelecidos
  (tenure > 3). É o que pontua e ordena os clientes na app (cliente individual e lote).
- `final_model.joblib` — **modelo core**, treinado com todos os tenures (população do
  enunciado). É o que se compara com o Churn Score IBM; nos clientes novos (tenure ≤ 3) dá
  apenas um valor indicativo, porque extrapola.

## Dados

- **Fonte:** dataset sintético *Telco Customer Churn* da IBM (5 ficheiros: demographics,
  location, population, services, status), um único extrato do 3.º trimestre, Califórnia.
- **7 043 clientes.** Validados na ingestão por contratos de dados (pandera): esquema, chaves,
  domínios, regras de negócio e estatísticas (`src/telco_churn/schemas.py`).
- **População de modelação:** Churned + Stayed (6 589 clientes, 28,4% de churn). Os 454
  *Joined* (1 a 3 meses de casa) não têm resposta e não são usados no treino.
- **Divisão:** 80% treino / 20% holdout, estratificada, ao nível do cliente (`random_state=42`).

## Features

32 variáveis originais (perfil, contrato e faturação, serviços, localização) e 4 derivadas,
calculadas linha a linha: nº de serviços adicionais, média histórica da fatura, diferença
entre a fatura atual e a média, e se houve reembolso.

**Excluídas:** colunas que codificam o resultado (Customer Status, Churn Label/Value/Category/
Reason), saídas de outros modelos (Churn Score, CLTV), Satisfaction Score (suspeita de ser
registada no contexto da saída: com ela o AUC sobe para 0,996, o que indica *leakage*) e
colunas redundantes ou de alta cardinalidade (City, Zip Code, etc.).

## Seleção do modelo

Nove modelos (regressão logística, árvore, kNN, MLP, Random Forest, Extra Trees, Gradient
Boosting, Soft Voting, Stacking) afinados com `RandomizedSearchCV` e comparados no mesmo 5-fold
estratificado, ao lado de três referências: taxa de churn (Dummy), uma **regra de negócio sem
ML** (Month-to-Month + tenure ≤ 12 meses + fibra ótica) e o **Churn Score da IBM**.
**Regra de parcimónia:** entre o melhor em AUC e os empatados com ele (diferença menor do que
um desvio-padrão entre folds), escolhe-se o mais simples. O holdout é usado uma única vez.

As métricas desta versão (CV e holdout) estão na secção seguinte, gerada a partir do
artefacto no momento da publicação.

## Como usar

```python
import joblib
from huggingface_hub import hf_hub_download

repo = "tvf007/EDSBG1TP2-model"
priority = joblib.load(hf_hub_download(repo_id=repo, filename="priority_model.joblib"))  # tenure > 3
core = joblib.load(hf_hub_download(repo_id=repo, filename="final_model.joblib"))
# bundle["model"]: Pipeline scikit-learn (pré-processador + classificador)
# bundle["features"]: colunas de entrada (as derivadas calculam-se com telco_churn.features.add_features)
# priority["risk_levels"]: níveis de risco absolutos (médio ≥ 0,30, alto ≥ 0,60)
# priority["capacity_cuts"]: P(churn) a partir da qual um cliente estabelecido está no
#                            top 10% / 20% dos clientes estabelecidos
```

Requer `scikit-learn==1.9.1` (a versão com que o artefacto foi gravado).

## Limitações

- **Clientes novos.** Os clientes recentes que ficaram são registados como *Joined* e saem da
  população de treino; os que saíram ficam como *Churned*. Por isso, no treino, quase todos os
  clientes com tenure ≤ 3 meses saíram, o que é um efeito da forma como os estados foram
  registados e não um comportamento. Para clientes novos o modelo extrapola; a app mostra um
  aviso e o projeto usa uma regra de onboarding em vez de P(churn).
- **Causalidade do desconto.** O modelo prevê risco, não o efeito de uma ação. A eficácia do
  desconto usada na app é um pressuposto e deve ser medida com um grupo de controlo.
- **Dados sintéticos IBM.** Os padrões podem não se transferir para dados reais. É um único
  extrato (Q3), sem datas, por isso não há validação temporal.
- **Comparação com a IBM.** O Churn Score da IBM é calculado com informação que não
  conhecemos. Em ROC AUC o modelo está empatado com ele (diferença dentro do desvio-padrão
  entre folds); é melhor em PR AUC e muito mais bem calibrado (Brier).
- **Importância das variáveis** mostra associação, não causa.

## Publicação

Publicado pelo GitHub Actions (`deploy-hf.yml`) **só** depois de a pipeline correr e de todos
os testes passarem (contratos de dados, pré-processamento, qualidade do modelo e app).
