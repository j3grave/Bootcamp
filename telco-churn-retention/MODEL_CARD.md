---
language: pt
library_name: sklearn
tags:
  - tabular-classification
  - churn
  - scikit-learn
---

# Telco Churn — modelo de risco de saída (EDSB 2026)

## Objetivo

Estimar a probabilidade de um cliente de telecomunicações sair, P(churn), para que a equipa de retenção contacte primeiro quem tem maior perda esperada, P(churn) × CLTV.

O repositório tem dois modelos, usados pela app Gradio do projeto:

- `priority_model.joblib` — **modelo de priorização**, treinado só com clientes estabelecidos (tenure > 3). Pontua e ordena os clientes na app.
- `final_model.joblib` — **modelo core**, treinado com todos os tenures. É o que se compara com o Churn Score da IBM. Nos clientes novos (tenure ≤ 3) dá só um valor indicativo, porque extrapola.

## Dados

- Dataset sintético *Telco Customer Churn* da IBM: 5 ficheiros, um extrato do Q3, Califórnia, 7.043 clientes.
- Validado na ingestão por contratos de dados (pandera): esquema, chaves, domínios, regras de negócio e estatísticas.
- População de treino: Churned + Stayed (6.589 clientes, 28,4% de churn). Os 454 *Joined* não têm resposta e ficam de fora.
- Divisão 80/20 estratificada ao nível do cliente (`random_state=42`).

## Features

32 variáveis originais (perfil, contrato e faturação, serviços, localização) e 4 derivadas: nº de serviços adicionais, média histórica da fatura, diferença entre a fatura atual e essa média, e se houve reembolso.

Ficam de fora as colunas que codificam o resultado (Customer Status, Churn Label/Value/Category/Reason), as saídas de outros modelos (Churn Score, CLTV), o Satisfaction Score (com ele o AUC sobe para 0,996, sinal de leakage) e as colunas redundantes ou de alta cardinalidade (City, Zip Code, etc.).

## Seleção

Nove modelos (regressão logística, árvore, kNN, MLP, Random Forest, Extra Trees, Gradient Boosting, Soft Voting, Stacking), afinados com `RandomizedSearchCV` e comparados no mesmo 5-fold estratificado. As referências são a taxa de churn (Dummy), uma regra de negócio sem ML (Month-to-Month, tenure ≤ 12 meses, fibra ótica) e o Churn Score da IBM.

Entre o melhor em AUC e os empatados com ele (diferença menor do que um desvio-padrão entre folds) escolhe-se o mais simples. O holdout é usado uma única vez.

As métricas desta versão estão na secção seguinte, gerada a partir do artefacto no momento da publicação.

## Como usar

```python
import joblib
from huggingface_hub import hf_hub_download

repo = "tvf007/EDSBG1TP2-model"
priority = joblib.load(hf_hub_download(repo_id=repo, filename="priority_model.joblib"))  # tenure > 3
core = joblib.load(hf_hub_download(repo_id=repo, filename="final_model.joblib"))
# bundle["model"]: Pipeline scikit-learn (pré-processador + classificador)
# bundle["features"]: colunas de entrada (as derivadas calculam-se com telco_churn.features.add_features)
# priority["risk_levels"]: níveis de risco (médio ≥ 0,30, alto ≥ 0,60)
# priority["capacity_cuts"]: P(churn) a partir da qual um cliente estabelecido está no top 10% / 20%
```

Requer `scikit-learn==1.9.1`, a versão com que os artefactos foram gravados.

## Limitações

- **Clientes novos.** Os recentes que ficam são registados como *Joined* e saem do treino; os que saem ficam como *Churned*. Por isso, no treino, quase todos os clientes com tenure ≤ 3 saíram. É um efeito do registo, não comportamento. Para estes clientes a app mostra um aviso e usa uma regra de onboarding.
- **Clientes estabelecidos: a IBM ordena melhor.** Com tenure > 3, o modelo de priorização tem AUC 0,899 contra 0,939 do Churn Score, e o IC 95% da diferença (bootstrap emparelhado) não inclui zero. É publicado porque dá probabilidades calibradas (Brier 0,097 contra 0,234), necessárias para P × CLTV, e usa só variáveis conhecidas. Não substitui o Churn Score como ordenação.
- **Base completa (modelo core).** Em ROC AUC está empatado com a IBM (diferença dentro do desvio-padrão entre folds); é melhor em PR AUC e muito mais bem calibrado.
- **Desconto.** O modelo prevê risco, não o efeito de uma ação. A eficácia do desconto na app é um pressuposto a medir com grupo de controlo.
- **Dados sintéticos e sem datas.** Os padrões podem não se transferir para dados reais e não há validação temporal.
- **Importância das variáveis** mostra associação, não causa.

## Publicação

Publicado pelo GitHub Actions (`deploy-hf.yml`) só depois de a pipeline correr e de os testes passarem. O modelo de priorização tem de bater a Dummy e a regra de negócio, ter sobreajuste < 0,05, Brier abaixo da IBM e níveis de risco calibrados.
