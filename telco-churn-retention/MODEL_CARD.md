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

O repositório tem um modelo, `final_model.joblib`, usado pela app Gradio do projeto. Nos clientes estabelecidos (tenure > 3) a P(churn) serve para ordenar a lista de contactos. Nos clientes novos (tenure ≤ 3) é só um valor indicativo, porque o modelo extrapola.

## Dados

- Dataset sintético *Telco Customer Churn* da IBM: 5 ficheiros, um extrato do Q3, Califórnia, 7.043 clientes.
- Validado na ingestão por contratos de dados (pandera): esquema, chaves, domínios, regras de negócio e estatísticas.
- População de treino: Churned + Stayed (6.589 clientes, 28,4% de churn). Os 454 *Joined* não têm resposta e ficam de fora.
- Divisão 80/20 estratificada ao nível do cliente (`random_state=42`).

## Features

32 variáveis originais (perfil, contrato e faturação, serviços, localização) e 4 derivadas: nº de serviços adicionais, média histórica da fatura, diferença entre a fatura atual e essa média, e se houve reembolso.

Ficam de fora as colunas que codificam o resultado (Customer Status, Churn Label/Value/Category/Reason), as saídas de outros modelos (Churn Score, CLTV), o Satisfaction Score (com ele o AUC sobe para 0,996, sinal de leakage) e as colunas redundantes ou de alta cardinalidade (City, Zip Code, etc.).

## Seleção

Quatro modelos (regressão logística, árvore de decisão, Random Forest, Gradient Boosting), afinados com `GridSearchCV` em grelhas pequenas e comparados no mesmo 5-fold estratificado. As referências são a taxa de churn (Dummy), uma regra de negócio sem ML (Month-to-Month, tenure ≤ 12 meses, fibra ótica) e o Churn Score da IBM.

Decidimos escolher o modelo de maior ROC AUC em CV, exceto se a regressão logística ficar a menos de 0,01 dele. O holdout é usado uma única vez.

As métricas desta versão estão na secção final, gerada a partir do artefacto no momento da publicação.

## Como usar

```python
import joblib
from huggingface_hub import hf_hub_download

bundle = joblib.load(hf_hub_download(repo_id="tvf007/EDSBG1TP2-model",
                                     filename="final_model.joblib"))
# bundle["model"]: Pipeline scikit-learn (pré-processador + classificador)
# bundle["features"]: colunas de entrada (as derivadas calculam-se com telco_churn.features.add_features)
# bundle["capacity_cut"]: P(churn) a partir da qual um cliente estabelecido está no top 20% de risco
```

Requer `scikit-learn==1.9.1`, a versão com que o artefacto foi gravado.

## Limitações

- **Clientes novos.** Os recentes que ficam são registados como *Joined* e saem do treino; os que saem ficam como *Churned*. Por isso, no treino, todos os clientes com tenure ≤ 3 saíram. É um efeito do registo, não comportamento. Para estes clientes a app mostra um aviso e usa uma regra de onboarding.
- **A IBM ordena melhor.** O Churn Score tem AUC mais alto, sobretudo nos clientes estabelecidos. Decidimos usar este modelo porque dá probabilidades calibradas (Brier muito mais baixo), necessárias para P × CLTV, e usa só variáveis conhecidas antes da saída. Não substitui o Churn Score como ordenação.
- **Desconto.** O modelo prevê risco, não o efeito de uma ação. A eficácia do desconto na app é um pressuposto a medir com grupo de controlo.
- **Dados sintéticos e sem datas.** Os padrões podem não se transferir para dados reais e não há validação temporal.
- **Importância das variáveis** mostra associação, não causa.

## Publicação

Publicado pelo GitHub Actions (`deploy-hf.yml`) só depois de a pipeline correr e de os testes passarem. O modelo tem de bater a Dummy e a regra de negócio, ter sobreajuste < 0,05 e um Brier abaixo do da IBM nos clientes estabelecidos.
