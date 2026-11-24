---
title: Retenção de Clientes Telco
emoji: 📉
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Risco de churn e decisão de desconto (EDSB 2026)
---

# Retenção de clientes — risco de churn

App do projeto Telco Customer Churn (EDSB 2026, NOVA IMS).

- **Cliente individual:** preencher as variáveis do cliente (por omissão, os valores típicos do treino) e obter P(churn), se o cliente está no top 20% de risco dos clientes estabelecidos, os principais fatores e a recomendação de desconto.
- **Lote:** carregar um CSV (ex.: `exemplo_lote.csv`) e descarregar duas listas:
  - clientes estabelecidos (tenure > 3), ordenados por P(churn) × CLTV;
  - clientes novos (tenure ≤ 3), por segmento de onboarding, com aviso de que o modelo extrapola.

Os clientes com tenure ≤ 3 não entram na prioridade de contacto. No treino, todos os clientes com tão poucos meses saíram por causa da forma como os estados foram registados, por isso a P(churn) deles é uma extrapolação.

**Desconto:** oferecer se P(churn) × eficácia × CLTV > desconto mensal × meses. A eficácia é um pressuposto: o modelo prevê o risco de saída, não o efeito do desconto.

O modelo vem do Model repo indicado em `MODEL_REPO`; o *model card* desse repositório tem os dados, as métricas e as limitações. Código: <https://github.com/j3grave/Bootcamp/tree/main/telco-churn-retention>.

Esta pasta é publicada pelo `deploy-hf.yml` depois de a pipeline e os testes passarem; o pacote `telco_churn` é copiado de `src/` no deploy.
