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
short_description: Risco de churn e decisão de desconto (EDSB 2026, Grupo 1)
---

# Retenção de clientes — risco de churn

App do projeto Telco Customer Churn (EDSB 2026, NOVA IMS, Grupo 1).

- **Cliente individual:** preencher as variáveis do cliente (os valores por omissão são os
  típicos do treino) e obter P(churn), o segmento de risco, os principais fatores e a
  recomendação de desconto.
- **Lote:** carregar um CSV (ex.: `exemplo_lote.csv`) e descarregar a lista de clientes
  ordenada por perda esperada = P(churn) × CLTV.

**Regra do desconto:** oferecer se P(churn) × eficácia × CLTV > desconto mensal × meses.
A eficácia é um **pressuposto**: o modelo prevê o risco de saída, não o efeito causal do
desconto.

O modelo vem do Model repo indicado na variável `MODEL_REPO` (ver o *model card* desse
repositório: dados, métricas e limitações). Código e pipeline:
<https://github.com/j3grave/Bootcamp/tree/main/telco-churn-retention>.

Esta pasta é publicada automaticamente pelo GitHub Actions (`deploy-hf.yml`) depois de a
pipeline e os testes passarem; o pacote `telco_churn` é copiado de `src/` no momento do deploy.
