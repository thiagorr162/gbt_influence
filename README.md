# Influência em GBDT com BoostIn

Este projeto treina um modelo LightGBM para prever o índice de refração (RI) e usa o BoostIn para estimar quanto cada ponto de treino ajuda ou prejudica as previsões.

A análise remove amostras com `RI > 3`, seleciona 50 pontos de teste por decil de RI e produz rankings e visualizações conjuntas e por decil. As figuras incluem versões log/symlog e lineares, além de projeções t-SNE.

## Como executar

O arquivo de entrada deve estar em `data/refractive_index.parquet`.

```bash
uv sync
uv run scripts/01_prepare_data.py
uv run scripts/02_train_model.py
uv run scripts/03_compute_boostin.py
uv run scripts/04_rank_influence.py
uv run scripts/05_compute_tsne.py
uv run scripts/06_plot_results.py
```

Os resultados intermediários são salvos em `artifacts/` e as imagens em `figures/`. Para refazer somente os gráficos após alterar sua aparência, execute apenas:

```bash
uv run scripts/06_plot_results.py
```

## Referência

Brophy, J.; Hammoudeh, Z.; Lowd, D. [Adapting and Evaluating Influence-Estimation Methods for Gradient-Boosted Decision Trees](https://www.jmlr.org/papers/v24/22-0449.html). *Journal of Machine Learning Research*, 24(154):1–48, 2023.

## Dashboard para novas composições

O dashboard aceita composições como `70sio2+20na2o+10cao`, prevê o RI e usa BoostIn para listar e visualizar os pontos de treino que ajudam ou atrapalham.

```bash
uv sync
uv run dashboard/run_dashboard.py
```

O lançador reutiliza o modelo e o t-SNE em `dashboard/artifacts/`. Se estiverem ausentes, executa o pequeno grid de LightGBM, retreina a melhor configuração em todos os dados e gera a projeção automaticamente.
