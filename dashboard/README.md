# Dashboard BoostIn

O dashboard recebe uma composição como `70sio2+20na2o+10cao`, prevê o índice de refração e mostra quais amostras de treino ajudam ou atrapalham essa previsão.

Execute:

```bash
uv sync
uv run python dashboard/run_dashboard.py
```

Na primeira execução, o lançador testa um pequeno grid de hiperparâmetros do LightGBM por RMSE de validação, escolhe a melhor e a retreina com todos os dados com `RI <= 3`. O modelo e o t-SNE ficam em `dashboard/artifacts/` e são verificados e reutilizados independentemente nas execuções seguintes.

Para forçar uma nova seleção, remova `dashboard/artifacts/` e execute novamente o lançador.
