from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
MODEL_FILES = [
    ARTIFACTS / "model.joblib",
    ARTIFACTS / "training_data.parquet",
    ARTIFACTS / "features.json",
    ARTIFACTS / "metadata.json",
    ARTIFACTS / "model_selection.csv",
]
TSNE_FILE = ARTIFACTS / "tsne_embedding.npy"

if not all(path.exists() for path in MODEL_FILES):
    print("Modelo do dashboard não encontrado. Selecionando e treinando o modelo...")
    subprocess.run([sys.executable, str(HERE / "prepare_model.py")], check=True)
else:
    print("Modelo do dashboard encontrado. Reutilizando os artefatos existentes.")

if not TSNE_FILE.exists():
    print("t-SNE não encontrado. Gerando a projeção...")
    subprocess.run([sys.executable, str(HERE / "compute_tsne.py")], check=True)
else:
    print("t-SNE encontrado. Reutilizando a projeção existente.")

subprocess.run(
    [sys.executable, "-m", "streamlit", "run", str(HERE / "app.py")],
    check=True,
)
