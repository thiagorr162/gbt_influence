from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm
import numpy as np
import pandas as pd
import streamlit as st
from tree_influence.explainers import BoostIn

from composition import format_composition, parse_composition, pretty_formula

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
TARGET = "refractive_index"
ID_COLUMN = "ID"

st.set_page_config(page_title="BoostIn para índice de refração", layout="wide")
st.title("Influência de composições em um modelo GBDT")
st.caption("Predição de índice de refração e atribuição dos pontos de treino com BoostIn")

required = [
    ARTIFACTS / "model.joblib",
    ARTIFACTS / "training_data.parquet",
    ARTIFACTS / "features.json",
    ARTIFACTS / "metadata.json",
    ARTIFACTS / "model_selection.csv",
    ARTIFACTS / "tsne_embedding.npy",
]
if not all(path.exists() for path in required):
    st.error("Os artefatos do dashboard ainda não foram preparados.")
    st.code("uv run python dashboard/prepare_model.py")
    st.stop()


@st.cache_resource(show_spinner="Carregando modelo e preparando o BoostIn...")
def load_resources():
    model = joblib.load(ARTIFACTS / "model.joblib")
    training = pd.read_parquet(ARTIFACTS / "training_data.parquet")
    with (ARTIFACTS / "features.json").open() as file:
        features = json.load(file)
    with (ARTIFACTS / "metadata.json").open() as file:
        metadata = json.load(file)
    selection = pd.read_csv(ARTIFACTS / "model_selection.csv")
    embedding = np.load(ARTIFACTS / "tsne_embedding.npy")
    X_train = training[features].to_numpy()
    y_train = training[TARGET].to_numpy()
    explainer = BoostIn().fit(model, X_train, y_train)
    return model, training, features, metadata, selection, embedding, explainer


def symmetric_log_norm(values):
    absolute = np.abs(values)
    nonzero = absolute[absolute > 0]
    limit = float(absolute.max())
    if limit == 0:
        limit = 1.0
    linear_threshold = float(np.quantile(nonzero, 0.10)) if len(nonzero) else 1.0
    return SymLogNorm(linthresh=linear_threshold, vmin=-limit, vmax=limit, base=10)


def robust_linear_norm(values):
    limit = float(np.quantile(np.abs(values), 0.95))
    if limit == 0:
        limit = 1.0
    return Normalize(vmin=-limit, vmax=limit, clip=True)


def make_table(training, features, influence, indices, classification):
    table = training.iloc[indices][[ID_COLUMN, TARGET]].copy()
    table["influence"] = influence[indices]
    table["classificação"] = classification
    table["composição"] = [
        format_composition(row, features)
        for _, row in training.iloc[indices].iterrows()
    ]
    table = table[["classificação", "influence", TARGET, ID_COLUMN, "composição"]]
    return table.rename(columns={TARGET: "RI de treino"}).reset_index(drop=True)


def plot_ri_influence(training, influence, reference_ri, top_indices, scale):
    if scale == "Symlog":
        norm = symmetric_log_norm(influence)
        extend = "neither"
    else:
        norm = robust_linear_norm(influence)
        extend = "both"

    fig, ax = plt.subplots(figsize=(10, 6))
    if scale == "Linear robusta":
        ax.set_facecolor("#eeeeee")
    points = ax.scatter(
        training[TARGET], influence,
        c=influence, cmap="RdBu", norm=norm,
        s=8, alpha=0.55,
    )
    ax.scatter(
        training.iloc[top_indices][TARGET], influence[top_indices],
        facecolors="none", edgecolors="black", s=75, linewidths=1,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(reference_ri, color="#6a3d9a", linestyle="--", linewidth=1.3, label="RI de referência")
    if scale == "Symlog":
        ax.set_yscale("symlog", linthresh=norm.linthresh)
    ax.set_xlabel("RI do ponto de treino")
    ax.set_ylabel("Influência BoostIn")
    ax.set_title(f"RI versus influência para a composição consultada — {scale}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.16, 1))
    fig.colorbar(points, ax=ax, label="Influência", extend=extend, pad=0.02)
    fig.tight_layout()
    return fig


def plot_tsne(embedding, influence, top_indices, query_position, scale):
    norm = symmetric_log_norm(influence) if scale == "Symlog" else robust_linear_norm(influence)
    extend = "neither" if scale == "Symlog" else "both"
    fig, ax = plt.subplots(figsize=(10, 7))
    if scale == "Linear robusta":
        ax.set_facecolor("#eeeeee")
    points = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=influence, cmap="RdBu", norm=norm,
        s=8, alpha=0.65,
    )
    ax.scatter(
        embedding[top_indices, 0], embedding[top_indices, 1],
        facecolors="none", edgecolors="black", s=75, linewidths=1,
    )
    ax.scatter(
        query_position[0], query_position[1],
        marker="*", color="#ffd92f", edgecolor="black",
        s=260, linewidth=1.2, label="composição consultada (aprox.)",
        zorder=5,
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.14, 1))
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"t-SNE colorido pela influência — {scale}")
    fig.colorbar(points, ax=ax, label="Influência", extend=extend)
    fig.tight_layout()
    return fig


def plot_top_influences(helpful, harmful):
    combined = pd.concat([harmful.iloc[::-1], helpful], ignore_index=True)
    colors = ["#b2182b" if value < 0 else "#2166ac" for value in combined["influence"]]
    labels = [str(value)[-24:] for value in combined[ID_COLUMN]]
    fig, ax = plt.subplots(figsize=(10, max(5, len(combined) * 0.32)))
    positions = np.arange(len(combined))
    ax.barh(positions, combined["influence"], color=colors, alpha=0.85)
    ax.set_yticks(positions, labels)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Influência BoostIn")
    ax.set_title("Pontos que mais ajudam e mais atrapalham")
    fig.tight_layout()
    return fig


model, training, features, metadata, selection, embedding, explainer = load_resources()

with st.sidebar:
    st.header("Visualização")
    top_n = st.slider("Quantidade por grupo", min_value=5, max_value=30, value=10)
    scale = st.radio("Escala", ["Symlog", "Linear robusta"])
    with st.expander("Modelo selecionado"):
        st.write("**Combinações testadas:**", metadata["grid_size"])
        st.write("**RMSE de validação:**", f"{metadata.get("validation_rmse", float("nan")):.5f}")
        st.write("**MAE de validação:**", f"{metadata.get("validation_mae", float("nan")):.5f}")
        st.dataframe(selection, hide_index=True)

with st.form("composition_form"):
    composition_text = st.text_input(
        "Composição",
        value="70sio2+20na2o+10cao",
        help="Use porcentagem seguida do óxido e separe os termos com +",
    )
    reference_text = st.text_input(
        "RI de referência opcional",
        placeholder="Deixe vazio para usar o RI predito",
    )
    normalize = st.checkbox("Normalizar a composição para 100%", value=True)
    submitted = st.form_submit_button("Prever e calcular influências", type="primary")

if submitted:
    try:
        composition, original_total, was_normalized = parse_composition(
            composition_text,
            features,
            normalize=normalize,
        )
        prediction = float(model.predict(composition.to_numpy())[0])
        if reference_text.strip():
            reference_ri = float(reference_text.replace(",", "."))
            reference_source = "informado pelo usuário"
        else:
            reference_ri = prediction
            reference_source = "própria previsão do modelo"

        with st.spinner("Calculando a influência de todas as amostras de treino..."):
            influence = explainer.get_local_influence(
                composition.to_numpy(),
                np.array([reference_ri]),
                verbose=0,
            )[:, 0]

        distances = np.linalg.norm(
            training[features].to_numpy() - composition.to_numpy()[0],
            axis=1,
        )
        neighbor_indices = np.argsort(distances)[:10]
        neighbor_distances = distances[neighbor_indices]
        if neighbor_distances[0] == 0:
            exact = neighbor_indices[neighbor_distances == 0]
            query_position = embedding[exact].mean(axis=0)
        else:
            weights = 1.0 / neighbor_distances
            query_position = np.average(embedding[neighbor_indices], axis=0, weights=weights)

        st.session_state["analysis"] = {
            "composition": composition,
            "original_total": original_total,
            "was_normalized": was_normalized,
            "prediction": prediction,
            "reference_ri": reference_ri,
            "reference_source": reference_source,
            "influence": influence,
            "query_position": query_position,
        }
    except ValueError as error:
        st.error(str(error))

if "analysis" not in st.session_state:
    st.info("Digite uma composição e clique em Prever e calcular influências.")
    st.stop()

analysis = st.session_state["analysis"]
composition = analysis["composition"]
prediction = analysis["prediction"]
reference_ri = analysis["reference_ri"]
influence = analysis["influence"]
query_position = analysis["query_position"]

if analysis["was_normalized"]:
    original_total = analysis["original_total"]
    st.info(f"A soma original era {original_total:.4g} e foi normalizada para 100%.")
elif not np.isclose(analysis["original_total"], 100.0):
    original_total = analysis["original_total"]
    st.warning(f"A composição soma {original_total:.4g}, fora do padrão de 100% usado nos dados.")

metric_1, metric_2, metric_3 = st.columns(3)
metric_1.metric("RI predito", f"{prediction:.5f}")
metric_2.metric("RI usado pelo BoostIn", f"{reference_ri:.5f}")
metric_3.metric("Soma da composição", f"{composition.to_numpy().sum():.2f}%")
reference_source = analysis["reference_source"]
st.caption(f"Referência da influência: {reference_source}.")

nonzero = composition.iloc[0][composition.iloc[0] > 0]
parsed = pd.DataFrame({
    "óxido": [pretty_formula(feature) for feature in nonzero.index],
    "porcentagem": nonzero.to_numpy(),
})
with st.expander("Composição interpretada", expanded=True):
    st.dataframe(parsed, hide_index=True, use_container_width=True)

helpful_indices = np.argsort(influence)[::-1]
helpful_indices = helpful_indices[influence[helpful_indices] > 0][:top_n]
harmful_indices = np.argsort(influence)
harmful_indices = harmful_indices[influence[harmful_indices] < 0][:top_n]
top_indices = np.concatenate([helpful_indices, harmful_indices])

helpful = make_table(training, features, influence, helpful_indices, "ajuda")
harmful = make_table(training, features, influence, harmful_indices, "atrapalha")

st.subheader("Pontos mais influentes")
left, right = st.columns(2)
with left:
    st.markdown("#### Ajudam a previsão")
    st.dataframe(helpful, hide_index=True, use_container_width=True)
with right:
    st.markdown("#### Atrapalham a previsão")
    st.dataframe(harmful, hide_index=True, use_container_width=True)

export = pd.concat([helpful, harmful], ignore_index=True)
st.download_button(
    "Baixar ranking em CSV",
    data=export.to_csv(index=False).encode("utf-8"),
    file_name="boostin_influences.csv",
    mime="text/csv",
)

chart_1, chart_2, chart_3 = st.tabs(["RI × influência", "t-SNE", "Ranking"])
with chart_1:
    st.pyplot(plot_ri_influence(training, influence, reference_ri, top_indices, scale))
with chart_2:
    st.pyplot(plot_tsne(embedding, influence, top_indices, query_position, scale))
    st.caption("A estrela é uma projeção aproximada pelos 10 vizinhos mais próximos no mapa t-SNE fixo.")
with chart_3:
    st.pyplot(plot_top_influences(helpful, harmful))

st.markdown("---")
st.caption(
    "Influência positiva indica que o ponto reduz a perda para o RI de referência; "
    "influência negativa indica que ele aumenta essa perda. Sem RI medido, a previsão "
    "é usada como pseudoalvo e a leitura passa a ser apoio ou oposição à previsão do modelo."
)
st.markdown(
    "Método: [Adapting and Evaluating Influence-Estimation Methods for Gradient-Boosted Decision Trees]"
    "(https://www.jmlr.org/papers/v24/22-0449.html)."
)
