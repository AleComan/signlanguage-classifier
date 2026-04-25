"""Streamlit frontend for selecting models and batch image prediction."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st
import torch
from dotenv import load_dotenv
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import (
    LoadedModel,
    discover_available_models,
    load_model_for_inference,
    predict_image_topk,
)


def _resolve_default_model_path() -> Path:
    """Resolve optional default model path from env var."""
    load_dotenv()
    configured = os.getenv("MODEL_PATH", "")
    if not configured:
        return PROJECT_ROOT / "artifacts"
    return (PROJECT_ROOT / configured).resolve()


def _format_model_label(path: Path) -> str:
    """Create friendly model selector label."""
    rel_path = path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path
    return str(rel_path)


@st.cache_resource(show_spinner=False)
def _load_model_cached(model_path: str, device_label: str) -> LoadedModel:
    """Cache loaded model artifacts."""
    device = torch.device(device_label)
    return load_model_for_inference(model_path=model_path, device=device)


def _render_model_metadata(loaded_model: LoadedModel) -> None:
    """Display human-readable metadata for selected model."""
    st.sidebar.subheader("Caracteristicas del modelo")
    metadata: dict[str, Any] = {
        "path": str(loaded_model.path),
        "kind": loaded_model.kind,
        "image_size": loaded_model.image_size,
        "num_classes": len(loaded_model.class_names),
    }
    metadata.update(loaded_model.metadata)
    st.sidebar.json(metadata)


def _render_predictions(image_name: str, predictions: list[tuple[str, float]]) -> None:
    """Render top-k predictions for one image."""
    st.markdown(f"**{image_name}**")
    for rank, (label, score) in enumerate(predictions, start=1):
        st.write(f"{rank}. **{label}** - {score * 100:.2f}%")


def _prediction_row(image_name: str, predictions: list[tuple[str, float]], top_k: int) -> dict[str, str]:
    """Build compact table row for one image prediction."""
    row: dict[str, str] = {"imagen": image_name}
    for rank in range(1, top_k + 1):
        if rank <= len(predictions):
            label, score = predictions[rank - 1]
            row[f"top_{rank}"] = f"{label} ({score * 100:.2f}%)"
        else:
            row[f"top_{rank}"] = "-"
    return row


def _init_session_state() -> None:
    """Initialize session keys used for uploader reset and image selection."""
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0
    if "selected_image_name" not in st.session_state:
        st.session_state["selected_image_name"] = None


def main() -> None:
    """Run Streamlit inference app."""
    st.set_page_config(page_title="Sign Language Classifier", layout="wide")
    _init_session_state()
    st.title("Sign Language Classifier")
    st.caption("Selecciona un modelo y sube una o varias imagenes para obtener predicciones top-k.")

    artifacts_root = PROJECT_ROOT / "artifacts"
    discovered_models = discover_available_models(artifacts_root)
    default_model_path = _resolve_default_model_path()
    if default_model_path.is_file() and default_model_path not in discovered_models:
        discovered_models.insert(0, default_model_path)

    if not discovered_models:
        st.warning(
            "No se encontraron modelos en artifacts/. Entrena primero y guarda un .joblib o .pt/.pth/.ckpt."
        )
        return

    model_options = [_format_model_label(path) for path in discovered_models]
    st.sidebar.header("Configuracion")
    selected_label = st.sidebar.selectbox("Modelo", options=model_options, index=0)
    selected_model_path = discovered_models[model_options.index(selected_label)]

    runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.caption(f"Runtime device: {runtime_device}")

    top_k = st.sidebar.slider("Top-k", min_value=1, max_value=5, value=5, step=1)

    uploader_col, clear_col = st.columns([4, 1])
    with uploader_col:
        uploaded_files = st.file_uploader(
            "Selecciona una o varias imagenes",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key=f"uploaded_images_{st.session_state['uploader_key']}",
        )
    with clear_col:
        st.write("")
        st.write("")
        if st.button("Borrar fotos", width="stretch"):
            st.session_state["uploader_key"] += 1
            st.session_state["selected_image_name"] = None
            st.rerun()

    if not uploaded_files:
        st.info("Sube al menos una imagen para iniciar la inferencia.")
        st.stop()

    device = torch.device(runtime_device)
    try:
        loaded_model = _load_model_cached(model_path=str(selected_model_path), device_label=str(device))
        _render_model_metadata(loaded_model)
    except Exception as error:
        st.error(f"No se pudo cargar el modelo seleccionado: {error}")
        st.stop()

    st.subheader("Predicciones")
    progress = st.progress(0, text="Clasificando imagenes...")
    rows: list[dict[str, str]] = []
    detailed_predictions: dict[str, list[tuple[str, float]]] = {}
    preview_images: dict[str, Image.Image] = {}

    total_files = len(uploaded_files)
    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        image = Image.open(uploaded_file).convert("RGB")
        try:
            predictions = predict_image_topk(
                loaded_model=loaded_model,
                image=image,
                k=top_k,
                device=device,
            )
        except Exception as error:
            st.error(f"No se pudo predecir {uploaded_file.name}: {error}")
            progress.progress(idx / total_files, text=f"Procesadas {idx}/{total_files} imagenes")
            continue

        rows.append(_prediction_row(uploaded_file.name, predictions, top_k=top_k))
        detailed_predictions[uploaded_file.name] = predictions
        preview_images[uploaded_file.name] = image
        progress.progress(idx / total_files, text=f"Procesadas {idx}/{total_files} imagenes")

    progress.empty()

    if not rows:
        st.warning("No se pudo clasificar ninguna imagen.")
        st.stop()

    st.dataframe(rows, width="stretch", hide_index=True)

    st.subheader("Miniaturas (clic para ver predicciones)")
    items = list(preview_images.items())
    cols = st.columns(5)
    for idx, (name, image) in enumerate(items):
        with cols[idx % 5]:
            top1_label, top1_score = detailed_predictions[name][0]
            st.image(image, caption=f"Top1: {top1_label} ({top1_score * 100:.1f}%)", width="stretch")
            if st.button(f"Ver {name}", key=f"btn_{idx}", width="stretch"):
                st.session_state["selected_image_name"] = name

    if st.session_state["selected_image_name"] not in detailed_predictions:
        st.session_state["selected_image_name"] = next(iter(detailed_predictions))

    selected_image_name = st.session_state["selected_image_name"]
    st.subheader(f"Predicciones: {selected_image_name}")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(preview_images[selected_image_name], caption=selected_image_name, width="stretch")
    with col2:
        _render_predictions(selected_image_name, detailed_predictions[selected_image_name])

    st.success(
        f"Clasificacion completada para {len(uploaded_files)} imagen(es) con el modelo {selected_label}."
    )


if __name__ == "__main__":
    main()
