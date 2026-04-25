"""Minimal Streamlit frontend for image classification inference."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
import torch
from dotenv import load_dotenv
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import load_torch_model, predict_image_topk


def _resolve_model_path() -> Path:
    """Resolve model path from env var with a sane default."""
    load_dotenv()
    configured = os.getenv("MODEL_PATH", "artifacts/checkpoints/latest.pt")
    return (PROJECT_ROOT / configured).resolve()


def main() -> None:
    """Run Streamlit inference app."""
    st.set_page_config(page_title="Sign Language Classifier", layout="centered")
    st.title("Sign Language Classifier")
    st.caption("Sube una imagen para obtener predicciones top-k.")

    model_path = _resolve_model_path()
    checkpoint_available = model_path.exists()

    if checkpoint_available:
        st.success(f"Checkpoint encontrado: {model_path}")
    else:
        st.warning(
            "Aun no hay checkpoint disponible. Entrena primero con alguno de los notebooks "
            "(notebooks/03_scratch_cnn.ipynb o notebooks/04_finetune.ipynb) y ajusta MODEL_PATH "
            "en tu .env si hace falta."
        )

    uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png", "webp"])
    top_k = st.slider("Top-k", min_value=1, max_value=5, value=3, step=1)

    if uploaded_file is None:
        st.info("Sube una imagen para iniciar la inferencia.")
        return

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_container_width=True)

    if not checkpoint_available:
        st.stop()

    device = torch.device("cpu")
    try:
        model, class_names, image_size = load_torch_model(checkpoint_path=model_path, device=device)
        predictions = predict_image_topk(
            model=model,
            class_names=class_names,
            image=image,
            image_size=image_size,
            k=top_k,
            device=device,
        )
    except Exception as error:
        st.error(f"No se pudo cargar o ejecutar el modelo: {error}")
        st.stop()

    st.subheader("Predicciones")
    for rank, (label, score) in enumerate(predictions, start=1):
        st.write(f"{rank}. **{label}** - {score * 100:.2f}%")


if __name__ == "__main__":
    main()
