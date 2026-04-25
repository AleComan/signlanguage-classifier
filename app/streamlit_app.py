"""Streamlit frontend for selecting models and batch image prediction."""

from __future__ import annotations

import os
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import Any
import base64

import streamlit as st
import torch
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageOps

try:
    import av
    from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

    WEBRTC_AVAILABLE = True
    WEBRTC_SENDRECV_MODE = WebRtcMode.SENDRECV
except Exception:
    WEBRTC_AVAILABLE = False
    WEBRTC_SENDRECV_MODE = "SENDRECV"

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


def _normalize_label(value: str) -> str:
    """Normalize labels to compare class names with filename-derived labels."""
    return re.sub(r"[\s\-_]+", "", value.strip().lower())


def _infer_true_label_from_filename(image_name: str, class_names: list[str]) -> str | None:
    """Try inferring the true class from common filename conventions."""
    normalized_map = {_normalize_label(class_name): class_name for class_name in class_names}
    stem = Path(image_name).stem
    candidates = [stem]
    candidates.extend([token for token in re.split(r"[\s\-_]+", stem) if token])

    for candidate in candidates:
        normalized_candidate = _normalize_label(candidate)
        if normalized_candidate in normalized_map:
            return normalized_map[normalized_candidate]
    return None


def _build_batch_stats(
    top1_predictions: dict[str, tuple[str, float]],
    class_names: list[str],
) -> dict[str, float | int]:
    """Compute aggregate metrics for the currently uploaded image batch."""
    total_images = len(top1_predictions)
    if total_images == 0:
        return {}

    confidences = [score for _, score in top1_predictions.values()]
    stats: dict[str, float | int] = {
        "total_images": total_images,
        "avg_confidence": (sum(confidences) / total_images) * 100,
    }

    evaluated = 0
    correct = 0
    for image_name, (predicted_label, _) in top1_predictions.items():
        inferred_label = _infer_true_label_from_filename(image_name, class_names)
        if inferred_label is None:
            continue
        evaluated += 1
        if _normalize_label(predicted_label) == _normalize_label(inferred_label):
            correct += 1

    if evaluated:
        accuracy = (correct / evaluated) * 100
        stats["evaluated_images"] = evaluated
        stats["accuracy"] = accuracy
        stats["error_rate"] = 100 - accuracy

    return stats


def _image_to_data_uri(image: Image.Image, width: int = 560, height: int = 320) -> str:
    """Convert PIL image to fixed-size data URI for consistent card rendering."""
    fitted = ImageOps.fit(image.convert("RGB"), (width, height), method=Image.Resampling.LANCZOS)
    buffer = BytesIO()
    fitted.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _init_session_state() -> None:
    """Initialize session keys used for uploader reset and image selection."""
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0
    if "selected_image_name" not in st.session_state:
        st.session_state["selected_image_name"] = None


def _render_batch_classification(
    loaded_model: LoadedModel,
    device: torch.device,
    top_k: int,
    selected_label: str,
) -> None:
    """Render batch image classification UI and results."""
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

    st.subheader("Predicciones")
    progress = st.progress(0, text="Clasificando imagenes...")
    rows: list[dict[str, str]] = []
    detailed_predictions: dict[str, list[tuple[str, float]]] = {}
    top1_predictions: dict[str, tuple[str, float]] = {}
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
        top1_predictions[uploaded_file.name] = predictions[0]
        preview_images[uploaded_file.name] = image
        progress.progress(idx / total_files, text=f"Procesadas {idx}/{total_files} imagenes")

    progress.empty()

    if not rows:
        st.warning("No se pudo clasificar ninguna imagen.")
        st.stop()

    batch_stats = _build_batch_stats(top1_predictions, loaded_model.class_names)

    st.subheader("Estadisticas del lote cargado")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Imagenes procesadas", int(batch_stats["total_images"]))
    c2.metric("Confianza media Top-1", f"{float(batch_stats['avg_confidence']):.2f}%")
    if "accuracy" in batch_stats and "error_rate" in batch_stats and "evaluated_images" in batch_stats:
        c3.metric("Accuracy", f"{float(batch_stats['accuracy']):.2f}%")
        c4.metric("Error", f"{float(batch_stats['error_rate']):.2f}%")
        st.caption(
            "Accuracy y error se calculan solo sobre imagenes con etiqueta inferible desde el nombre del archivo "
            "(por ejemplo: `A_01.jpg`, `hello_sample.png`)."
        )
    else:
        c3.metric("Accuracy", "N/A")
        c4.metric("Error", "N/A")
        st.caption(
            "No se pudo inferir etiqueta real desde los nombres de archivo, por eso accuracy/error no estan disponibles."
        )

    st.subheader("Tabla de predicciones")
    st.dataframe(rows, width="stretch", hide_index=True)

    st.subheader("Miniaturas con Top-5")
    st.markdown(
        """
        <style>
        .pred-card {
            border: 1px solid rgba(250, 250, 250, 0.2);
            border-radius: 12px;
            padding: 14px 16px;
            margin-bottom: 12px;
            display: grid;
            grid-template-columns: minmax(0, 2.15fr) minmax(190px, 0.9fr);
            gap: 16px;
            background: rgba(255, 255, 255, 0.02);
        }
        .pred-image-wrap {
            width: 100%;
            height: 250px;
            border-radius: 10px;
            overflow: hidden;
            background: #101217;
        }
        .pred-image-wrap img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }
        .pred-filename {
            margin-top: 6px;
            text-align: center;
            font-size: 0.88rem;
            color: rgba(255, 255, 255, 0.72);
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .pred-side {
            height: 250px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            padding: 2px 8px 2px 4px;
        }
        .pred-title {
            font-weight: 700;
            margin-bottom: 10px;
        }
        .pred-list {
            margin: 0;
            padding: 0;
            list-style: none;
            display: grid;
            grid-template-rows: repeat(5, minmax(0, 1fr));
            gap: 2px;
            height: 100%;
        }
        .pred-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 14px;
            font-size: 0.95rem;
            line-height: 1.2;
            padding: 3px 4px;
        }
        .pred-rank {
            font-weight: 650;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .pred-score {
            color: rgba(255, 255, 255, 0.86);
            font-variant-numeric: tabular-nums;
            white-space: nowrap;
            min-width: 70px;
            text-align: right;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    items = list(preview_images.items())
    gallery_cols = st.columns(2)
    for idx, (name, image) in enumerate(items):
        with gallery_cols[idx % 2]:
            image_uri = _image_to_data_uri(image)
            pred_items = []
            for rank in range(1, 6):
                if rank <= len(detailed_predictions[name]):
                    label, score = detailed_predictions[name][rank - 1]
                    pred_items.append(
                        (
                            f"{rank}. {label}",
                            f"{score * 100:.2f}%",
                        )
                    )
                else:
                    pred_items.append((f"{rank}. -", "-"))
            pred_list_html = "".join(
                (
                    "<li class='pred-item'>"
                    f"<span class='pred-rank'>{label}</span>"
                    f"<span class='pred-score'>{score}</span>"
                    "</li>"
                )
                for label, score in pred_items
            )
            st.markdown(
                (
                    "<div class='pred-card'>"
                    "<div>"
                    f"<div class='pred-image-wrap'><img src='{image_uri}' alt='{name}'></div>"
                    f"<div class='pred-filename' title='{name}'>{name}</div>"
                    "</div>"
                    "<div class='pred-side'>"
                    "<div class='pred-title'>Top-5</div>"
                    f"<ul class='pred-list'>{pred_list_html}</ul>"
                    "</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

    st.success(
        f"Clasificacion completada para {len(uploaded_files)} imagen(es) con el modelo {selected_label}."
    )


def _render_top_mode_buttons() -> str:
    """Render top horizontal nav buttons to switch app module."""
    if "selected_mode" not in st.session_state:
        st.session_state["selected_mode"] = "Clasificacion por imagenes"

    img_col, video_col = st.columns(2)
    with img_col:
        if st.button(
            "Clasificacion por imagenes",
            type="primary" if st.session_state["selected_mode"] == "Clasificacion por imagenes" else "secondary",
            width="stretch",
        ):
            st.session_state["selected_mode"] = "Clasificacion por imagenes"
            st.rerun()
    with video_col:
        if st.button(
            "Video en tiempo real",
            type="primary" if st.session_state["selected_mode"] == "Video en tiempo real" else "secondary",
            width="stretch",
        ):
            st.session_state["selected_mode"] = "Video en tiempo real"
            st.rerun()

    return st.session_state["selected_mode"]


def _render_realtime_video(
    loaded_model: LoadedModel,
    device: torch.device,
    top_k: int,
) -> None:
    """Render realtime webcam inference page."""
    st.subheader("Prediccion en tiempo real (camara)")
    st.caption(
        "Inicia la camara para ver las top-5 predicciones superpuestas sobre el video. "
        "Puedes reducir frecuencia de inferencia para mejorar rendimiento."
    )

    if not WEBRTC_AVAILABLE:
        st.error(
            "No esta disponible el modulo de video en tiempo real. "
            "Instala `streamlit-webrtc` y reinicia la app."
        )
        return

    frame_stride = st.slider(
        "Frecuencia de inferencia (cada N frames)",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
    )
    st.markdown(
        """
        <style>
        video {
            max-height: 420px !important;
            width: auto !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    class SignLanguageVideoProcessor(VideoProcessorBase):
        """Video processor that overlays model predictions on frames."""

        loaded_model_ref = loaded_model
        device_ref = device
        top_k_ref = top_k
        frame_stride_ref = frame_stride

        def __init__(self) -> None:
            self.frame_count = 0
            self.last_predictions: list[tuple[str, float]] = [("Detectando...", 0.0)]

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_image().convert("RGB")
            self.frame_count += 1

            if self.frame_count % max(self.frame_stride_ref, 1) == 0:
                try:
                    predictions = predict_image_topk(
                        loaded_model=self.loaded_model_ref,
                        image=image,
                        k=self.top_k_ref,
                        device=self.device_ref,
                    )
                    if predictions:
                        self.last_predictions = predictions
                except Exception:
                    self.last_predictions = [("Error de inferencia", 0.0)]

            draw = ImageDraw.Draw(image)
            lines = ["Top-5 predicciones"]
            for rank in range(1, self.top_k_ref + 1):
                if rank <= len(self.last_predictions):
                    label, score = self.last_predictions[rank - 1]
                    lines.append(f"{rank}. {label} ({score * 100:.1f}%)")
                else:
                    lines.append(f"{rank}. -")

            max_line_chars = max(len(line) for line in lines)
            rect_width = min(720, 24 + max_line_chars * 8)
            rect_height = 18 + len(lines) * 22
            draw.rectangle((10, 10, rect_width, 10 + rect_height), fill=(0, 0, 0))

            for i, line in enumerate(lines):
                y = 16 + i * 22
                if i == 1:
                    # Simulate bold for top-1 by drawing text twice with slight offset.
                    draw.text((16, y), line, fill=(255, 255, 0))
                    draw.text((17, y), line, fill=(255, 255, 0))
                elif i == 0:
                    draw.text((16, y), line, fill=(255, 255, 255))
                else:
                    draw.text((16, y), line, fill=(220, 220, 220))
            return av.VideoFrame.from_image(image)

    model_key = Path(str(loaded_model.path)).name
    video_col, info_col = st.columns([3, 1])
    with video_col:
        webrtc_streamer(
            key=f"sign-realtime-{model_key}-{top_k}-{frame_stride}",
            mode=WEBRTC_SENDRECV_MODE,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=SignLanguageVideoProcessor,
            async_processing=True,
        )
    with info_col:
        st.info(
            "Tip: si el video va lento, sube `Frecuencia de inferencia` para ejecutar el modelo cada mas frames."
        )


def main() -> None:
    """Run Streamlit inference app."""
    st.set_page_config(page_title="Sign Language Classifier", layout="wide")
    _init_session_state()
    section = _render_top_mode_buttons()

    st.title("Sign Language Classifier")
    st.caption("Selecciona un modelo y el modulo de inferencia: imagenes o video en tiempo real.")

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
    top_k = 5

    device = torch.device(runtime_device)
    try:
        loaded_model = _load_model_cached(model_path=str(selected_model_path), device_label=str(device))
        _render_model_metadata(loaded_model)
    except Exception as error:
        st.error(f"No se pudo cargar el modelo seleccionado: {error}")
        st.stop()

    if section == "Clasificacion por imagenes":
        _render_batch_classification(
            loaded_model=loaded_model,
            device=device,
            top_k=top_k,
            selected_label=selected_label,
        )
    else:
        _render_realtime_video(
            loaded_model=loaded_model,
            device=device,
            top_k=top_k,
        )


if __name__ == "__main__":
    main()
