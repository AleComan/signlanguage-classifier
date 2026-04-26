"""Streamlit frontend for selecting models and batch image prediction."""

from __future__ import annotations

import os
import re
import sys
from collections import Counter
from datetime import datetime
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
    """Create friendly model selector label without full path."""
    parent_name = path.parent.name
    if parent_name and parent_name != path.anchor:
        return f"{path.name} ({parent_name})"
    return path.name


def _build_model_labels(paths: list[Path]) -> list[str]:
    """Build readable labels and disambiguate duplicates."""
    base_labels = [_format_model_label(path) for path in paths]
    counts = Counter(base_labels)
    labels: list[str] = []
    for path, base_label in zip(paths, base_labels):
        if counts[base_label] == 1:
            labels.append(base_label)
            continue
        # If two models share same filename/parent, include one extra parent level.
        grandparent = path.parent.parent.name if path.parent.parent != path.parent else ""
        if grandparent:
            labels.append(f"{path.name} ({grandparent}/{path.parent.name})")
        else:
            labels.append(base_label)
    return labels


@st.cache_resource(show_spinner=False)
def _load_model_cached(model_path: str, device_label: str) -> LoadedModel:
    """Cache loaded model artifacts."""
    device = torch.device(device_label)
    return load_model_for_inference(model_path=model_path, device=device)


def _render_model_metadata(loaded_model: LoadedModel) -> None:
    """Display human-readable metadata for selected model."""
    st.sidebar.subheader("Caracteristicas del modelo")
    file_stats = loaded_model.path.stat()
    metadata: dict[str, Any] = {
        "file_name": loaded_model.path.name,
        "file_size_mb": round(file_stats.st_size / (1024 * 1024), 2),
        "last_modified": datetime.fromtimestamp(file_stats.st_mtime).strftime("%Y-%m-%d %H:%M"),
        "kind": loaded_model.kind,
        "image_size": loaded_model.image_size,
        "num_classes": len(loaded_model.class_names),
    }
    metadata.update(loaded_model.metadata)

    st.sidebar.markdown(
        """
        <style>
        .model-meta-card {
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 12px;
            padding: 10px 12px;
            margin-bottom: 12px;
            background: rgba(255, 255, 255, 0.02);
        }
        .model-meta-row {
            display: flex;
            justify-content: space-between;
            gap: 10px;
            padding: 5px 0;
            border-bottom: 1px dashed rgba(255, 255, 255, 0.12);
            font-size: 0.9rem;
        }
        .model-meta-row:last-child {
            border-bottom: none;
        }
        .model-meta-key {
            color: rgba(255, 255, 255, 0.70);
        }
        .model-meta-value {
            text-align: right;
            font-weight: 600;
            color: rgba(255, 255, 255, 0.94);
            word-break: break-word;
        }
        .class-chip-wrap {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 6px;
        }
        .class-chip {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            background: rgba(255, 255, 255, 0.07);
            font-size: 0.78rem;
            line-height: 1.4;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        f"""
        <div class="model-meta-card">
            <div class="model-meta-row"><span class="model-meta-key">Archivo</span><span class="model-meta-value">{metadata.get("file_name", "-")}</span></div>
            <div class="model-meta-row"><span class="model-meta-key">Tipo</span><span class="model-meta-value">{metadata.get("kind", "-")}</span></div>
            <div class="model-meta-row"><span class="model-meta-key">Tamano</span><span class="model-meta-value">{metadata.get("file_size_mb", "-")} MB</span></div>
            <div class="model-meta-row"><span class="model-meta-key">Ultima modificacion</span><span class="model-meta-value">{metadata.get("last_modified", "-")}</span></div>
            <div class="model-meta-row"><span class="model-meta-key">Image size</span><span class="model-meta-value">{metadata.get("image_size", "-")}</span></div>
            <div class="model-meta-row"><span class="model-meta-key">Num clases</span><span class="model-meta-value">{metadata.get("num_classes", "-")}</span></div>
            <div class="model-meta-row"><span class="model-meta-key">Familia</span><span class="model-meta-value">{metadata.get("model_family", "-")}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if loaded_model.class_names:
        class_chips = "".join(f"<span class='class-chip'>{class_name}</span>" for class_name in loaded_model.class_names)
        st.sidebar.markdown("**Clases**")
        st.sidebar.markdown(f"<div class='class-chip-wrap'>{class_chips}</div>", unsafe_allow_html=True)


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
    """Convert PIL image to fixed-size data URI without cropping original content."""
    # Use contain (not fit) to preserve full image; pad the remaining area.
    contained = ImageOps.contain(image.convert("RGB"), (width, height), method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), color=(16, 18, 23))
    x = (width - contained.width) // 2
    y = (height - contained.height) // 2
    canvas.paste(contained, (x, y))
    buffer = BytesIO()
    canvas.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _init_session_state() -> None:
    """Initialize session keys used for uploader reset and image selection."""
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0
    if "selected_image_name" not in st.session_state:
        st.session_state["selected_image_name"] = None
    if "uploaded_images_data" not in st.session_state:
        # Stored as list[(filename, bytes)] so images persist across UI mode switches.
        st.session_state["uploaded_images_data"] = []


def _render_loaded_images_right_panel(persisted_images: list[tuple[str, bytes]]) -> None:
    """Render a floating right panel to manage session images."""
    st.markdown(
        """
        <style>
        /* Keep open panel visually anchored below the trigger button. */
        div[data-testid="stPopover"] [data-baseweb="popover"] {
            transform-origin: top right !important;
            margin-top: 8px !important;
            animation: images-popover-enter 180ms ease-out;
        }
        @keyframes images-popover-enter {
            from {
                opacity: 0;
                transform: translateY(-8px) scale(0.99);
            }
            to {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
        }
        /* Match uploader vertical rhythm so trigger aligns with upload block. */
        div[data-testid="stPopover"] > div > button {
            min-height: 42px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.popover(f"Imagenes cargadas ({len(persisted_images)})", width="stretch"):
        if st.button(
            "Borrar todas",
            width="stretch",
            disabled=not persisted_images,
            key="clear_all_images_right_panel",
        ):
            st.session_state["uploader_key"] += 1
            st.session_state["selected_image_name"] = None
            st.session_state["uploaded_images_data"] = []
            st.rerun()

        if not persisted_images:
            st.caption("No hay imagenes cargadas.")
            return

        st.caption("Pulsa `✕` para quitar una imagen concreta.")
        # Fixed-height list prevents popover from growing and flipping above the trigger.
        list_container = st.container(height=360, border=False)
        with list_container:
            for idx, (name, _) in enumerate(persisted_images):
                name_col, remove_col = st.columns([5, 1])
                with name_col:
                    st.caption(f"{idx + 1}. {name}")
                with remove_col:
                    if st.button("✕", key=f"remove_image_{idx}_{name}", width="stretch"):
                        # Remove by position to handle possible duplicate filenames.
                        st.session_state["uploaded_images_data"] = [
                            image_item for image_pos, image_item in enumerate(persisted_images) if image_pos != idx
                        ]
                        # Reset uploader widget so removed files are not re-added on rerun.
                        st.session_state["uploader_key"] += 1
                        st.session_state["selected_image_name"] = None
                        st.rerun()


def _render_batch_classification(
    loaded_model: LoadedModel,
    device: torch.device,
    top_k: int,
    selected_label: str,
) -> None:
    """Render batch image classification UI and results."""
    uploader_col, right_panel_col = st.columns([4, 1])
    with uploader_col:
        uploaded_files = st.file_uploader(
            "Selecciona una o varias imagenes",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key=f"uploaded_images_{st.session_state['uploader_key']}",
        )

    persisted_images: list[tuple[str, bytes]] = st.session_state["uploaded_images_data"]
    if uploaded_files:
        # Merge new picks with session-persisted images so users can keep managing them across mode switches.
        order = [name for name, _ in persisted_images]
        images_by_name = {name: data for name, data in persisted_images}
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            images_by_name[file_name] = uploaded_file.getvalue()
            if file_name not in order:
                order.append(file_name)
        st.session_state["uploaded_images_data"] = [(name, images_by_name[name]) for name in order]
        # Clear uploader selected-file chips to avoid redundant duplicated UI.
        st.session_state["uploader_key"] += 1
        st.rerun()

    with right_panel_col:
        # Add vertical spacer so the trigger sits aligned with uploader widget.
        st.markdown("<div style='height: 1.95rem;'></div>", unsafe_allow_html=True)
        _render_loaded_images_right_panel(persisted_images)

    if not persisted_images:
        st.info("Sube al menos una imagen para iniciar la inferencia.")
        st.stop()

    st.subheader("Predicciones")
    progress = st.progress(0, text="Clasificando imagenes...")
    rows: list[dict[str, str]] = []
    detailed_predictions: dict[str, list[tuple[str, float]]] = {}
    top1_predictions: dict[str, tuple[str, float]] = {}
    preview_images: dict[str, Image.Image] = {}

    total_files = len(persisted_images)
    for idx, (image_name, image_bytes) in enumerate(persisted_images, start=1):
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        try:
            predictions = predict_image_topk(
                loaded_model=loaded_model,
                image=image,
                k=top_k,
                device=device,
            )
        except Exception as error:
            st.error(f"No se pudo predecir {image_name}: {error}")
            progress.progress(idx / total_files, text=f"Procesadas {idx}/{total_files} imagenes")
            continue

        rows.append(_prediction_row(image_name, predictions, top_k=top_k))
        detailed_predictions[image_name] = predictions
        top1_predictions[image_name] = predictions[0]
        preview_images[image_name] = image
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
        f"Clasificacion completada para {len(persisted_images)} imagen(es) con el modelo {selected_label}."
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

    model_options = _build_model_labels(discovered_models)
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
