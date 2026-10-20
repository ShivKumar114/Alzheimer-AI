#!/usr/bin/env python3
"""Local Streamlit interface for the saved MRI ResNet18 experiment."""

from __future__ import annotations

import io
import hashlib
import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "checkpoints" / "mri_resnet18" / "best_model.pth"
METRICS_PATH = ROOT / "checkpoints" / "mri_resnet18" / "metrics.json"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_BATCH_IMAGES = 20_000
MAX_BATCH_TOTAL_BYTES = 500 * 1024 * 1024
INFERENCE_BATCH_SIZE = 16
SUPPORTED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tif", "tiff"}
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

st.set_page_config(
    page_title="MRI Classifier | Research Interface",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
      color-scheme: light;
      --ink: #142b3d;
      --muted: #52697b;
      --teal: #087f78;
      --line: #d8e2e9;
      --canvas: #f4f7fb;
      --surface: #ffffff;
    }
    html, body, [class*="css"] {
      font-family: Inter, "Segoe UI", Arial, sans-serif !important;
      color: var(--ink) !important;
    }
    [data-testid="stAppViewContainer"], [data-testid="stMain"],
    [data-testid="stHeader"] { background: var(--canvas) !important; }
    [data-testid="stHeader"] { border-bottom: 1px solid var(--line); }
    h1, h2, h3, h4 { color: var(--ink) !important; letter-spacing: -.025em; }
    p, li, label, [data-testid="stMarkdownContainer"] { color: var(--ink); }
    .block-container { max-width: 1320px; padding: 2rem clamp(1rem, 3vw, 3rem) 4rem; }

    [data-testid="stSidebar"] {
      background: #eaf0f5 !important;
      border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] label p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
      color: #193448 !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
      padding: .45rem .55rem;
      border-radius: 9px;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover { background: #dce7ee; }

    .hero {
      background: linear-gradient(120deg, #123c4a 0%, #176c6d 58%, #328a80 100%);
      color: #fff; padding: clamp(1.5rem, 3vw, 2.6rem); border-radius: 20px;
      margin: 0 0 1.8rem; box-shadow: 0 14px 34px rgba(18, 60, 74, .14);
    }
    .hero h1 { color: #fff !important; font-size: clamp(2rem, 4vw, 2.8rem); margin: 0 0 .5rem; }
    .hero p { color: #e6f3f1 !important; font-size: 1.05rem; margin: 0; max-width: 780px; line-height: 1.7; }
    .eyebrow { text-transform: uppercase; letter-spacing: .14em; font-size: .74rem;
      font-weight: 750; color: #c5e9e2 !important; margin-bottom: .65rem; }

    .metric-card {
      padding: 1.1rem 1.2rem; border: 1px solid var(--line); border-radius: 16px;
      background: var(--surface); min-height: 112px; box-shadow: 0 5px 18px rgba(30, 55, 75, .045);
    }
    .metric-label { color: var(--muted) !important; font-size: .84rem; font-weight: 650; }
    .metric-value { color: var(--ink) !important; font-size: 1.8rem; font-weight: 800; margin-top: .3rem; }
    .metric-note { color: var(--muted) !important; font-size: .78rem; margin-top: .1rem; }
    .section-kicker { color: var(--teal) !important; font-size: .75rem; text-transform: uppercase;
      letter-spacing: .14em; font-weight: 750; margin-bottom: .3rem; }
    .notice, .warning {
      padding: 1rem 1.15rem; border-radius: 0 12px 12px 0; line-height: 1.6;
    }
    .notice { background: #eaf3f7; border-left: 4px solid #28738b; color: #244c60 !important; }
    .warning { background: #fff5df; border-left: 4px solid #bd8213; color: #624710 !important; }
    .class-chip { display: inline-block; margin: .2rem .3rem .2rem 0; padding: .45rem .75rem;
      border-radius: 999px; background: #e1f0ed; color: #145c57 !important; font-weight: 650; font-size: .84rem; }

    /* Explicit text colors prevent dark-mode browser preferences from making
       Streamlit metric cards white-on-white. */
    div[data-testid="stMetric"] {
      background: var(--surface) !important; color: var(--ink) !important;
      padding: 1rem 1.1rem; border: 1px solid var(--line); border-radius: 16px;
      box-shadow: 0 5px 18px rgba(30, 55, 75, .045);
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *,
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] *,
    [data-testid="stMetricDelta"], [data-testid="stMetricDelta"] * {
      color: var(--ink) !important;
    }
    [data-testid="stMetricLabel"] { color: var(--muted) !important; }
    .stButton > button {
      min-height: 2.8rem; border-radius: 10px; font-weight: 700;
      border: 1px solid #087f78; transition: transform .15s ease, box-shadow .15s ease;
    }
    .stButton > button[kind="primary"] {
      background: #087f78 !important; color: #fff !important;
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 5px 14px rgba(8, 127, 120, .18); }
    [data-testid="stFileUploader"] section {
      background: #fff !important; border: 1px dashed #9bb0bf !important;
      border-radius: 14px; color: var(--ink) !important;
    }
    [data-testid="stFileUploader"] section *, [data-testid="stFileUploader"] small {
      color: var(--ink) !important;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
      border: 1px solid var(--line); border-radius: 12px; overflow: hidden;
    }
    [data-testid="stAlert"] { border-radius: 12px; }
    hr { border-color: var(--line) !important; }

    @media (max-width: 768px) {
      .block-container { padding: 1.1rem 1rem 2.5rem; }
      .hero { border-radius: 15px; }
      .hero h1 { font-size: 1.9rem; }
      .metric-card { min-height: auto; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def read_metrics() -> tuple[dict | None, str | None]:
    if not METRICS_PATH.is_file():
        return None, f"Metrics file not found: `{METRICS_PATH.relative_to(ROOT)}`. Performance results are unavailable."
    try:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        if not isinstance(metrics, dict) or not isinstance(metrics.get("test"), dict):
            raise ValueError("Expected a JSON object containing a `test` section.")
        required = {"accuracy", "macro_f1", "loss"}
        missing = required.difference(metrics["test"])
        if missing:
            raise ValueError(f"The saved test section is missing: {', '.join(sorted(missing))}.")
        return metrics, None
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return None, f"Could not read saved metrics: {exc}"


@st.cache_resource(show_spinner="Loading the trained ResNet18 checkpoint…")
def load_model() -> tuple[torch.nn.Module, list[str], torch.device]:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH.relative_to(ROOT)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    class_names = checkpoint.get("class_names")
    state_dict = checkpoint.get("model_state_dict")
    if checkpoint.get("architecture") != "torchvision.resnet18":
        raise ValueError("Checkpoint architecture is missing or is not torchvision ResNet18.")
    if not isinstance(class_names, list) or len(class_names) != 4 or not all(
        isinstance(name, str) and name for name in class_names
    ):
        raise ValueError("Checkpoint must contain the ordered names for four classes.")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a model_state_dict.")
    if int(checkpoint.get("image_size", -1)) != IMAGE_SIZE:
        raise ValueError("Checkpoint image size does not match the training pipeline (224×224).")
    if tuple(checkpoint.get("normalization_mean", ())) != IMAGENET_MEAN or tuple(
        checkpoint.get("normalization_std", ())
    ) != IMAGENET_STD:
        raise ValueError("Checkpoint normalization values do not match the training pipeline.")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model, class_names, device


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def render_overview(metrics: dict | None, metric_error: str | None) -> None:
    st.markdown('<div class="section-kicker">Project overview</div>', unsafe_allow_html=True)
    st.header("A focused image classification study")
    st.write(
        "This interface presents a trained image-only MRI classifier built with transfer learning. "
        "It maps a single 2D image to one of the four labels learned in the saved checkpoint."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### Architecture")
        st.write("Torchvision ResNet18 with an ImageNet-initialized backbone and a four-output classification head.")
    with c2:
        st.markdown("### Prediction labels")
        try:
            _, names, _ = load_model()
            st.markdown("".join(f'<span class="class-chip">{name}</span>' for name in names), unsafe_allow_html=True)
        except Exception:
            st.write("Mild Impairment · Moderate Impairment · No Impairment · Very Mild Impairment")
            st.caption("Labels shown from the saved experiment metrics; checkpoint could not be loaded.")
    with c3:
        st.markdown("### Runtime device")
        st.write("CUDA GPU" if torch.cuda.is_available() else "CPU")
        st.caption("Inference uses CUDA when available and CPU otherwise.")
    if metrics:
        test = metrics["test"]
        left, right = st.columns([1, 1])
        with left:
            metric_card("Held-out test accuracy", f'{test["accuracy"]:.2%}', "Recorded in metrics.json")
        with right:
            metric_card("Held-out test macro-F1", f'{test["macro_f1"]:.3f}', "Recorded in metrics.json")
    elif metric_error:
        st.info(metric_error)
    st.markdown(
        '<div class="warning"><strong>Research prototype.</strong> This output is a model prediction '
        "for research and demonstration only. It is not a medical diagnosis or a substitute for review "
        "by a qualified clinician.</div>",
        unsafe_allow_html=True,
    )


def clear_image_payload() -> None:
    """Clear the in-memory upload and result without changing widget state."""
    for key in ("mri_image_bytes", "mri_image_name", "mri_image_sha256", "mri_prediction"):
        st.session_state.pop(key, None)


def clear_uploaded_image() -> None:
    """Clear upload data and reset the uploader after an explicit button click."""
    clear_image_payload()
    # Streamlit otherwise keeps the uploader's own value after the button rerun.
    st.session_state.pop("mri_image_uploader", None)


def render_prediction() -> None:
    st.markdown('<div class="section-kicker">Single image inference</div>', unsafe_allow_html=True)
    st.header("Predict an MRI image")
    st.write("Upload one supported 2D image (20 MB maximum). It is processed in memory and is not saved to disk.")
    if st.session_state.get("mri_image_bytes") is not None:
        st.button("Remove image / Start over", on_click=clear_uploaded_image)

    uploader_key = "mri_image_uploader"
    uploader_state_existed = uploader_key in st.session_state
    uploaded = st.file_uploader(
        "MRI image file",
        type=sorted(SUPPORTED_EXTENSIONS),
        help="Accepted formats: JPG, JPEG, PNG, BMP, TIF, TIFF. Maximum file size: 20 MB.",
        key=uploader_key,
    )

    if uploaded is None and uploader_state_existed and st.session_state.get("mri_image_bytes") is not None:
        # A cleared, still-mounted uploader means the user clicked its built-in X.
        # If navigation removed the widget, its key is absent before it is rendered;
        # in that case the saved in-memory image remains available.
        clear_image_payload()
    elif uploaded is not None:
        if Path(uploaded.name).suffix.lower().lstrip(".") not in SUPPORTED_EXTENSIONS:
            st.error("Unsupported file type. Choose a JPG, JPEG, PNG, BMP, or TIFF image.")
            return
        if uploaded.size > MAX_UPLOAD_BYTES:
            st.error("This file is larger than 20 MB. Please upload a smaller image.")
            return
        candidate_bytes = uploaded.getvalue()
        candidate_hash = hashlib.sha256(candidate_bytes).hexdigest()
        if candidate_hash != st.session_state.get("mri_image_sha256"):
            st.session_state["mri_image_bytes"] = candidate_bytes
            st.session_state["mri_image_name"] = uploaded.name
            st.session_state["mri_image_sha256"] = candidate_hash
            st.session_state.pop("mri_prediction", None)

    image_bytes = st.session_state.get("mri_image_bytes")
    if image_bytes is None:
        st.markdown('<div class="notice">Your image preview and prediction will appear here after upload.</div>', unsafe_allow_html=True)
        return
    filename = st.session_state.get("mri_image_name", "Uploaded MRI image")
    image_hash = st.session_state.get("mri_image_sha256") or hashlib.sha256(image_bytes).hexdigest()
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            source.verify()
        with Image.open(io.BytesIO(image_bytes)) as source:
            preview = source.convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as exc:
        st.error(f"The uploaded file could not be read as an image. Try another file. Details: {exc}")
        return

    preview_col, details_col = st.columns([1, 1.35], gap="large")
    with preview_col:
        st.image(preview, caption=f"Preview · {filename}", use_container_width=True)
        st.caption(f"{preview.width} × {preview.height} pixels · RGB conversion applied")
    with details_col:
        st.markdown("### Ready for inference")
        st.write("The image will be resized to 224 × 224 and normalized with ImageNet channel statistics.")
        if st.button("Run model prediction", type="primary", use_container_width=True, key="run_mri_prediction"):
            try:
                model, class_names, device = load_model()
                preprocess = transforms.Compose([
                    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ])
                tensor = preprocess(preview).unsqueeze(0).to(device)
                with torch.inference_mode():
                    probabilities = torch.softmax(model(tensor), dim=1)[0].detach().cpu().tolist()
                ranked = sorted(zip(class_names, probabilities), key=lambda pair: pair[1], reverse=True)
                label, score = ranked[0]
                st.session_state["mri_prediction"] = {
                    "image_sha256": image_hash,
                    "filename": filename,
                    "label": label,
                    "score": score,
                    "ranked": ranked,
                    "device": str(device),
                }
            except FileNotFoundError as exc:
                st.error(f"{exc}. Place the trained best_model.pth at checkpoints/mri_resnet18/best_model.pth.")
            except Exception as exc:
                st.error(f"The saved model could not be loaded or used for prediction: {exc}")

    result = st.session_state.get("mri_prediction")
    if result and result["image_sha256"] == image_hash:
        st.divider()
        st.markdown('<div class="section-kicker">Model output</div>', unsafe_allow_html=True)
        left, right = st.columns([.85, 1.4], gap="large")
        with left:
            st.markdown("### Predicted class")
            st.subheader(result["label"])
            st.metric("Model score", f'{result["score"]:.1%}')
            st.caption(f'Inference ran on {result["device"]}. A score is not clinical certainty.')
        with right:
            frame = pd.DataFrame(result["ranked"], columns=["Class", "Model score"])
            frame["Model score"] *= 100
            st.markdown("### Scores across all classes")
            st.bar_chart(frame, x="Class", y="Model score", horizontal=True)


def _make_thumbnail(image: Image.Image) -> bytes:
    thumbnail = image.copy()
    thumbnail.thumbnail((128, 128))
    output = io.BytesIO()
    thumbnail.save(output, format="JPEG", quality=55, optimize=True)
    return output.getvalue()


def _decode_batch_image(image_bytes: bytes) -> tuple[torch.Tensor, bytes, int, int]:
    preprocess = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    with Image.open(io.BytesIO(image_bytes)) as source:
        if source.width * source.height > 50_000_000:
            raise ValueError("Image dimensions exceed the 50 megapixel processing limit")
        source.verify()
    with Image.open(io.BytesIO(image_bytes)) as source:
        rgb = source.convert("RGB")
        width, height = rgb.size
        thumbnail = _make_thumbnail(rgb)
        tensor = preprocess(rgb)
    return tensor, thumbnail, width, height


def _flush_batch(
    model: torch.nn.Module,
    device: torch.device,
    pending: list[dict],
    rows: list[dict],
    thumbnails: dict[int, bytes],
    class_names: list[str],
) -> None:
    if not pending:
        return
    try:
        batch_tensor = torch.stack([item["tensor"] for item in pending]).to(device)
        with torch.inference_mode():
            probabilities = torch.softmax(model(batch_tensor), dim=1).detach().cpu().tolist()
    except torch.cuda.OutOfMemoryError:
        if device.type != "cuda" or len(pending) <= 1:
            raise
        torch.cuda.empty_cache()
        midpoint = len(pending) // 2
        _flush_batch(model, device, pending[:midpoint], rows, thumbnails, class_names)
        _flush_batch(model, device, pending[midpoint:], rows, thumbnails, class_names)
        pending.clear()
        return
    for item, scores in zip(pending, probabilities):
        predicted_index = max(range(len(scores)), key=scores.__getitem__)
        row = {
            "Index": item["index"],
            "Image": item["name"],
            "Status": "Analyzed",
            "Predicted class": class_names[predicted_index],
            "Top model score (%)": round(scores[predicted_index] * 100, 3),
            "Width": item["width"],
            "Height": item["height"],
            "Issue": "",
        }
        row.update({f"Score · {name} (%)": round(score * 100, 3) for name, score in zip(class_names, scores)})
        rows.append(row)
        thumbnails[item["index"]] = item["thumbnail"]
    pending.clear()


def run_batch_inference(
    files,
    total: int,
    model: torch.nn.Module,
    class_names: list[str],
    device: torch.device,
) -> tuple[list[dict], dict[int, bytes]]:
    """Process image byte records in small batches; retain only rows and thumbnails."""
    rows: list[dict] = []
    thumbnails: dict[int, bytes] = {}
    pending: list[dict] = []
    progress = st.progress(0, text="Preparing batch…")
    status = st.empty()
    completed = 0

    for index, (name, image_bytes, read_error) in enumerate(files, start=1):
        completed = index
        if read_error:
            rows.append({
                "Index": index, "Image": name, "Status": "Failed", "Predicted class": "",
                "Top model score (%)": None, "Width": None, "Height": None,
                "Issue": read_error,
            })
        elif len(image_bytes) > MAX_UPLOAD_BYTES:
            rows.append({
                "Index": index, "Image": name, "Status": "Failed", "Predicted class": "",
                "Top model score (%)": None, "Width": None, "Height": None,
                "Issue": "Image exceeds the 20 MB per-image limit.",
            })
        else:
            try:
                tensor, thumbnail, width, height = _decode_batch_image(image_bytes)
                pending.append({
                    "index": index, "name": name, "tensor": tensor,
                    "thumbnail": thumbnail, "width": width, "height": height,
                })
            except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError, RuntimeError) as exc:
                rows.append({
                    "Index": index, "Image": name, "Status": "Failed", "Predicted class": "",
                    "Top model score (%)": None, "Width": None, "Height": None,
                    "Issue": f"Image could not be read: {type(exc).__name__}.",
                })
        if len(pending) >= INFERENCE_BATCH_SIZE:
            _flush_batch(model, device, pending, rows, thumbnails, class_names)
        progress.progress(min(completed / total, 1.0), text=f"Checked {completed:,} of {total:,} files…")
        if completed % 25 == 0 or completed == total:
            failures = sum(row["Status"] == "Failed" for row in rows)
            status.caption(f"Processed {completed:,}/{total:,} · {failures:,} failed so far")

    _flush_batch(model, device, pending, rows, thumbnails, class_names)
    rows.sort(key=lambda row: row["Index"])
    progress.empty()
    status.empty()
    return rows, thumbnails


def clear_batch_report() -> None:
    for key in ("mri_batch_report", "mri_batch_thumbnails", "mri_batch_name", "mri_batch_device"):
        st.session_state.pop(key, None)
    st.session_state.pop("mri_batch_images_upload", None)
    st.session_state.pop("mri_batch_zip_upload", None)
    st.session_state.pop("mri_batch_selected_image", None)
    st.session_state.pop("mri_batch_report_page", None)


def render_batch_analysis() -> None:
    st.markdown('<div class="section-kicker">Batch image research analysis</div>', unsafe_allow_html=True)
    st.header("Analyze multiple MRI images")
    st.write(
        "Select up to 20,000 supported 2D images at once, or upload one ZIP archive. "
        "Files are read in memory; the app does not extract or save the uploaded images."
    )
    st.markdown(
        '<div class="warning"><strong>Research use only.</strong> Results are image-level model estimates, '
        'not a diagnosis, clinical assessment, or patient-level Alzheimer report.</div>',
        unsafe_allow_html=True,
    )
    st.caption("Supported: JPG, JPEG, PNG, BMP, TIF, TIFF · 20 MB per image · up to 20,000 images · 500 MB total batch limit.")

    source_mode = st.radio("Upload method", ["Select multiple images", "Upload a ZIP archive"], horizontal=True, key="mri_batch_source_mode")
    uploaded_files = None
    uploaded_zip = None
    image_infos = []
    zip_size = 0
    if source_mode == "Select multiple images":
        uploaded_files = st.file_uploader(
            "Choose MRI image files", type=sorted(SUPPORTED_EXTENSIONS), accept_multiple_files=True,
            help="Select multiple supported 2D image files. Each file is limited to 20 MB.",
            key="mri_batch_images_upload",
        )
        candidates = uploaded_files or []
    else:
        uploaded_zip = st.file_uploader(
            "Choose one ZIP archive", type=["zip"],
            help="The ZIP can contain images in nested folders; files are not extracted to disk.",
            key="mri_batch_zip_upload",
        )
        candidates = []

    if uploaded_zip is not None:
        try:
            with zipfile.ZipFile(uploaded_zip) as archive:
                image_infos = [
                    info for info in archive.infolist()
                    if not info.is_dir()
                    and Path(info.filename).suffix.lower().lstrip(".") in SUPPORTED_EXTENSIONS
                    and "__MACOSX" not in Path(info.filename).parts
                ]
                zip_images = len(image_infos)
                zip_size = sum(info.file_size for info in image_infos)
            if zip_images > MAX_BATCH_IMAGES:
                st.error(f"This archive contains {zip_images:,} supported images; the limit is {MAX_BATCH_IMAGES:,}.")
            elif zip_size > MAX_BATCH_TOTAL_BYTES:
                st.error("The archive's images exceed the 500 MB uncompressed batch limit.")
            elif zip_images == 0:
                st.error("No supported image files were found inside this ZIP archive.")
            else:
                st.caption(f"Found {zip_images:,} supported images · {zip_size / (1024 ** 2):.1f} MB uncompressed · other archive files will be ignored.")
        except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
            image_infos, st.session_state["mri_batch_zip_error"] = [], f"This ZIP could not be opened: {type(exc).__name__}."
            st.error(st.session_state["mri_batch_zip_error"])

    if uploaded_files:
        direct_total = sum(file.size for file in uploaded_files)
        if len(uploaded_files) > MAX_BATCH_IMAGES:
            st.error(f"You selected {len(uploaded_files):,} images; the limit is {MAX_BATCH_IMAGES:,}.")
        elif direct_total > MAX_BATCH_TOTAL_BYTES:
            st.error("The selected files exceed the 500 MB total batch limit.")
        else:
            st.caption(f"Selected {len(uploaded_files):,} images · {direct_total / (1024 ** 2):.1f} MB total.")

    left, right = st.columns([1, 1])
    with left:
        start_batch = st.button("Analyze selected images", type="primary", use_container_width=True, key="start_mri_batch")
    with right:
        if st.session_state.get("mri_batch_report"):
            st.button("Clear batch results", use_container_width=True, on_click=clear_batch_report, key="clear_mri_batch")

    if start_batch:
        st.session_state.pop("mri_batch_selected_image", None)
        st.session_state.pop("mri_batch_report_page", None)
        if source_mode == "Select multiple images":
            if not uploaded_files:
                st.warning("Select one or more images first.")
            elif len(uploaded_files) > MAX_BATCH_IMAGES or sum(file.size for file in uploaded_files) > MAX_BATCH_TOTAL_BYTES:
                st.error("The selection exceeds the image-count or total-size limit.")
            else:
                file_items = (
                    (file.name, b"", "Image exceeds the 20 MB per-image limit.")
                    if file.size > MAX_UPLOAD_BYTES else (file.name, file.getvalue(), None)
                    for file in uploaded_files
                )
                _run_and_store_batch(file_items, len(uploaded_files), f"{len(uploaded_files)} selected images")
        elif uploaded_zip is None:
            st.warning("Choose a ZIP archive first.")
        elif not image_infos or len(image_infos) > MAX_BATCH_IMAGES or zip_size > MAX_BATCH_TOTAL_BYTES:
            st.error("The ZIP archive does not contain a valid batch within the stated limits.")
        else:
            try:
                with zipfile.ZipFile(uploaded_zip) as archive:
                    def zip_items():
                        for info in image_infos:
                            if info.file_size > MAX_UPLOAD_BYTES:
                                yield info.filename, b"", "Image exceeds the 20 MB per-image limit."
                                continue
                            try:
                                yield info.filename, archive.read(info), None
                            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                                yield info.filename, b"", f"Could not read ZIP member: {type(exc).__name__}."
                    _run_and_store_batch(zip_items(), len(image_infos), uploaded_zip.name)
            except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
                st.error(f"Could not process this ZIP archive: {type(exc).__name__}.")

    _render_batch_report()


def _run_and_store_batch(files, total: int, batch_name: str) -> None:
    try:
        model, class_names, device = load_model()
        rows, thumbnails = run_batch_inference(files, total, model, class_names, device)
        st.session_state["mri_batch_report"] = rows
        st.session_state["mri_batch_thumbnails"] = thumbnails
        st.session_state["mri_batch_name"] = batch_name
        st.session_state["mri_batch_device"] = str(device)
        st.success(f"Batch complete: {sum(row['Status'] == 'Analyzed' for row in rows):,} analyzed, {sum(row['Status'] == 'Failed' for row in rows):,} failed.")
    except FileNotFoundError as exc:
        st.error(f"{exc}. Add the trained checkpoint at checkpoints/mri_resnet18/best_model.pth.")
    except Exception as exc:
        st.error(f"Batch analysis could not finish: {type(exc).__name__}: {exc}")


def _render_batch_report() -> None:
    rows = st.session_state.get("mri_batch_report")
    if not rows:
        st.info("After analysis, the complete per-image and comparative report will appear here and remain available when you return to this page.")
        return

    frame = pd.DataFrame(rows).sort_values("Index")
    analyzed = frame[frame["Status"] == "Analyzed"].copy()
    failed = frame[frame["Status"] == "Failed"]
    st.divider()
    st.markdown('<div class="section-kicker">Batch report</div>', unsafe_allow_html=True)
    st.subheader(st.session_state.get("mri_batch_name", "Image batch"))
    st.caption(f"Inference device: {st.session_state.get('mri_batch_device', 'unknown')} · Individual images and results remain in this browser session until cleared or the session ends.")
    summaries = st.columns(4)
    summaries[0].metric("Images checked", f"{len(frame):,}")
    summaries[1].metric("Analyzed", f"{len(analyzed):,}")
    summaries[2].metric("Could not read", f"{len(failed):,}")
    summaries[3].metric("Predicted classes", f"{analyzed['Predicted class'].nunique() if not analyzed.empty else 0}")

    if not analyzed.empty:
        st.markdown("### Batch comparison")
        counts = analyzed["Predicted class"].value_counts().rename_axis("Predicted class").rename("Image count").reindex(
            [column.removeprefix("Score · ").removesuffix(" (%)") for column in analyzed.columns if column.startswith("Score · ")], fill_value=0
        )
        distribution = counts.rename_axis("Class").reset_index()
        distribution["Share (%)"] = (distribution["Image count"] / len(analyzed) * 100).round(2)
        st.markdown("#### Predicted class distribution")
        st.bar_chart(distribution, x="Class", y="Image count")
        st.dataframe(distribution, use_container_width=True, hide_index=True)
        score_columns = [column for column in analyzed.columns if column.startswith("Score · ")]
        comparison = analyzed.groupby("Predicted class")[score_columns].mean().rename(columns=lambda name: name.removeprefix("Score · ").removesuffix(" (%)"))
        st.markdown("#### Mean class scores by predicted group")
        st.caption("Rows group images by the model's predicted class; values are mean softmax scores (%), not calibrated probabilities.")
        st.dataframe(comparison.style.format("{:.1f}"), use_container_width=True)

        st.markdown("#### Per-image score comparison")
        st.caption("Browse the full report in pages. Each row includes the predicted class and score for every class.")
        page_size = 100
        page_count = max(1, (len(analyzed) + page_size - 1) // page_size)
        page_number = st.number_input(
            f"Report page · {page_count} pages, {page_size} images per page",
            min_value=1, max_value=page_count, value=1, step=1, key="mri_batch_report_page",
        )
        start = (int(page_number) - 1) * page_size
        page_frame = analyzed.iloc[start:start + page_size].drop(columns=["Index", "Issue"]).set_index("Image")
        st.dataframe(page_frame, use_container_width=True, hide_index=False, height=420)

        st.markdown("### Individual image report")
        image_indices = analyzed["Index"].astype(int).tolist()
        selected_index = st.selectbox(
            "Choose an image", image_indices,
            format_func=lambda index: f"{index}. {frame.loc[frame['Index'] == index, 'Image'].iloc[0]}",
            key="mri_batch_selected_image",
        )
        selected = frame.loc[frame["Index"] == selected_index].iloc[0]
        image_col, result_col = st.columns([1, 1.3], gap="large")
        with image_col:
            thumbnail = st.session_state.get("mri_batch_thumbnails", {}).get(int(selected_index))
            if thumbnail:
                st.image(thumbnail, caption=str(selected["Image"]), use_container_width=True)
            st.caption(f"Image dimensions: {int(selected['Width'])} × {int(selected['Height'])} pixels")
        with result_col:
            st.markdown(f"**Predicted class:** {selected['Predicted class']}")
            st.metric("Top model score", f"{selected['Top model score (%)']:.1f}%")
            individual_scores = pd.DataFrame({
                "Class": [column.removeprefix("Score · ").removesuffix(" (%)") for column in score_columns],
                "Model score (%)": [float(selected[column]) for column in score_columns],
            }).sort_values("Model score (%)", ascending=False)
            st.bar_chart(individual_scores, x="Class", y="Model score (%)", horizontal=True)

    if not failed.empty:
        st.markdown("### Files that could not be analyzed")
        st.dataframe(failed[["Image", "Issue"]], use_container_width=True, hide_index=True)

    csv_data = frame.drop(columns=["Index"]).to_csv(index=False).encode("utf-8-sig")
    json_frame = frame.drop(columns=["Index"]).astype(object).where(pd.notna(frame.drop(columns=["Index"])), None)
    json_records = json_frame.to_dict(orient="records")
    for record in json_records:
        for key, value in record.items():
            if pd.isna(value):
                record[key] = None
            elif hasattr(value, "item"):
                record[key] = value.item()
    json_data = json.dumps({
        "report_type": "research_image_classification_batch",
        "batch": st.session_state.get("mri_batch_name", "Image batch"),
        "device": st.session_state.get("mri_batch_device", "unknown"),
        "images_checked": len(frame),
        "images_analyzed": len(analyzed),
        "images_failed": len(failed),
        "clinical_interpretation": None,
        "results": json_records,
    }, indent=2, ensure_ascii=False).encode("utf-8")
    csv_col, json_col = st.columns(2)
    with csv_col:
        st.download_button("Download per-image CSV", csv_data, "mri_batch_results.csv", "text/csv", use_container_width=True)
    with json_col:
        st.download_button("Download detailed JSON report", json_data, "mri_batch_report.json", "application/json", use_container_width=True)
    st.markdown(
        '<div class="warning"><strong>Interpretation:</strong> This report summarizes model outputs for individual 2D images. '
        'It does not diagnose Alzheimer’s disease, combine slices into a patient-level assessment, or establish clinical certainty.</div>',
        unsafe_allow_html=True,
    )


def render_performance(metrics: dict | None, metric_error: str | None) -> None:
    st.markdown('<div class="section-kicker">Saved experiment results</div>', unsafe_allow_html=True)
    st.header("Model performance")
    st.caption("All values below are loaded from checkpoints/mri_resnet18/metrics.json.")
    if metric_error:
        st.error(metric_error)
        st.info("No demo or replacement performance values are shown.")
        return
    if not metrics:
        st.error("Metrics are unavailable; no results can be displayed.")
        return
    test = metrics["test"]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Test accuracy", f'{test["accuracy"]:.2%}')
    with c2:
        st.metric("Test macro-F1", f'{test["macro_f1"]:.4f}')
    with c3:
        st.metric("Test loss", f'{test["loss"]:.4f}')
    sizes = metrics.get("split_sizes")
    if isinstance(sizes, dict):
        st.markdown("#### Recorded split sizes")
        cols = st.columns(3)
        for col, split in zip(cols, ("train", "validation", "test")):
            with col:
                if split in sizes:
                    st.metric(f"{split.title()} images", f'{int(sizes[split]):,}')
    if metrics.get("best_epoch") is not None:
        st.caption(f'Best validation epoch: {metrics["best_epoch"]} · selected by validation macro-F1.')

    report = test.get("classification_report", {})
    names = metrics.get("class_names", [])
    rows = []
    for name in names:
        item = report.get(name)
        if isinstance(item, dict):
            rows.append({
                "Class": name,
                "Precision": item.get("precision"),
                "Recall": item.get("recall"),
                "F1-score": item.get("f1-score"),
                "Support": int(item["support"]) if item.get("support") is not None else None,
            })
    st.markdown("#### Per-class test metrics")
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("Class").style.format({
            "Precision": "{:.3f}", "Recall": "{:.3f}", "F1-score": "{:.3f}", "Support": "{:,.0f}"
        }), use_container_width=True)
    else:
        st.info("Per-class report is not present in the saved metrics file.")

    matrix = test.get("confusion_matrix")
    if matrix is not None and names and len(matrix) == len(names):
        try:
            values = pd.DataFrame(matrix, index=names, columns=names).astype(int)
            st.markdown("#### Held-out test confusion matrix")
            st.caption("Rows are true classes; columns are predicted classes. Counts are taken from the saved test evaluation.")
            fig, ax = plt.subplots(figsize=(8.4, 5.2))
            fig.patch.set_facecolor("white")
            image = ax.imshow(values.to_numpy(), cmap="GnBu")
            ax.set_xticks(range(len(names)), labels=names, rotation=25, ha="right")
            ax.set_yticks(range(len(names)), labels=names)
            ax.set_xlabel("Predicted class")
            ax.set_ylabel("True class")
            ax.set_title("Test set · image counts", loc="left", fontweight="bold", pad=14)
            threshold = max(values.to_numpy().max(), 1) / 2
            for row in range(len(names)):
                for col in range(len(names)):
                    ax.text(col, row, f"{values.iloc[row, col]:,}", ha="center", va="center",
                            color="white" if values.iloc[row, col] > threshold else "#163440", fontsize=10)
            fig.colorbar(image, ax=ax, fraction=.046, pad=.04, label="Number of images")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except (TypeError, ValueError, IndexError) as exc:
            st.error(f"Saved confusion matrix could not be displayed: {exc}")
    else:
        st.info("A compatible confusion matrix is not present in the saved metrics file.")


def render_method() -> None:
    st.markdown('<div class="section-kicker">How to read this experiment</div>', unsafe_allow_html=True)
    st.header("Method and limitations")
    st.markdown("### Data split and model selection")
    st.write(
        "The training script creates a stratified validation split from the training image folders (20% by default, "
        "with seed 42). The best checkpoint is selected using validation macro-F1, with validation loss as a tie-breaker. "
        "The separate supplied test folder is held out during fitting and model selection, then evaluated once after "
        "restoring the selected checkpoint. The UI displays the split counts and test results recorded by that run."
    )
    st.markdown("### Image preprocessing")
    st.write(
        "The classifier accepts individual 2D image files. Each image is converted to RGB, resized directly to 224 × 224, "
        "converted to a tensor, and normalized per channel with ImageNet mean (0.485, 0.456, 0.406) and standard deviation "
        "(0.229, 0.224, 0.225). Training used random horizontal flips and rotations up to 8 degrees; validation, test, and "
        "this prediction interface use resize and normalization only."
    )
    st.markdown("### Limitations")
    st.write(
        "This is an image-level research prototype. It does not analyze full 3D MRI volumes, NIfTI files, patient history, "
        "or other clinical information. Performance depends on the training data and may not generalize to different scanners, "
        "sites, or populations. The model score is a softmax output, not calibrated certainty. Predictions must not be used "
        "to diagnose, screen, or guide treatment."
    )


metrics, metrics_error = read_metrics()
with st.sidebar:
    st.markdown("## MRI classifier")
    st.caption("Research interface · ResNet18")
    page = st.radio(
        "Navigate",
        ["Overview", "Predict MRI", "Batch Analysis", "Model Performance", "Method and Limitations"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Active device**")
    st.write("CUDA" if torch.cuda.is_available() else "CPU")
    st.caption("The prediction output is a model score, not a diagnosis.")

st.markdown(
    '<div class="hero"><div class="eyebrow">MRI image classification · research prototype</div>'
    '<h1>Alzheimer’s MRI Classifier</h1>'
    '<p>Explore the saved ResNet18 experiment, review held-out results, and analyze one or many 2D images.</p></div>',
    unsafe_allow_html=True,
)

if page == "Overview":
    render_overview(metrics, metrics_error)
elif page == "Predict MRI":
    render_prediction()
elif page == "Batch Analysis":
    render_batch_analysis()
elif page == "Model Performance":
    render_performance(metrics, metrics_error)
else:
    render_method()
