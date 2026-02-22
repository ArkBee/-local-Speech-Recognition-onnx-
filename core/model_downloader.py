import logging
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

REPO_ID = "istupakov/gigaam-v3-onnx"

# Files needed for each model variant
MODEL_FILES = {
    "v3_rnnt": [
        "v3_rnnt_encoder.onnx",
        "v3_rnnt_decoder.onnx",
        "v3_rnnt_joint.onnx",
    ],
    "v3_ctc": [
        "v3_ctc.onnx",
    ],
    "v3_e2e_rnnt": [
        "v3_e2e_rnnt_encoder.onnx",
        "v3_e2e_rnnt_decoder.onnx",
        "v3_e2e_rnnt_joint.onnx",
        "v3_e2e_rnnt_vocab.txt",
    ],
    "v3_e2e_ctc": [
        "v3_e2e_ctc.onnx",
        "v3_e2e_ctc_vocab.txt",
    ],
}

MODEL_LABELS = {
    "v3_rnnt": "v3 RNNT (по умолчанию)",
    "v3_ctc": "v3 CTC (быстрый)",
    "v3_e2e_rnnt": "v3 E2E RNNT",
    "v3_e2e_ctc": "v3 E2E CTC",
}


def check_models_exist(model_dir: Path, model_type: str = "v3_rnnt") -> bool:
    files = MODEL_FILES.get(model_type, MODEL_FILES["v3_rnnt"])
    return all((model_dir / f).exists() for f in files)


def download_models(
    model_dir: Path,
    model_type: str = "v3_rnnt",
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> bool:
    """Download ONNX models from HuggingFace.

    Args:
        model_dir: Directory to save models to.
        model_type: Which model variant to download.
        progress_callback: Optional callback(filename, current_idx, total)
            for GUI progress updates.

    Returns:
        True if all files downloaded successfully.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        return False

    files = MODEL_FILES.get(model_type, MODEL_FILES["v3_rnnt"])
    model_dir.mkdir(parents=True, exist_ok=True)

    for idx, filename in enumerate(files):
        dest = model_dir / filename
        if dest.exists():
            logger.info("Already exists: %s", filename)
            if progress_callback:
                progress_callback(filename, idx + 1, len(files))
            continue

        logger.info("Downloading %s from %s ...", filename, REPO_ID)
        if progress_callback:
            progress_callback(filename, idx, len(files))

        try:
            # Download to HF cache, then copy to model_dir
            cached = hf_hub_download(repo_id=REPO_ID, filename=filename)
            import shutil
            shutil.copy2(cached, str(dest))
            logger.info("Downloaded: %s -> %s", filename, dest)
        except OSError as e:
            if "getaddrinfo" in str(e) or "Errno 11001" in str(e):
                logger.error("No internet connection: %s", e)
            else:
                logger.error("Network error downloading %s: %s", filename, e)
            return False
        except Exception as e:
            error_msg = str(e).lower()
            if any(w in error_msg for w in ("connect", "timeout", "network", "resolve")):
                logger.error(
                    "Нет подключения к интернету. "
                    "Проверьте сеть и перезапустите приложение. (%s)", e
                )
            else:
                logger.error("Failed to download %s: %s", filename, e)
            return False

        if progress_callback:
            progress_callback(filename, idx + 1, len(files))

    return True
