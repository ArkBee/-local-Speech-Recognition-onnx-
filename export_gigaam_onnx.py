"""Utility script to export a GigaAM ASR model to ONNX format.

Run this script from the project root inside the Python environment that
has PyTorch and gigaam installed. The exported ONNX files will be stored
under `models/onnx` and can later be loaded without requiring PyTorch.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import gigaam


DEFAULT_MODEL = "rnnt"
DEFAULT_VERSION = "v2"
ONNX_DIR = Path("models") / "onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export GigaAM model to ONNX")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=[
            "ctc", "rnnt",
            "v1_ctc", "v1_rnnt",
            "v2_ctc", "v2_rnnt",
            "v3_ctc", "v3_rnnt",
            "v3_e2e_ctc", "v3_e2e_rnnt"
        ],
        help="Model variant to export (default: rnnt)",
    )
    parser.add_argument(
        "--output",
        default=str(ONNX_DIR),
        help="Target directory for ONNX files (default: models/onnx)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("export_gigaam_onnx")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading GigaAM model '%s'...", args.model)
    model = gigaam.load_model(args.model)

    logger.info("Exporting ONNX files to %s...", output_dir)
    model.to_onnx(str(output_dir))

    # Export tokenizer if it exists (for e2e models)
    try:
        if hasattr(model, "cfg") and "decoding" in model.cfg:
            # Check both 'tokenizer_file' and 'model_path'
            tokenizer_file = model.cfg.decoding.get("tokenizer_file") or model.cfg.decoding.get("model_path")
            if tokenizer_file:
                tokenizer_path = Path(tokenizer_file)
                if tokenizer_path.exists():
                    import shutil
                    target_tokenizer = output_dir / f"{args.model}_tokenizer.model"
                    logger.info("Copying tokenizer from %s to %s...", tokenizer_path, target_tokenizer)
                    shutil.copy(tokenizer_path, target_tokenizer)
                else:
                    logger.warning("Tokenizer file not found at %s", tokenizer_path)
    except Exception as tok_err:
        logger.warning("Failed to export tokenizer: %s", tok_err)

    logger.info("Export completed successfully.")


if __name__ == "__main__":
    main()
