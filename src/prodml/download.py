"""Model artifact downloader supporting Google Drive sources."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from prodml.config import get_settings

logger = logging.getLogger(__name__)

GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1yeJcg-nrci3BDq01p9-pjtciv1c4g9rc"
GDRIVE_FOLDER_ID = "1yeJcg-nrci3BDq01p9-pjtciv1c4g9rc"

FILE_IDS: dict[str, str] = {
    "config.json": "1T1KBobXpqQ3I6D1P2eWMa14oEHZ0vL3L",
    "model.safetensors": "1-LjK_ineRXRMqs6ZWszISimuQpzYJQ4i",
    "tokenizer.json": "1udNgNrk-HbpLDbk4jUzU3zTGB4UPIM5I",
    "tokenizer_config.json": "1s43JYX99ldlbQH7NVEXWiNvAnQNUX0u6",
}


def is_model_present(model_dir: Path | str) -> bool:
    """Check if all essential model artifact files exist and are non-empty."""
    target = Path(model_dir)
    if not target.exists() or not target.is_dir():
        return False

    required_files = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    return all(
        (target / fname).exists() and (target / fname).stat().st_size > 0
        for fname in required_files
    )


def download_model(
    model_dir: Path | str | None = None,
    folder_url: str = GDRIVE_FOLDER_URL,
    force: bool = False,
) -> Path:
    """Download model weights from Google Drive into the target directory."""
    target_dir = Path(model_dir or get_settings().model.model_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if not force and is_model_present(target_dir):
        logger.info("All model artifacts already present in %s. Skipping download.", target_dir)
        return target_dir

    logger.info(
        "Downloading model artifacts from Google Drive (%s) into %s...",
        folder_url,
        target_dir,
    )

    try:
        import gdown

        logger.info("Attempting folder download via gdown...")
        gdown.download_folder(url=folder_url, output=str(target_dir), quiet=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Folder download encountered issue: %s. Falling back to individual file IDs...",
            exc,
        )
        import gdown

        for filename, file_id in FILE_IDS.items():
            dest = target_dir / filename
            if not force and dest.exists() and dest.stat().st_size > 0:
                logger.info("File %s already exists, skipping.", filename)
                continue
            logger.info("Downloading %s (ID: %s)...", filename, file_id)
            gdown.download(id=file_id, output=str(dest), quiet=False)

    if not is_model_present(target_dir):
        raise RuntimeError(
            f"Model download completed but artifacts verification failed in {target_dir}"
        )

    # Ensure files are readable by non-root containers
    for path in target_dir.rglob("*"):
        try:
            path.chmod(0o644 if path.is_file() else 0o755)
        except OSError:
            pass

    logger.info("Model artifacts successfully downloaded and verified in %s.", target_dir)
    return target_dir


def main() -> None:
    """CLI entrypoint for prodml-download."""
    parser = argparse.ArgumentParser(
        description="Download fine-tuned ProdML model weights from Google Drive."
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Target directory to save model weights (default: outputs/final_model)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-download even if files already exist",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    download_model(model_dir=args.output_dir, force=args.force)


if __name__ == "__main__":
    main()
