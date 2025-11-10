"""Run the Svelte dashboard backend (FastAPI + static assets)."""

import argparse
import logging
import sys
from pathlib import Path

# add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn  # noqa: E402

from src.ui.server import create_app  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """main entry point for ui server."""
    parser = argparse.ArgumentParser(description="run the Svelte dashboard backend")
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://127.0.0.1:5000",
        help="api server url (default: http://127.0.0.1:5000)",
    )
    parser.add_argument(
        "--classification-model",
        type=str,
        help="path to trained classification model (joblib) - fallback if api unavailable",
    )
    parser.add_argument(
        "--survival-model",
        type=str,
        help="path to trained survival model (joblib) - fallback if api unavailable",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="port to bind to (default: 7860)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--static-dir",
        type=str,
        default=str(PROJECT_ROOT / "ui-frontend" / "dist"),
        help="path to compiled svelte build (default: ui-frontend/dist)",
    )

    args = parser.parse_args()

    logger.info("initializing dashboard backend...")
    logger.info(f"api url: {args.api_url}")
    logger.info(f"static dir: {args.static_dir}")
    if args.classification_model:
        logger.info(f"classification model: {args.classification_model}")
    if args.survival_model:
        logger.info(f"survival model: {args.survival_model}")

    app = create_app(
        api_url=args.api_url,
        classification_model_path=args.classification_model,
        survival_model_path=args.survival_model,
        static_dir=Path(args.static_dir),
    )

    bind_host = "0.0.0.0" if args.host == "127.0.0.1" else args.host  # nosec B104
    logger.info(f"launching fastapi server on {bind_host}:{args.port}")

    uvicorn.run(
        app,
        host=bind_host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
