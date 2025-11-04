"""run gradio ui dashboard."""

import argparse
import logging
import sys
from pathlib import Path

# add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.dashboard import create_dashboard  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """main entry point for ui server."""
    parser = argparse.ArgumentParser(description="run gradio ui dashboard")
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
        "--share",
        action="store_true",
        help="create public gradio share link",
    )

    args = parser.parse_args()

    logger.info("initializing dashboard...")
    logger.info(f"api url: {args.api_url}")
    if args.classification_model:
        logger.info(f"classification model: {args.classification_model}")
    if args.survival_model:
        logger.info(f"survival model: {args.survival_model}")

    # create dashboard
    dashboard = create_dashboard(
        api_url=args.api_url,
        classification_model_path=args.classification_model,
        survival_model_path=args.survival_model,
    )

    # launch
    # bind to 0.0.0.0 when running in docker to allow external access
    bind_host = "0.0.0.0" if args.host == "127.0.0.1" else args.host  # nosec B104
    logger.info(f"launching dashboard on {bind_host}:{args.port}")
    dashboard.launch(
        server_name=bind_host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
