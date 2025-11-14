"""Command-line entrypoints for flare_plus utilities."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import dotenv

# ensure project root is importable for src.* modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flare_plus.ingestion.run_donki_ingestion import run_ingestion  # noqa: E402


def _parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def ingest_donki_flares(args: argparse.Namespace) -> None:
    """Handle ingest-donki-flares command."""
    start = _parse_date(args.start)
    end = _parse_date(args.end) if args.end else None

    stats = run_ingestion(start, end, api_key=args.api_key, days_per_chunk=args.chunk_days)
    print(f"Ingestion complete: {stats}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="flare+ CLI utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest-donki-flares", help="Ingest solar flares from NASA DONKI")
    ingest.add_argument("--start", required=True, help="Start date YYYY-MM-DD (e.g., 2010-01-01)")
    ingest.add_argument("--end", help="End date YYYY-MM-DD (defaults to today)")
    ingest.add_argument("--api-key", default="DEMO_KEY", help="NASA API key (default DEMO_KEY)")
    ingest.add_argument(
        "--chunk-days",
        type=int,
        default=30,
        help="Number of days per API chunk (max 30 per NASA docs)",
    )
    ingest.set_defaults(func=ingest_donki_flares)

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    dotenv.load_dotenv(PROJECT_ROOT / ".env", override=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
