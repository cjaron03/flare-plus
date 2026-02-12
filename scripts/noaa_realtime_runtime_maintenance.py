#!/usr/bin/env python
"""runtime file maintenance for realtime NOAA vs flare+ monitoring."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


RUNTIME_DEFAULT = "scripts/runtime"
DEFAULT_PATTERNS = ("*.log", "*.csv", "*.json")


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def truncate_file_tail_in_place(path: Path, keep_bytes: int) -> Tuple[int, int]:
    """truncate file in-place while preserving the newest tail bytes."""
    if keep_bytes < 0:
        raise ValueError("keep_bytes must be >= 0")

    original_size = path.stat().st_size
    if original_size <= keep_bytes:
        return original_size, original_size

    with path.open("rb+") as handle:
        if keep_bytes == 0:
            tail = b""
        else:
            handle.seek(original_size - keep_bytes)
            tail = handle.read(keep_bytes)

        handle.seek(0)
        if tail:
            handle.write(tail)
        handle.truncate(len(tail))
        handle.flush()
        os.fsync(handle.fileno())

    return original_size, len(tail)


def maintain_runtime_files(args: argparse.Namespace) -> Dict[str, object]:
    runtime_dir = resolve_path(args.runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(days=args.retention_days)

    scanned = 0
    deleted = 0
    deleted_paths: List[str] = []
    truncated = 0
    truncated_paths: List[Dict[str, object]] = []

    max_bytes = int(args.max_file_mb * 1024 * 1024)
    keep_bytes = int(args.keep_tail_mb * 1024 * 1024)
    if keep_bytes > max_bytes:
        keep_bytes = max_bytes

    for pattern in args.patterns:
        for path in runtime_dir.glob(pattern):
            if not path.is_file():
                continue
            scanned += 1

            stat = path.stat()
            modified_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            if modified_at < cutoff:
                if not args.dry_run:
                    path.unlink(missing_ok=True)
                deleted += 1
                deleted_paths.append(str(path))
                continue

            if stat.st_size > max_bytes:
                if args.dry_run:
                    truncated += 1
                    truncated_paths.append(
                        {
                            "path": str(path),
                            "before_bytes": stat.st_size,
                            "after_bytes": keep_bytes,
                            "dry_run": True,
                        }
                    )
                else:
                    before, after = truncate_file_tail_in_place(path=path, keep_bytes=keep_bytes)
                    truncated += 1
                    truncated_paths.append(
                        {
                            "path": str(path),
                            "before_bytes": before,
                            "after_bytes": after,
                            "dry_run": False,
                        }
                    )

    return {
        "generated_at_utc": now_utc.isoformat(),
        "runtime_dir": str(runtime_dir),
        "patterns": list(args.patterns),
        "retention_days": args.retention_days,
        "max_file_mb": args.max_file_mb,
        "keep_tail_mb": args.keep_tail_mb,
        "dry_run": bool(args.dry_run),
        "scanned_files": scanned,
        "deleted_files": deleted,
        "deleted_paths": deleted_paths,
        "truncated_files": truncated,
        "truncated_paths": truncated_paths,
    }


def write_json(path_str: Optional[str], payload: Dict[str, object]) -> Optional[Path]:
    if not path_str:
        return None
    output_path = resolve_path(path_str)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str))
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="runtime maintenance for realtime NOAA monitor artifacts")
    parser.add_argument("--runtime-dir", type=str, default=RUNTIME_DEFAULT)
    parser.add_argument("--retention-days", type=int, default=45)
    parser.add_argument("--max-file-mb", type=float, default=20.0)
    parser.add_argument("--keep-tail-mb", type=float, default=5.0)
    parser.add_argument("--patterns", type=str, nargs="+", default=list(DEFAULT_PATTERNS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json-out", type=str, default="scripts/runtime/noaa_runtime_maintenance.json")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.retention_days <= 0:
        raise ValueError("--retention-days must be > 0")
    if args.max_file_mb <= 0:
        raise ValueError("--max-file-mb must be > 0")
    if args.keep_tail_mb < 0:
        raise ValueError("--keep-tail-mb must be >= 0")
    if not args.patterns:
        raise ValueError("--patterns must include at least one glob")


def main() -> None:
    args = parse_args()
    validate_args(args)
    result = maintain_runtime_files(args)
    out_path = write_json(args.json_out, result)

    print("=" * 88)
    print("REALTIME NOAA RUNTIME MAINTENANCE")
    print("=" * 88)
    print(f"Runtime dir: {result['runtime_dir']}")
    print(f"Dry run: {result['dry_run']}")
    print(f"Scanned files: {result['scanned_files']}")
    print(f"Deleted files: {result['deleted_files']}")
    print(f"Truncated files: {result['truncated_files']}")
    if out_path is not None:
        print(f"JSON report: {out_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
