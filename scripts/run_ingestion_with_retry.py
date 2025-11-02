#!/usr/bin/env python3
"""
run data ingestion with retry logic for nightly workflows.
"""

import sys
import time
import logging
from typing import Tuple

# setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_ingestion_with_retry(max_retries: int = 3, retry_delay: int = 60) -> Tuple[bool, str]:
    """
    run ingestion with retry logic.

    args:
        max_retries: maximum number of retry attempts
        retry_delay: delay between retries in seconds

    returns:
        tuple of (success, error_message)
    """
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"ingestion attempt {attempt}/{max_retries}")

            # import here to avoid loading modules on script load
            from src.data.ingestion import DataIngestionPipeline

            pipeline = DataIngestionPipeline()
            results = pipeline.run_incremental_update()

            # check if ingestion was successful
            success = all(result.get("status") == "success" for result in results.values())

            if success:
                logger.info("ingestion successful")
                return True, None
            else:
                failed = [name for name, result in results.items() if result.get("status") == "failure"]
                error_msg = f"ingestion failed for: {', '.join(failed)}"
                logger.warning(error_msg)

                if attempt < max_retries:
                    logger.info(f"retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    return False, error_msg

        except Exception as e:
            error_msg = f"ingestion error: {str(e)}"
            logger.error(error_msg, exc_info=True)

            if attempt < max_retries:
                logger.info(f"retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                return False, error_msg

    return False, "max retries exceeded"


if __name__ == "__main__":
    success, error = run_ingestion_with_retry()

    if success:
        logger.info("✓ ingestion completed successfully")
        sys.exit(0)
    else:
        logger.error(f"✗ ingestion failed: {error}")
        sys.exit(1)
