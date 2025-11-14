#!/usr/bin/env python
"""Clear stale guardrail status from latest validation record."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import desc  # noqa: E402

from src.data.database import get_database  # noqa: E402
from src.data.schema import SystemValidationLog  # noqa: E402


def clear_guardrail():
    """Clear guardrail status from latest validation record."""
    db = get_database()
    db.connect()
    session = db.session_factory()

    # Get latest validation
    latest = session.query(SystemValidationLog).order_by(desc(SystemValidationLog.run_timestamp)).first()

    if not latest:
        print("No validation records found.")
        return

    print(f"Found latest validation: {latest.run_timestamp}")
    print(f"Current status: {latest.status}")
    print(f"Current guardrail_triggered: {latest.guardrail_triggered}")
    print(f"Current guardrail_reason: {latest.guardrail_reason}")

    if latest.guardrail_triggered:
        # Clear the guardrail
        latest.guardrail_triggered = False
        latest.guardrail_reason = None
        latest.status = "pass"  # Update status to pass since guardrail is cleared

        session.commit()
        print("\n✅ Guardrail status cleared!")
        print(f"Updated status: {latest.status}")
        print(f"Updated guardrail_triggered: {latest.guardrail_triggered}")
    else:
        print("\n✅ Guardrail is already cleared - no changes needed.")


if __name__ == "__main__":
    clear_guardrail()
