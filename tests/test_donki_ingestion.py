"""Tests for DONKI ingestion helpers."""

from datetime import datetime

import pandas as pd

from flare_plus.ingestion.donki_client import DonkiClient
from flare_plus.ingestion.normalize_donki import normalize_donki_flares
from flare_plus.ingestion.store_flares import upsert_solar_flares
from src.data.schema import SolarFlareEvent


def test_normalize_donki_flares_parses_datetimes():
    raw = [
        {
            "flrID": "2024-01-ABC",
            "beginTime": "2024-01-01T00:00Z",
            "peakTime": "2024-01-01T00:05Z",
            "endTime": "2024-01-01T00:10Z",
            "classType": "M1.0",
            "sourceLocation": "N15E30",
            "activeRegionNum": "1234",
            "instruments": [{"displayName": "GOES"}],
            "linkedEvents": [{"activityID": "TEST"}],
        }
    ]

    df = normalize_donki_flares(raw)
    assert df.loc[0, "external_id"] == "2024-01-ABC"
    assert df.loc[0, "source"] == "nasa_donki"
    assert df.loc[0, "active_region_num"] == 1234
    assert df.loc[0, "begin_time"] == datetime(2024, 1, 1, 0, 0)
    assert df.loc[0, "peak_time"] == datetime(2024, 1, 1, 0, 5)
    assert df.loc[0, "end_time"] == datetime(2024, 1, 1, 0, 10)
    assert df.loc[0, "instruments"].startswith("[")
    assert df.loc[0, "linked_events"].startswith("[")


def test_upsert_solar_flares_is_idempotent(db_session):
    payload = {
        "external_id": "2024-01-ABC",
        "source": "nasa_donki",
        "begin_time": datetime(2024, 1, 1, 0, 0),
        "peak_time": datetime(2024, 1, 1, 0, 5),
        "end_time": datetime(2024, 1, 1, 0, 10),
        "class_type": "M1.0",
        "source_location": "N15E30",
        "active_region_num": 1234,
        "instruments": "[]",
        "linked_events": "[]",
        "raw_payload": "{}",
    }
    df = pd.DataFrame([payload])

    stats_first = upsert_solar_flares(df, session=db_session)
    assert stats_first["records_inserted"] == 1
    assert stats_first["records_updated"] == 0

    record = db_session.query(SolarFlareEvent).filter_by(external_id="2024-01-ABC").first()
    assert record is not None
    assert record.class_type == "M1.0"

    df_second = df.copy()
    df_second.loc[0, "class_type"] = "M1.2"
    stats_second = upsert_solar_flares(df_second, session=db_session)
    assert stats_second["records_inserted"] == 0
    assert stats_second["records_updated"] == 1

    record_updated = db_session.query(SolarFlareEvent).filter_by(external_id="2024-01-ABC").first()
    assert record_updated.class_type == "M1.2"


def test_donki_client_chunking(monkeypatch):
    client = DonkiClient(api_key="demo")
    calls = []

    def fake_fetch(start, end):
        calls.append((start, end))
        return []

    monkeypatch.setattr(client, "fetch_flares", fake_fetch)

    chunks = list(
        client.iter_flares_chunked(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 20),
            days_per_chunk=10,
        )
    )

    assert len(chunks) == 2
    assert calls == [("2024-01-01", "2024-01-11"), ("2024-01-12", "2024-01-20")]
