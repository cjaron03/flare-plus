"""tests for database functionality."""

from datetime import datetime

from src.data.schema import GOESXRayFlux, SolarRegion, FlareEvent, DataIngestionLog


def test_goes_xray_flux_model(db_session):
    """test goes xray flux model creation and retrieval."""
    flux = GOESXRayFlux(
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        flux_short=1.5e-6,
        flux_long=3.2e-6,
        satellite="GOES-16",
        data_quality="good",
    )

    db_session.add(flux)
    db_session.commit()

    retrieved = db_session.query(GOESXRayFlux).filter_by(timestamp=datetime(2024, 1, 1, 12, 0, 0)).first()

    assert retrieved is not None
    assert retrieved.flux_short == 1.5e-6
    assert retrieved.flux_long == 3.2e-6
    assert retrieved.satellite == "GOES-16"


def test_solar_region_model(db_session):
    """test solar region model creation."""
    region = SolarRegion(
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        region_number=12345,
        latitude=15.0,
        longitude=30.0,
        mcintosh_class="Dkc",
        mount_wilson_class="beta-gamma",
        area=120,
        magnetic_type="beta-gamma",
    )

    db_session.add(region)
    db_session.commit()

    retrieved = db_session.query(SolarRegion).filter_by(region_number=12345).first()

    assert retrieved is not None
    assert retrieved.latitude == 15.0
    assert retrieved.longitude == 30.0


def test_flare_event_model(db_session):
    """test flare event model creation."""
    flare = FlareEvent(
        start_time=datetime(2024, 1, 1, 12, 0, 0),
        peak_time=datetime(2024, 1, 1, 12, 15, 0),
        end_time=datetime(2024, 1, 1, 12, 30, 0),
        flare_class="M5.5",
        class_category="M",
        class_magnitude=5.5,
        active_region=12345,
        source="noaa",
        verified=True,
    )

    db_session.add(flare)
    db_session.commit()

    retrieved = db_session.query(FlareEvent).filter_by(flare_class="M5.5").first()

    assert retrieved is not None
    assert retrieved.class_category == "M"
    assert retrieved.class_magnitude == 5.5


def test_ingestion_log_model(db_session):
    """test data ingestion log model."""
    log = DataIngestionLog(
        source_name="test_source",
        status="success",
        records_fetched=100,
        records_inserted=95,
        records_updated=5,
        duration_seconds=1.5,
    )

    db_session.add(log)
    db_session.commit()

    retrieved = db_session.query(DataIngestionLog).filter_by(source_name="test_source").first()

    assert retrieved is not None
    assert retrieved.status == "success"
    assert retrieved.records_fetched == 100
