"""database schema definitions for flare+ data storage."""

from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, Index, UniqueConstraint, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class GOESXRayFlux(Base):  # type: ignore[misc,valid-type]
    """goes x-ray flux measurements (0.5-4.0Å and 1.0-8.0Å bands)."""

    __tablename__ = "flare_goes_xray_flux"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, unique=True, index=True)

    # xrs channels
    flux_short = Column(Float, nullable=True)  # 0.5-4.0 angstrom (short wavelength)
    flux_long = Column(Float, nullable=True)  # 1.0-8.0 angstrom (long wavelength)

    # metadata
    satellite = Column(String(50))  # e.g., GOES-16, GOES-17
    data_quality = Column(String(20))  # good, poor, missing

    # ingestion tracking
    ingested_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<GOESXRayFlux(timestamp={self.timestamp}, flux_long={self.flux_long})>"


class SolarRegion(Base):  # type: ignore[misc,valid-type]
    """active solar regions (sunspot groups) from noaa swpc."""

    __tablename__ = "flare_solar_regions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    region_number = Column(Integer, nullable=False)

    # location
    latitude = Column(Float)  # heliographic latitude
    longitude = Column(Float)  # heliographic longitude

    # sunspot classification
    mcintosh_class = Column(String(10))  # e.g., Dkc, Ekc
    mount_wilson_class = Column(String(10))  # magnetic class (alpha, beta, beta-gamma, etc.)

    # region characteristics
    area = Column(Integer)  # area in millionths of solar hemisphere
    num_sunspots = Column(Integer)

    # magnetic complexity
    magnetic_type = Column(String(20))

    # ingestion tracking
    ingested_at = Column(DateTime, default=datetime.utcnow)

    # composite unique constraint
    __table_args__ = (
        UniqueConstraint("region_number", "timestamp", name="uq_region_timestamp"),
        Index("ix_region_timestamp", "region_number", "timestamp"),
    )

    def __repr__(self):
        return f"<SolarRegion(region={self.region_number}, timestamp={self.timestamp})>"


class FlareEvent(Base):  # type: ignore[misc,valid-type]
    """observed solar flare events for labeling training data."""

    __tablename__ = "flare_events"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # timing
    start_time = Column(DateTime, nullable=False, index=True)
    peak_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)

    # classification
    flare_class = Column(String(5), nullable=False)  # e.g., C1.2, M5.5, X2.1
    class_category = Column(String(1), nullable=False, index=True)  # B, C, M, X
    class_magnitude = Column(Float, nullable=False)  # numeric part

    # location
    active_region = Column(Integer)  # associated region number
    location = Column(String(20))  # heliographic coordinates

    # source tracking
    source = Column(String(50))  # noaa, nasa, etc.
    verified = Column(Boolean, default=False)

    # ingestion tracking
    ingested_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<FlareEvent(class={self.flare_class}, peak={self.peak_time})>"


class SolarMagnetogram(Base):  # type: ignore[misc,valid-type]
    """solar magnetogram data - magnetic field measurements by region."""

    __tablename__ = "flare_solar_magnetogram"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    region_number = Column(Integer, nullable=False)

    # magnetic field measurements
    magnetic_field_strength = Column(Float)  # gauss (G) or tesla
    magnetic_field_polarity = Column(String(10))  # positive, negative, mixed
    magnetic_complexity = Column(String(20))  # alpha, beta, beta-gamma, beta-gamma-delta

    # location on solar disk
    latitude = Column(Float)  # heliographic latitude
    longitude = Column(Float)  # heliographic longitude
    solar_radius = Column(Float)  # distance from disk center (0-1)

    # source metadata
    source = Column(String(50), default="noaa_swpc")  # noaa_swpc, sdo_hmi, etc.
    data_quality = Column(String(20))  # good, fair, poor

    # ingestion tracking
    ingested_at = Column(DateTime, default=datetime.utcnow)

    # composite unique constraint
    __table_args__ = (
        UniqueConstraint("region_number", "timestamp", name="uq_magnetogram_region_timestamp"),
        Index("ix_magnetogram_region_timestamp", "region_number", "timestamp"),
    )

    def __repr__(self):
        return f"<SolarMagnetogram(region={self.region_number}, timestamp={self.timestamp})>"


class DataIngestionLog(Base):  # type: ignore[misc,valid-type]
    """log of data ingestion runs for monitoring and debugging."""

    __tablename__ = "flare_ingestion_log"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # run info
    run_timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    source_name = Column(String(100), nullable=False)  # endpoint name

    # execution details
    status = Column(String(20), nullable=False)  # success, failure, partial
    records_fetched = Column(Integer, default=0)
    records_inserted = Column(Integer, default=0)
    records_updated = Column(Integer, default=0)

    # time range processed
    data_start_time = Column(DateTime)
    data_end_time = Column(DateTime)

    # error tracking
    error_message = Column(String(500))

    # performance
    duration_seconds = Column(Float)

    def __repr__(self):
        return f"<DataIngestionLog(source={self.source_name}, status={self.status})>"


class PredictionLog(Base):  # type: ignore[misc,valid-type]
    """log of model predictions for monitoring and outcome tracking."""

    __tablename__ = "flare_prediction_log"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # prediction info
    prediction_timestamp = Column(DateTime, nullable=False, index=True)
    observation_timestamp = Column(DateTime, nullable=False)  # when features were computed
    prediction_type = Column(String(20), nullable=False)  # 'classification' or 'survival'
    region_number = Column(Integer, nullable=True)

    # model info
    model_type = Column(String(50))  # e.g., 'gradient_boosting', 'cox', 'gb'
    window_hours = Column(Integer, nullable=True)  # for classification predictions

    # prediction results (stored as json-encodable dict)
    predicted_class = Column(String(10), nullable=True)  # for classification
    class_probabilities = Column(String(1000), nullable=True)  # json string of probabilities
    probability_distribution = Column(String(2000), nullable=True)  # json string for survival
    hazard_score = Column(Float, nullable=True)  # for survival predictions

    # outcome tracking
    actual_flare_class = Column(String(10), nullable=True)
    actual_flare_time = Column(DateTime, nullable=True)
    outcome_recorded_at = Column(DateTime, nullable=True)

    # metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<PredictionLog(id={self.id}, type={self.prediction_type}, timestamp={self.prediction_timestamp})>"


class SystemValidationLog(Base):  # type: ignore[misc,valid-type]
    """log of system validation runs and guardrail status."""

    __tablename__ = "flare_system_validation_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    validation_type = Column(String(50), default="system", nullable=False)
    status = Column(String(20), nullable=False)  # pass / fail
    guardrail_triggered = Column(Boolean, default=False, nullable=False)
    guardrail_reason = Column(String(500))
    details = Column(Text)  # optional json payload with more info
    initiated_by = Column(String(100))  # optional identifier of who triggered the run

    def __repr__(self):
        return (
            f"<SystemValidationLog(validation_type={self.validation_type}, status={self.status}, "
            f"guardrail_triggered={self.guardrail_triggered}, run_timestamp={self.run_timestamp})>"
        )


class SolarFlareEvent(Base):  # type: ignore[misc,valid-type]
    """Solar flare events from external sources (e.g., NASA DONKI).

    Stores raw payload for traceability and has a uniqueness constraint on
    (external_id, source) to support idempotent upserts.
    """

    __tablename__ = "solar_flare_events"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # identity and source
    external_id = Column(String(100), nullable=False, index=True)
    source = Column(String(50), nullable=False, index=True)

    # timing
    begin_time = Column(DateTime, nullable=True, index=True)
    peak_time = Column(DateTime, nullable=True, index=True)
    end_time = Column(DateTime, nullable=True, index=True)

    # classification and metadata
    class_type = Column(String(10), nullable=True)  # e.g., C1.2, M5.5, X2.1
    source_location = Column(String(50), nullable=True)  # e.g., N15E30
    active_region_num = Column(Integer, nullable=True)

    # json-like payloads stored as text for portability
    instruments = Column(Text, nullable=True)  # JSON string list
    linked_events = Column(Text, nullable=True)  # JSON string list
    raw_payload = Column(Text, nullable=True)  # full event payload JSON

    # audit
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("external_id", "source", name="uq_solar_flare_events_ext_source"),
        Index("ix_solar_flare_events_peak_time", "peak_time"),
    )

    def __repr__(self):
        return f"<SolarFlareEvent(external_id={self.external_id}, source={self.source}, peak_time={self.peak_time})>"
