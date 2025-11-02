"""database schema definitions for flare+ data storage."""

from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, Index, UniqueConstraint
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
