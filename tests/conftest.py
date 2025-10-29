"""pytest configuration and fixtures."""

import os
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.data.schema import Base
from src.data.database import Database


@pytest.fixture(scope="session")
def test_db_engine():
    """create test database engine."""
    # use DB_HOST from env (set to 'postgres' in docker-compose)
    db_host = os.getenv("DB_HOST", "localhost")
    db_name = "flare_prediction_test"
    
    # first connect to default postgres db to create test db if needed
    admin_url = f"postgresql://postgres:postgres@{db_host}:5432/postgres"
    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    
    # create test database if it doesn't exist
    with admin_engine.connect() as conn:
        result = conn.execute(
            text(f"SELECT 1 FROM pg_database WHERE datname='{db_name}'")
        )
        if not result.fetchone():
            conn.execute(text(f"CREATE DATABASE {db_name}"))
    
    admin_engine.dispose()
    
    # now connect to test database
    db_url = os.getenv(
        "TEST_DATABASE_URL",
        f"postgresql://postgres:postgres@{db_host}:5432/{db_name}"
    )
    engine = create_engine(db_url)
    
    # create all tables
    Base.metadata.create_all(engine)
    
    yield engine
    
    # cleanup
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(test_db_engine):
    """create a new database session for a test."""
    connection = test_db_engine.connect()
    transaction = connection.begin()
    
    session_factory = sessionmaker(bind=connection)
    session = session_factory()
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def mock_goes_xrs_data():
    """mock goes xrs flux data."""
    return [
        {
            "time_tag": "2024-01-01T00:00:00Z",
            "energy": "0.05-0.4nm",
            "flux": 1.5e-6,
            "satellite": "GOES-16"
        },
        {
            "time_tag": "2024-01-01T00:00:00Z",
            "energy": "0.1-0.8nm",
            "flux": 3.2e-6,
            "satellite": "GOES-16"
        }
    ]


@pytest.fixture
def mock_solar_region_data():
    """mock solar region data."""
    return [
        {
            "Number": 12345,
            "Location": "N15E30",
            "Carlon": 30,
            "Lat": 15,
            "Area": 120,
            "MagType": "beta-gamma"
        }
    ]

