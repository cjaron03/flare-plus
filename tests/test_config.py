"""tests for configuration management."""

from src.config import CONFIG, DatabaseConfig, DataConfig


def test_config_loaded():
    """test that configuration is loaded."""
    assert CONFIG is not None
    assert 'data_ingestion' in CONFIG
    assert 'database' in CONFIG
    assert 'model' in CONFIG


def test_database_config():
    """test database configuration."""
    conn_string = DatabaseConfig.get_connection_string()
    
    assert conn_string is not None
    assert 'postgresql://' in conn_string
    assert DatabaseConfig.PORT == 5432


def test_data_config():
    """test data ingestion configuration."""
    assert DataConfig.CACHE_HOURS > 0
    assert DataConfig.UPDATE_INTERVAL > 0
    assert DataConfig.ENDPOINTS is not None
    assert 'goes_xrs_7day' in DataConfig.ENDPOINTS
    assert 'solar_regions' in DataConfig.ENDPOINTS


def test_endpoints_are_valid_urls():
    """test that configured endpoints are valid urls."""
    for key, url in DataConfig.ENDPOINTS.items():
        assert url.startswith('http://') or url.startswith('https://'), \
            f"endpoint {key} has invalid url: {url}"

