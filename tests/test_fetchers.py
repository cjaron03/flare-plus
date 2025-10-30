"""tests for data fetchers."""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
import requests

from src.data.fetchers import NOAAFetcher, GOESXRayFetcher, SolarRegionFetcher, MagnetogramFetcher


def test_noaa_fetcher_initialization():
    """test noaa fetcher initialization."""
    fetcher = NOAAFetcher(timeout=15)
    assert fetcher.timeout == 15
    assert fetcher.session is not None


def test_goes_xray_fetcher_initialization():
    """test goes xrs fetcher initialization."""
    fetcher = GOESXRayFetcher()
    assert fetcher.session is not None


@patch('src.data.fetchers.NOAAFetcher.fetch_json')
def test_goes_xray_fetcher_with_mock_data(mock_fetch, mock_goes_xrs_data):
    """test goes xrs fetcher with mocked data."""
    mock_fetch.return_value = mock_goes_xrs_data
    
    fetcher = GOESXRayFetcher()
    result = fetcher.fetch_recent_flux(days=7)
    
    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert 'timestamp' in result.columns
    assert 'flux_short' in result.columns or 'flux_long' in result.columns


def test_solar_region_fetcher_initialization():
    """test solar region fetcher initialization."""
    fetcher = SolarRegionFetcher()
    assert fetcher.session is not None


@patch('src.data.fetchers.NOAAFetcher.fetch_json')
def test_solar_region_fetcher_with_mock_data(mock_fetch, mock_solar_region_data):
    """test solar region fetcher with mocked data."""
    mock_fetch.return_value = mock_solar_region_data
    
    fetcher = SolarRegionFetcher()
    result = fetcher.fetch_current_regions()
    
    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    assert 'timestamp' in result.columns


@patch('requests.Session.get')
def test_noaa_fetcher_handles_request_failure(mock_get):
    """test that fetcher handles request failures gracefully."""
    # mock a connection error
    mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")
    
    fetcher = NOAAFetcher()
    result = fetcher.fetch_json("https://invalid-url-that-does-not-exist.com/data")
    
    assert result is None


def test_magnetogram_fetcher_initialization():
    """test magnetogram fetcher initialization."""
    fetcher = MagnetogramFetcher()
    assert fetcher.session is not None


def test_magnetogram_fetcher_extracts_from_regions(mock_solar_region_data):
    """test magnetogram fetcher extracts data from regions dataframe."""
    import pandas as pd
    from datetime import datetime
    
    # create a regions dataframe similar to what SolarRegionFetcher would produce
    regions_df = pd.DataFrame([
        {
            "region_number": 12345,
            "latitude": 15.0,
            "longitude": 30.0,
            "magnetic_type": "beta-gamma",
            "timestamp": datetime.utcnow()
        }
    ])
    
    fetcher = MagnetogramFetcher()
    result = fetcher.fetch_magnetogram_from_regions(regions_df)
    
    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert 'region_number' in result.columns
    assert 'magnetic_complexity' in result.columns
    assert 'magnetic_field_polarity' in result.columns
    assert 'latitude' in result.columns
    assert 'source' in result.columns
    assert result.iloc[0]['magnetic_complexity'] == "beta-gamma"
    assert result.iloc[0]['source'] == "noaa_swpc"


def test_magnetogram_fetcher_handles_empty_regions():
    """test magnetogram fetcher handles empty regions gracefully."""
    import pandas as pd
    
    fetcher = MagnetogramFetcher()
    result = fetcher.fetch_magnetogram_from_regions(pd.DataFrame())
    
    assert result is None


def test_magnetogram_fetcher_parses_complexity():
    """test magnetic complexity parsing."""
    fetcher = MagnetogramFetcher()
    
    assert fetcher._parse_magnetic_complexity("beta-gamma-delta") == "beta-gamma-delta"
    assert fetcher._parse_magnetic_complexity("Beta-Gamma") == "beta-gamma"
    assert fetcher._parse_magnetic_complexity("alpha") == "alpha"
    assert fetcher._parse_magnetic_complexity("beta") == "beta"
    assert fetcher._parse_magnetic_complexity("") == "unknown"
    assert fetcher._parse_magnetic_complexity(None) == "unknown"


def test_magnetogram_fetcher_parses_polarity():
    """test magnetic polarity parsing."""
    fetcher = MagnetogramFetcher()
    
    assert fetcher._parse_polarity("beta-gamma") == "mixed"
    assert fetcher._parse_polarity("beta-gamma-delta") == "mixed"
    assert fetcher._parse_polarity("alpha") == "unknown"
    assert fetcher._parse_polarity("") == "unknown"
    assert fetcher._parse_polarity(None) == "unknown"

