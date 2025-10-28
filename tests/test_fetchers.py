"""tests for data fetchers."""

import pytest
from unittest.mock import Mock, patch
import pandas as pd

from src.data.fetchers import NOAAFetcher, GOESXRayFetcher, SolarRegionFetcher


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


def test_noaa_fetcher_handles_request_failure():
    """test that fetcher handles request failures gracefully."""
    fetcher = NOAAFetcher()
    
    # test with invalid url
    result = fetcher.fetch_json("https://invalid-url-that-does-not-exist.com/data")
    
    assert result is None

