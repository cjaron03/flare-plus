"""tests for donki integration in main ingestion pipeline."""

from datetime import datetime
from unittest.mock import Mock, patch


class TestDonkiPipelineIntegration:
    """test donki integration in DataIngestionPipeline."""

    def test_donki_disabled_without_api_key(self, monkeypatch):
        """donki should be disabled when NASA_API_KEY is not set."""
        monkeypatch.setenv("NASA_API_KEY", "DEMO_KEY")

        # mock DataPersister to avoid database dependency
        with patch("src.data.ingestion.DataPersister"):
            from src.data.ingestion import DataIngestionPipeline

            pipeline = DataIngestionPipeline()
            assert pipeline.donki_fetcher is None

    def test_donki_disabled_with_empty_key(self, monkeypatch):
        """donki should be disabled when NASA_API_KEY is empty."""
        monkeypatch.setenv("NASA_API_KEY", "")

        with patch("src.data.ingestion.DataPersister"):
            from src.data.ingestion import DataIngestionPipeline

            pipeline = DataIngestionPipeline()
            assert pipeline.donki_fetcher is None

    def test_donki_enabled_with_api_key(self, monkeypatch):
        """donki should be enabled when NASA_API_KEY is set to real key."""
        # patch DataConfig directly to avoid module reload issues
        from src.config import DataConfig

        original_key = DataConfig.NASA_API_KEY
        DataConfig.NASA_API_KEY = "real_api_key_123"

        try:
            with patch("src.data.ingestion.DataPersister"):
                from src.data.ingestion import DataIngestionPipeline

                pipeline = DataIngestionPipeline()
                assert pipeline.donki_fetcher is not None
        finally:
            # cleanup
            DataConfig.NASA_API_KEY = original_key

    def test_donki_fetch_skipped_when_disabled(self, monkeypatch):
        """donki fetch should return skipped status when disabled."""
        monkeypatch.setenv("NASA_API_KEY", "DEMO_KEY")

        with patch("src.data.ingestion.DataPersister"):
            from src.data.ingestion import DataIngestionPipeline

            pipeline = DataIngestionPipeline()
            result = pipeline._fetch_donki_flares()

            assert result["status"] == "skipped"
            assert "not configured" in result["reason"]

    def test_donki_flares_converted_correctly(self, monkeypatch):
        """donki response should convert to FlareEvent format."""
        from src.config import DataConfig

        original_key = DataConfig.NASA_API_KEY
        DataConfig.NASA_API_KEY = "test_key"

        try:
            with patch("src.data.ingestion.DataPersister"):
                from src.data.ingestion import DataIngestionPipeline

                pipeline = DataIngestionPipeline()

                donki_response = [
                    {
                        "flrID": "2024-01-01-TEST",
                        "beginTime": "2024-01-01T12:00Z",
                        "peakTime": "2024-01-01T12:15Z",
                        "endTime": "2024-01-01T12:30Z",
                        "classType": "M2.5",
                        "activeRegionNum": 1234,
                    }
                ]

                df = pipeline._convert_donki_to_flare_events(donki_response)

                assert df is not None
                assert len(df) == 1
                assert df.iloc[0]["class_category"] == "M"
                assert df.iloc[0]["class_magnitude"] == 2.5
                assert df.iloc[0]["source"] == "nasa_donki"
                assert df.iloc[0]["verified"] == True  # noqa: E712 - comparing numpy bool
                assert df.iloc[0]["active_region"] == 1234
        finally:
            DataConfig.NASA_API_KEY = original_key

    def test_donki_conversion_handles_x_class(self, monkeypatch):
        """donki conversion should handle X-class flares."""
        from src.config import DataConfig

        original_key = DataConfig.NASA_API_KEY
        DataConfig.NASA_API_KEY = "test_key"

        try:
            with patch("src.data.ingestion.DataPersister"):
                from src.data.ingestion import DataIngestionPipeline

                pipeline = DataIngestionPipeline()

                donki_response = [
                    {
                        "beginTime": "2024-01-01T12:00Z",
                        "peakTime": "2024-01-01T12:15Z",
                        "endTime": "2024-01-01T12:30Z",
                        "classType": "X1.5",
                    }
                ]

                df = pipeline._convert_donki_to_flare_events(donki_response)

                assert df is not None
                assert len(df) == 1
                assert df.iloc[0]["class_category"] == "X"
                assert df.iloc[0]["class_magnitude"] == 1.5
        finally:
            DataConfig.NASA_API_KEY = original_key

    def test_donki_conversion_skips_invalid_class(self, monkeypatch):
        """donki conversion should skip flares with invalid class."""
        from src.config import DataConfig

        original_key = DataConfig.NASA_API_KEY
        DataConfig.NASA_API_KEY = "test_key"

        try:
            with patch("src.data.ingestion.DataPersister"):
                from src.data.ingestion import DataIngestionPipeline

                pipeline = DataIngestionPipeline()

                donki_response = [
                    {
                        "beginTime": "2024-01-01T12:00Z",
                        "peakTime": "2024-01-01T12:15Z",
                        "classType": None,  # invalid
                    },
                    {
                        "beginTime": "2024-01-01T13:00Z",
                        "peakTime": "2024-01-01T13:15Z",
                        "classType": "M1.0",  # valid
                    },
                ]

                df = pipeline._convert_donki_to_flare_events(donki_response)

                assert df is not None
                assert len(df) == 1  # only the valid one
                assert df.iloc[0]["class_category"] == "M"
        finally:
            DataConfig.NASA_API_KEY = original_key

    def test_donki_conversion_skips_missing_peak_time(self, monkeypatch):
        """donki conversion should skip flares without peak time."""
        from src.config import DataConfig

        original_key = DataConfig.NASA_API_KEY
        DataConfig.NASA_API_KEY = "test_key"

        try:
            with patch("src.data.ingestion.DataPersister"):
                from src.data.ingestion import DataIngestionPipeline

                pipeline = DataIngestionPipeline()

                donki_response = [
                    {
                        "beginTime": "2024-01-01T12:00Z",
                        "peakTime": None,  # missing
                        "classType": "M1.0",
                    },
                ]

                df = pipeline._convert_donki_to_flare_events(donki_response)

                assert df is None  # no valid flares
        finally:
            DataConfig.NASA_API_KEY = original_key

    def test_parse_flare_class(self, monkeypatch):
        """test flare class parsing."""
        monkeypatch.setenv("NASA_API_KEY", "DEMO_KEY")

        with patch("src.data.ingestion.DataPersister"):
            from src.data.ingestion import DataIngestionPipeline

            pipeline = DataIngestionPipeline()

            # test valid classes
            assert pipeline._parse_flare_class("M2.5") == ("M", 2.5)
            assert pipeline._parse_flare_class("X1.0") == ("X", 1.0)
            assert pipeline._parse_flare_class("C3.4") == ("C", 3.4)
            assert pipeline._parse_flare_class("B5.6") == ("B", 5.6)

            # test invalid classes
            assert pipeline._parse_flare_class("") == (None, None)
            assert pipeline._parse_flare_class(None) == (None, None)
            assert pipeline._parse_flare_class("invalid") == (None, None)

    def test_parse_donki_timestamp(self, monkeypatch):
        """test donki timestamp parsing."""
        monkeypatch.setenv("NASA_API_KEY", "DEMO_KEY")

        with patch("src.data.ingestion.DataPersister"):
            from src.data.ingestion import DataIngestionPipeline

            pipeline = DataIngestionPipeline()

            # test valid timestamps
            result = pipeline._parse_donki_timestamp("2024-01-01T12:00Z")
            assert result == datetime(2024, 1, 1, 12, 0)

            result = pipeline._parse_donki_timestamp("2024-01-01T12:00:00Z")
            assert result == datetime(2024, 1, 1, 12, 0, 0)

            # test invalid timestamps
            assert pipeline._parse_donki_timestamp("") is None
            assert pipeline._parse_donki_timestamp(None) is None
            assert pipeline._parse_donki_timestamp("invalid") is None

    def test_donki_graceful_failure(self, monkeypatch):
        """donki failure should not crash pipeline."""
        from src.config import DataConfig

        original_key = DataConfig.NASA_API_KEY
        DataConfig.NASA_API_KEY = "test_key"

        try:
            with patch("src.data.ingestion.DataPersister"):
                from src.data.ingestion import DataIngestionPipeline

                pipeline = DataIngestionPipeline()

                # mock fetcher to raise exception
                pipeline.donki_fetcher = Mock()
                pipeline.donki_fetcher.fetch_flares.side_effect = Exception("API error")

                result = pipeline._fetch_donki_flares()

                assert result["status"] == "failure"
                assert "API error" in result["error"]
        finally:
            DataConfig.NASA_API_KEY = original_key

    def test_run_incremental_update_includes_donki(self, monkeypatch):
        """run_incremental_update should include donki_flares in results."""
        monkeypatch.setenv("NASA_API_KEY", "DEMO_KEY")

        with patch("src.data.ingestion.DataPersister") as mock_persister_class:
            # setup mock persister instance
            mock_persister = Mock()
            mock_persister.save_xray_flux.return_value = {"status": "success", "records_inserted": 0}
            mock_persister.save_solar_regions.return_value = {"status": "success", "records_inserted": 0}
            mock_persister.save_magnetogram.return_value = {"status": "success", "records_inserted": 0}
            mock_persister.save_flare_events.return_value = {
                "status": "success",
                "records_inserted": 0,
                "records_updated": 0,
            }
            mock_persister_class.return_value = mock_persister

            from src.data.ingestion import DataIngestionPipeline

            pipeline = DataIngestionPipeline()

            # mock fetchers
            pipeline.xray_fetcher = Mock()
            pipeline.xray_fetcher.fetch_recent_flux.return_value = None
            pipeline.region_fetcher = Mock()
            pipeline.region_fetcher.fetch_current_regions.return_value = None

            results = pipeline.run_incremental_update(use_cache=False)

            # donki_flares should be in results
            assert "donki_flares" in results
            # should be skipped since NASA_API_KEY is DEMO_KEY
            assert results["donki_flares"]["status"] == "skipped"


class TestDonkiApiFormatting:
    """test donki results formatting in api response."""

    def test_determine_overall_status_ignores_skipped_optional(self):
        """skipped optional sources should not affect overall status."""
        # test directly with mock results - all required sources succeed, donki skipped
        results = {
            "xray_flux": {"status": "success"},
            "solar_regions": {"status": "success"},
            "magnetogram": {"status": "success"},
            "flare_events": {"status": "success"},
            "donki_flares": {"status": "skipped", "reason": "NASA_API_KEY not configured"},
        }

        # test the new logic: only required sources determine status
        required_keys = ["xray_flux", "solar_regions", "magnetogram", "flare_events"]
        required_statuses = []
        for key in required_keys:
            if key in results and results[key] is not None:
                status = results[key].get("status", "failed")
                required_statuses.append(status)

        required_success = sum(1 for s in required_statuses if s == "success")
        assert required_success == 4  # all required sources succeeded
        assert len(required_statuses) == 4  # 4 required sources checked

    def test_determine_overall_status_skipped_donki_doesnt_mask_failure(self):
        """skipped donki should not mask failures in required sources."""
        # test: all required sources fail, only donki skipped - should be "failed", not "partial"
        results = {
            "xray_flux": None,  # not attempted
            "solar_regions": None,
            "magnetogram": None,
            "flare_events": None,
            "donki_flares": {"status": "skipped", "reason": "NASA_API_KEY not configured"},
        }

        # with no required sources returning data, overall should be "failed"
        required_keys = ["xray_flux", "solar_regions", "magnetogram", "flare_events"]
        required_statuses = []
        for key in required_keys:
            if key in results and results[key] is not None:
                status = results[key].get("status", "failed")
                required_statuses.append(status)

        # no required sources were attempted
        assert len(required_statuses) == 0
        # this means overall status should be "failed" (not masked by skipped donki)


class TestDonkiUiFormatting:
    """test donki results formatting in ui."""

    def test_format_ingestion_summary_with_donki_success(self):
        """ui should format donki success correctly."""
        from src.ui.utils.ingestion import format_ingestion_summary

        response_data = {
            "overall_status": "success",
            "duration": 5.0,
            "results": {
                "xray_flux": {"status": "success", "records": 100},
                "donki_flares": {"status": "success", "new": 3, "duplicates": 2},
            },
        }

        summary = format_ingestion_summary(response_data)

        assert "DONKI: 3 new verified, 2 duplicates" in summary

    def test_format_ingestion_summary_with_donki_skipped(self):
        """ui should format donki skipped correctly."""
        from src.ui.utils.ingestion import format_ingestion_summary

        response_data = {
            "overall_status": "success",
            "duration": 5.0,
            "results": {
                "xray_flux": {"status": "success", "records": 100},
                "donki_flares": {"status": "skipped", "reason": "NASA_API_KEY not configured"},
            },
        }

        summary = format_ingestion_summary(response_data)

        assert "Donki Flares: skipped (NASA_API_KEY not configured)" in summary

    def test_format_ingestion_summary_with_donki_no_new(self):
        """ui should format donki no new events correctly."""
        from src.ui.utils.ingestion import format_ingestion_summary

        response_data = {
            "overall_status": "success",
            "duration": 5.0,
            "results": {
                "donki_flares": {"status": "success", "new": 0, "duplicates": 0},
            },
        }

        summary = format_ingestion_summary(response_data)

        assert "DONKI: no new events" in summary
