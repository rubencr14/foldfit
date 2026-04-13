"""Tests for RCSB antibody fetcher."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import json

from finetuning.data.antibody_fetcher import RCSBAntibodyFetcher


class TestRCSBAntibodyFetcher:
    """Tests for the RCSB PDB antibody fetcher."""

    def test_init(self):
        """Fetcher should initialize with cache dir and resolution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = RCSBAntibodyFetcher(
                cache_dir=Path(tmpdir), max_resolution=2.5
            )
            assert fetcher.max_resolution == 2.5
            assert fetcher.cache_dir == Path(tmpdir)

    def test_build_search_query_structure(self):
        """Search query should have the correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = RCSBAntibodyFetcher(
                cache_dir=Path(tmpdir), max_resolution=3.0
            )
            query = fetcher._build_search_query(max_results=100)

            assert query["return_type"] == "entry"
            assert query["query"]["type"] == "group"
            assert query["query"]["logical_operator"] == "and"
            assert query["request_options"]["paginate"]["rows"] == 100

    def test_build_search_query_resolution_value(self):
        """Search query should use the configured resolution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = RCSBAntibodyFetcher(
                cache_dir=Path(tmpdir), max_resolution=2.0
            )
            query = fetcher._build_search_query(max_results=50)

            # Find resolution node
            resolution_node = None
            for node in query["query"]["nodes"]:
                if node.get("type") == "terminal":
                    params = node.get("parameters", {})
                    if "resolution" in params.get("attribute", ""):
                        resolution_node = node
                        break

            assert resolution_node is not None
            assert resolution_node["parameters"]["value"] == 2.0

    @patch("finetuning.data.antibody_fetcher.urllib.request.urlopen")
    def test_search_parses_response(self, mock_urlopen):
        """Search should parse PDB IDs from the API response."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "result_set": [
                {"identifier": "1IGT"},
                {"identifier": "7FAE"},
                {"identifier": "8ABC"},
            ]
        }).encode("utf-8")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = RCSBAntibodyFetcher(cache_dir=Path(tmpdir))
            pdb_ids = fetcher.search(max_results=10)

        assert pdb_ids == ["1IGT", "7FAE", "8ABC"]

    def test_load_pdb_ids_from_file(self):
        """Should load PDB IDs from text file, skipping comments."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("# Comment line\n")
            f.write("1IGT\n")
            f.write("7fae\n")
            f.write("\n")
            f.write("# Another comment\n")
            f.write("8abc\n")
            f.flush()

            ids = RCSBAntibodyFetcher.load_pdb_ids_from_file(f.name)

        assert ids == ["1IGT", "7FAE", "8ABC"]
