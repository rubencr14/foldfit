"""RCSB PDB Search API client for fetching antibody structures."""

import json
import logging
import urllib.request
import urllib.error
from pathlib import Path

logger = logging.getLogger(__name__)

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download"


class RCSBAntibodyFetcher:
    """Queries RCSB PDB for antibody/immunoglobulin structures.

    Uses the RCSB Search API v2 to find structures matching antibody
    criteria, then downloads them in mmCIF format.

    Args:
        cache_dir: Directory to cache downloaded structure files.
        max_resolution: Maximum crystallographic resolution in Angstroms.
    """

    def __init__(self, cache_dir: Path, max_resolution: float = 3.5):
        self.cache_dir = Path(cache_dir)
        self.max_resolution = max_resolution

    def search(self, max_results: int = 1000) -> list[str]:
        """Search RCSB for antibody PDB IDs.

        Args:
            max_results: Maximum number of PDB IDs to return.

        Returns:
            List of PDB ID strings (e.g., ["1IGT", "7FAE", ...]).
        """
        query = self._build_search_query(max_results)
        query_json = json.dumps(query).encode("utf-8")

        request = urllib.request.Request(
            RCSB_SEARCH_URL,
            data=query_json,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            logger.error(f"RCSB search failed with HTTP {e.code}: {e.reason}")
            raise
        except urllib.error.URLError as e:
            logger.error(f"RCSB search connection error: {e.reason}")
            raise

        pdb_ids = [
            hit["identifier"] for hit in result.get("result_set", [])
        ]
        logger.info(
            f"Found {len(pdb_ids)} antibody structures "
            f"(max_resolution={self.max_resolution}A)"
        )
        return pdb_ids

    def download(
        self, pdb_ids: list[str], file_format: str = "cif"
    ) -> list[Path]:
        """Download structure files from RCSB PDB.

        Already-cached files are skipped.

        Args:
            pdb_ids: List of PDB IDs to download.
            file_format: File format to download ('cif' or 'pdb').

        Returns:
            List of paths to downloaded (or cached) files.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        downloaded_paths = []
        extension = f".{file_format}"

        for pdb_id in pdb_ids:
            pdb_id_lower = pdb_id.lower()
            filename = f"{pdb_id_lower}{extension}"
            filepath = self.cache_dir / filename

            if filepath.exists():
                downloaded_paths.append(filepath)
                continue

            url = f"{RCSB_DOWNLOAD_URL}/{pdb_id_lower}{extension}"
            try:
                urllib.request.urlretrieve(url, filepath)
                downloaded_paths.append(filepath)
                logger.debug(f"Downloaded {pdb_id} -> {filepath}")
            except urllib.error.URLError as e:
                logger.warning(f"Failed to download {pdb_id}: {e}")

        logger.info(
            f"Downloaded {len(downloaded_paths)}/{len(pdb_ids)} structures "
            f"to {self.cache_dir}"
        )
        return downloaded_paths

    def _build_search_query(self, max_results: int) -> dict:
        """Build RCSB Search API v2 JSON query for antibody structures.

        The query combines:
        - Keyword filter for "ANTIBODY" or "IMMUNE SYSTEM"
        - Resolution cutoff
        - Protein polymer type
        """
        return {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {
                        "type": "group",
                        "logical_operator": "or",
                        "nodes": [
                            {
                                "type": "terminal",
                                "service": "text",
                                "parameters": {
                                    "attribute": "struct_keywords.pdbx_keywords",
                                    "operator": "contains_words",
                                    "value": "ANTIBODY",
                                },
                            },
                            {
                                "type": "terminal",
                                "service": "text",
                                "parameters": {
                                    "attribute": "struct_keywords.pdbx_keywords",
                                    "operator": "contains_words",
                                    "value": "IMMUNE SYSTEM",
                                },
                            },
                            {
                                "type": "terminal",
                                "service": "text",
                                "parameters": {
                                    "attribute": "struct_keywords.pdbx_keywords",
                                    "operator": "contains_words",
                                    "value": "IMMUNOGLOBULIN",
                                },
                            },
                        ],
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entry_info.resolution_combined",
                            "operator": "less_or_equal",
                            "value": self.max_resolution,
                        },
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "entity_poly.rcsb_entity_polymer_type",
                            "operator": "exact_match",
                            "value": "Protein",
                        },
                    },
                ],
            },
            "return_type": "entry",
            "request_options": {
                "results_content_type": ["experimental"],
                "paginate": {"start": 0, "rows": max_results},
                "sort": [
                    {
                        "sort_by": "rcsb_entry_info.resolution_combined",
                        "direction": "asc",
                    }
                ],
            },
        }

    @staticmethod
    def load_pdb_ids_from_file(path: str | Path) -> list[str]:
        """Load PDB IDs from a text file (one ID per line).

        Lines starting with '#' are treated as comments and ignored.
        Empty lines are also ignored.

        Args:
            path: Path to the text file.

        Returns:
            List of PDB ID strings.
        """
        ids = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    ids.append(line.upper())
        return ids
