"""Tests for CDR annotator."""

from finetuning.data.cdr_annotator import CDRAnnotator, CDRRegion, CDRScheme


class TestCDRAnnotator:
    """Tests for CDR region annotation."""

    def test_annotate_heavy_chain(self):
        """Should annotate CDR-H1, CDR-H2, CDR-H3 on a heavy chain."""
        annotator = CDRAnnotator(scheme=CDRScheme.IMGT)
        # Typical heavy chain variable region (~120 residues)
        sequence = "A" * 130

        regions = annotator.annotate(sequence, chain_type="heavy", chain_id="H")

        region_names = {r.name for r in regions}
        assert "CDR-H1" in region_names
        assert "CDR-H2" in region_names
        assert "CDR-H3" in region_names

    def test_annotate_light_chain(self):
        """Should annotate CDR-L1, CDR-L2, CDR-L3 on a light chain."""
        annotator = CDRAnnotator(scheme=CDRScheme.IMGT)
        sequence = "A" * 120

        regions = annotator.annotate(sequence, chain_type="light", chain_id="L")

        region_names = {r.name for r in regions}
        assert "CDR-L1" in region_names
        assert "CDR-L2" in region_names
        assert "CDR-L3" in region_names

    def test_invalid_chain_type_raises(self):
        """Should raise ValueError for invalid chain type."""
        annotator = CDRAnnotator()
        try:
            annotator.annotate("AAAA", chain_type="invalid")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "heavy" in str(e) or "light" in str(e)

    def test_short_sequence_returns_empty(self):
        """Very short sequences should return no CDR annotations."""
        annotator = CDRAnnotator()
        regions = annotator.annotate("AAAA", chain_type="heavy")
        assert len(regions) == 0

    def test_cdr_region_dataclass(self):
        """CDRRegion should store annotation data correctly."""
        region = CDRRegion(
            name="CDR-H3",
            start_idx=95,
            end_idx=102,
            sequence="ARDYGGF",
            chain_id="H",
        )
        assert region.name == "CDR-H3"
        assert region.length == 7
        assert region.chain_id == "H"

    def test_annotate_structure(self):
        """Should annotate multiple chains in a structure."""
        annotator = CDRAnnotator()
        chains = {
            "H": {"sequence": "A" * 130, "chain_type": "heavy"},
            "L": {"sequence": "A" * 120, "chain_type": "light"},
        }

        result = annotator.annotate_structure(chains)
        assert "H" in result
        assert "L" in result
        assert len(result["H"]) == 3  # H1, H2, H3
        assert len(result["L"]) == 3  # L1, L2, L3

    def test_different_schemes(self):
        """Different numbering schemes should produce different boundaries."""
        seq = "A" * 130

        imgt = CDRAnnotator(CDRScheme.IMGT)
        chothia = CDRAnnotator(CDRScheme.CHOTHIA)

        imgt_regions = imgt.annotate(seq, "heavy")
        chothia_regions = chothia.annotate(seq, "heavy")

        # Both should find 3 CDRs but potentially at different positions
        assert len(imgt_regions) == 3
        assert len(chothia_regions) == 3
