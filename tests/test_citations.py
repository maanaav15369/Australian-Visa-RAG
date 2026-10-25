from visa_intel.citations import FALLBACK_TEXT, extract_citations, is_insufficient
from visa_intel.contracts import Jurisdiction

RESULTS = {
    "ids": [["c1", "c2", "c3"]],
    "documents": [["a", "b", "c"]],
    "metadatas": [[
        {"title": "Doc One", "status": "CURRENT", "version": 7, "line_start": 10,
         "line_end": 20, "source_url": "https://example.org/one"},
        {"title": "", "status": None},
        {},
    ]],
}


def cite(text):
    return extract_citations(text, RESULTS, Jurisdiction.AU)


def test_single_marker_maps_to_real_chunk():
    [c] = cite("Claim. [S1]")
    assert (c.marker, c.chunk_id, c.title, c.legal_status) == ("S1", "c1", "Doc One", "CURRENT")
    assert c.version == "7"
    assert c.provision == "lines 10-20"
    assert c.source_url == "https://example.org/one"


def test_grouped_and_repeated_markers():
    assert [c.marker for c in cite("A [S1, S2]. B [S1]. C [S3]")] == ["S1", "S2", "S3"]


def test_unknown_marker_is_ignored_not_invented():
    assert cite("Claim [S9]") == []
    assert cite("Claim [S0]") == []


def test_empty_metadata_gives_none_not_blank_strings():
    c = cite("x [S2]")[0]
    assert c.title is None and c.legal_status is None and c.provision is None


def test_no_markers():
    assert cite("Plain text with no markers.") == []


def test_empty_results_do_not_crash():
    assert extract_citations("x [S1]", {"ids": [[]], "metadatas": [[]]}, Jurisdiction.AU) == []
    assert extract_citations("x [S1]", {}, Jurisdiction.AU) == []


def test_insufficient_detection():
    assert is_insufficient(FALLBACK_TEXT)
    assert is_insufficient("  " + FALLBACK_TEXT + " More.")
    assert not is_insufficient("Some answer [S1]")
