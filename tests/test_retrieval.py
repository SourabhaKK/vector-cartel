def test_matches_metadata_filter_no_filter_matches_everything():
    from src.retrieval import _matches_metadata_filter

    chunk = {"text": "x", "metadata": {"vendor": "Siemens"}, "score": 0.0}

    assert _matches_metadata_filter(chunk, None) is True
    assert _matches_metadata_filter(chunk, {}) is True


def test_matches_metadata_filter_matching_vendor():
    from src.retrieval import _matches_metadata_filter

    chunk = {
        "text": "x",
        "metadata": {"doc": "CISA Advisory ICSA-26-111-02", "vendor": "Siemens"},
        "score": 0.0,
    }

    assert _matches_metadata_filter(chunk, {"vendor": "Siemens"}) is True


def test_matches_metadata_filter_vendor_is_case_insensitive():
    from src.retrieval import _matches_metadata_filter

    chunk = {"text": "x", "metadata": {"vendor": "Siemens"}, "score": 0.0}

    assert _matches_metadata_filter(chunk, {"vendor": "siemens"}) is True
    assert _matches_metadata_filter(chunk, {"vendor": "SIEMENS"}) is True


def test_matches_metadata_filter_non_matching_vendor_excluded():
    from src.retrieval import _matches_metadata_filter

    chunk = {"text": "x", "metadata": {"vendor": "Siemens"}, "score": 0.0}

    assert _matches_metadata_filter(chunk, {"vendor": "ABB"}) is False


def test_matches_metadata_filter_matching_date():
    from src.retrieval import _matches_metadata_filter

    chunk = {"text": "x", "metadata": {"date": "2026-04-21"}, "score": 0.0}

    assert _matches_metadata_filter(chunk, {"date": "2026-04-21"}) is True


def test_matches_metadata_filter_combined_keys_all_must_match():
    from src.retrieval import _matches_metadata_filter

    chunk = {
        "text": "x",
        "metadata": {"vendor": "Siemens", "date": "2026-04-21"},
        "score": 0.0,
    }

    assert (
        _matches_metadata_filter(chunk, {"vendor": "Siemens", "date": "2026-04-21"})
        is True
    )
    assert (
        _matches_metadata_filter(chunk, {"vendor": "Siemens", "date": "2099-01-01"})
        is False
    )


def test_matches_metadata_filter_excludes_chunks_missing_the_key():
    from src.retrieval import _matches_metadata_filter

    # NIST/ATT&CK chunks generally have no "vendor" key at all -- a vendor
    # filter must exclude them, not crash or silently include them.
    nist_chunk = {
        "text": "y",
        "metadata": {"doc": "NIST SP 800-82 Rev. 3", "section": "5.2.3"},
        "score": 0.0,
    }

    assert _matches_metadata_filter(nist_chunk, {"vendor": "Siemens"}) is False


def test_dedupe_candidates_drops_later_exact_text_duplicates():
    """
    Regression test: MITRE ATT&CK mitigation text is frequently copy-pasted
    verbatim across dozens of technique IDs (e.g. "Human User Authentication"
    appears identically under 20 different T-codes). Left undeduped, these
    near-identical vectors flood the fused candidate pool and crowd out
    chunks with unique, on-topic content -- verified live: T0831's unique
    "Manipulation of Control" description chunk lost to 8 duplicate "Human
    User Authentication" chunks in the final top-5 for a query asking
    specifically about control manipulation. Keeping the first (best-ranked)
    occurrence of each exact text and dropping the rest restores diversity
    to the pool without dropping any unique information.
    """
    from src.retrieval import _dedupe_candidates_by_text

    chunks = [
        {"text": "duplicate body", "metadata": {"section": "T0800"}, "score": 0.0},
        {"text": "unique body", "metadata": {"section": "T0831"}, "score": 0.0},
        {"text": "duplicate body", "metadata": {"section": "T0816"}, "score": 0.0},
        {"text": "duplicate body", "metadata": {"section": "T0821"}, "score": 0.0},
    ]

    result = _dedupe_candidates_by_text(chunks)

    assert len(result) == 2
    assert result[0]["metadata"]["section"] == "T0800"
    assert result[1]["metadata"]["section"] == "T0831"


def test_dedupe_candidates_drops_duplicates_with_different_headers():
    """
    Real ATT&CK chunks wrap each duplicate mitigation paragraph in a
    different per-technique header, e.g.
    "[MITRE ATT&CK T0800 - ...]\n\n**Human User Authentication**\n..." vs
    "[MITRE ATT&CK T0816 - ...]\n\n**Human User Authentication**\n...".
    Full-string comparison would miss these; dedup must compare the body
    after the header.
    """
    from src.retrieval import _dedupe_candidates_by_text

    chunks = [
        {
            "text": "[MITRE ATT&CK T0800 - Activate Firmware Update Mode]\n\n**Human User Authentication**\nRequire user authentication.",
            "metadata": {"section": "T0800"},
            "score": 0.0,
        },
        {
            "text": "[MITRE ATT&CK T0816 - Device Restart/Shutdown]\n\n**Human User Authentication**\nRequire user authentication.",
            "metadata": {"section": "T0816"},
            "score": 0.0,
        },
        {
            "text": "[MITRE ATT&CK T0831 - Manipulation of Control]\n\nAdversaries may manipulate physical process control.",
            "metadata": {"section": "T0831"},
            "score": 0.0,
        },
    ]

    result = _dedupe_candidates_by_text(chunks)

    assert len(result) == 2
    assert result[0]["metadata"]["section"] == "T0800"
    assert result[1]["metadata"]["section"] == "T0831"


def test_dedupe_candidates_no_duplicates_returns_unchanged():
    from src.retrieval import _dedupe_candidates_by_text

    chunks = [
        {"text": "a", "metadata": {"section": "T0800"}, "score": 0.0},
        {"text": "b", "metadata": {"section": "T0831"}, "score": 0.0},
    ]

    result = _dedupe_candidates_by_text(chunks)

    assert result == chunks


def test_dedupe_candidates_empty_list():
    from src.retrieval import _dedupe_candidates_by_text

    assert _dedupe_candidates_by_text([]) == []
