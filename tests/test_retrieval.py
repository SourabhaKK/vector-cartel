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
