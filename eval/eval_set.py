"""
eval_set.py -- 20-item evaluation set for the SecureOps Assistant retriever.

Per the brief's Tier 2 direction ("No measurement of quality -> 15-25 item
test set; retrieval hit rate + groundedness"). Every ``expected_doc``/
``expected_section`` below was verified against the real, indexed corpus
(chunks.jsonl) before writing the question -- not guessed -- so a failed
hit-rate check reflects a genuine retrieval gap, not a bad ground truth.

Each item is one of:
  - "doc"     : the question should retrieve >=1 chunk from expected_doc
  - "section" : the question should retrieve a chunk matching
                expected_doc AND expected_section specifically
  - "refusal" : the question is out-of-corpus-scope; retrieval succeeding
                or failing is irrelevant -- this category is for the
                synthesis-level honesty check (run separately, needs an
                LLM call), not the retrieval hit-rate metric.
"""

EVAL_SET = [
    # --- NIST SP 800-82 Rev. 3 (5 items) -----------------------------------
    {
        "id": "nist82-01",
        "question": "What does NIST recommend regarding remote access to OT networks?",
        "kind": "section",
        "expected_doc": "NIST SP 800-82 Rev. 3",
        "expected_section": "6.2.10",
    },
    {
        "id": "nist82-02",
        "question": "What is the difference between IT security and OT security priorities?",
        "kind": "section",
        "expected_doc": "NIST SP 800-82 Rev. 3",
        "expected_section": "2.3",
    },
    {
        "id": "nist82-03",
        "question": "What encryption-based technologies should be used to protect remote access sessions to OT?",
        "kind": "section",
        "expected_doc": "NIST SP 800-82 Rev. 3",
        "expected_section": "1.4.1",
    },
    {
        "id": "nist82-04",
        "question": "What is configuration control in the context of OT systems?",
        "kind": "section",
        "expected_doc": "NIST SP 800-82 Rev. 3",
        "expected_section": "128",
    },
    {
        "id": "nist82-05",
        "question": "How should organizations apply the principle of least privilege to OT remote access controls?",
        "kind": "doc",
        "expected_doc": "NIST SP 800-82 Rev. 3",
    },
    # --- NIST CSF 2.0 (4 items) ---------------------------------------------
    {
        "id": "csf-01",
        "question": "How should an organization establish its cybersecurity risk management strategy and organizational context under the Govern function?",
        "kind": "section",
        "expected_doc": "NIST CSF 2.0",
        "expected_section": "GV.OC-01",
    },
    {
        "id": "csf-02",
        "question": "How should access permissions and entitlements be managed according to the principle of least privilege?",
        "kind": "section",
        "expected_doc": "NIST CSF 2.0",
        "expected_section": "PR.AA-05",
    },
    {
        "id": "csf-03",
        "question": "Where should an organization receive cyber threat intelligence from?",
        "kind": "section",
        "expected_doc": "NIST CSF 2.0",
        "expected_section": "ID.RA-02",
    },
    {
        "id": "csf-04",
        "question": "What are the functions of the NIST Cybersecurity Framework 2.0?",
        "kind": "doc",
        "expected_doc": "NIST CSF 2.0",
    },
    # --- CISA ICS Advisories (6 items, real CVEs/vendors from the corpus) --
    {
        "id": "cisa-01",
        "question": "What vulnerability affects Siemens products in advisory ICSA-26-134-14, and what is CVE-2025-22871?",
        "kind": "doc",
        "expected_doc": "CISA Advisory ICSA-26-134-14",
    },
    {
        "id": "cisa-02",
        "question": "What does CVE-2026-8805 affect in Mitsubishi Electric products?",
        "kind": "doc",
        "expected_doc": "CISA Advisory ICSA-26-169-05",
    },
    {
        "id": "cisa-03",
        "question": "What is the vulnerability described in CVE-2025-3465 for ABB products?",
        "kind": "doc",
        "expected_doc": "CISA Advisory ICSA-26-139-01",
    },
    {
        "id": "cisa-04",
        "question": "What vulnerabilities affect Siemens products in advisory ICSA-26-111-07 (CVE-2020-26140, CVE-2020-26146)?",
        "kind": "doc",
        "expected_doc": "CISA Advisory ICSA-26-111-07",
    },
    {
        "id": "cisa-05",
        "question": "What is CVE-2024-41975 and which ABB advisory does it appear in?",
        "kind": "doc",
        "expected_doc": "CISA Advisory ICSA-26-132-04",
    },
    {
        "id": "cisa-06",
        "question": "Summarise recent advisories affecting Siemens industrial products.",
        "kind": "doc",
        "expected_doc": "CISA Advisory",
    },
    # --- MITRE ATT&CK for ICS (4 items) -------------------------------------
    {
        "id": "attck-01",
        "question": "Which ATT&CK for ICS techniques involve manipulation of control logic?",
        "kind": "doc",
        "expected_doc": "MITRE ATT&CK T0831",
    },
    {
        "id": "attck-02",
        "question": "What is the ATT&CK for ICS technique for manipulating the I/O image of a PLC?",
        "kind": "doc",
        "expected_doc": "MITRE ATT&CK T0835",
    },
    {
        "id": "attck-03",
        "question": "What is the ATT&CK for ICS technique involving modifying a parameter to impair process control?",
        "kind": "doc",
        "expected_doc": "MITRE ATT&CK T0836",
    },
    {
        "id": "attck-04",
        "question": "What mitigations involve requiring human user authentication before accepting device commands?",
        "kind": "doc",
        "expected_doc": "MITRE ATT&CK",
    },
    # --- Honesty test / out-of-scope refusal (1 item, synthesis-level) -----
    {
        "id": "honesty-01",
        "question": "What is our company's firewall configuration?",
        "kind": "refusal",
        "expected_doc": None,
    },
]
