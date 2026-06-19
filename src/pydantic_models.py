from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


SourceType = Literal[
    "NIST_SP_800_82",
    "NIST_CSF_2_0",
    "CISA_ICS_ADVISORY",
    "MITRE_ATTACK_ICS",
    "OTHER",
]

SourcePreference = Literal["NIST", "CISA", "MITRE", "ANY", "N/A"]

QueryRoute = Literal[
    "PENDING",
    "NIST_GUIDANCE",
    "CISA_ADVISORY",
    "MITRE_ATTACK_ICS",
    "COMPANY_SPECIFIC_UNKNOWN",
    "GENERAL_OT_SECURITY",
    "OUT_OF_SCOPE",
]


class AttackICSMetadata(BaseModel):
    technique_id: str = "N/A"
    name: str = "N/A"
    tactics: list[str] = Field(default_factory=list)
    tactic_ids: list[str] = Field(default_factory=list)
    is_subtechnique: bool | None = None
    parent_technique: str = "N/A"
    tactic_source : str = "N/A"
    url : str = 'N/A'


class AdvisoriesMetadata(BaseModel):
    alert_code: str = "N/A"
    title: str = "N/A"
    url: str = "N/A"
    release_date: str = "N/A"
    vendor: str = "N/A"
    cvss_version: str = "N/A"
    cvss_score: float | None = None
    sectors: list[str] = Field(default_factory=list)
    countries: str = "N/A"
    source: str = "CISA ICS Advisory"


class SecureOpsDocumentMetadata(BaseModel):
    source_type: SourceType = "OTHER"
    source_name: str = "N/A"
    document_title: str = "N/A"
    section_or_page: str = "N/A"
    source_path: str = "N/A"
    url: str = "N/A"

    advisory_id: str = "N/A"
    vendor: str = "N/A"

class SecureOpsQueryClassify(BaseModel):
    routing_label: QueryRoute = "PENDING"
    source_preference: SourcePreference = "ANY"
    vendor: str = "N/A"
    products: list[str] = Field(default_factory=list)
    cve_ids: list[str] = Field(default_factory=list)
    mitre_technique_ids: list[str] = Field(default_factory=list)
    mitre_technique_names: list[str] = Field(default_factory=list)
    date_filter: str = "N/A"
    severity_filter: str = "N/A"
    topic_keywords: list[str] = Field(default_factory=list)
    restructured_query: str = ""
    should_retrieve: bool = True
    needs_clarification: bool = False