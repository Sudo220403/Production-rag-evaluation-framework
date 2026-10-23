"""
Multi-Domain RAG Benchmark Corpus & Ground-Truth QA Dataset.
Covers 3 distinct document architectures where chunking and embedding choices
exhibit measurable differences in retrieval precision and answer faithfulness:
  1. Technical Architecture & API Spec (`technical_docs`)
  2. Financial & Regulatory 10-K Filing (`financial_sec_filing`)
  3. Biomedical / Clinical Trial Manuscript (`clinical_research`)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

from src.config import DATA_DIR


@dataclass
class DocumentCorpusItem:
    doc_id: str
    doc_type: str
    title: str
    content: str


@dataclass
class BenchmarkQASample:
    sample_id: str
    doc_id: str
    doc_type: str
    question: str
    ground_truth: str
    required_facts: List[str] = field(default_factory=list)
    reasoning_category: str = "multi-clause-factual"


TECHNICAL_SPEC_CONTENT = """# ChronosMesh Distributed Consensus & API Gateway Specification (v4.2)

## 1. Consensus Engine & Raft Lease Parameters
ChronosMesh coordinates state replication across regional control planes using a modified Raft consensus protocol with leader lease read-optimizations. Under normal operation inside a single availability zone, the default heartbeat interval (`raft_heartbeat_interval_ms`) is set to 60ms, and the election timeout (`raft_election_timeout_ms`) is randomized between 300ms and 450ms.

However, when operating in Multi-Region Active-Active topology (`topology_mode = "global_mesh"`), cross-continental WAN jitter requires strict lease calibration. Specifically, `quorum_lease_duration_ms` must be configured to 850ms while `max_clock_drift_tolerance_ms` is capped at 12ms via Precision Time Protocol (PTP) hardware sync. If hardware PTP synchronization degrades and observed clock drift exceeds 12ms for three consecutive epochs, the leader node MUST immediately revoke local read leases, step down to follower state, and force all client reads through linearizable quorum (`read_consistency = "quorum_strict"`), which adds an average penalty of 42ms per read request.

## 2. Adaptive Rate Limiting & Token Bucket Burst Matrix
The ChronosMesh Edge Gateway enforces tenant isolation using a hierarchical two-tier token bucket algorithm backed by local eBPF counters and asynchronous Redis-Cluster state synchronization every 25ms.

For Tier-1 Enterprise tenants, the sustained rate limit is 12,500 requests per second (RPS) per regional ingress pop, with a maximum burst capacity (`burst_bucket_ceiling`) of 18,000 tokens over a 4-second sliding window. When a Tier-1 tenant exceeds 18,000 tokens within the 4-second window, the gateway does NOT immediately drop traffic with HTTP 429; instead, it activates Priority Shedding Mode (`shed_policy = "degrade_non_critical"`), which drops background batch telemetry (`X-Priority-Class: P3`) while permitting transactional traffic (`P0` and `P1`) up to a hard ceiling of 21,500 RPS before returning HTTP 429 with `Retry-After: 8`.

## 3. Vector Index Compaction & Write-Ahead Log (WAL) Recovery
Storage nodes persist incoming mutations to an append-only Write-Ahead Log (`wal_segment_size_mb = 256`) using `fdatasync` batching every 5ms or 64KB of payload. Background HNSW graph compaction triggers automatically when tombstone ratio (`deleted_vectors / total_vectors`) surpasses 18.5% or when uncompacted segment count reaches 24 segments. During compaction, memory allocation is strictly bounded to 35% of host RAM (`compaction_mem_limit_pct = 0.35`) to prevent OOM-killer preemption of live query workers.

## 4. Circuit Breaker & Tail-Latency Hedging
To bound p99 tail latency across downstream microservice shards, the gateway dispatches hedged read requests when a shard's rolling 30-second p95 latency exceeds 115ms. Hedged requests are capped at 6% of total outbound traffic (`hedge_budget_max_ratio = 0.06`) to prevent retry amplification storms during partial brownouts. If downstream error rate (HTTP 502/503/504) reaches 8.0% over a 15-second window, the circuit breaker trips into `HALF_OPEN` probe state for 45 seconds, allowing exactly 1 probe request every 500ms.
"""

FINANCIAL_FILING_CONTENT = """# Apex Cloud Infrastructure Corp — Form 10-K Annual Financial & Covenant Report (FY2025)

## Item 7. Revenue Disaggregation & Annual Recurring Revenue (ARR)
For the fiscal year ended December 31, 2025, Apex Cloud Infrastructure Corp recorded Consolidated Total Revenue of $1,482.6 million, representing a year-over-year increase of 24.8% compared to $1,188.0 million in FY2024. Revenue is disaggregated across three primary operating segments:
1. Core Compute & Sovereign AI Clusters: $964.2 million (65.0% of total revenue), growing 31.4% YoY.
2. Managed Vector & Database Platform (DBaaS): $378.9 million (25.6% of total revenue), growing 19.2% YoY.
3. Professional Migration & Compliance Advisory: $139.5 million (9.4% of total revenue), declining 2.1% YoY due to partner offloading.

Net Dollar Retention (NDR) across customers with Annual Contract Value (ACV) above $100,000 stood at 118.4% at year-end, down from 121.0% in FY2024 primarily due to seat consolidation in mid-market accounts during Q2 2025.

## Item 7A. Non-GAAP Adjusted EBITDA & Footnote Reconciliations
GAAP Operating Income for FY2025 was $194.8 million (13.1% GAAP operating margin). Reconciled Non-GAAP Adjusted EBITDA reached $412.5 million (27.8% Adjusted EBITDA margin).

CRITICAL FOOTNOTE (Note 14-B — Excluded & Capped Adjustments): In computing Consolidated Adjusted EBITDA for Senior Credit Facility compliance, stock-based compensation expense of $142.3 million and amortization of acquired intangibles of $48.7 million are added back in full. However, restructuring and datacenter decommissioning charges ($26.7 million incurred in FY2025) are subject to a strict contractual add-back cap equal to 5.0% of Consolidated Adjusted EBITDA (or $20.6 million maximum). Consequently, $6.1 million of excess decommissioning expense CANNOT be added back under the Credit Agreement definition, resulting in Covenant-Defined EBITDA of $406.4 million.

## Item 8. Senior Secured Revolving Credit Facility & Debt Covenants
OnAugust 14, 2024, the Company entered into a $750.0 million Senior Secured Credit Facility maturing August 14, 2029, bearing interest at SOFR + 225 bps (with a 0.50% SOFR floor). Under Section 6.12 of the Credit Agreement, the Company must maintain a Maximum Consolidated Net Leverage Ratio (`Consolidated Net Debt / Covenant-Defined EBITDA`) not to exceed 3.25x tested at the end of each fiscal quarter.

Exception Provision: Immediately following a Material Permitted Acquisition exceeding $150.0 million in cash consideration, the Maximum Consolidated Net Leverage Ratio covenant temporarily steps up to 3.75x for four consecutive fiscal quarters ("Covenant Holiday Period"), after which it reverts to 3.25x. In addition, the Company must maintain a Minimum Interest Coverage Ratio (`Covenant-Defined EBITDA / Cash Interest Expense`) of at least 4.00x at all times without exception. As of December 31, 2025, actual Net Leverage stood at 1.84x and Interest Coverage stood at 7.65x.

## Item 9. Deferred Revenue & Remaining Performance Obligations (RPO)
Total Remaining Performance Obligations (RPO) as of December 31, 2025 totaled $2,840.0 million, of which current RPO (expected to be recognized as revenue within the next 12 months) accounted for 54.0% ($1,533.6 million), while non-current RPO accounted for 46.0% ($1,306.4 million).
"""

CLINICAL_RESEARCH_CONTENT = """# Phase III Multicenter Randomized Trial of Nexaferon-Alpha in Refractory Autoimmune Nephritis (NEPHRO-302 Study)

## Abstract & Study Design
The NEPHRO-302 trial (NCT05891204) is a double-blind, placebo-controlled, 52-week Phase III study evaluating subcutaneous Nexaferon-Alpha (180 mcg weekly) plus standard-of-care mycophenolate mofetil (MMF) versus placebo plus MMF in 640 adult patients with biopsy-proven Class III/IV refractory autoimmune nephritis across 78 clinical centers. Patients were stratified at baseline by urinary protein-to-creatinine ratio (UPCR >= 3.5 g/g vs. < 3.5 g/g) and baseline estimated glomerular filtration rate (eGFR Stage 2 vs. Stage 3a/3b).

## Primary & Secondary Efficacy Endpoints (Week 52)
At Week 52, the primary endpoint of Complete Renal Response (CRR — defined as UPCR < 0.5 g/g, stabilization of eGFR within 10% of baseline, and corticosteroid taper to <= 5 mg/day prednisone equivalent between Weeks 40 and 52 without rescue therapy) was achieved by 48.4% (155/320) of patients in the Nexaferon-Alpha arm compared to 26.9% (86/320) in the placebo arm, yielding an adjusted hazard ratio (HR) of 0.58 for renal flare or treatment failure (95% CI: 0.44 to 0.76; p = 0.00014).

Subgroup Analysis by Baseline Renal Function:
- In patients with preserved baseline filtration (eGFR >= 60 mL/min/1.73m2, n = 388), CRR rates were 54.1% with Nexaferon-Alpha vs. 29.4% with placebo (absolute risk difference +24.7%).
- In patients with moderate-to-severe renal impairment (eGFR 30 to <60 mL/min/1.73m2, n = 252), CRR rates were 39.7% with Nexaferon-Alpha vs. 23.0% with placebo (absolute risk difference +16.7%). Patients with baseline eGFR < 30 mL/min/1.73m2 were excluded from enrollment due to delayed renal clearance of the pegylated carrier moiety.

## Safety Profile, Hepatic Monitoring & Dose Modification Protocol
Treatment-emergent adverse events (TEAEs) occurred in 81.6% of Nexaferon-Alpha recipients versus 76.2% of placebo recipients. Serious adverse events (SAEs) occurred in 14.1% vs. 15.9%, respectively.

Hepatic Transaminitis Protocol (Section 5.4): Grade 3 asymptomatic elevation in alanine aminotransferase (ALT > 5.0x to 8.0x Upper Limit of Normal [ULN] without hyperbilirubinemia) was observed in 4.7% (15/320) of patients receiving Nexaferon-Alpha. Per mandatory protocol rules, any patient exhibiting ALT > 5.0x ULN with total bilirubin < 2.0x ULN must suspend dosing for 14 days and repeat hepatic panels every 72 hours; once ALT resolves to <= 2.5x ULN, Nexaferon-Alpha is resumed at a reduced step-down dose of 120 mcg weekly. However, if ALT exceeds 3.0x ULN concurrently with total bilirubin >= 2.0x ULN (Hy's Law criteria), study drug must be permanently discontinued immediately without rechallenge.

## Pharmacokinetics & Immunogenicity
Steady-state trough concentrations (`C_trough_ss`) of 42.8 ng/mL (CV 21.4%) were reached by Week 8. Anti-drug antibodies (ADAs) developed in 6.2% (20/320) of treated patients by Week 52, of whom only 1.9% (6/320) exhibited neutralizing antibodies (`NAb+`), with no statistically significant reduction in Week 52 CRR rates among ADA-positive subjects (p = 0.41).
"""


def get_corpus_documents() -> List[DocumentCorpusItem]:
    """Returns the 3 domain documents used across the evaluation benchmark."""
    return [
        DocumentCorpusItem(
            doc_id="doc_tech_01",
            doc_type="technical_docs",
            title="ChronosMesh Distributed Consensus & API Gateway Specification (v4.2)",
            content=TECHNICAL_SPEC_CONTENT.strip(),
        ),
        DocumentCorpusItem(
            doc_id="doc_fin_01",
            doc_type="financial_sec_filing",
            title="Apex Cloud Infrastructure Corp — Form 10-K Annual Financial Report (FY2025)",
            content=FINANCIAL_FILING_CONTENT.strip(),
        ),
        DocumentCorpusItem(
            doc_id="doc_clin_01",
            doc_type="clinical_research",
            title="NEPHRO-302 Phase III Trial of Nexaferon-Alpha in Autoimmune Nephritis",
            content=CLINICAL_RESEARCH_CONTENT.strip(),
        ),
    ]


def get_benchmark_qa_samples() -> List[BenchmarkQASample]:
    """
    Returns 12 multi-clause ground-truth evaluation questions (4 per document type).
    Designed so that naive fixed-window chunking often bisects conditions, qualifiers,
    or footnotes from their core figures, allowing rigorous measurement of faithfulness
    and hallucination rate across chunking strategies.
    """
    return [
        # --- Technical Docs (4 samples) ---
        BenchmarkQASample(
            sample_id="tech_q1",
            doc_id="doc_tech_01",
            doc_type="technical_docs",
            question="In Multi-Region Active-Active topology, what are the required quorum lease duration and clock drift tolerance, and what happens if clock drift exceeds tolerance for three consecutive epochs?",
            ground_truth="In global_mesh topology, quorum_lease_duration_ms is 850ms and max_clock_drift_tolerance_ms is capped at 12ms via PTP sync. If clock drift exceeds 12ms for three consecutive epochs, the leader immediately revokes local read leases, steps down to follower, and forces all client reads through linearizable quorum (quorum_strict), adding an average penalty of 42ms per read.",
            required_facts=["850ms", "12ms", "follower", "quorum_strict", "42ms"],
            reasoning_category="conditional-failure-recovery",
        ),
        BenchmarkQASample(
            sample_id="tech_q2",
            doc_id="doc_tech_01",
            doc_type="technical_docs",
            question="How does the ChronosMesh Edge Gateway handle a Tier-1 Enterprise tenant that exceeds 18,000 burst tokens within a 4-second sliding window?",
            ground_truth="Instead of immediately returning HTTP 429, it activates Priority Shedding Mode (degrade_non_critical), dropping P3 background batch telemetry while permitting P0 and P1 transactional traffic up to a hard ceiling of 21,500 RPS before returning HTTP 429 with Retry-After: 8.",
            required_facts=["18,000", "degrade_non_critical", "P3", "21,500", "Retry-After: 8"],
            reasoning_category="multi-threshold-policy",
        ),
        BenchmarkQASample(
            sample_id="tech_q3",
            doc_id="doc_tech_01",
            doc_type="technical_docs",
            question="Under what exact conditions does background HNSW graph compaction trigger, and how is memory usage bounded during compaction?",
            ground_truth="Background HNSW compaction triggers when the tombstone ratio surpasses 18.5% or when uncompacted segment count reaches 24 segments. Memory allocation is bounded to 35% of host RAM (compaction_mem_limit_pct = 0.35) to prevent OOM-killer preemption.",
            required_facts=["18.5%", "24 segments", "35%", "0.35"],
            reasoning_category="system-threshold-lookup",
        ),
        BenchmarkQASample(
            sample_id="tech_q4",
            doc_id="doc_tech_01",
            doc_type="technical_docs",
            question="When are hedged read requests dispatched by the gateway, what is the maximum traffic budget for hedging, and when does the circuit breaker trip into HALF_OPEN state?",
            ground_truth="Hedged read requests are dispatched when a shard's rolling 30-second p95 latency exceeds 115ms, capped at 6% of outbound traffic (0.06). The circuit breaker trips into HALF_OPEN state for 45 seconds (allowing 1 probe every 500ms) if the HTTP 502/503/504 error rate reaches 8.0% over a 15-second window.",
            required_facts=["115ms", "6%", "8.0%", "45 seconds", "500ms"],
            reasoning_category="multi-clause-factual",
        ),
        # --- Financial SEC Filing (4 samples) ---
        BenchmarkQASample(
            sample_id="fin_q1",
            doc_id="doc_fin_01",
            doc_type="financial_sec_filing",
            question="What was Apex Cloud Infrastructure Corp's FY2025 Consolidated Total Revenue and how much revenue did each of the three operating segments contribute?",
            ground_truth="Consolidated Total Revenue in FY2025 was $1,482.6 million (+24.8% YoY), comprising $964.2 million (65.0%) from Core Compute & Sovereign AI Clusters, $378.9 million (25.6%) from Managed Vector & Database Platform (DBaaS), and $139.5 million (9.4%) from Professional Migration & Compliance Advisory.",
            required_facts=["$1,482.6 million", "$964.2 million", "$378.9 million", "$139.5 million"],
            reasoning_category="tabular-aggregation",
        ),
        BenchmarkQASample(
            sample_id="fin_q2",
            doc_id="doc_fin_01",
            doc_type="financial_sec_filing",
            question="Why does Covenant-Defined EBITDA ($406.4 million) differ from reported Non-GAAP Adjusted EBITDA ($412.5 million) in FY2025?",
            ground_truth="Under Note 14-B, restructuring and datacenter decommissioning charges ($26.7 million) are subject to a contractual add-back cap of 5.0% of Consolidated Adjusted EBITDA ($20.6 million maximum). Therefore, $6.1 million of excess decommissioning expense cannot be added back, reducing $412.5 million to $406.4 million.",
            required_facts=["$412.5 million", "$26.7 million", "5.0%", "$6.1 million", "$406.4 million"],
            reasoning_category="footnote-reconciliation",
        ),
        BenchmarkQASample(
            sample_id="fin_q3",
            doc_id="doc_fin_01",
            doc_type="financial_sec_filing",
            question="What is the standard Maximum Consolidated Net Leverage Ratio covenant under the Senior Secured Credit Facility, when can it step up to 3.75x, and what was the actual ratio at year-end FY2025?",
            ground_truth="The standard Maximum Consolidated Net Leverage Ratio is 3.25x tested quarterly. Following a Material Permitted Acquisition exceeding $150.0 million in cash consideration, it steps up to 3.75x for four consecutive fiscal quarters. As of December 31, 2025, actual Net Leverage stood at 1.84x.",
            required_facts=["3.25x", "$150.0 million", "3.75x", "four consecutive", "1.84x"],
            reasoning_category="covenant-exception-logic",
        ),
        BenchmarkQASample(
            sample_id="fin_q4",
            doc_id="doc_fin_01",
            doc_type="financial_sec_filing",
            question="What were Total Remaining Performance Obligations (RPO) as of December 31, 2025, and how is RPO split between current (next 12 months) and non-current portions?",
            ground_truth="Total RPO as of December 31, 2025 was $2,840.0 million, split into 54.0% ($1,533.6 million) current RPO recognized within 12 months and 46.0% ($1,306.4 million) non-current RPO.",
            required_facts=["$2,840.0 million", "54.0%", "$1,533.6 million", "$1,306.4 million"],
            reasoning_category="financial-breakdown",
        ),
        # --- Clinical Research (4 samples) ---
        BenchmarkQASample(
            sample_id="clin_q1",
            doc_id="doc_clin_01",
            doc_type="clinical_research",
            question="In the NEPHRO-302 trial, what were the Week 52 Complete Renal Response (CRR) rates in the Nexaferon-Alpha and placebo arms, and what were the hazard ratio and p-value?",
            ground_truth="At Week 52, Complete Renal Response (CRR) was achieved by 48.4% (155/320) in the Nexaferon-Alpha arm versus 26.9% (86/320) in the placebo arm, with an adjusted hazard ratio (HR) of 0.58 (95% CI: 0.44 to 0.76; p = 0.00014).",
            required_facts=["48.4%", "26.9%", "0.58", "0.00014"],
            reasoning_category="clinical-endpoint-statistics",
        ),
        BenchmarkQASample(
            sample_id="clin_q2",
            doc_id="doc_clin_01",
            doc_type="clinical_research",
            question="How did Week 52 CRR rates compare between patients with preserved eGFR (>= 60) and moderate-to-severe renal impairment (eGFR 30 to <60), and why were eGFR < 30 patients excluded?",
            ground_truth="In preserved filtration (eGFR >= 60), CRR was 54.1% vs. 29.4% (+24.7% difference). In eGFR 30 to <60, CRR was 39.7% vs. 23.0% (+16.7% difference). Patients with eGFR < 30 were excluded due to delayed renal clearance of the pegylated carrier moiety.",
            required_facts=["54.1%", "39.7%", "+16.7%", "pegylated carrier"],
            reasoning_category="subgroup-stratification",
        ),
        BenchmarkQASample(
            sample_id="clin_q3",
            doc_id="doc_clin_01",
            doc_type="clinical_research",
            question="What is the required dose modification protocol for a patient who develops ALT > 5.0x ULN with total bilirubin < 2.0x ULN versus a patient meeting Hy's Law criteria?",
            ground_truth="For ALT > 5.0x ULN with bilirubin < 2.0x ULN, dosing must be suspended for 14 days with hepatic panels every 72 hours, and resumed at a reduced dose of 120 mcg weekly once ALT resolves to <= 2.5x ULN. If ALT > 3.0x ULN with total bilirubin >= 2.0x ULN (Hy's Law), the drug must be permanently discontinued immediately without rechallenge.",
            required_facts=["14 days", "72 hours", "120 mcg", "2.5x ULN", "permanently discontinued"],
            reasoning_category="clinical-safety-branching",
        ),
        BenchmarkQASample(
            sample_id="clin_q4",
            doc_id="doc_clin_01",
            doc_type="clinical_research",
            question="What steady-state trough concentration was reached by Week 8, what percentage of patients developed anti-drug antibodies (ADAs) and neutralizing antibodies, and did ADAs significantly impact CRR?",
            ground_truth="Steady-state trough concentration (C_trough_ss) reached 42.8 ng/mL (CV 21.4%) by Week 8. ADAs developed in 6.2% (20/320) of patients and neutralizing antibodies (NAb+) in 1.9% (6/320), with no statistically significant reduction in Week 52 CRR rates (p = 0.41).",
            required_facts=["42.8 ng/mL", "6.2%", "1.9%", "0.41"],
            reasoning_category="pharmacokinetics-immunogenicity",
        ),
    ]


def persist_dataset_artifacts(target_dir: Path = DATA_DIR) -> Dict[str, str]:
    """Writes corpus documents and QA benchmark JSON to `data/` for reproducibility."""
    target_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = target_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    docs = get_corpus_documents()
    for doc in docs:
        (corpus_dir / f"{doc.doc_id}_{doc.doc_type}.md").write_text(
            doc.content, encoding="utf-8"
        )

    qa_samples = [asdict(s) for s in get_benchmark_qa_samples()]
    qa_path = target_dir / "benchmark_qa_dataset.json"
    qa_path.write_text(json.dumps(qa_samples, indent=2), encoding="utf-8")

    return {
        "corpus_dir": str(corpus_dir),
        "qa_dataset_path": str(qa_path),
        "num_documents": str(len(docs)),
        "num_questions": str(len(qa_samples)),
    }
