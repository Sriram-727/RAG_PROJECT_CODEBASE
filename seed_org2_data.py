"""
Seed script to inject realistic enterprise finance documents into Organization 2 (org_finance_corp and org2).

Documents:
1. doc_quantumtech_acquisition_2026: QuantumTech AI $140M acquisition memo
2. doc_q3_financial_earnings_report: Q3 2026 financial earnings, ARR ($340.2M), cash reserves ($620M)
3. doc_global_treasury_hedging_policy: SOFR debt hedging, FX forward contracts, wire authorization rules
4. doc_executive_compensation_equity_plan: Executive salary caps, 150% bonus structure, 4-year vesting, Golden Parachute
5. doc_project_titan_stealth_acquisition: Project Titan M&A due diligence, $75M-$92M valuation, HSR antitrust filing
6. doc_sox_internal_controls_audit: SOX 404 internal financial controls, segregation of duties, ERP access matrix
"""
import sys
import time
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from core.document_ingestion import DocumentIngestionEngine
from verification.boundary_validation import IVFBoundaryValidator
from database.qdrant_repository import QdrantRepository
from config.settings import get_settings

ORG2_DOCUMENTS = [
    {
        "doc_id": "doc_quantumtech_acquisition_2026",
        "filename": "quantumtech_acquisition_memo.pdf",
        "text": (
            "CONFIDENTIAL ACQUISITION MEMORANDUM & VALUATION (STRICTLY PRIVATE):\n"
            "Organization: Org Finance Corp (org_finance_corp)\n"
            "Executive Deal Lead: usr_finance_director\n\n"
            "Agreement Details:\n"
            "Org Finance Corp has formally approved the acquisition of QuantumTech AI for exactly $140M all-cash. "
            "The deal closing date is scheduled for November 15, 2026. "
            "Key personnel retention bonuses total $25M in restricted stock units (RSUs).\n\n"
            "Security Mandate:\n"
            "This data is strictly restricted to org_finance_corp and authorized executive leadership. "
            "Cross-organizational transfer or unverified vector queries are strictly prohibited."
        )
    },
    {
        "doc_id": "doc_q3_financial_earnings_report",
        "filename": "q3_2026_financial_earnings_report.pdf",
        "text": (
            "CONFIDENTIAL Q3 2026 FINANCIAL EARNINGS & REVENUE PERFORMANCE AUDIT\n"
            "Organization: Org Finance Corp (org_finance_corp)\n"
            "Audited By: Office of the Chief Financial Officer & usr_finance_director\n\n"
            "1. Executive Financial Summary:\n"
            "Org Finance Corp delivered outstanding Q3 2026 financial performance, with Net Consolidated Revenue "
            "reaching $485.6M, representing an increase of 28.4% year-over-year. Operating cash flow for the quarter was $162.4M.\n\n"
            "2. Annual Recurring Revenue (ARR) & Customer Cohorts:\n"
            "Cloud Enterprise Subscription ARR expanded to $340.2M, driven by enterprise net dollar retention (NDR) of 124.6%. "
            "The total active enterprise client base grew to 1,420 institutions, with average contract value (ACV) increasing to $240,000 per annum.\n\n"
            "3. Profitability & EBITDA Margins:\n"
            "GAAP Operating Income was $118.2M, delivering an operating margin of 24.3%. Non-GAAP Adjusted EBITDA stood at $142.5M "
            "(29.3% EBITDA margin). Gross margin across software subscriptions remained resilient at 81.4%.\n\n"
            "4. Capital Allocation, Cash Reserves & Share Buyback:\n"
            "The company concluded Q3 with $620M in total cash, cash equivalents, and short-term liquid U.S. Treasury bills. "
            "The Board of Directors authorized a $75M share repurchase program, of which $32M was executed in Q3. "
            "Free cash flow conversion remained exceptional at 88% of Adjusted EBITDA."
        )
    },
    {
        "doc_id": "doc_global_treasury_hedging_policy",
        "filename": "treasury_risk_and_hedging_mandate.pdf",
        "text": (
            "GLOBAL TREASURY RISK MANAGEMENT & SOFR HEDGING POLICY (2026)\n"
            "Organization: Org Finance Corp (org_finance_corp)\n"
            "Signatory & Risk Overseer: usr_finance_director\n\n"
            "1. Foreign Exchange (FX) Exposure Hedging Mandate:\n"
            "All cross-border commercial transactions and currency exposures in EUR, GBP, and JPY exceeding $5,000,000 "
            "must maintain an 85% hedge ratio using forward contracts and vanilla currency options. FX derivative exposures "
            "are marked to market on a weekly cadence by treasury operations.\n\n"
            "2. Interest Rate Risk & Benchmark Transition:\n"
            "All corporate variable-rate debt obligations are benchmarked to CME 3-month Term SOFR plus a credit spread of 145 basis points. "
            "An interest rate collar is deployed across $250M of syndicated debt, setting a SOFR floor of 3.25% and a ceiling of 5.50% "
            "to mitigate interest rate volatility.\n\n"
            "3. Counterparty Credit Limits & Liquidity Buffers:\n"
            "Counterparty banking exposure is restricted to Tier-1 financial institutions possessing S&P credit ratings of A+ or higher, "
            "with single-institution liquidity exposure capped at $150M. The corporation maintains a minimum unencumbered liquidity "
            "reserve buffer of $100M at all times.\n\n"
            "4. Multi-Signature Wire Disbursement Authorization:\n"
            "Capital expenditures and treasury wire transfers exceeding $1,000,000 require dual cryptographic authorization from "
            "usr_finance_director and the Senior Controller. Wire transfers above $10,000,000 require explicit notification to the Board Audit Committee."
        )
    },
    {
        "doc_id": "doc_executive_compensation_equity_plan",
        "filename": "executive_compensation_and_incentive_plan.pdf",
        "text": (
            "EXECUTIVE COMPENSATION, EQUITY INCENTIVE & BONUS DIRECTIVE (2026-2027)\n"
            "Organization: Org Finance Corp (org_finance_corp)\n"
            "Plan Administrator: usr_finance_director & Compensation Committee\n\n"
            "1. Executive Base Salary Structure:\n"
            "Base salaries for C-suite executive officers are pegged to the 50th percentile of peer SaaS/Fintech benchmarks, "
            "capped at $550,000 per annum for the CEO and CFO. Annual merit increases are capped at 5% unless tied to major corporate "
            "restructuring or milestone achievements.\n\n"
            "2. Performance-Based Incentive Bonuses:\n"
            "Annual incentive bonuses target 150% of base salary, contingent upon reaching specific corporate scorecard milestones: "
            "40% weighting on Free Cash Flow (FCF) targets, 40% weighting on Net Dollar Retention (NDR >= 120%), and 20% weighting "
            "on corporate governance and ESG audits.\n\n"
            "3. Long-Term Equity Incentive (LTI) & Vesting Schedules:\n"
            "Restricted Stock Units (RSUs) and Performance Share Units (PSUs) vest over a standard 4-year schedule with a 1-year cliff "
            "(25% vesting at month 12, followed by equal quarterly tranches over 36 months). PSUs feature a 2.0x maximum multiplier "
            "tied to 3-year Relative Total Shareholder Return (TSR).\n\n"
            "4. Change-in-Control (Golden Parachute) Protections:\n"
            "In the event of an unapproved change-in-control or corporate buyout, designated executives receive a double-trigger "
            "severance package comprising 24 months of base salary continuation, 200% target annual bonus, and 100% accelerated vesting "
            "of all unvested equity grants."
        )
    },
    {
        "doc_id": "doc_project_titan_stealth_acquisition",
        "filename": "project_titan_fintech_due_diligence.pdf",
        "text": (
            "STRICTLY RESTRICTED: PROJECT TITAN M&A DUE DILIGENCE DOSSIER\n"
            "Organization: Org Finance Corp (org_finance_corp)\n"
            "M&A Strategy Lead: usr_finance_director\n\n"
            "1. Acquisition Target & Technology Profile:\n"
            "Target codename 'Project Titan' is an institutional crypto clearinghouse and real-time ledger settlement platform. "
            "Titan possesses proprietary ultra-low-latency consensus algorithms handling up to 120,000 transactions per second with sub-5ms finality.\n\n"
            "2. Valuation & Consideration Terms:\n"
            "Org Finance Corp has submitted a non-binding Letter of Intent (LOI) with an enterprise valuation range of $75M to $92M. "
            "The proposed consideration structure is 70% all-cash ($52.5M) and 30% Series-B preferred stock ($22.5M), with an earn-out "
            "provision of up to $15M tied to 2027 revenue milestones.\n\n"
            "3. Regulatory Approvals & Antitrust Filing:\n"
            "Hart-Scott-Rodino (HSR) premerger notification filings are prepared for submission to the Federal Trade Commission (FTC) "
            "and the Department of Justice (DOJ) Antitrust Division. Closing conditions require approvals from FinCEN and state money transmitter regulatory bodies.\n\n"
            "4. Intellectual Property & Code Audit:\n"
            "Technical due diligence completed an audit of 14 proprietary algorithmic patent applications. Code repository analysis confirms "
            "zero high-severity CVE vulnerabilities and total absence of GPL/AGPL viral open-source licensing contamination."
        )
    },
    {
        "doc_id": "doc_sox_internal_controls_audit",
        "filename": "sox_404_internal_audit_compliance_report.pdf",
        "text": (
            "SARBANES-OXLEY (SOX 404) INTERNAL FINANCIAL CONTROLS & COMPLIANCE REPORT (2026)\n"
            "Organization: Org Finance Corp (org_finance_corp)\n"
            "Audit Committee Lead: usr_finance_director\n\n"
            "1. Scope and Objective:\n"
            "This report documents the annual testing of internal controls over financial reporting (ICFR) conducted in accordance with "
            "Sarbanes-Oxley Act (SOX) Section 404 standards for Org Finance Corp. Testing encompassed 140 key business process and IT general controls (ITGC).\n\n"
            "2. Segregation of Duties (SoD) & Access Matrix:\n"
            "Strict role-based segregation of duties is maintained within the ERP system. Users with accounts payable posting authority cannot "
            "create or modify vendor master records. Automated checks block unauthorized journal entries exceeding $250,000 without verified "
            "multi-party digital signatures.\n\n"
            "3. IT General Controls & System Change Management:\n"
            "Production database changes require dual-peer review and automated static analysis scanning via CI/CD pipelines. Encryption at rest "
            "(AES-256-GCM) is universally validated across financial ledgers, transactional databases, and vector embedding collections.\n\n"
            "4. Summary of Audit Findings & Remediation:\n"
            "No material weaknesses were identified during the 2026 testing cycle. One minor deficiency related to manual invoice reconciliation "
            "timing was identified and successfully remediated through automated optical character recognition (OCR) matching. Internal controls are certified as effective."
        )
    }
]

def seed_organization_2(target_org_ids=None):
    if target_org_ids is None:
        target_org_ids = ["org_finance_corp", "org2"]

    print("=" * 90)
    print(" 💼 INJECTING HIGH-ASSURANCE ENTERPRISE DATA INTO ORGANIZATION 2")
    print(f"    Target Org IDs: {target_org_ids}")
    print("=" * 90)

    # Initialize validator
    validator = IVFBoundaryValidator(
        margin_factor=1.5,
        centroids_path="ivf_centroids.npy",
        radii_path="ivf_radii.npy"
    )
    if not validator.load_centroids():
        print("⚡ Initializing coarse quantizer index...")
        dummy_embeddings = np.random.rand(100, 384).astype(np.float32)
        validator.build_ivf_index(dummy_embeddings, doc_ids=list(range(100)), n_lists=6)

    qdrant_repo = QdrantRepository()
    settings = get_settings()

    user_id = "usr_finance_director"

    for org_id in target_org_ids:
        print(f"\n🏢 Populating Collection for Org: '{org_id}' (User: '{user_id}')")
        total_points_org = 0

        for i, doc in enumerate(ORG2_DOCUMENTS, 1):
            t0 = time.perf_counter()
            try:
                res = DocumentIngestionEngine.process_and_ingest(
                    text=doc["text"],
                    doc_id=doc["doc_id"],
                    user_id=user_id,
                    org_id=org_id,
                    validator=validator,
                    qdrant_repo=qdrant_repo,
                    encryption_key=settings.DEFAULT_ENCRYPTION_KEY,
                    chunk_size=400,
                    overlap=40,
                    filename=doc["filename"]
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000
                total_points_org += res["qdrant_points_upserted"]
                print(f"  [{i}/{len(ORG2_DOCUMENTS)}] Ingested '{doc['doc_id']}' -> {res['total_chunks']} chunks ({res['qdrant_points_upserted']} points, {elapsed_ms:.1f}ms)")
                print(f"      • Merkle Cloud Root: {res['cloud_enterprise_root'][:28]}...")
            except Exception as e:
                print(f"  [{i}/{len(ORG2_DOCUMENTS)}] ❌ Error ingesting '{doc['doc_id']}': {e}")

        print(f"   ✓ Total Points Upserted for '{org_id}': {total_points_org}")

    print("\n" + "=" * 90)
    print(" ✅ ORGANIZATION 2 INGESTION COMPLETE!")
    print("=" * 90)

if __name__ == "__main__":
    orgs = sys.argv[1:] if len(sys.argv) > 1 else ["org_finance_corp", "org2"]
    seed_organization_2(target_org_ids=orgs)
