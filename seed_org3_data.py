"""
Seed script to inject realistic enterprise healthcare and clinical research data into Organization 3 (org_healthcare_plus).

Documents:
1. doc_hipaa_phi_compliance_protocol: HIPAA patient privacy, encryption standards, BAA contracts, 6-year audit trail
2. doc_clinical_trial_oncology_phase3: Phase III OncoShield-7 trial, Overall Survival HR 0.68, FDA BLA priority review
3. doc_telehealth_ehr_interoperability: HL7 FHIR R4 interoperability, Epic/Cerner API, RPM IoT telemetry triggers
4. doc_genomic_biobank_research_governance: ACMG/AMP variant classification, NovaSeq X WGS pipeline, consent revocation
5. doc_hospital_emergency_preparedness_plan: Level 1 trauma surge capacity, 96-hour auxiliary power, VIE liquid oxygen supply
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

ORG3_DOCUMENTS = [
    {
        "doc_id": "doc_hipaa_phi_compliance_protocol",
        "filename": "hipaa_phi_patient_privacy_protocol.pdf",
        "text": (
            "HEALTHCARE PLUS COMPLIANCE DIRECTIVE: HIPAA PRIVACY & PHI SECURITY STANDARDS (2026)\n"
            "Organization: Healthcare Plus (org_healthcare_plus)\n"
            "Compliance Officer: usr_clinical_director\n\n"
            "1. Protected Health Information (PHI) Encryption Mandate:\n"
            "All Electronic Protected Health Information (ePHI), Electronic Health Records (EHR), genomic variant data, "
            "and medical diagnostic imaging (DICOM) files must enforce AES-256-GCM authenticated encryption at rest and "
            "TLS 1.3 in transit. Symmetric encryption keys must be held in FIPS 140-3 Level 3 Hardware Security Modules (HSMs).\n\n"
            "2. Business Associate Agreements (BAAs) & Vendor Due Diligence:\n"
            "No third-party cloud service, API pipeline, or external analytics engine may ingest or process ePHI without an "
            "executed Business Associate Agreement (BAA) and annual SOC2 Type II healthcare security attestation.\n\n"
            "3. Audit Log Accounting & Retention Period:\n"
            "All clinical workforce access, queries, and disclosure events are cryptographically logged in tamper-evident "
            "Merkle audit trees. Records must be retained for exactly 6 years from the date of creation in compliance with "
            "45 CFR § 164.530(j)(2).\n\n"
            "4. Minimum Necessary Rule & Role-Based Access:\n"
            "Attending physicians, clinical nurses, and researchers are restricted to the minimum necessary patient data required "
            "for their active departmental shift. Cross-departmental record access requires emergency break-glass authorization."
        )
    },
    {
        "doc_id": "doc_clinical_trial_oncology_phase3",
        "filename": "oncology_phase3_clinical_trial_results.pdf",
        "text": (
            "PHASE III ONCOLOGY CLINICAL TRIAL FINAL PROTOCOL & EFFICACY REPORT\n"
            "Organization: Healthcare Plus (org_healthcare_plus)\n"
            "Principal Investigator: usr_clinical_director\n\n"
            "1. Study Overview & Patient Cohort:\n"
            "Protocol OncoShield-7 (ClinicalTrials.gov Identifier: NCT05891234) completed its international, double-blind, "
            "randomized Phase III trial evaluating novel bispecific immunotherapy in 840 patients with metastatic solid tumors "
            "across 42 academic medical centers.\n\n"
            "2. Primary & Secondary Clinical Endpoints:\n"
            "The primary endpoint of Overall Survival (OS) demonstrated a statistically significant Hazard Ratio (HR) of 0.68 "
            "(95% CI: 0.54-0.81, p < 0.001). Median Progression-Free Survival (PFS) was 18.4 months in the experimental arm versus "
            "9.2 months in the standard-of-care control arm. Objective Response Rate (ORR) was 52.4%.\n\n"
            "3. Safety Profile & Tolerability:\n"
            "Grade 3 or 4 treatment-emergent adverse events (TEAEs) occurred in only 14.2% of participants, with cytokine release "
            "syndrome (CRS) managed effectively using standard tocilizumab protocols. No treatment-related fatalities occurred.\n\n"
            "4. Regulatory Filing & Commercialization Timeline:\n"
            "Healthcare Plus has submitted a Biologics License Application (BLA) under FDA Accelerated Approval pathway with "
            "Priority Review designation, targeting formal commercial launch in Q4 2026."
        )
    },
    {
        "doc_id": "doc_telehealth_ehr_interoperability",
        "filename": "telehealth_ehr_fhir_interoperability_spec.pdf",
        "text": (
            "TELEHEALTH & ELECTRONIC HEALTH RECORD (EHR) INTEROPERABILITY SPECIFICATION\n"
            "Organization: Healthcare Plus (org_healthcare_plus)\n"
            "Chief Medical Information Officer: usr_clinical_director\n\n"
            "1. HL7 FHIR Interoperability Architecture:\n"
            "The enterprise clinical data exchange is built upon HL7 FHIR Release 4 (R4) RESTful APIs. Clinical document architectures "
            "seamlessly exchange patient demographics, medication orders, and observation bundles between Epic Systems, Cerner Millennium, "
            "and the proprietary Healthcare Plus Telehealth platform.\n\n"
            "2. Remote Patient Monitoring (RPM) Telemetry:\n"
            "Continuous biometric telemetry streams from cellular-connected patient medical devices (continuous glucose monitors, "
            "pulse oximeters, ECG patches) ingest directly into real-time FHIR Observation resources via mTLS-secured endpoints.\n\n"
            "3. Automated Critical Care Escalation:\n"
            "Algorithmic clinical decision support triggers automated emergency alerts to the on-call attending cardiologist within "
            "45 seconds when patient telemetry indicates sustained ventricular tachycardia or oxygen saturation drops below 88%.\n\n"
            "4. Cryptographic Authentication & Token Expiry:\n"
            "All clinician and third-party developer access enforces SMART on FHIR OAuth 2.0 with asymmetric PKCE tokens. Access "
            "tokens expire after 30 minutes, requiring silent biometric refresh."
        )
    },
    {
        "doc_id": "doc_genomic_biobank_research_governance",
        "filename": "genomics_biobank_research_governance.pdf",
        "text": (
            "INSTITUTIONAL REVIEW BOARD (IRB) GENOMICS BIOBANK RESEARCH GOVERNANCE\n"
            "Organization: Healthcare Plus (org_healthcare_plus)\n"
            "Biobank Director: usr_clinical_director\n\n"
            "1. Next-Generation Sequencing (NGS) Infrastructure:\n"
            "The Healthcare Plus Genomic Core facility operates an ultra-high-throughput Whole Genome Sequencing (WGS) pipeline on "
            "Illumina NovaSeq X platforms, delivering 30x minimum mean coverage depth with Q30 quality scores exceeding 92%.\n\n"
            "2. Clinical Variant Interpretation & ACMG Standards:\n"
            "Germline and somatic genetic variants are annotated and classified according to American College of Medical Genetics and "
            "Genomics (ACMG/AMP) guidelines. Any novel variant of uncertain significance (VUS) undergoes algorithmic structural "
            "prediction modeling before entry into the curated biobank.\n\n"
            "3. Patient Consent Tracking & Digital Revocation:\n"
            "All biological specimens are collected under IRB-approved broad research consent protocols. Patients retain the legal "
            "right to digital consent revocation at any time via the patient portal, which triggers automated cryptographic shredding "
            "of all genomic vector representations within 72 hours."
        )
    },
    {
        "doc_id": "doc_hospital_emergency_preparedness_plan",
        "filename": "hospital_mass_casualty_emergency_plan.pdf",
        "text": (
            "LEVEL 1 TRAUMA & DISASTER EMERGENCY PREPAREDNESS STANDARD OPERATING PROCEDURE\n"
            "Organization: Healthcare Plus (org_healthcare_plus)\n"
            "Emergency Operations Director: usr_clinical_director\n\n"
            "1. Incident Command System (ICS) & Surge Capacity:\n"
            "In mass-casualty events, the Incident Command System automatically converts ambulatory surgery centers and post-anesthesia "
            "care units (PACU) into 60 additional fully monitored Level 1 trauma ICU beds within 120 minutes of code activation.\n\n"
            "2. Emergency Power & Life Safety Systems:\n"
            "Dual-redundant 2.5 MW Caterpillar emergency diesel generators ensure 96 hours of uninterrupted full-facility electrical "
            "power with 12,000 gallons of subterranean fuel storage, maintaining all ventilators, surgical suites, and cold bio-storage.\n\n"
            "3. Medical Gas & Bulk Oxygen Reserve:\n"
            "The primary vacuum insulated evaporator (VIE) liquid oxygen bulk tank maintains a 14-day supply at maximum peak surge "
            "consumption, supplemented by an emergency manifold reserve containing 120 high-pressure H-cylinders."
        )
    }
]

def seed_organization_3(target_org_id="org_healthcare_plus"):
    print("=" * 90)
    print(f" 🏥 INJECTING HEALTHCARE & CLINICAL RESEARCH DATA INTO ORGANIZATION 3 ('{target_org_id}')")
    print("=" * 90)

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

    user_id = "usr_clinical_director"
    total_points = 0

    for i, doc in enumerate(ORG3_DOCUMENTS, 1):
        t0 = time.perf_counter()
        try:
            res = DocumentIngestionEngine.process_and_ingest(
                text=doc["text"],
                doc_id=doc["doc_id"],
                user_id=user_id,
                org_id=target_org_id,
                validator=validator,
                qdrant_repo=qdrant_repo,
                encryption_key=settings.DEFAULT_ENCRYPTION_KEY,
                chunk_size=400,
                overlap=40,
                filename=doc["filename"]
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            total_points += res["qdrant_points_upserted"]
            print(f"  [{i}/{len(ORG3_DOCUMENTS)}] Ingested '{doc['doc_id']}' -> {res['total_chunks']} chunks ({res['qdrant_points_upserted']} points, {elapsed_ms:.1f}ms)")
            print(f"      • Merkle Cloud Root: {res['cloud_enterprise_root'][:28]}...")
        except Exception as e:
            print(f"  [{i}/{len(ORG3_DOCUMENTS)}] ❌ Error ingesting '{doc['doc_id']}': {e}")

    print(f"\n   ✓ Total Points Upserted for Org 3 ('{target_org_id}'): {total_points}")
    print("=" * 90)
    print(" ✅ ORGANIZATION 3 INGESTION COMPLETE!")
    print("=" * 90)

if __name__ == "__main__":
    seed_organization_3()
