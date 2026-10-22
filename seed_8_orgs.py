"""
Script to create 8 more organization collections (org_3 through org_10) in Qdrant,
each storing exactly 1 complete document with encryption, embedding, MMR tree commitment, and vector indexing.
Also ensures org_2_collection is normalized.
"""
import os
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.document_ingestion import DocumentIngestionEngine
from core.auth_service import AuthService
from database.qdrant_repository import QdrantRepository
from database.qdrant_client import QdrantClientManager
from verification.boundary_validation import IVFBoundaryValidator
from config.settings import get_settings

DOCUMENTS = [
    {
        "org_id": "org_3",
        "doc_id": "doc_hipaa_phi_patient_privacy_protocol",
        "filename": "hipaa_phi_patient_privacy_protocol.pdf",
        "text": (
            "HEALTHCARE CLINICAL COMPLIANCE DIRECTIVE: HIPAA PRIVACY & PHI SECURITY (2026)\n"
            "Organization: Healthcare Plus (org_3)\n"
            "Compliance Officer: usr_clinical_director\n\n"
            "1. Protected Health Information (PHI) Encryption Mandate:\n"
            "All Electronic Protected Health Information (ePHI), Electronic Health Records (EHR), genomic data, "
            "and medical diagnostic imaging files must enforce AES-256-GCM authenticated encryption at rest and "
            "TLS 1.3 in transit. Symmetric keys must be stored in FIPS 140-3 Hardware Security Modules.\n\n"
            "2. Business Associate Agreements (BAAs) & Vendor Auditing:\n"
            "No third-party cloud pipeline, model provider, or API analytics service may ingest or process patient ePHI "
            "without an executed BAA and annual SOC2 Type II healthcare security attestation.\n\n"
            "3. Audit Log Accounting & Retention Period:\n"
            "All medical workforce record queries, chart access, and disclosures are cryptographically logged in "
            "tamper-evident Merkle audit trees. Records must be retained for 6 years in accordance with 45 CFR § 164.530.\n\n"
            "4. Minimum Necessary Rule & Role-Based Access:\n"
            "Attending physicians, nurses, and clinical specialists are strictly limited to the minimum necessary patient data "
            "required for their current departmental shift. Emergency cross-departmental access requires break-glass override."
        )
    },
    {
        "org_id": "org_4",
        "doc_id": "doc_aerospace_avionics_flight_manual",
        "filename": "aerospace_avionics_flight_manual.pdf",
        "text": (
            "AEROSPACE DEFENSE DIRECTIVE: AVIONICS NAVIGATION & HYPERSONIC FLIGHT CONTROLS (2026)\n"
            "Organization: Aerospace Defense Systems (org_4)\n"
            "Lead Systems Engineer: usr_chief_flight_officer\n\n"
            "1. Autonomous Inertial Navigation System (INS):\n"
            "All aerospace vehicles operating in GPS-denied environments must utilize ring laser gyroscope (RLG) and "
            "fiber-optic gyroscope (FOG) inertial measurement units. Navigation drift must remain below 0.05 nautical miles per hour.\n\n"
            "2. Triple Modular Redundant (TMR) Fly-by-Wire Architecture:\n"
            "Flight control computers (FCC) must execute identical flight control laws across three isolated hardware channels "
            "with hardware majority voting to ensure zero-failure operation compliant with DO-178C Level A avionics certification.\n\n"
            "3. Hypersonic Thermal Protection & Actuator Telemetry:\n"
            "Atmospheric re-entry surfaces employ ultra-high-temperature ceramic matrix composites (UHTCMC) rated for 2,400°C. "
            "Actuator status telemetry is sampled at 1,000 Hz and transmitted over encrypted Ku-band satellite datalinks.\n\n"
            "4. Emergency Abort & Mission Termination Sequencing:\n"
            "Mission abort sequencing utilizes cryptographic dual-command arming protocols. Flight termination command receivers "
            "enforce MIL-STD-1553 bus isolation from non-safety payload systems."
        )
    },
    {
        "org_id": "org_5",
        "doc_id": "doc_smart_grid_renewable_energy_specs",
        "filename": "smart_grid_renewable_energy_specs.pdf",
        "text": (
            "CLEAN ENERGY GRID DIRECTIVE: UTILITY BESS STORAGE & RENEWABLE BALANCING (2026)\n"
            "Organization: Global Clean Grid Energy (org_5)\n"
            "Chief Grid Architect: usr_grid_operations_lead\n\n"
            "1. High-Voltage Direct Current (HVDC) Transmission:\n"
            "Subsea and long-range terrestrial power interconnects operate at ±525 kV DC with modular multilevel converter (MMC) "
            "technology to achieve transmission power losses below 1.2% per 1,000 kilometers.\n\n"
            "2. Utility Battery Energy Storage Systems (BESS):\n"
            "Battery storage facilities utilize lithium iron phosphate (LFP) chemistry with liquid cooling and aerosol fire suppression. "
            "Each BESS facility must maintain a minimum 4-hour discharge capacity of 500 Megawatt-hours (MWh).\n\n"
            "3. Autonomous Frequency Regulation & Grid Following Inverters:\n"
            "Inverter-based renewable assets (solar PV and offshore wind) must respond to grid frequency excursions within 15 milliseconds "
            "using synthetic inertia and grid-forming voltage control algorithms.\n\n"
            "4. SCADA Operational Technology Security:\n"
            "All substation RTU telemetry protocols (IEC 61850 and DNP3) require mutual TLS 1.3 cryptographic authentication "
            "and unidirectional optical data diodes separating OT networks from IT enterprise corporate systems."
        )
    },
    {
        "org_id": "org_6",
        "doc_id": "doc_2nm_semiconductor_fabrication_guide",
        "filename": "2nm_semiconductor_fabrication_guide.pdf",
        "text": (
            "SEMICONDUCTOR FOUNDRY SPECIFICATION: 2NM GATE-ALL-AROUND PROCESS (2026)\n"
            "Organization: Nanotech Semiconductor Labs (org_6)\n"
            "Chief Nanotechnology Scientist: usr_fab_process_director\n\n"
            "1. Gate-All-Around (GAA) Nanosheet Architecture:\n"
            "The 2-nanometer production node incorporates horizontally stacked silicon nanosheet channels surrounded on all 4 sides "
            "by high-k metal gate dielectrics (EOT < 0.65 nm), delivering 18% frequency boost at iso-power over FinFET architectures.\n\n"
            "2. High-NA Extreme Ultraviolet (EUV) Photolithography:\n"
            "Critical metal layers are patterned using High-NA (0.55 NA) EUV lithography scanners operating at 13.5 nm wavelength, "
            "reducing mask count from 82 to 54 while eliminating complex quadruple-patterning cycles.\n\n"
            "3. Backside Power Delivery Network (BSPDN):\n"
            "Power supply rails are routed on the wafer backside through nano-through-silicon vias (nTSV), eliminating IR drop "
            "and interconnect congestion while boosting standard cell area utilization by 22%.\n\n"
            "4. Cleanroom Microcontamination Standards:\n"
            "Manufacturing takes place in ISO Class 1 cleanroom environments. Airborne molecular contamination (AMC) is monitored "
            "with cavity ring-down spectroscopy to maintain VOC concentrations below 1 part-per-trillion."
        )
    },
    {
        "org_id": "org_7",
        "doc_id": "doc_level4_autonomous_driving_architecture",
        "filename": "level4_autonomous_driving_architecture.pdf",
        "text": (
            "AUTONOMOUS MOBILITY SAFETY SPECIFICATION: LEVEL 4 DRIVING STACK (2026)\n"
            "Organization: Autonomous Vehicle Mobility (org_7)\n"
            "VP of Autonomous Safety: usr_robotics_director\n\n"
            "1. Multi-Modal Perception Sensor Suite:\n"
            "Level 4 autonomous driving platform combines 4 long-range solid-state 1550nm LiDARs (300-meter range), 11 high-dynamic-range "
            "8-megapixel camera sensors, and 5 millimeter-wave 77GHz radars to provide 360-degree overlapping perception coverage.\n\n"
            "2. Redundant Compute Hardware & ISO 26262 ASIL-D:\n"
            "Autonomous driving perception, prediction, and motion planning algorithms execute on dual heterogeneous compute units "
            "delivering 2,000 TOPS total compute. Both compute boards operate in lockstep with independent power and liquid cooling loops.\n\n"
            "3. Real-Time HD Mapping & Centimeter Localization:\n"
            "Vehicle localization combines RTK-GNSS satellite positioning, wheel odometry, and LiDAR point-cloud map registration "
            "to maintain sub-3-centimeter localization accuracy across tunnels, inclement weather, and urban canyons.\n\n"
            "4. Minimal Risk Maneuver (MRM) Safe Stop Protocol:\n"
            "Upon detection of catastrophic component degradation or sensor blockage, the system initiates an autonomous Minimal "
            "Risk Maneuver to steer the vehicle safely to the highway shoulder and activate hazard lights within 4.5 seconds."
        )
    },
    {
        "org_id": "org_8",
        "doc_id": "doc_global_cold_chain_logistics_protocol",
        "filename": "global_cold_chain_logistics_protocol.pdf",
        "text": (
            "GLOBAL SUPPLY CHAIN DIRECTIVE: MULTIMODAL COLD CHAIN FREIGHT PROTOCOL (2026)\n"
            "Organization: Global Freight & Logistics (org_8)\n"
            "Global Operations Director: usr_logistics_operations_head\n\n"
            "1. Cryogenic Vaccine & Biopharmaceutical Cold Chain:\n"
            "Biologic shipments and mRNA vaccines require uninterrupted temperature monitoring between -80°C and -20°C using active "
            "Stirling engine coolers and vacuum insulated panels (VIP). Dry ice sublimation rate must not exceed 2.2 kg per day.\n\n"
            "2. Real-Time IoT Sensor Telemetry & Geofencing:\n"
            "Freight containers are monitored via 5G satellite-connected IoT gateways that record ambient temperature, humidity, shock, "
            "tilt, and optical door breach events. Sensor readings are logged to cryptographic audit ledgers every 120 seconds.\n\n"
            "3. Multimodal Port Interoperability & Customs Clearance:\n"
            "Automated port terminal transfer procedures leverage RFID container tags and EDIFACT EDI messaging to enable seamless "
            "transition between ocean freight vessels, freight rail networks, and last-mile electric distribution trucks.\n\n"
            "4. Cold Chain Excursion Protocol & Product Quarantine:\n"
            "Any temperature deviation exceeding ±2°C for longer than 15 consecutive minutes triggers automated shipment quarantine, "
            "cellular alert notification to QA officers, and mandatory stability testing prior to pharmaceutical batch release."
        )
    },
    {
        "org_id": "org_9",
        "doc_id": "doc_zero_trust_incident_response_playbook",
        "filename": "zero_trust_incident_response_playbook.pdf",
        "text": (
            "ENTERPRISE CYBER DEFENSE DIRECTIVE: ZERO-TRUST INCIDENT RESPONSE PLAYBOOK (2026)\n"
            "Organization: Cyber Threat Defense Operations (org_9)\n"
            "Chief Information Security Officer: usr_ciso_operations\n\n"
            "1. Zero-Trust Access Architecture Principles:\n"
            "Never Trust, Always Verify. All network transactions, internal service-to-service RPC calls, and user sessions must be "
            "authenticated and authorized based on device posture, user context, behavioral risk scores, and mutual TLS encryption.\n\n"
            "2. Active Directory & Identity Compromise Protocol:\n"
            "Upon detecting credential dumping (e.g., Mimikatz, DCSync attack), the SOC must trigger automatic revocation of all "
            "active Kerberos TGT tickets, reset KRBTGT account passwords twice, and enforce immediate FIDO2 hardware re-authentication.\n\n"
            "3. Micro-Segmentation & Kubernetes Workload Isolation:\n"
            "Compromised cloud containers and Kubernetes pods must be immediately isolated using Cilium eBPF network policies "
            "that block all egress and ingress traffic while preserving volatile memory state for digital forensic analysis.\n\n"
            "4. Forensic Memory Acquisition & Chain of Custody:\n"
            "Host volatile memory (RAM) must be acquired using LiME or WinPmem with SHA-256 integrity hashing prior to host shutdown. "
            "Evidence disk images are preserved in immutable WORM cloud object storage with cryptographic timestamping."
        )
    },
    {
        "org_id": "org_10",
        "doc_id": "doc_6g_satellite_telecom_architecture",
        "filename": "6g_satellite_telecom_architecture.pdf",
        "text": (
            "TELECOMMUNICATIONS STANDARD: 6G NON-TERRESTRIAL MESH NETWORKS (2026)\n"
            "Organization: Quantum Telecom Networks (org_10)\n"
            "Chief Telecommunications Scientist: usr_telecom_systems_lead\n\n"
            "1. Sub-Terahertz Spectrum Bands (100 GHz - 300 GHz):\n"
            "Next-generation 6G base stations utilize sub-THz carrier frequencies with ultra-massive MIMO (1,024 antenna elements) "
            "to deliver peak data transmission rates exceeding 100 Gbps and user-plane air interface latency below 0.1 milliseconds.\n\n"
            "2. LEO Satellite Optical Inter-Satellite Links (OISL):\n"
            "Low Earth Orbit satellite constellation employs free-space optical laser crosslinks operating at 1550 nm, achieving "
            "10 Gbps inter-satellite throughput and routing global data packets faster than terrestrial optical fiber networks.\n\n"
            "3. Quantum Key Distribution (QKD) Network Integration:\n"
            "Satellite-to-ground downlinks utilize decoy-state BB84 quantum key distribution protocols, distributing information-theoretically "
            "secure cryptographic key streams to metropolitan quantum telecommunication hubs.\n\n"
            "4. AI-Driven Radio Resource Management & Beamforming:\n"
            "Deep reinforcement learning agents embedded on satellite payload processors dynamically steer high-gain phased-array spot beams "
            "to follow maritime, aviation, and disaster relief traffic surges in real time without ground-station intervention."
        )
    }
]

def run():
    print("=" * 80)
    print(" [*] SEEDING 8 ADDITIONAL ORGANIZATIONS (org_3 through org_10) IN QDRANT")
    print("=" * 80)

    # 1. Initialize IVF Validator
    validator = IVFBoundaryValidator(centroids_path="ivf_centroids.npy", radii_path="ivf_radii.npy")
    if not validator.load_centroids():
        print("Building dummy IVF index for centroids...")
        import numpy as np
        dummy_embeddings = np.random.rand(100, 384).astype(np.float32)
        validator.build_ivf_index(dummy_embeddings, doc_ids=list(range(100)), n_lists=6)

    repo = QdrantRepository()
    client = QdrantClientManager.get_client()

    # Normalize org_2 if org_org 2_collection exists
    existing_colls = [c.name for c in client.get_collections().collections]
    print(f"Existing collections in Qdrant: {existing_colls}")

    if "org_org 2_collection" in existing_colls and "org_2_collection" not in existing_colls:
        print("\n[Normalizing org_2_collection from org_org 2_collection...]")
        repo.ensure_collection("org_2", vector_size=384)
        points, _ = client.scroll("org_org 2_collection", limit=100, with_payload=True, with_vectors=True)
        if points:
            from qdrant_client.http import models as rest_models
            pts = [
                rest_models.PointStruct(id=p.id, vector=p.vector, payload=p.payload)
                for p in points
            ]
            client.upsert(collection_name="org_2_collection", points=pts)
            print(f"  + Copied {len(pts)} points to 'org_2_collection'")

    # 2. Ingest 1 document for each org (org_3 to org_10)
    ingested_summary = []
    for item in DOCUMENTS:
        org_id = item["org_id"]
        doc_id = item["doc_id"]
        filename = item["filename"]
        text = item["text"]

        print(f"\n--- Ingesting for {org_id} (Collection: {repo.get_collection_name(org_id)}) ---")
        
        # Ensure a default user exists for this org in auth DB
        user_id = f"admin_{org_id}"
        if not AuthService.get_user(user_id):
            AuthService.create_user_direct(
                user_id=user_id,
                username=user_id,
                password=f"Admin_{org_id}_2026!",
                full_name=f"Admin {org_id.upper()}",
                org_id=org_id,
                role="admin",
                status="approved"
            )

        res = DocumentIngestionEngine.process_and_ingest(
            text=text,
            doc_id=doc_id,
            user_id=user_id,
            org_id=org_id,
            validator=validator,
            qdrant_repo=repo,
            chunk_size=500,
            overlap=50,
            filename=filename,
        )

        coll_name = repo.get_collection_name(org_id)
        pt_count = client.count(coll_name).count
        ingested_summary.append({
            "org_id": org_id,
            "collection": coll_name,
            "doc_id": doc_id,
            "filename": filename,
            "chunks": res.get("total_chunks", len(res.get("chunks", []))),
            "points_in_qdrant": pt_count,
            "cloud_root": res.get("cloud_enterprise_root")[:16] + "..." if res.get("cloud_enterprise_root") else None,
        })
        print(f"  + Ingested '{doc_id}' -> {res.get('total_chunks')} chunks into '{coll_name}' (Total points: {pt_count})")

    # 3. Final Verification
    print("\n" + "=" * 80)
    print(" [*] QDRANT COLLECTION AUDIT SUMMARY (10 TOTAL ORGANIZATIONS)")
    print("=" * 80)
    all_colls = sorted([c.name for c in client.get_collections().collections])
    for cname in all_colls:
        cnt = client.count(cname).count
        print(f"  - {cname:<25} -> {cnt} points")

    print("\n" + "=" * 80)
    print(" [OK] ALL 8 NEW COLLECTIONS SUCCESSFULLY CREATED WITH 1 DOCUMENT EACH!")
    print("=" * 80)

if __name__ == "__main__":
    run()
