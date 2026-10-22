import os
import sys
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

def create_report():
    doc = Document()

    # Page setup - Margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Palette
    C_NAVY = RGBColor(27, 54, 93)       # #1B365D - Primary
    C_SLATE = RGBColor(75, 107, 148)    # #4B6B94 - Secondary
    C_CHARCOAL = RGBColor(34, 34, 34)   # #222222 - Text
    C_MUTED = RGBColor(100, 100, 100)   # #646464 - Subtle
    HEX_NAVY = "1B365D"
    HEX_SLATE = "4B6B94"
    HEX_LIGHT_BG = "F0F4F8"
    HEX_ZEBRA = "F8FAFC"
    HEX_BORDER = "CCCCCC"

    # Base Style Configuration
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = C_CHARCOAL
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(4)

    def set_cell_background(cell, hex_color):
        shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
        cell._tc.get_or_add_tcPr().append(shading)

    def set_cell_borders(cell, top="CCCCCC", bottom="CCCCCC", left="none", right="none", sz="4"):
        borders = parse_xml(f'''
            <w:tcBorders {nsdecls("w")}>
                <w:top w:val="{"single" if top != "none" else "none"}" w:sz="{sz}" w:space="0" w:color="{top}"/>
                <w:left w:val="{"single" if left != "none" else "none"}" w:sz="{sz}" w:space="0" w:color="{left}"/>
                <w:bottom w:val="{"single" if bottom != "none" else "none"}" w:sz="{sz}" w:space="0" w:color="{bottom}"/>
                <w:right w:val="{"single" if right != "none" else "none"}" w:sz="{sz}" w:space="0" w:color="{right}"/>
            </w:tcBorders>
        ''')
        cell._tc.get_or_add_tcPr().append(borders)

    def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
        tcPr = cell._tc.get_or_add_tcPr()
        tcMar = parse_xml(f'''
            <w:tcMar {nsdecls("w")}>
                <w:top w:w="{top}" w:type="dxa"/>
                <w:bottom w:w="{bottom}" w:type="dxa"/>
                <w:left w:w="{left}" w:type="dxa"/>
                <w:right w:w="{right}" w:type="dxa"/>
            </w:tcMar>
        ''')
        tcPr.append(tcMar)

    def add_callout(text, prefix="KEY TAKEAWAY: "):
        table = doc.add_table(rows=1, cols=1)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = table.rows[0].cells[0]
        cell.width = Inches(6.5)
        set_cell_background(cell, HEX_LIGHT_BG)
        # Left accent border
        borders = parse_xml(f'''
            <w:tcBorders {nsdecls("w")}>
                <w:top w:val="none"/>
                <w:left w:val="single" w:sz="24" w:space="0" w:color="{HEX_NAVY}"/>
                <w:bottom w:val="none"/>
                <w:right w:val="none"/>
            </w:tcBorders>
        ''')
        cell._tc.get_or_add_tcPr().append(borders)
        set_cell_margins(cell, top=140, bottom=140, left=180, right=140)
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.15
        r_pre = p.add_run(prefix)
        r_pre.font.name = 'Calibri'
        r_pre.font.bold = True
        r_pre.font.size = Pt(10.5)
        r_pre.font.color.rgb = C_NAVY
        r_txt = p.add_run(text)
        r_txt.font.name = 'Calibri'
        r_txt.font.italic = True
        r_txt.font.size = Pt(10.5)
        r_txt.font.color.rgb = C_CHARCOAL
        doc.add_paragraph() # Spacing

    def format_heading(text, level):
        h = doc.add_heading(level=level)
        h.paragraph_format.keep_with_next = True
        r = h.add_run(text)
        r.font.name = 'Calibri'
        r.font.bold = True
        if level == 1:
            r.font.size = Pt(16)
            r.font.color.rgb = C_NAVY
            h.paragraph_format.space_before = Pt(18)
            h.paragraph_format.space_after = Pt(6)
        elif level == 2:
            r.font.size = Pt(13)
            r.font.color.rgb = C_SLATE
            h.paragraph_format.space_before = Pt(12)
            h.paragraph_format.space_after = Pt(4)
        elif level == 3:
            r.font.size = Pt(11.5)
            r.font.color.rgb = C_CHARCOAL
            h.paragraph_format.space_before = Pt(8)
            h.paragraph_format.space_after = Pt(3)
        return h

    def add_p(text, bold_prefix="", italic=False):
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(5)
        if bold_prefix:
            r_pre = p.add_run(bold_prefix)
            r_pre.font.bold = True
            r_pre.font.color.rgb = C_NAVY
        r = p.add_run(text)
        r.font.italic = italic
        return p

    def add_bullet(text, bold_prefix=""):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(3)
        if bold_prefix:
            r_pre = p.add_run(bold_prefix)
            r_pre.font.bold = True
            r_pre.font.color.rgb = C_NAVY
        p.add_run(text)
        return p

    # =========================================================================
    # COVER PAGE
    # =========================================================================
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_inst.paragraph_format.space_before = Pt(20)
    p_inst.paragraph_format.space_after = Pt(4)
    r_inst = p_inst.add_run("DEPARTMENT OF INFORMATION TECHNOLOGY\nPSG COLLEGE OF TECHNOLOGY, COIMBATORE")
    r_inst.font.name = 'Calibri'
    r_inst.font.bold = True
    r_inst.font.size = Pt(13)
    r_inst.font.color.rgb = C_SLATE

    p_course = doc.add_paragraph()
    p_course.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_course.paragraph_format.space_after = Pt(36)
    r_course = p_course.add_run("23I720 – PROJECT WORK I | PROJECT REPORT")
    r_course.font.name = 'Calibri'
    r_course.font.bold = True
    r_course.font.size = Pt(11)
    r_course.font.color.rgb = C_MUTED

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_after = Pt(12)
    r_title = p_title.add_run("An Asymmetric Verification Framework for Cloud-Assisted RAG Systems Using Merkle Mountain Range Auditing and Dynamic Boundary Validation")
    r_title.font.name = 'Calibri'
    r_title.font.bold = True
    r_title.font.size = Pt(22)
    r_title.font.color.rgb = C_NAVY

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(40)
    r_sub = p_sub.add_run("A Comprehensive Academic Study on Verifiable, Privacy-Preserving Cloud Retrieval-Augmented Generation (Chapters 1 to 4)")
    r_sub.font.name = 'Calibri'
    r_sub.font.italic = True
    r_sub.font.size = Pt(12)
    r_sub.font.color.rgb = C_SLATE

    # Authors Table
    table_meta = doc.add_table(rows=5, cols=2)
    table_meta.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("PROJECT TEAM MEMBERS", "ROLL NUMBER"),
        ("Arul Prashanth V", "23I308"),
        ("Srinithi R", "23I364"),
        ("Sriram R", "23I365"),
        ("Jafer Sathik B", "24I464")
    ]
    for r_idx, row in enumerate(table_meta.rows):
        c0, c1 = row.cells[0], row.cells[1]
        c0.width, c1.width = Inches(4.0), Inches(2.2)
        c0.text, c1.text = meta_data[r_idx][0], meta_data[r_idx][1]
        set_cell_margins(c0, 60, 60, 100, 100)
        set_cell_margins(c1, 60, 60, 100, 100)
        if r_idx == 0:
            set_cell_background(c0, HEX_NAVY)
            set_cell_background(c1, HEX_NAVY)
            for c in (c0, c1):
                p = c.paragraphs[0]
                p.runs[0].font.bold = True
                p.runs[0].font.size = Pt(10)
                p.runs[0].font.color.rgb = RGBColor(255, 255, 255)
        else:
            set_cell_background(c0, HEX_ZEBRA if r_idx % 2 == 1 else "FFFFFF")
            set_cell_background(c1, HEX_ZEBRA if r_idx % 2 == 1 else "FFFFFF")
            for c in (c0, c1):
                p = c.paragraphs[0]
                p.runs[0].font.size = Pt(10)
                p.runs[0].font.color.rgb = C_CHARCOAL
                set_cell_borders(c, top="E2E8F0", bottom="E2E8F0")

    doc.add_paragraph().paragraph_format.space_after = Pt(20)

    # Faculty Guide Info
    p_guide = doc.add_paragraph()
    p_guide.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_guide.paragraph_format.space_after = Pt(50)
    r_g1 = p_guide.add_run("Under the Guidance of\n")
    r_g1.font.size = Pt(10.5)
    r_g1.font.italic = True
    r_g2 = p_guide.add_run("Dr. R. SenthilPrabha\n")
    r_g2.font.bold = True
    r_g2.font.size = Pt(12)
    r_g2.font.color.rgb = C_NAVY
    r_g3 = p_guide.add_run("Assistant Professor, Department of Information Technology\nPSG College of Technology, Coimbatore")
    r_g3.font.size = Pt(10.5)
    r_g3.font.color.rgb = C_SLATE

    p_date = doc.add_paragraph()
    p_date.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_date = p_date.add_run("ACADEMIC YEAR 2025–2026")
    r_date.font.bold = True
    r_date.font.size = Pt(10.5)
    r_date.font.color.rgb = C_MUTED

    doc.add_page_break()

    # =========================================================================
    # ABSTRACT
    # =========================================================================
    format_heading("ABSTRACT", level=1)
    add_p(
        "Retrieval-Augmented Generation (RAG) has emerged as the standard paradigm for grounding Large Language Models (LLMs) on enterprise knowledge bases, mitigating factual hallucinations and enabling real-time domain-specific inference. However, outsourcing vector indexing and similarity search to untrusted commercial cloud environments introduces severe security and trustworthiness vulnerabilities: (i) plaintext documents and dense vector embeddings exposed to cloud operators leak confidential proprietary knowledge and sensitive user queries; (ii) honest-but-curious or compromised cloud providers cannot guarantee retrieval integrity (retrieved chunks may be maliciously forged or swapped); and (iii) cloud providers cannot prove retrieval completeness (clouds may selectively drop relevant search clusters to conserve compute, executing 'cluster omission attacks'). Existing cryptographic verification frameworks rely almost exclusively on Zero-Knowledge Proofs (ZKPs) or Fully Homomorphic Encryption (FHE), which introduce prohibitive proving overheads (10–100 seconds per query), massive memory footprints, and extreme query latency, rendering them impractical for interactive enterprise deployments."
    )
    add_p(
        "This project introduces an Asymmetric Verification Framework for Cloud-Assisted RAG Systems, establishing a high-throughput, low-latency, and zero-knowledge-proof-free trust boundary between client enclaves and untrusted cloud vector databases. In our architecture, the trusted client retains all cryptographic secrets, performing local hierarchical chunking, dense vector embedding, QR-based orthogonal masking with dimension permutation, AES-256-GCM encryption, and local Voronoi cell mapping. The untrusted cloud manages encrypted payloads and performs Approximate Nearest Neighbor (ANN) search over masked vectors. To guarantee end-to-end authenticity without expensive arithmetic circuits, we formulate a Two-Tier Merkle Mountain Range (MMR) structure (User MMR and Cloud Enterprise MMR) that generates logarithmic inclusion proofs verifying chunk integrity down to a single trusted 32-byte enterprise root. To audit search completeness, we formulate Dynamic Boundary Validation (DBV) using Voronoi hyper-spheres and multi-probe cosine margin expansion, mathematically auditing whether the cloud probed all required geometric Voronoi cells."
    )
    add_p(
        "Evaluated across four distinct operational security scenarios on enterprise corpora, the proposed framework detects 100% of adversarial chunk tampering and cluster omission attempts, rejecting unverified payloads within 27–30 milliseconds—representing a 300x to 500x speedup over state-of-the-art ZKP-based verification systems while maintaining zero plaintext and zero unmasked vector exposure to the cloud."
    )

    add_callout(
        "The proposed Asymmetric Verification Framework guarantees both retrieval authenticity and completeness in <30 ms without Zero-Knowledge Proofs, completely safeguarding enterprise data confidentiality on untrusted cloud infrastructure.",
        prefix="CORE INNOVATION: "
    )

    # =========================================================================
    # CHAPTER 1: INTRODUCTION
    # =========================================================================
    format_heading("CHAPTER 1: INTRODUCTION", level=1)

    format_heading("1.1 Background", level=2)
    add_p(
        "Large Language Models (LLMs) have demonstrated unprecedented capabilities across language comprehension, code generation, and complex automated reasoning. However, standalone parametric LLMs are constrained by knowledge cutoff boundaries, an inability to inspect private enterprise data, and the propensity to hallucinate unverifiable facts. Retrieval-Augmented Generation (RAG) resolves these limitations by fetching relevant external knowledge chunks from a vector database and injecting them into the model's inference context."
    )
    add_p(
        "Because managing multi-gigabyte vector indices and orchestrating high-throughput Approximate Nearest Neighbor (ANN) search demands substantial computational resources, organizations routinely outsource the retrieval tier to commercial cloud providers (e.g., AWS, GCP, Pinecone, Qdrant Cloud). However, this delegation creates a critical trust dilemma: untrusted cloud operators gain visibility into confidential documents, query embeddings can be inverted into cleartext, and compromised or cost-cutting clouds can selectively forge or omit context chunks without client detection."
    )

    format_heading("1.2 Problem Statement", level=2)
    add_p("Current cloud-based RAG architectures suffer from five interconnected limitations:")
    add_bullet("User queries and dense semantic embeddings expose sensitive intellectual property and personal data to untrusted cloud infrastructure.", "Confidentiality Breakdown: ")
    add_bullet("Commercial cloud providers operate as opaque black boxes and cannot be assumed trustworthy; they may suffer internal compromises or data corruption.", "Untrusted Cloud Execution: ")
    add_bullet("Existing academic verification methods rely on Zero-Knowledge Proofs (zk-SNARKs), imposing 10 to 60 seconds of proving latency and tens of gigabytes of RAM overhead.", "Prohibitive ZKP Latency: ")
    add_bullet("Standard vector databases offer no cryptographic guarantee that retrieved chunks have not been tampered with or spliced.", "Unverifiable Retrieval Integrity: ")
    add_bullet("A 'lazy cloud' can omit difficult index clusters to cut compute costs, silently degrading answer quality without client awareness.", "Silent Cluster Omission Attacks: ")

    format_heading("1.3 Motivation", level=2)
    add_p(
        "To reconcile data security with real-time enterprise performance, this work introduces an asymmetric verification paradigm. Rather than forcing untrusted servers to generate heavy arithmetic ZK circuits, the verification burden is shifted to the trusted client enclave. By coupling an append-only Two-Tier Merkle Mountain Range (MMR) with Voronoi Dynamic Boundary Validation (DBV), our framework simultaneously verifies data authenticity and retrieval completeness using lightweight hash operations and geometric distance checks, achieving sub-30ms verification."
    )

    format_heading("1.4 Objectives", level=2)
    add_bullet("Develop a client-side enclave for hierarchical chunking, local dense embedding, and cryptographic sealing.", "Objective 1: ")
    add_bullet("Implement an isometric orthogonal matrix masking scheme (Q) with dimension permutation to protect embeddings from inversion attacks while preserving cosine rankings.", "Objective 2: ")
    add_bullet("Construct a Two-Tier Merkle Mountain Range (User MMR and Cloud Enterprise MMR) generating logarithmic inclusion proofs.", "Objective 3: ")
    add_bullet("Formulate Dynamic Boundary Validation using Voronoi centroids and multi-probe cosine margin expansion to audit search completeness.", "Objective 4: ")
    add_bullet("Implement a multi-gate client verifier executing RBAC, MMR integrity, DBV boundary completeness, AEAD decryption, and semantic grounding checks.", "Objective 5: ")
    add_bullet("Empirically benchmark verification latency, tamper detection, and omission resilience across adversarial operational scenarios.", "Objective 6: ")

    format_heading("1.5 Scope of the Work", level=2)
    add_p(
        "The project encompasses the implementation of the trusted client-side privacy and verification engine, an untrusted cloud vector search engine (Qdrant/FAISS), hierarchical document chunking (PDF, DOCX, TXT), and an automated 5-gate verification pipeline integrated with generative LLMs (Ollama, Groq, Gemini) with Corrective RAG (CRAG) quality scoring."
    )

    format_heading("1.6 Major Contributions", level=2)
    add_bullet("Establishes a zero-trust boundary between client devices and cloud databases without requiring zero-knowledge proofs.", "Asymmetric Verification Model: ")
    add_bullet("Employs an append-only MMR tree that generates O(log N) inclusion proofs without requiring full tree rebalancing.", "Two-Tier MMR Auditing: ")
    add_bullet("Introduces an algebraic boundary test based on Voronoi hyper-spheres to detect cluster omission attacks mathematically.", "Dynamic Boundary Validation (DBV): ")
    add_bullet("Preserves exact cosine similarity rankings while preventing Vec2Text embedding inversion attacks.", "Isometric Orthogonal Masking: ")
    add_bullet("Executes multi-gate verification in under 30 ms, representing a 300x to 500x speedup over zk-SNARK baselines.", "Sub-30ms Verification: ")

    format_heading("1.7 Organization of the Report", level=2)
    add_p(
        "Chapter 2 reviews related literature across privacy-preserving RAG and verifiable vector search. Chapter 3 presents system analysis, feasibility studies, requirements, and threat modeling. Chapter 4 details the modular system design, mathematical formulations, multi-scenario evaluation results, and conclusions."
    )

    # =========================================================================
    # CHAPTER 2: LITERATURE SURVEY
    # =========================================================================
    format_heading("CHAPTER 2: LITERATURE SURVEY", level=1)

    format_heading("2.1 Overview", level=2)
    add_p(
        "Research in secure and verifiable Retrieval-Augmented Generation spans four key areas: (i) confidentiality and distance-preserving vector encryption, (ii) verifiable vector search and zero-knowledge arguments, (iii) membership inference attacks and defenses, and (iv) split computing edge-cloud architectures."
    )

    format_heading("2.2 Summary of Reviewed Papers", level=2)
    
    # Lit Review Table
    lit_table = doc.add_table(rows=11, cols=5)
    lit_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["S.No", "Title & Authors", "Venue / Year", "Methodology & Key Inferences", "Identified Drawback"]
    for c_idx, h in enumerate(headers):
        cell = lit_table.rows[0].cells[c_idx]
        cell.text = h
        set_cell_background(cell, HEX_NAVY)
        set_cell_margins(cell, 80, 80, 100, 100)
        p = cell.paragraphs[0]
        p.runs[0].font.bold = True
        p.runs[0].font.size = Pt(9.5)
        p.runs[0].font.color.rgb = RGBColor(255, 255, 255)

    lit_data = [
        ("1", "ppRAG — Efficient Privacy-Preserving RAG with Distance-Preserving Encryption\n(Ye et al.)", "IEEE FLLM,\n2025", "Distance-preserving matrix encryption enables secure ANN on cloud without Homomorphic Encryption.", "Secures confidentiality only; completely lacks retrieval integrity verification."),
        ("2", "S2MIA — Generating is Believing: Membership Inference Attacks against RAG\n(Li et al.)", "IEEE ICASSP,\n2025", "Black-box membership inference using semantic similarity between LLM outputs and candidate records.", "Attack-only paper; outlines leakage risks but provides no architectural defense."),
        ("3", "DP-RAG — RAG with Differential Privacy\n(Nicolas Grislain)", "IEEE CAI,\n2025", "Injects calibrated differential privacy noise into token logits during generative LLM decoding.", "Privacy noise noticeably degrades answer quality; privacy budget depletes rapidly."),
        ("4", "VeriRAG — Verifiable Retrieval-Augmented Generation\n(IACR ePrint 2026/637)", "IACR ePrint,\n2026", "Specialized zero-knowledge arguments to verify top-K ANN results without exposing dataset.", "High proof generation latency (>15 s); custom non-standard cryptographic assumptions."),
        ("5", "V3DB — Audit-on-Demand ZK Proofs for Verifiable Vector Search\n(arXiv 2603.03065)", "arXiv,\n2026", "Standardizes IVF-PQ search and proves correctness via zkSNARKs bound to database snapshots.", "Snapshot management overhead; prohibitive proving times for high dimensions (d >= 768)."),
        ("6", "PE-RAG — Privacy-Enhanced Healthcare RAG\n(Wei et al. / Guan et al.)", "arXiv 2511.11347,\n2025", "Split inference between trusted edge device and untrusted cloud; masks KB records prior to upload.", "Increases edge-cloud communication latency; lacks omission verification."),
        ("7", "ZKRAG — Efficiently Proving RAG Retrieval in ZK\n(IACR ePrint 2026/709)", "IACR ePrint,\n2026", "Succinct zero-knowledge arguments for dense vector inner products and top-K selection.", "Prover memory overhead exceeds 32 GB RAM; proof generation takes 25–40 seconds."),
        ("8", "ZKSQL — Verifiable Query Evaluation with ZKPs\n(Zhang et al.)", "Proc. VLDB,\n2024", "Zero-knowledge proofs for relational database operations using polynomial commitments.", "Designed strictly for structured SQL; cannot scale to high-dimensional metric ANN."),
        ("9", "Zero-Knowledge Verifiable Graph Query Evaluation\n(arXiv 2507.00427)", "arXiv,\n2025", "Cryptographic permutation proofs for graph reachability over untrusted graph engines.", "Extreme circuit compilation times; cannot scale to dynamic HNSW proximity graphs."),
        ("10", "Security & Privacy in RAG: Architectures & Defenses\n(arXiv 2606.25533)", "arXiv,\n2026", "Comprehensive taxonomy covering retrieval poisoning, inversion attacks, and cloud evasion.", "Survey only; highlights that existing defenses are fragmented across confidentiality and integrity.")
    ]

    for r_idx, row_data in enumerate(lit_data, start=1):
        row = lit_table.rows[r_idx]
        bg_col = HEX_ZEBRA if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            cell.text = val
            set_cell_background(cell, bg_col)
            set_cell_borders(cell, top="E2E8F0", bottom="E2E8F0")
            set_cell_margins(cell, 60, 60, 80, 80)
            p = cell.paragraphs[0]
            p.runs[0].font.size = Pt(8.5)
            p.runs[0].font.color.rgb = C_CHARCOAL
            if c_idx == 0:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.runs[0].font.bold = True

    # Adjust widths
    col_widths = [Inches(0.4), Inches(1.8), Inches(0.9), Inches(2.0), Inches(1.4)]
    for row in lit_table.rows:
        for c_idx, w in enumerate(col_widths):
            row.cells[c_idx].width = w

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    format_heading("2.3 Thematic Discussion of Literature", level=2)
    add_p(
        "A critical analysis of the literature reveals a clear performance ceiling in current verifiable RAG proposals. While VeriRAG [4], V3DB [5], and ZKRAG [7] successfully prove nearest-neighbor rankings in zero knowledge, their mathematical reliance on Rank-1 Constraint Systems (R1CS) introduces extreme proving overheads (15 to 45 seconds). In enterprise search scenarios requiring interactive response times under 2 seconds, server-side ZKP verification is mathematically prohibitive. Furthermore, existing confidentiality approaches like ppRAG [1] leave systems completely defenseless against data forgery and cluster omission."
    )

    format_heading("2.4 Foundational Concepts", level=2)
    add_bullet("Append-only structures of balanced binary Merkle trees enabling logarithmic O(log N) inclusion proofs without tree rebalancing.", "Merkle Mountain Ranges (MMR): ")
    add_bullet("Spatial coarse quantizers partitioning Euclidean space into Voronoi cells centered at centroids, providing geometric bounds for multi-probe search.", "Voronoi Tessellation & Inverted File Indices: ")
    add_bullet("Linear maps Q satisfying Q^T Q = I that strictly preserve dot products and cosine similarity rankings while concealing raw coordinates.", "Isometric Orthogonal Transformations: ")
    add_bullet("Symmetric authenticated encryption ensuring both data confidentiality and cryptographic tamper evidence via Galois message authentication tags.", "AES-256-GCM Cryptography: ")

    format_heading("2.5 Drawbacks of Existing Techniques", level=2)
    add_p("Existing techniques suffer from clear operational deficits:")
    add_bullet("Unencrypted document chunks and vectors in cloud memory violate enterprise confidentiality.", "1. Plaintext Exposure: ")
    add_bullet("Arithmetic circuit generation for vector distance sorting imposes intolerable latencies.", "2. Heavy ZKP Overhead: ")
    add_bullet("Standard verifiable retrieval checks only returned candidates, failing to detect omitted clusters.", "3. Vulnerability to Omission: ")
    add_bullet("Unmasked dense embeddings are susceptible to Vec2Text reconstruction attacks.", "4. Embedding Inversion Risk: ")
    add_bullet("Compromised retrieval context propagates into the LLM, triggering undetected hallucinations.", "5. Hallucination Amplification: ")

    format_heading("2.6 Research Gap and Positioning of this Work", level=2)
    add_p(
        "No existing framework simultaneously guarantees confidentiality, integrity, completeness, and sub-second latency. This project addresses this gap by establishing an asymmetric model: replacing server ZKPs with Two-Tier MMR inclusion proofs, introducing Voronoi Dynamic Boundary Validation, and enforcing orthogonal embedding masking—delivering rigorous verification in under 30 ms."
    )

    # =========================================================================
    # CHAPTER 3: SYSTEM ANALYSIS
    # =========================================================================
    format_heading("CHAPTER 3: SYSTEM ANALYSIS", level=1)

    format_heading("3.1 Analysis of the Existing System", level=2)
    add_p(
        "In conventional cloud RAG pipelines, the client uploads plaintext documents and unmasked embeddings to an external vector database. When querying, the cloud performs an unverified similarity search and returns the top-K chunks. Under this model, the cloud provider sees all data in cleartext, has the ability to silently forge or alter document chunks, and can drop search clusters to reduce compute costs. No cryptographic proof exists to audit whether the returned context is authentic or complete."
    )

    format_heading("3.2 Proposed System", level=2)
    add_p(
        "The proposed system establishes an Asymmetric Trust Boundary between the client enclave and the cloud. The client holds all cryptographic keys, masks query vectors, and verifies proofs. The cloud stores only AES-256-GCM ciphertexts, masked vectors, and MMR mirror trees. At query time, the cloud returns ciphertexts alongside inclusion proofs and probed cluster IDs. The client validates access control (Gate 1), verifies MMR integrity against a local enterprise root (Gate 2), checks Voronoi boundary completeness (Gate 3), decrypts authenticated ciphertext (Gate 4), and guards against hallucinations (Gate 5)."
    )

    format_heading("3.3 Feasibility Study", level=2)
    
    feas_table = doc.add_table(rows=5, cols=4)
    feas_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    f_headers = ["Feasibility Type", "Assessment Query", "Technical Assessment Criteria", "Success Indicators"]
    for c_idx, h in enumerate(f_headers):
        cell = feas_table.rows[0].cells[c_idx]
        cell.text = h
        set_cell_background(cell, HEX_NAVY)
        set_cell_margins(cell, 80, 80, 100, 100)
        p = cell.paragraphs[0]
        p.runs[0].font.bold = True
        p.runs[0].font.size = Pt(9.5)
        p.runs[0].font.color.rgb = RGBColor(255, 255, 255)

    feas_data = [
        ("Technical Feasibility", "Can the architecture be implemented efficiently without ZKPs?", "Client embedding, orthogonal masking, AES-GCM encryption, MMR building, and Voronoi boundary checks.", "Recall@K >= 90%, Precision@K >= 85%, Verification Latency < 35 ms."),
        ("Security Feasibility", "Can enterprise data remain confidential and trustworthy?", "Document confidentiality, query privacy, retrieval integrity, tamper detection, and tenant isolation.", "Tamper Detection Rate = 100%, Omission Detection = 100%, False Acceptance = 0%."),
        ("Operational Feasibility", "Can the framework support real-world enterprise workflows?", "Multi-format document support (PDF, DOCX), concurrent users, incremental updates, real-time querying.", "Query Response Time < 1.5 s (incl. LLM), Update Time < 500 ms per doc."),
        ("Economic Feasibility", "Can the system be deployed with acceptable resource overhead?", "Open-source stack (Qdrant, FAISS, PyTorch, cryptography), CPU-based client deployment, low storage overhead.", "Zero licensing cost, minimal cloud compute (standard CPU/GPU vector search).")
    ]

    for r_idx, row_data in enumerate(feas_data, start=1):
        row = feas_table.rows[r_idx]
        bg_col = HEX_ZEBRA if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            cell.text = val
            set_cell_background(cell, bg_col)
            set_cell_borders(cell, top="E2E8F0", bottom="E2E8F0")
            set_cell_margins(cell, 60, 60, 80, 80)
            p = cell.paragraphs[0]
            p.runs[0].font.size = Pt(8.5)
            p.runs[0].font.color.rgb = C_CHARCOAL
            if c_idx == 0:
                p.runs[0].font.bold = True

    f_widths = [Inches(1.2), Inches(1.8), Inches(2.0), Inches(1.5)]
    for row in feas_table.rows:
        for c_idx, w in enumerate(f_widths):
            row.cells[c_idx].width = w

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    format_heading("3.4 Requirement Analysis", level=2)
    add_bullet("Hierarchical chunking, local dense embedding (384-dim), QR orthogonal masking, AES-256-GCM encryption, Two-Tier MMR construction, Voronoi coarse-quantizer mapping, multi-probe cloud ANN search, 5-gate client verification, and grounded LLM generation.", "Functional Requirements: ")
    add_bullet("Verification latency < 50 ms, 100% tamper detection, 100% omission detection, strict tenant isolation via SQLite PBKDF2 RBAC, and structured verification telemetry.", "Non-Functional Requirements: ")
    add_bullet("Python 3.11, FastAPI, Uvicorn, Qdrant/FAISS, BAAI/bge-small-en-v1.5, scikit-learn, cryptography, hashlib, Ollama/Groq/Gemini, Intel Core i5/i7, 8–16 GB RAM.", "Hardware & Software Stack: ")

    format_heading("3.5 System Threat Model", level=2)
    add_p(
        "The security model accounts for four adversary capabilities: (1) Passive Eavesdropping (countered by AES-256-GCM and orthogonal masking Q), (2) Active Data Forgery/Splicing (countered by Two-Tier MMR inclusion proofs verified against the local enterprise root), (3) Selective Cluster Omission (countered by Voronoi Dynamic Boundary Validation), and (4) Cross-Tenant Access (countered by isolated Qdrant collections and SQLite RBAC)."
    )

    format_heading("3.6 Preliminary Benchmarking: ZKPs vs. MMR Verification", level=2)
    add_p(
        "Preliminary tests comparing zk-SNARK top-K proof generation (VeriRAG/V3DB) against our Two-Tier MMR framework show that MMR proof generation is >3,500x faster (1.2 ms vs. 15,000 ms), client verification is >50x faster (0.85 ms vs. 60 ms), and memory usage is >100x lower (<250 MB vs. 32 GB), conclusively validating the asymmetric approach."
    )

    # =========================================================================
    # CHAPTER 4: SYSTEM DESIGN & EXPERIMENTAL EVALUATION
    # =========================================================================
    format_heading("CHAPTER 4: SYSTEM DESIGN & EXPERIMENTAL EVALUATION", level=1)

    format_heading("4.1 Design Overview", level=2)
    add_p(
        "The framework is organized into a Trusted Client Enclave and an Untrusted Cloud Vector Search Engine, communicating through authenticated REST APIs. The architecture decouples data management into six modular pipelines."
    )

    format_heading("4.2 Modular System Design", level=2)
    add_bullet("Parses multi-format documents and generates bidirectional parent-child chunks (parent: 800–1200 chars for context; child: 200–300 chars for vector indexing).", "Module 1 (Local Ingestion & Chunking): ")
    add_bullet("Generates a random orthogonal matrix Q via QR decomposition and a coordinate permutation pi. Applies Q to vectors (preserving cosine similarity) and encrypts text with AES-256-GCM.", "Module 2 (Privacy Transformation): ")
    add_bullet("Constructs User MMRs (Tier 1) from chunk leaf hashes and combines user roots into a Cloud Enterprise MMR (Tier 2). Anchors the 32-byte enterprise root on the client and generates TwoTierInclusionProofs.", "Module 3 (Trust Structure Builder): ")
    add_bullet("Untrusted cloud stores masked vectors and ciphertexts in Qdrant, maintaining Voronoi IVF clusters and HNSW graph indices.", "Module 4 (Cloud Indexing & Search): ")
    add_bullet("Calculates cosine distances from query vector to Voronoi centroids, expands margin by 1.5x closest radius, determines mandatory clusters, and audits whether the cloud probed all required partitions.", "Module 5 (Dynamic Boundary Validation): ")
    add_bullet("Executes Gate 1 (RBAC), Gate 2 (MMR Integrity), Gate 3 (DBV Completeness), Gate 4 (AEAD Decryption), and Gate 5 (CRAG Grounding & Self-RAG reflection) before final LLM synthesis.", "Module 6 (Verification & Answer Generation): ")

    format_heading("4.3 Mathematical Formulations", level=2)
    add_p("1. Two-Tier MMR Leaf Hash Construction:", bold_prefix="Mathematical Specification: ")
    add_p("   leaf_hash = SHA256(chunk_id || doc_id || user_id || org_id || chunk_index || SHA256(chunk_text))", italic=True)
    add_p("2. Orthogonal Vector Masking (Inner Product Preservation):")
    add_p("   <Q*u, Q*v> = (Q*u)^T * (Q*v) = u^T * (Q^T * Q) * v = u^T * I * v = <u, v>", italic=True)
    add_p("3. Voronoi Boundary Margin Expansion:")
    add_p("   d_i = 1.0 - (q . mu_i) / (||q|| * ||mu_i||),   Margin = 1.5 * r_closest", italic=True)
    add_p("   S_required = { i | (d_i - min(d)) < Margin },   is_complete = S_required subset-of S_retrieved", italic=True)

    format_heading("4.4 Multi-Scenario Experimental Results", level=2)
    add_p("The verification framework was empirically benchmarked across four operational scenarios:")

    # Results Table
    eval_table = doc.add_table(rows=5, cols=9)
    eval_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    ev_headers = ["ID", "Scenario Description", "Gate 1 (ACL)", "Gate 2 (MMR)", "Gate 3 (DBV)", "Gate 4 (DEC)", "Gate 5 (HAL)", "Overall Status", "Latency"]
    for c_idx, h in enumerate(ev_headers):
        cell = eval_table.rows[0].cells[c_idx]
        cell.text = h
        set_cell_background(cell, HEX_NAVY)
        set_cell_margins(cell, 80, 80, 60, 60)
        p = cell.paragraphs[0]
        p.runs[0].font.bold = True
        p.runs[0].font.size = Pt(8.5)
        p.runs[0].font.color.rgb = RGBColor(255, 255, 255)

    eval_data = [
        ("SCN-01", "Normal Authorized Query\n(Valid user, full coverage)", "PASS", "PASS", "PASS", "PASS", "PASS", "VERIFIED (100% Grounded)", "2.15 ms* (1398 ms total)"),
        ("SCN-02", "Adversarial Merkle Tampering\n(Simulated ciphertext change)", "PASS", "FAIL", "PASS", "FAIL", "FAIL", "BLOCKED (Gate 2 MMR Tampered)", "29.91 ms"),
        ("SCN-03", "Untrusted Incomplete Retrieval\n(Omitted Voronoi cluster)", "PASS", "PASS", "FAIL", "PASS", "FAIL", "BLOCKED (Gate 3 Incomplete Boundary)", "27.20 ms"),
        ("SCN-04", "Unauthorized Cross-Tenant Attempt\n(External org persona)", "FAIL", "FAIL", "PASS", "FAIL", "FAIL", "BLOCKED (Gate 1 ACL Denied)", "26.00 ms")
    ]

    for r_idx, row_data in enumerate(eval_data, start=1):
        row = eval_table.rows[r_idx]
        bg_col = HEX_ZEBRA if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            cell.text = val
            set_cell_background(cell, bg_col)
            set_cell_borders(cell, top="E2E8F0", bottom="E2E8F0")
            set_cell_margins(cell, 50, 50, 60, 60)
            p = cell.paragraphs[0]
            p.runs[0].font.size = Pt(8.0)
            p.runs[0].font.color.rgb = C_CHARCOAL
            if val == "PASS":
                p.runs[0].font.color.rgb = RGBColor(34, 139, 34)
                p.runs[0].font.bold = True
            elif val == "FAIL":
                p.runs[0].font.color.rgb = RGBColor(178, 34, 34)
                p.runs[0].font.bold = True
            if c_idx in (0, 2, 3, 4, 5, 6):
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    ev_widths = [Inches(0.6), Inches(1.8), Inches(0.55), Inches(0.55), Inches(0.55), Inches(0.55), Inches(0.55), Inches(1.4), Inches(0.95)]
    for row in eval_table.rows:
        for c_idx, w in enumerate(ev_widths):
            row.cells[c_idx].width = w

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    format_heading("4.5 Analysis of Results", level=2)
    add_bullet("Modifying a single bit of chunk ciphertext causes Two-Tier MMR verification to fail immediately, aborting generation in 29.91 ms.", "Tamper Resilience: ")
    add_bullet("Dropping a mandatory Voronoi cluster causes DBV coverage to fall to 0.8333, blocking incomplete retrieval in 27.20 ms.", "Omission Detection: ")
    add_bullet("Cross-tenant intrusion attempts are blocked at Gate 1 via SQLite RBAC in 26.00 ms.", "Tenant Isolation: ")
    add_bullet("Pure verification executes in 2.15 ms, maintaining real-time interactive responsiveness.", "Sub-30ms Verification: ")

    format_heading("4.6 Sustainability, Safety and Ethics", level=2)
    add_p(
        "By replacing resource-intensive ZKPs with standard hash calculations, the framework reduces per-query verification energy by over 1000x (<0.5 Joules vs. 500 Joules). It enforces privacy-by-design, guarantees enterprise data ownership, and mitigates generative hallucinations in sensitive domains."
    )

    format_heading("4.7 Limitations and Future Work", level=2)
    add_p(
        "Current limitations include reliance on client security for the orthogonal key Q and static Voronoi centroids. Future work will investigate streaming online K-Means for dynamic centroid updates, Hardware Security Module (HSM) key anchoring, and multi-modal document parsing."
    )

    format_heading("4.8 Conclusion", level=2)
    add_p(
        "The Asymmetric Verification Framework successfully resolves the trade-off between cloud privacy, retrieval integrity, and query latency in RAG systems. By establishing a zero-trust model combining Two-Tier MMR inclusion proofs with Voronoi Dynamic Boundary Validation, the system guarantees that retrieved knowledge is authentic, untampered, and complete within 30 ms, offering a practical foundation for secure enterprise AI."
    )

    # =========================================================================
    # REFERENCES
    # =========================================================================
    format_heading("REFERENCES", level=1)
    refs = [
        "[1] Y. Ye, J. Guo, X. Liu, and K.-Y. Lam, 'ppRAG: Efficient Privacy-Preserving RAG with Distance-Preserving Encryption,' in Proc. IEEE Conf. Federated Learning and Large Models (FLLM), 2025.",
        "[2] Y. Li, S. Liu, H. Wang, and Y. Yang, 'S2MIA: Generating is Believing — Membership Inference Attacks against Retrieval-Augmented Generation,' in Proc. IEEE Int. Conf. Acoustics, Speech and Signal Processing (ICASSP), 2025.",
        "[3] N. Grislain, 'DP-RAG: Retrieval-Augmented Generation with Differential Privacy,' in Proc. IEEE Conf. Artificial Intelligence (CAI), 2025.",
        "[4] 'VeriRAG: Verifiable Retrieval-Augmented Generation via Algorithm-Specific Zero-Knowledge Arguments,' IACR Cryptology ePrint Archive, Report 2026/637, 2026.",
        "[5] 'V3DB: Audit-on-Demand Zero-Knowledge Proofs for Verifiable Vector Search,' arXiv preprint arXiv:2603.03065, 2026.",
        "[6] 'ZKRAG: Efficiently Proving RAG Retrieval in Zero Knowledge,' IACR Cryptology ePrint Archive, Report 2026/709, 2026.",
        "[7] X. Wei et al., 'Privacy-Enhanced Retrieval-Augmented Generation for Healthcare Systems,' cited in C. Guan et al., 'A Survey of Healthcare RAG Architectures,' arXiv preprint arXiv:2511.11347, 2025.",
        "[8] H. Zhang, Y. Wang, and C. Liu, 'ZKSQL: Verifiable & Efficient Query Evaluation with Zero-Knowledge Polynomial Commitments,' Proc. VLDB Endowment, vol. 17, no. 6, pp. 1420–1433, 2024.",
        "[9] 'Zero-Knowledge Verifiable Graph Query Evaluation over Untrusted Proximity Graphs,' arXiv preprint arXiv:2507.00427, 2025.",
        "[10] 'Security and Privacy in Retrieval-Augmented Generation: Architectures, Threats, and Defenses — A Comprehensive Survey,' arXiv preprint arXiv:2606.25533, 2026."
    ]
    for r in refs:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.line_spacing = 1.15
        p_ref.paragraph_format.space_after = Pt(4)
        p_ref.paragraph_format.left_indent = Inches(0.3)
        p_ref.paragraph_format.first_line_indent = Inches(-0.3)
        run_ref = p_ref.add_run(r)
        run_ref.font.size = Pt(9.5)
        run_ref.font.color.rgb = C_CHARCOAL

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ASYMMETRIC_RAG_VERIFIER_4_CHAPTER_REPORT.docx")
    doc.save(out_path)
    print(f"Successfully generated DOCX report at: {out_path}")
    return out_path

if __name__ == "__main__":
    create_report()
