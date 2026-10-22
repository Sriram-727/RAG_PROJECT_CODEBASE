# Unified RAG Verifier

This project provides a unified architecture for Retrieval-Augmented Generation (RAG) verification. It combines Qdrant vector retrieval, Dynamic Boundary Validation (DBV), two-tier Merkle Mountain Range (MMR) cryptographic proofs, and semantic hallucination detection.

## Prerequisites

Make sure you have Python 3.9+ installed on your machine.

## Setup and Installation

1. **Install Dependencies:**
   Run the following command to install all required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the API Server:**
   Start the FastAPI server using Uvicorn. Run this from the `unified_rag_verifier` directory:
   ```bash
   uvicorn main:app --host 127.0.0.1 --port 8000 --reload
   ```

3. **Explore the API:**
   Once the server is running, you can access the interactive Swagger UI documentation in your browser at:
   [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

   From there, you can easily test the `/api/v1/query-verify` endpoint.

4. **Access the Interactive Web Dashboard:**
   Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser to view the interactive verification dashboard with one-click test attack simulations and live metrics.

## Demo & Architecture Documentation

For a complete word-for-word demo presentation script, module-by-module code walkthrough, and architectural diagrams, please see:
- 📖 [COMPLETE_DEMO_AND_EXPLANATION_GUIDE.md](file:///c:/Users/srira/.gemini/antigravity-ide/scratch/unified_rag_verifier/COMPLETE_DEMO_AND_EXPLANATION_GUIDE.md)
- 📄 [PROJECT_PRESENTATION_DOCUMENT.md](file:///c:/Users/srira/.gemini/antigravity-ide/scratch/unified_rag_verifier/PROJECT_PRESENTATION_DOCUMENT.md)
- 🌐 [PROJECT_PRESENTATION_DOCUMENT.html](file:///c:/Users/srira/.gemini/antigravity-ide/scratch/unified_rag_verifier/PROJECT_PRESENTATION_DOCUMENT.html)
- 📝 [PROJECT_PRESENTATION_DOCUMENT.docx](file:///c:/Users/srira/.gemini/antigravity-ide/scratch/unified_rag_verifier/PROJECT_PRESENTATION_DOCUMENT.docx)

## Running the Automated Test

If you want to run a quick automated integration test that spins up the server, submits a test query, and prints the result, you can use the included test script (make sure the Uvicorn server is *not* already running on port 8000 when you do this):

```bash
python test_query.py
```

