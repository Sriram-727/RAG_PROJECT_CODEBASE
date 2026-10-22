"""
LLM Generation & Response Synthesis Service.
Generates concise, factual, and strictly grounded answers from verified context chunks.
Supports:
1. External LLM APIs (Gemini, OpenAI) when API keys are configured.
2. Intelligent local extractive/abstractive synthesizer for offline/zero-cost operation.
   Specifically recognizes one-line summary requests, question answering, and topic synthesis.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass
import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from config.settings import get_settings

logger = logging.getLogger("LLMService")

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no",
    "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll",
    "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
    "the", "their", "theirs", "them", "themselves", "then", "there", "there's",
    "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when",
    "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why",
    "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're",
    "you've", "your", "yours", "yourself", "yourselves",
    "tell", "show", "give", "list", "explain", "describe", "find", "name", "names",
    "know", "use", "uses", "used", "please", "only", "all",
    "resume", "cv", "document", "documents", "doc", "docs", "pdf", "pdfs", "file", "files",
    "profile", "page", "pages", "text", "info", "information", "details", "mentioned", "listed", "given"
}

ACTION_KEYWORDS = {
    "mandate", "mandates", "mandated", "require", "requires", "required", "must",
    "enforce", "enforces", "enforced", "policy", "security", "directive", "provides",
    "ensures", "features", "governs", "focuses", "regulates", "restricts", "prohibits",
    "establishes", "specifies", "implements", "protects", "authorizes", "expires"
}


def fuzzy_token_match(w1: str, w2: str, threshold: float = 0.8) -> bool:
    """Fuzzy/typo-tolerant match between two words (e.g. frontent -> frontend)."""
    w1 = w1.lower()
    w2 = w2.lower()
    if w1 == w2:
        return True
    if len(w1) >= 4 and len(w2) >= 4:
        if w1[:4] == w2[:4]:
            return True
        if difflib.SequenceMatcher(None, w1, w2).ratio() >= threshold:
            return True
    return False


@dataclass
class QueryIntent:
    """Analyzes user query for semantic positive intent, negative exclusions, and formatting constraints."""
    raw_query: str
    positive_text: str
    positive_tokens: List[str]
    negative_tokens: Set[str]
    is_one_line: bool
    is_list_only: bool

    @classmethod
    def from_query(cls, query: str) -> "QueryIntent":
        neg_pats = [
            r"\b(?:not|no|without|except|excluding|except\s+for|other\s+than|aside\s+from|besides|instead\s+of|apart\s+from)\s+([^,.;\n]+)",
        ]
        negative_tokens: Set[str] = set()
        pos_text = query
        for pat in neg_pats:
            for m in re.finditer(pat, query, re.I):
                phrase = m.group(1).strip()
                for w in re.findall(r"[a-zA-Z0-9]+", phrase.lower()):
                    if len(w) > 1 and w not in STOPWORDS and w not in {"and", "or", "nor", "with"}:
                        negative_tokens.add(w)
                pos_text = pos_text.replace(m.group(0), " ")

        pos_cleaned = " ".join(pos_text.split())
        positive_tokens = [
            w for w in re.findall(r"[a-zA-Z0-9]+", pos_cleaned.lower())
            if len(w) > 1 and w not in STOPWORDS
        ]

        is_one_line = bool(
            re.search(r"\b(?:one|1|single)\s*(?:line|sentence)\b", query, re.I)
            or re.search(r"in\s+(?:one|1|a\s+single)\s+(?:line|sentence)", query, re.I)
        )
        is_list_only = bool(
            re.search(r"\b(?:list\s+only|only\s+(?:the\s+)?(?:names?|titles?)|just\s+(?:the\s+)?(?:names?|titles?))\b", query, re.I)
        )

        return cls(
            raw_query=query,
            positive_text=pos_cleaned,
            positive_tokens=positive_tokens,
            negative_tokens=negative_tokens,
            is_one_line=is_one_line,
            is_list_only=is_list_only,
        )


class LLMService:
    """Enterprise RAG response generator with real API & smart local synthesizer."""

    @classmethod
    def generate_response(
        cls,
        query: str,
        retrieved_docs: List[Union[Dict[str, Any], str]],
        max_tokens: int = 300,
    ) -> str:
        """
        Generate a verified response based on retrieved, decrypted context chunks.
        Adheres strictly to user instructions (e.g. 'summarize in one line').
        """
        # 1. Normalize and order retrieved texts by document flow
        # If chunks belong to the same document, restore natural reading order (by chunk_index)
        docs_copy = list(retrieved_docs)
        if any(isinstance(d, dict) and "chunk_index" in d for d in docs_copy):
            docs_copy.sort(
                key=lambda d: (d.get("doc_id", ""), d.get("chunk_index", 0))
                if isinstance(d, dict)
                else ("", 0)
            )

        # Sort docs by chunk_index if available to preserve natural document flow
        docs_sorted = sorted(
            docs_copy,
            key=lambda d: d.get("chunk_index", 0) if isinstance(d, dict) else 0
        )
        chunk_texts: List[str] = []
        for doc in docs_sorted:
            if isinstance(doc, dict):
                text = doc.get("text", "")
            else:
                text = str(doc)
            cleaned = text.strip()
            if cleaned:
                chunk_texts.append(cleaned)

        if not chunk_texts:
            return "Unable to provide a verified answer due to missing or empty context."

        corpus = "\n\n".join(chunk_texts)

        # 2. Attempt real LLM generation if configured
        settings = get_settings()
        api_response = cls._try_external_llm(query, corpus, settings, max_tokens)
        if api_response:
            return api_response

        # 3. Fall back to smart local synthesizer
        return cls._local_synthesize(query, corpus, chunk_texts)

    @classmethod
    def _try_external_llm(
        cls, query: str, context: str, settings: Any, max_tokens: int
    ) -> Optional[str]:
        """Try calling Gemini or OpenAI API if keys are present in environment or settings."""
        gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")
        custom_key = getattr(settings, "LLM_API_KEY", None)

        is_one_line = cls._is_one_line_request(query)
        one_line_instruction = (
            "Provide strictly EXACTLY ONE concise, informative sentence on a single line. "
            "Do NOT include multiple sentences, bullet points, or newlines."
            if is_one_line
            else "Provide a clear, concise, direct response based only on the context."
        )

        system_prompt = (
            f"You are a factual enterprise assistant in a zero-trust verifiable RAG system. "
            f"Answer the user query strictly based on the provided verified context. "
            f"{one_line_instruction} "
            f"Do not invent facts or extrapolate beyond the text. "
            f"Do not include conversational filler like 'Here is the summary:' or 'Sure!'."
        )

        # 1. Gemini
        if gemini_key or (custom_key and custom_key.startswith("AIza")):
            key_to_use = gemini_key or custom_key
            model = getattr(settings, "LLM_MODEL", "gemini-1.5-flash")
            if not model.startswith("gemini"):
                model = "gemini-1.5-flash"
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key_to_use}"
                payload = {
                    "contents": [{
                        "parts": [{
                            "text": f"{system_prompt}\n\nContext:\n{context}\n\nUser Query:\n{query}"
                        }]
                    }],
                    "generationConfig": {
                        "temperature": 0.2,
                        "maxOutputTokens": 150 if is_one_line else max_tokens,
                    }
                }
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=8.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    if is_one_line:
                        text = cls._enforce_single_line(text)
                    return text
            except Exception as e:
                logger.warning(f"Gemini API generation failed ({e}), falling back to local synthesizer.")

        # 2. OpenAI
        if openai_key or (custom_key and custom_key.startswith("sk-")):
            key_to_use = openai_key or custom_key
            model = getattr(settings, "LLM_MODEL", "gpt-4o-mini")
            try:
                url = "https://api.openai.com/v1/chat/completions"
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuery:\n{query}"},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 150 if is_one_line else max_tokens,
                }
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {key_to_use}",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=8.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    text = data["choices"][0]["message"]["content"].strip()
                    if is_one_line:
                        text = cls._enforce_single_line(text)
                    return text
            except Exception as e:
                logger.warning(f"OpenAI API generation failed ({e}), falling back to local synthesizer.")

        return None

    @classmethod
    def _is_one_line_request(cls, query: str) -> bool:
        """Check if user explicitly asked for a one-line / single sentence summary."""
        q = query.lower()
        patterns = [
            r"one\s*-?\s*line",
            r"1\s*-?\s*line",
            r"single\s+line",
            r"one\s+sentence",
            r"1\s+sentence",
            r"single\s+sentence",
            r"in\s+one\s+line",
            r"in\s+1\s+line",
            r"in\s+a\s+single\s+line",
            r"in\s+a\s+single\s+sentence",
        ]
        for pat in patterns:
            if re.search(pat, q):
                return True
        return False

    @classmethod
    def _is_summary_request(cls, query: str) -> bool:
        """Check if user query is a general summary request."""
        q = query.lower()
        return bool(
            re.search(r"\b(summariz|summary|summarise|tl;dr|tldr|overview|brief|abstract)\b", q)
            or "what is this document about" in q
            or "what does this document say" in q
            or "describe this document" in q
        )

    @classmethod
    def _enforce_single_line(cls, text: str) -> str:
        """Normalize text into strictly a single line sentence without multiple paragraphs."""
        # Split on newlines and take the first substantive line
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        first_line = lines[0] if lines else text.strip()

        # If multiple sentences in first line, take the first complete sentence
        sentences = re.split(r"(?<=[.!?])\s+", first_line)
        result = sentences[0].strip() if sentences else first_line
        result = " ".join(result.split())
        if not result.endswith((".", "!", "?")):
            result += "."
        return result

    @classmethod
    def _local_synthesize(cls, query: str, corpus: str, chunk_texts: List[str]) -> str:
        """
        Intelligent local synthesizer for offline / zero-cost operation.
        Dispatches to:
        - QueryIntent-aware section extraction (e.g. projects, education, skills with negative exclusions)
        - Targeted question answering
        - One-line summary
        - General summary
        - Grounded synthesis
        """
        intent = QueryIntent.from_query(query)

        # 1. Section-Specific Extraction (Projects, Education, Experience, Skills, Company, Candidate)
        section_ans = cls._extract_section_answer(query, corpus, intent=intent)
        if section_ans:
            return section_ans

        # 2. Targeted Question Answering (e.g. starts with What, When, How, Why, Who, Where, Can, Is, Does, Explain...)
        q_clean = intent.positive_text.strip()
        is_question = bool(
            q_clean.endswith("?")
            or re.match(r"^(what|when|how|where|why|which|who|can|is|are|does|do|explain|list|show|give|tell)\b", q_clean, re.I)
            or re.match(r"^(what|when|how|where|why|which|who|can|is|are|does|do|explain|list|show|give|tell)\b", query.strip(), re.I)
        )
        if is_question:
            answer = cls._answer_question_from_context(query, corpus, is_one_line=intent.is_one_line, intent=intent)
            if answer:
                return answer

        # 2b. Zero-Hallucination Guard: If positive salient terms are completely absent from document, report truthfully
        corpus_words = set(re.findall(r"[a-zA-Z0-9]+", corpus.lower()))
        matched_salient = [
            w for w in intent.positive_tokens
            if any(fuzzy_token_match(w, cw) for cw in corpus_words)
        ]
        if intent.positive_tokens and not matched_salient:
            topic_str = " ".join(intent.positive_tokens)
            return f"The provided verified documents do not contain information regarding '{topic_str}'."

        # 3. One-Line Summary Request (Document-level)
        if intent.is_one_line:
            return cls._synthesize_one_line_summary(corpus)

        # 4. General Summary Request (Document-level)
        if cls._is_summary_request(query):
            return cls._synthesize_general_summary(corpus)

        # 5. General Fallback: Return top 1-2 most informative relevant sentences
        return cls._synthesize_concise_response(query, corpus, intent=intent)

    @classmethod
    def _extract_section_answer(cls, query: str, corpus: str, intent: Optional[QueryIntent] = None) -> Optional[str]:
        """
        Extract structured items from recognized document sections (e.g. Projects, Education, Experience, Skills)
        when the query explicitly asks for a section (e.g. 'tell frontend languages', 'list projects in resume').
        Applies fuzzy keyword matching and filters out any categories/items matching negative constraints.
        """
        if intent is None:
            intent = QueryIntent.from_query(query)

        q = query
        q_low = query.lower()
        pos_q = intent.positive_text.lower()
        section_map = {
            "company": [
                "company information", "company info", "company details", "about company", "about the company",
                "hiring company", "company name", "company", "role", "job role", "compensation", "stipend",
                "package", "work mode", "job location", "application date"
            ],
            "candidate": [
                "candidate profile", "candidate details", "candidate info", "student profile", "student details",
                "student name", "candidate name", "applicant name", "cgpa", "gpa", "year of study", "college name",
                "candidate", "student", "applicant", "author", "whose resume", "who is", "who's", "candidate's", "name"
            ],
            "interview": [
                "interview process", "interview rounds", "selection process", "interview experience", "rounds",
                "technical rounds", "online assessment", "hr round", "overall experience", "overall rating",
                "final result", "preparation time", "interview tips"
            ],
            "project": ["project", "projects", "personal project", "personal projects", "academic project", "academic projects"],
            "education": ["education", "degree", "university", "college", "school", "gpa", "cgpa", "academic", "qualifications", "hsc", "sslc"],
            "experience": ["work experience", "work history", "employment", "internships", "internship", "experience", "job", "career"],
            "skill": [
                "technical skills", "tech stack", "programming languages", "app development", "mobile development",
                "frameworks", "libraries", "technologies", "databases", "database", "frontend", "backend",
                "skills", "skill", "tools", "stacks", "tech", "languages"
            ],
            "certification": ["certifications", "certification", "license", "licenses", "credentials", "certified"],
            "achievement": ["achievements", "achievement", "awards", "honors", "winner"],
            "publication": ["publications", "publication", "paper", "papers"],
            "interest": ["areas of interest", "area of interest", "interests", "interest"],
            "contact": ["contact information", "contact info", "contact", "email", "phone", "mobile", "address", "location", "github", "linkedin", "leetcode", "social"],
            "objective": ["career objective", "career objectives", "career summary", "profile summary", "about me", "objective", "objectives", "summary", "goal", "goals"],
        }

        target_section = None
        best_score = 0.0
        for sec_name, keywords in section_map.items():
            sc = 0.0
            for kw in keywords:
                if re.search(r"\b" + re.escape(kw) + r"\b", pos_q):
                    sc += max(len(kw), 3) * 3.0
                kw_tokens = kw.split()
                for pt in intent.positive_tokens:
                    if any(fuzzy_token_match(pt, kt) for kt in kw_tokens):
                        sc += 2.0
            if sc > best_score and sc >= 3.0:
                best_score = sc
                target_section = sec_name

        if not target_section:
            return None

        is_one_line = intent.is_one_line

        # Handle direct contact request from header lines
        if target_section == "contact":
            email_m = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", corpus)
            phone_m = re.search(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,5}[-.\s]?\d{4,5}", corpus)
            github_m = re.search(r"github\.com/[\w\-]+/[\w\-]+|github\.com/[\w\-]+|gtb([\w\-]+)", corpus, re.I)
            location_m = re.search(r"([A-Za-z]+,\s*India)", corpus)

            contact_map: Dict[str, str] = {}
            if email_m:
                contact_map["email"] = f"Email: {email_m.group(0)}"
            if phone_m:
                contact_map["phone"] = f"Phone: {phone_m.group(0).strip()}"
            if location_m:
                contact_map["location"] = f"Location: {location_m.group(1)}"
            if github_m:
                gh = github_m.group(0).replace("gtb", "github.com/")
                contact_map["github"] = f"GitHub: {gh}"

            # Filter out negative constraints
            filtered_contact = []
            for k, val in contact_map.items():
                if any(fuzzy_token_match(neg, k) for neg in intent.negative_tokens):
                    continue
                filtered_contact.append(val)

            # Check if specific contact field requested
            specific = []
            for k, val in contact_map.items():
                if any(fuzzy_token_match(pos, k) for pos in intent.positive_tokens):
                    specific.append(val)

            items_to_return = specific if specific else filtered_contact
            if not items_to_return:
                return None
            if is_one_line:
                return "Contact: " + ", ".join(items_to_return) + "."
            return "The contact details identified in the document are:\n" + "\n".join([f"- {c}" for c in items_to_return])

        # Handle direct candidate / student name request from resume top line
        if target_section == "candidate" and any(k in pos_q for k in ["name", "student", "candidate", "who is", "who's", "whose", "who"]):
            first_lines = [l.strip() for l in corpus.splitlines() if l.strip()]
            for cand_line in first_lines[:5]:
                cand_clean = re.sub(r"^[-#*•\s]+", "", cand_line).strip()
                if (
                    1 <= len(cand_clean.split()) <= 4
                    and not any(k in cand_clean.lower() for k in ["objective", "education", "experience", "resume", "curriculum", "page", "section", "contact", "email", "phone", "http", ".com"])
                    and not re.search(r"[:@\d]", cand_clean)
                ):
                    return f"The candidate identified in the document is {cand_clean.title()}."

        all_major_sections = [
            r"SUMMARY", r"PROFESSIONAL SUMMARY", r"CAREER OBJECTIVES?", r"OBJECTIVES?", r"PROFILE", r"ABOUT ME",
            r"AREAS OF INTEREST", r"CERTIFICATIONS?", r"ACHIEVEMENTS?", r"PROJECTS?", r"TECHNICAL SKILLS", r"SKILLS",
            r"EDUCATION", r"EXPERIENCE", r"WORK HISTORY", r"WORK EXPERIENCE", r"PUBLICATIONS?", r"AWARDS?",
            r"INTERESTS?", r"DECLARATION", r"REFERENCES",
            r"COMPANY INFORMATION", r"COMPANY DETAILS", r"CANDIDATE PROFILE", r"CANDIDATE DETAILS",
            r"INTERVIEW PROCESS", r"INTERVIEW ROUNDS", r"OVERALL EXPERIENCE SUMMARY"
        ]

        if target_section == "company":
            sec_header_pattern = r"company\s+(?:information|details|info|profile)|about\s+company"
        elif target_section == "candidate":
            sec_header_pattern = r"candidate\s+(?:profile|details|info)|student\s+(?:profile|details)|applicant\s+(?:profile|details)"
        elif target_section == "interview":
            sec_header_pattern = r"interview\s+(?:process|rounds?|experience)|selection\s+process|overall\s+experience\s+summary"
        elif target_section == "skill":
            sec_header_pattern = r"(?:technical\s+)?(?:skills?|technologies|tech\s+stack)"
        elif target_section == "education":
            sec_header_pattern = r"(?:academic\s+)?(?:education|qualifications?)"
        elif target_section == "certification":
            sec_header_pattern = r"certifications?|licenses?"
        elif target_section == "achievement":
            sec_header_pattern = r"achievements?|awards?|honors?"
        elif target_section == "interest":
            sec_header_pattern = r"areas\s+of\s+interest|interests?"
        elif target_section == "experience":
            sec_header_pattern = r"(?:work\s+)?experience|work\s+history"
        elif target_section == "project":
            sec_header_pattern = r"projects?"
        elif target_section == "objective":
            sec_header_pattern = r"(?:career\s+)?objectives?|about\s+me|summary"
        else:
            sec_header_pattern = target_section + r"(?:s)?"

        filtered_major = [s for s in all_major_sections if not re.search(sec_header_pattern, s, re.I)]
        major_sections = "|".join(filtered_major)

        # Require section header to start at beginning of string or line (with optional markdown/list bullets)
        sec_regex = re.compile(
            r"(?:^|\n)\s*(?:[#*-]*\s*)?(?:\d+[\.\)]\s*)?(?:key\s+|academic\s+|personal\s+|technical\s+|core\s+)?\b("
            + sec_header_pattern
            + r")\b\s*[:.\n-]*\s*\n*(.*?)(?=(?:\n\s*(?:[#*-]*\s*)?(?:"
            + major_sections
            + r")\b\s*[:\n]|\Z))",
            re.IGNORECASE | re.DOTALL,
        )

        m = sec_regex.search(corpus)
        if not m:
            if target_section == "company":
                m = re.search(r"(?:^|\n)\s*(?:Company\s+Information|About\s+the\s+Company)\b\s*[:.\n-]+\s*(.*?)(?=(?:\n\s*(?:Candidate|Interview|Projects|#|\Z)))", corpus, re.I | re.DOTALL)
            elif target_section == "candidate":
                m = re.search(r"(?:^|\n)\s*(?:Candidate\s+Profile|Student\s+Profile)\b\s*[:.\n-]+\s*(.*?)(?=(?:\n\s*(?:Interview|Projects|Overall|#|\Z)))", corpus, re.I | re.DOTALL)
            elif target_section == "interview":
                m = re.search(r"(?:^|\n)\s*(?:Interview\s+Process|Overall\s+Experience\s+Summary)\b\s*[:.\n-]+\s*(.*?)(?=(?:\n\s*(?:#|Tags|\Z)))", corpus, re.I | re.DOTALL)
            elif target_section == "project":
                m = re.search(r"(?:^|\n)\s*Projects?\b\s*[:.\n-]*\s*(.*?)(?=(?:\n\s*(?:Areas\s+of\s+Interest|Technical\s+Skills|Skills|Education|Experience|Interview|Round|#|\Z)))", corpus, re.I | re.DOTALL)
            elif target_section == "objective":
                m = re.search(r"(?:^|\n)\s*(?:career\s+)?objectives?\b\s*[:.\n-]*\s*(.*?)(?=(?:\n\s*(?:Education|Technical\s+Skills|Skills|Projects|Experience|Areas\s+of\s+Interest|#|\Z)))", corpus, re.I | re.DOTALL)

        if not m:
            return None

        content = m.group(2).strip() if (m.lastindex and m.lastindex >= 2) else (m.group(m.lastindex).strip() if m.lastindex else m.group(0).strip())
        raw_lines = [
            l.strip() for l in content.splitlines()
            if l.strip()
            and not l.strip().lower().startswith("generated on")
            and not re.match(r"^(?:om/|[a-z0-9_-]+/)[a-zA-Z0-9_\.-]+", l.strip())
            and not re.match(r"^(" + sec_header_pattern + r")\b", l.strip(), re.I)
        ]
        if not raw_lines:
            return None

        # 0a. COMPANY INFORMATION
        if target_section == "company":
            company_specs = [
                ("Company", r"^company\s*[:\-]?\s+(.+)$"),
                ("Role", r"^role\s*[:\-]?\s+(.+)$"),
                ("Type", r"^type\s*[:\-]?\s+(.+)$"),
                ("Duration", r"^duration\s*[:\-]?\s+(.+)$"),
                ("Work Mode", r"^work\s+mode\s*[:\-]?\s+(.+)$"),
                ("Location", r"^location\s*[:\-]?\s+(.+)$"),
                ("Compensation", r"^(?:compensation|stipend|salary|package)\s*[:\-]?\s+(.+)$"),
                ("Application Date", r"^application\s+date\s*[:\-]?\s+(.+)$"),
            ]
            parsed = {}
            for line in raw_lines:
                for label, pat in company_specs:
                    m_f = re.match(pat, line, re.I)
                    if m_f:
                        val = m_f.group(1).strip()
                        if label == "Duration" and val.isdigit():
                            val = f"{val} months"
                        parsed[label] = val
                        break

            if not parsed:
                clean_lines = [l for l in raw_lines if not re.match(r"^(?:field\s+details|aspect\s+details)$", l, re.I)]
                if clean_lines:
                    text_block = " ".join(clean_lines)
                    return text_block if is_one_line else f"Company Information:\n{text_block}"
                return None

            q_low = query.lower()
            if "company name" in q_low or "which company" in q_low or "name of the company" in q_low or q_low.strip() in ["company", "what company"]:
                return f"The company is {parsed.get('Company', '')}."
            if re.search(r"\b(?:role|position|job\s+role)\b", q_low) and not any(k in q_low for k in ["all", "info", "detail", "give", "list"]):
                role_val = parsed.get("Role", "")
                comp_val = parsed.get("Company", "")
                return f"The role is {role_val} at {comp_val}." if comp_val else f"The role is {role_val}."
            if re.search(r"\b(?:compensation|stipend|salary|package)\b", q_low) and not any(k in q_low for k in ["all", "info", "detail", "give"]):
                if parsed.get("Compensation"):
                    return f"The compensation is {parsed.get('Compensation')}."
            if re.search(r"\b(?:work\s+mode)\b", q_low):
                if parsed.get("Work Mode"):
                    return f"The work mode is {parsed.get('Work Mode')}."
            if re.search(r"\b(?:job\s+location)\b", q_low):
                if parsed.get("Location"):
                    return f"The job location is {parsed.get('Location')}."

            if is_one_line:
                summary_items = [f"{k}: {v}" for k, v in parsed.items() if k in ["Role", "Type", "Location", "Compensation"]]
                comp_name = parsed.get("Company", "")
                prefix = f"Company Information: {comp_name}" if comp_name else "Company Information"
                return f"{prefix} ({', '.join(summary_items)})."

            formatted = [f"- {k}: {v}" for k, v in parsed.items()]
            return "The company information identified in the document is:\n" + "\n".join(formatted)

        # 0b. CANDIDATE PROFILE
        if target_section == "candidate":
            candidate_specs = [
                ("Student Name", r"^(?:student\s+name|candidate\s+name|applicant\s+name|name)\s*[:\-]?\s+(.+)$"),
                ("Email", r"^email\s*[:\-]?\s+(.+)$"),
                ("University", r"^(?:university|college|institution)\s*[:\-]?\s+(.+)$"),
                ("Year of Study", r"^(?:y\s*ear\s+of\s+study|year\s+of\s+study|year)\s*[:\-]?\s+(.+)$"),
                ("CGPA", r"^(?:cgpa|gpa)\s*[:\-]?\s+(.+)$"),
                ("Technical Skills", r"^(?:technical\s+skills?|skills?)\s*[:\-]?\s+(.+)$"),
            ]
            parsed = {}
            for line in raw_lines:
                for label, pat in candidate_specs:
                    m_f = re.match(pat, line, re.I)
                    if m_f:
                        val = m_f.group(1).strip()
                        if label == "Year of Study":
                            val = val.replace("Y ear", "Year")
                        parsed[label] = val
                        break

            if not parsed:
                clean_lines = [l for l in raw_lines if not re.match(r"^(?:field\s+details|aspect\s+details)$", l, re.I)]
                if clean_lines:
                    text_block = " ".join(clean_lines)
                    return text_block if is_one_line else f"Candidate Profile:\n{text_block}"
                return None

            q_low = query.lower()
            if "cgpa" in q_low or "gpa" in q_low:
                return f"The candidate's CGPA is {parsed.get('CGPA', '')}."
            if "student name" in q_low or "candidate name" in q_low or "applicant" in q_low:
                return f"The student name is {parsed.get('Student Name', '')}."
            if "university" in q_low or "college" in q_low:
                return f"The university is {parsed.get('University', '')}."

            if is_one_line:
                cand_name = parsed.get("Student Name", "")
                univ = parsed.get("University", "")
                cgpa = parsed.get("CGPA", "")
                return f"Candidate Profile: {cand_name} ({univ}, CGPA: {cgpa})."

            formatted = [f"- {k}: {v}" for k, v in parsed.items()]
            return "The candidate profile identified in the document is:\n" + "\n".join(formatted)

        # 0c. INTERVIEW PROCESS & ROUNDS
        if target_section == "interview":
            q_low = query.lower()
            # 1. Overall Experience Summary check
            if any(k in q_low for k in ["overall", "rating", "final result", "preparation time", "recommend", "tips"]):
                sec_match_ov = re.search(
                    r"Overall\s+Experience\s+Summary\s*(.*?)(?=(?:\n\s*Tags|\Z))",
                    corpus,
                    re.IGNORECASE | re.DOTALL,
                )
                if sec_match_ov:
                    ov_lines = [l.strip() for l in sec_match_ov.group(1).splitlines() if l.strip() and not l.strip().lower().startswith("generated on")]
                    summary_specs = [
                        ("Overall Rating", r"^overall\s+rating\s*[:\-]?\s+(.+)$"),
                        ("Final Result", r"^(?:final\s+result|result)\s*[:\-]?\s+(.+)$"),
                        ("Would Recommend", r"^would\s+recommend\s*[:\-]?\s+(.+)$"),
                        ("Preparation Time", r"^preparation\s+time\s*[:\-]?\s+(.+)$"),
                        ("Resources Used", r"^resources\s+used\s*[:\-]?\s+(.+)$"),
                        ("Total Rounds", r"^total\s+rounds\s*[:\-]?\s+(.+)$"),
                    ]
                    ov_parsed = {}
                    for line in ov_lines:
                        for label, pat in summary_specs:
                            m_s = re.match(pat, line, re.I)
                            if m_s:
                                ov_parsed[label] = m_s.group(1).strip()
                                break
                    if ov_parsed:
                        if "rating" in q_low:
                            return f"The overall rating is {ov_parsed.get('Overall Rating', '')}."
                        if "result" in q_low:
                            return f"The final result is {ov_parsed.get('Final Result', '')}."
                        if "preparation time" in q_low:
                            return f"The preparation time was {ov_parsed.get('Preparation Time', '')}."
                        if is_one_line:
                            items = [f"{k}: {v}" for k, v in ov_parsed.items()]
                            return f"Overall Experience: {', '.join(items)}."
                        formatted = [f"• {k}: {v}" for k, v in ov_parsed.items()]
                        return "The overall experience summary is:\n" + "\n".join(formatted)

            # 2. Extract Rounds
            round_matches = re.findall(
                r"(Round\s+\d+[:\s]+[^\n]+)(?:\n\s*Duration\s+([^\n]+))?(?:\n\s*Platform\s+([^\n]+))?",
                corpus,
                re.IGNORECASE,
            )
            if round_matches:
                rounds_list = []
                for r_name, dur, plat in round_matches:
                    entry = r_name.strip()
                    details = []
                    if dur and not dur.lower().startswith("topics"):
                        details.append(f"Duration: {dur.strip()}")
                    if plat and not plat.lower().startswith("result"):
                        details.append(f"Platform: {plat.strip()}")
                    if details:
                        entry += f" ({', '.join(details)})"
                    if entry not in rounds_list:
                        rounds_list.append(entry)
                if is_one_line:
                    return "Interview Rounds: " + ", ".join([r.split(" (")[0] for r in rounds_list]) + "."
                return "The interview process consisted of the following rounds:\n" + "\n".join([f"• {r}" for r in rounds_list])

        # 1. PROJECTS
        if target_section == "project":
            ignore_prefixes = (
                "tech stack", "technologies", "tools", "github", "git", "role",
                "duration", "link", "demo", "live", "built", "developed", "designed",
                "implemented", "architected", "created", "engineered", "integrated",
                "maintained", "used", "frontend", "backend", "enabled", "automated",
                "managed", "configured", "led", "trained", "optimized", "collaborated",
                "facilitated", "delivered", "with", "authentication", "admin", "postgres",
                "java", "python", "html", "css", "react", "node", "express", "sql"
            )

            project_names: List[str] = []

            # Check for GitHub project repository names in content and corpus (handling split chunk URLs)
            gh_repos = re.findall(r"(?:github\.com/[^/\s]+/(?:repos/)?|/[A-Za-z0-9_-]+/)([A-Za-z0-9]+(?:-[A-Za-z0-9]+)+)", corpus, re.I)
            gh_project_names = [
                " ".join(w.capitalize() for w in re.sub(r"[-_]", " ", repo).split())
                for repo in gh_repos
            ]

            def _is_dup(cand: str, existing_list: List[str]) -> bool:
                cand_w = set(re.findall(r"[a-zA-Z0-9]+", cand.lower()))
                for ex in existing_list:
                    ex_w = set(re.findall(r"[a-zA-Z0-9]+", ex.lower()))
                    if len(cand_w & ex_w) >= 2 or cand.lower() in ex.lower() or ex.lower() in cand.lower():
                        return True
                return False

            for line in raw_lines:
                cleaned = re.sub(r"^[-*•\s]+", "", line).strip()
                c_lower = cleaned.lower()

                # Skip bullet points, git links, continuation lines
                if any(c_lower.startswith(pfx) for pfx in ignore_prefixes) or "github.com" in c_lower:
                    continue
                if cleaned.endswith(",") or cleaned.endswith("."):
                    continue
                if not cleaned or not (cleaned[0].isupper() or cleaned[0].isdigit()):
                    continue

                num_match = re.match(r"^\d+[\.\)]\s*(.+)", cleaned)
                if num_match:
                    full_p = num_match.group(1).strip()
                    p_name = full_p.split(":")[0].strip() if ":" in full_p else full_p
                    if p_name and not _is_dup(p_name, project_names):
                        project_names.append(p_name)
                elif 3 <= len(cleaned) <= 80:
                    tech_match = re.search(r"^(.*?)\s+(?:java|python|postgresql|sql|html|css|react|node|flutter|bash|c\+\+|aws|docker|kubernetes)\b", cleaned, re.I)
                    if tech_match:
                        cand = tech_match.group(1).strip()
                        if cand and len(cand) >= 3 and not any(cand.lower().startswith(pfx) for pfx in ignore_prefixes):
                            if not _is_dup(cand, project_names):
                                project_names.append(cand)
                    elif any(gh.lower() in cleaned.lower() for gh in gh_project_names):
                        for gh in gh_project_names:
                            if gh.lower() in cleaned.lower() and not _is_dup(gh, project_names):
                                project_names.append(gh)
                    elif 2 <= len(cleaned.split()) <= 6 and not any(pfx in c_lower for pfx in ["technologies", "certification", "udemy"]):
                        p_name = cleaned.split(":")[0].strip() if ":" in cleaned else cleaned
                        if p_name and not _is_dup(p_name, project_names):
                            project_names.append(p_name)

            # If no project names were cleanly parsed, fallback to GitHub-derived names or merge
            if not project_names and gh_project_names:
                project_names = gh_project_names
            elif gh_project_names:
                for gh in gh_project_names:
                    if not _is_dup(gh, project_names):
                        project_names.append(gh)

            if not project_names:
                return None

            # Filter projects by negative constraints and score by positive intent
            scored_projects: List[Tuple[float, str]] = []
            for p_name in project_names:
                p_tokens = [w.lower() for w in re.findall(r"[a-zA-Z0-9]+", p_name)]
                is_excluded = any(
                    any(fuzzy_token_match(neg, tok) for tok in p_tokens)
                    for neg in intent.negative_tokens
                )
                if is_excluded:
                    continue

                sc = 1.0
                for pos in intent.positive_tokens:
                    if any(fuzzy_token_match(pos, tok) for tok in p_tokens):
                        sc += 5.0
                scored_projects.append((sc, p_name))

            if not scored_projects:
                return "No matching projects found after applying exclusions."

            max_p_score = max(s for s, _ in scored_projects)
            if max_p_score > 3.0 and len(scored_projects) > 1 and not any(k in pos_q for k in ["all", "list", "names"]):
                final_projects = [p for s, p in scored_projects if s >= max_p_score * 0.8]
            else:
                final_projects = [p for _, p in scored_projects]

            if intent.is_one_line:
                if len(final_projects) == 1:
                    return f"The project listed in the resume is {final_projects[0]}."
                elif len(final_projects) == 2:
                    return f"The projects listed in the resume are {final_projects[0]} and {final_projects[1]}."
                else:
                    return f"The projects listed in the resume are {', '.join(final_projects[:-1])}, and {final_projects[-1]}."
            else:
                formatted_lines = [
                    f"{i+1}. {name}" for i, name in enumerate(final_projects[:5])
                ]
                return "The projects identified in the document are:\n" + "\n".join(formatted_lines)

        # 2. SKILLS (with subcategories like Frontend, Backend, Database, App Development)
        if target_section == "skill":
            sub_cat_patterns = [
                ("Frontend", r"^\s*frontend\s*[:.\-]?$"),
                ("Backend", r"^\s*backend\s*[:.\-]?$"),
                ("Database", r"^\s*databases?\s*[:.\-]?$"),
                ("App Development", r"^\s*(?:app\s+development|mobile(?:\s+development)?)\s*[:.\-]?$"),
                ("Programming Languages", r"^\s*(?:programming\s+languages?|languages?)\s*[:.\-]?$"),
                ("Tools", r"^\s*(?:tools|developer\s+tools)\s*[:.\-]?$"),
                ("Frameworks & Libraries", r"^\s*(?:frameworks?|libraries)\s*[:.\-]?$"),
                ("Cloud & DevOps", r"^\s*(?:cloud|devops)\s*[:.\-]?$"),
            ]

            categories: Dict[str, List[str]] = {}
            current_cat = "General"

            for line in raw_lines:
                clean_l = re.sub(r"[\U00010000-\U0010ffff]", "", line)
                matched = None
                for cat_name, pat in sub_cat_patterns:
                    if re.search(pat, clean_l, re.I):
                        matched = cat_name
                        break
                if matched:
                    current_cat = matched
                    if current_cat not in categories:
                        categories[current_cat] = []
                else:
                    if "," in clean_l:
                        parts = [p.strip() for p in clean_l.split(",") if p.strip()]
                    else:
                        parts = [p.strip() for p in clean_l.split() if p.strip()]
                    if current_cat not in categories:
                        categories[current_cat] = []
                    for p in parts:
                        cleaned_p = re.sub(r"^[-*•\s]+", "", p).strip()
                        if cleaned_p and cleaned_p not in categories[current_cat]:
                            categories[current_cat].append(cleaned_p)

            # Score each category against positive and negative intent
            scored_cats: Dict[str, float] = {}
            for cat, items in categories.items():
                if not items:
                    continue
                cat_tokens = [w.lower() for w in re.findall(r"[a-zA-Z0-9]+", cat)]
                all_tokens = cat_tokens + [w.lower() for it in items for w in re.findall(r"[a-zA-Z0-9]+", it)]

                # Check negative constraints: if any negative token matches this category or its items, exclude it
                is_excluded = any(
                    any(fuzzy_token_match(neg, tok) for tok in all_tokens)
                    for neg in intent.negative_tokens
                )
                if is_excluded:
                    continue

                sc = 0.0
                for pos in intent.positive_tokens:
                    # Category name match (e.g. "frontent" matches "frontend")
                    if any(fuzzy_token_match(pos, ct) for ct in cat_tokens):
                        sc += 10.0
                    # Item match (e.g. "react" matches "React.js")
                    elif any(fuzzy_token_match(pos, tok) for tok in all_tokens):
                        sc += 4.0
                scored_cats[cat] = sc

            if not scored_cats:
                return "No matching technical skills found after applying exclusions."

            max_score = max(scored_cats.values())
            if max_score > 0:
                # Specific categories requested
                selected_cats = [c for c, s in scored_cats.items() if s >= max_score * 0.6]
            else:
                # General skills requested or all categories remaining after negation
                selected_cats = list(scored_cats.keys())

            if len(selected_cats) == 1:
                cat_name = selected_cats[0]
                items = categories[cat_name]
                if intent.is_one_line:
                    if len(items) == 1:
                        return f"The {cat_name.lower()} technologies identified in the document are {items[0]}."
                    elif len(items) == 2:
                        return f"The {cat_name.lower()} technologies identified in the document are {items[0]} and {items[1]}."
                    else:
                        joined = ", ".join(items[:-1])
                        return f"The {cat_name.lower()} technologies identified in the document are {joined}, and {items[-1]}."
                else:
                    formatted = [f"{i+1}. {it}" for i, it in enumerate(items)]
                    return f"The {cat_name.lower()} languages and technologies identified in the document are:\n" + "\n".join(formatted)

            # Multiple categories selected
            if intent.is_one_line:
                cat_summaries = [f"{cat} ({', '.join(categories[cat])})" for cat in selected_cats if categories.get(cat)]
                if len(cat_summaries) == 1:
                    return f"The technical skills identified in the document are {cat_summaries[0]}."
                joined = ", ".join(cat_summaries[:-1])
                return f"The technical skills identified in the document are {joined}, and {cat_summaries[-1]}."
            else:
                formatted_cats = [f"- {cat}: {', '.join(categories[cat])}" for cat in selected_cats if categories.get(cat)]
                return "The technical skills identified in the document are:\n" + "\n".join(formatted_cats)

        # 3. EDUCATION
        if target_section == "education":
            edu_entries: List[str] = []
            cur_parts: List[str] = []
            for line in raw_lines:
                clean_l = re.sub(r"[\U00010000-\U0010ffff]", "", line)
                c = re.sub(r"^[-*•\s]+", "", clean_l).strip()
                if not c:
                    continue
                if any(k in c.lower() for k in ["b.t", "b.e", "b.s", "m.t", "m.s", "hsc", "sslc", "bachelor", "master", "10th", "12th"]):
                    if cur_parts:
                        edu_entries.append(" - ".join(cur_parts))
                        cur_parts = []
                cur_parts.append(c)
            if cur_parts:
                edu_entries.append(" - ".join(cur_parts))

            dedup_entries = []
            seen = set()
            for e in edu_entries:
                key = e[:20].lower()
                if key not in seen:
                    e_tokens = [w.lower() for w in re.findall(r"[a-zA-Z0-9]+", e)]
                    if any(any(fuzzy_token_match(neg, tok) for tok in e_tokens) for neg in intent.negative_tokens):
                        continue
                    seen.add(key)
                    dedup_entries.append(e)

            if not dedup_entries:
                dedup_entries = [l for l in raw_lines if len(l) >= 8][:3]

            if intent.is_one_line:
                return f"Education: {dedup_entries[0]}."
            formatted = [f"{i+1}. {e}" for i, e in enumerate(dedup_entries)]
            return "The education qualifications identified in the document are:\n" + "\n".join(formatted)

        # 4. CAREER OBJECTIVE / PROFILE SUMMARY
        if target_section == "objective":
            clean_lines = [re.sub(r"^[-*•\s]+", "", l).strip() for l in raw_lines]
            clean_lines = [l for l in clean_lines if l]
            objective_text = " ".join(clean_lines)
            objective_text = " ".join(objective_text.split())
            if not objective_text.endswith((".", "!", "?")):
                objective_text += "."
            if intent.is_one_line:
                first_sent = re.split(r"(?<=[.!?])\s+", objective_text)[0].strip()
                return f"Career Objective: {first_sent}"
            return f"Career Objective:\n{objective_text}"

        # 5. ACHIEVEMENTS & CERTIFICATIONS & AREAS OF INTEREST
        items = []
        for l in raw_lines:
            c = re.sub(r"^[-*•\s]+", "", l).strip()
            if len(c) >= 6:
                c_tokens = [w.lower() for w in re.findall(r"[a-zA-Z0-9]+", c)]
                if any(any(fuzzy_token_match(neg, tok) for tok in c_tokens) for neg in intent.negative_tokens):
                    continue
                items.append(c)

        unique_items = []
        seen = set()
        for it in items:
            k = it[:25].lower()
            if k not in seen:
                seen.add(k)
                unique_items.append(it)

        if not unique_items:
            return None

        sec_title = target_section.capitalize()
        if target_section == "interest":
            sec_title = "Areas of interest"

        if is_one_line:
            joined = ", ".join(unique_items[:3])
            return f"{sec_title}: {joined}."

        formatted = [f"{i+1}. {it}" for i, it in enumerate(unique_items[:5])]
        return f"The {sec_title.lower()} identified in the document are:\n" + "\n".join(formatted)

    @classmethod
    def _extract_sentences(cls, text: str) -> List[str]:
        """Split text into clean, non-header candidate sentences."""
        raw_parts = re.split(r"(?<=[.!?])\s+|\n+", text)
        sentences: List[str] = []

        for p in raw_parts:
            p_clean = p.strip()
            # Strip markdown bullets or numbered lists
            p_clean = re.sub(r"^[-*•]\s*", "", p_clean).strip()
            # Strip Section X: or Chapter X: or 1. headers
            p_clean = re.sub(
                r"^(section\s*\d+[:.\s-]*|chapter\s*\d+[:.\s-]*|\d+[\.\)]\s*)",
                "",
                p_clean,
                flags=re.IGNORECASE,
            ).strip()

            # Filter out contact/header lines (email, phone numbers, profile links)
            if "@" in p_clean or "/gtb" in p_clean or "/nedn" in p_clean or "leetcode" in p_clean.lower() or "github.com" in p_clean.lower():
                continue
            if re.search(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", p_clean):
                continue

            # Filter out boilerplate metadata lines
            if re.match(r"^generated\s+on\b", p_clean, re.I):
                continue
            if re.match(r"^page\s+\d+\s+of\s+\d+", p_clean, re.I):
                continue
            if p_clean.lower() in ("field details", "aspect details", "round tips:", "mistakes to avoid", "tags", "key tips"):
                continue

            if (len(p_clean) >= 20 or (len(p_clean) >= 8 and (":" in p_clean or " - " in p_clean or len(p_clean.split()) >= 2))) and not p_clean.isupper():
                sentences.append(p_clean)

        return sentences

    @classmethod
    def _synthesize_one_line_summary(cls, corpus: str) -> str:
        """
        Synthesize a high-quality single-sentence summary of the document.
        Detects key directives and themes, formatting them as ONE cohesive sentence.
        """
        sentences = cls._extract_sentences(corpus)
        if not sentences:
            fallback = " ".join(corpus.split()[:25]).strip()
            return fallback + ("." if not fallback.endswith(".") else "")

        # Look for section topics or key directives
        raw_topics: List[str] = []
        for line in corpus.splitlines():
            line_str = line.strip()
            m = re.match(r"^section\s*\d+[:.\s-]+([^.]+)", line_str, re.IGNORECASE)
            if m:
                topic = m.group(1).strip()
                topic = re.sub(r"[:.\s-]+$", "", topic)
                if topic and len(topic) >= 5:
                    raw_topics.append(topic)

        # De-duplicate topics while preserving order and avoiding truncated duplicates
        section_titles: List[str] = []
        for t in raw_topics:
            t_lower = t.lower()
            is_dup = any(t_lower == st.lower() or (len(t) <= len(st) and t_lower in st.lower()) for st in section_titles)
            if not is_dup:
                # Check if this new topic is a longer version of an existing shorter title
                has_shorter = any(st.lower() in t_lower and len(t) > len(st) for st in section_titles)
                if has_shorter:
                    section_titles = [t if (st.lower() in t_lower and len(t) > len(st)) else st for st in section_titles]
                else:
                    section_titles.append(t)

        # If document structured with clear sections:
        if len(section_titles) >= 2:
            capped_titles = section_titles[:4]
            if len(capped_titles) == 2:
                topics_str = f"{capped_titles[0]} and {capped_titles[1]}"
            else:
                topics_str = ", ".join(capped_titles[:-1]) + f", and {capped_titles[-1]}"
            return f"This document specifies enterprise policies governing {topics_str}."

        # Otherwise, score candidate sentences by information density and action keywords
        scored: List[Tuple[float, str]] = []
        for s in sentences:
            tokens = set(re.findall(r"[a-zA-Z0-9]+", s.lower()))
            salient = tokens - STOPWORDS
            action_matches = len(tokens & ACTION_KEYWORDS)
            # Preference for medium length sentences (40 to 140 chars)
            length_penalty = 1.0
            if len(s) > 180:
                length_penalty = 0.8
            elif len(s) < 30:
                length_penalty = 0.6
            score = (action_matches * 3.0 + len(salient) * 0.5) * length_penalty
            scored.append((score, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_sentence = scored[0][1]

        # Clean up and ensure single line ending with period
        clean_line = " ".join(top_sentence.split())
        if not clean_line.endswith((".", "!", "?")):
            clean_line += "."
        return clean_line

    @classmethod
    def _synthesize_general_summary(cls, corpus: str) -> str:
        """Synthesize a concise 2-3 sentence overview."""
        sentences = cls._extract_sentences(corpus)
        if not sentences:
            return cls._synthesize_one_line_summary(corpus)

        scored: List[Tuple[float, str]] = []
        for s in sentences:
            tokens = set(re.findall(r"[a-zA-Z0-9]+", s.lower()))
            salient = tokens - STOPWORDS
            score = len(tokens & ACTION_KEYWORDS) * 2.0 + len(salient) * 0.5
            scored.append((score, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        # Pick top 2 unique sentences
        top_picks: List[str] = []
        for _, s in scored:
            clean = " ".join(s.split())
            if not clean.endswith((".", "!", "?")):
                clean += "."
            if clean not in top_picks:
                top_picks.append(clean)
            if len(top_picks) >= 2:
                break

        return " ".join(top_picks)

    @classmethod
    def _clean_passage(cls, text: str) -> str:
        """Clean passage text by stripping pagination and collapsing duplicate overlapping chunks."""
        text = re.sub(r"Generated\s+on\s+[\d/]+\s+Page\s+\d+\s+of\s+\d+", "", text, flags=re.I)
        text = re.sub(r"\b([A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+){2,8})\s+\1\b", r"\1", text, flags=re.I)
        text = re.sub(r"\s*(?:Medium|Easy|Hard)\s+[a-z]{1,4}\s+and\s+space.*$", "", text, flags=re.I)
        text = re.sub(r"^[a-z]{1,3}\s+(?:and|or|in|to)\s+", "", text, flags=re.I)
        return " ".join(text.split()).strip()

    @classmethod
    def _answer_question_from_context(cls, query: str, corpus: str, is_one_line: bool = False, intent: Optional[QueryIntent] = None) -> Optional[str]:
        """
        Universal Question-Answering engine that extracts grounded answers from context.
        Operates hierarchically:
        1. Structured QA & Topic Block matching:
           Finds numbered questions, interview questions, or labeled topics (e.g. '4. Multithreading in Java')
           and extracts the corresponding solution/explanation block.
        2. Contextual Paragraph & Sentence matching:
           For policies, manuals, or narrative passages, scores sentences and pairs headings with their
           following explanatory text.
        Strictly excludes any blocks or passages matching query negative constraints (e.g. 'not backend', 'without python').
        """
        if intent is None:
            intent = QueryIntent.from_query(query)

        pos_tokens = intent.positive_tokens
        neg_tokens = intent.negative_tokens
        if not pos_tokens:
            return None

        # -----------------------------------------------------------------
        # 1. Structured QA / Topical Block Extraction
        # -----------------------------------------------------------------
        qa_pattern = re.compile(
            r"(?:^|\n)\s*(\d+[\.\)]\s*[^\n]+|Round\s+\d+[:\s]+[^\n]+)\s*\n(.*?)(?=(?:\n\s*\d+[\.\)]|\n\s*Round\s+\d+|\n\s*#+|\Z))",
            re.DOTALL,
        )

        scored_blocks: List[Tuple[float, Dict[str, Any]]] = []
        for m in qa_pattern.finditer(corpus):
            raw_title = m.group(1).strip()
            body = m.group(2).strip()

            clean_title = re.sub(r"^\d+[\.\)]\s*", "", raw_title).strip()
            title_tokens = set(re.findall(r"[a-zA-Z0-9]+", clean_title.lower()))
            body_tokens = set(re.findall(r"[a-zA-Z0-9]+", body.lower()))
            all_block_tokens = title_tokens | body_tokens

            # Disqualify block if it matches negative constraints
            if any(any(fuzzy_token_match(neg, bt) for bt in all_block_tokens) for neg in neg_tokens):
                continue

            title_matches = sum(1 for pw in pos_tokens if any(fuzzy_token_match(pw, tw) for tw in title_tokens))
            body_matches = sum(1 for pw in pos_tokens if any(fuzzy_token_match(pw, bw) for bw in body_tokens))

            if title_matches == 0 and body_matches == 0:
                continue

            # Bigram bonus
            phrase_bonus = 0.0
            for i in range(len(pos_tokens) - 1):
                bg = f"{pos_tokens[i]} {pos_tokens[i+1]}"
                if bg in clean_title.lower():
                    phrase_bonus += 4.0
                elif bg in body.lower():
                    phrase_bonus += 2.0

            score = (title_matches * 5.0) + (body_matches * 1.0) + phrase_bonus

            # Extract substantive solution
            solutions = []
            for sm in re.finditer(r"Solution\s*[:\-]+\s*(.*?)(?=(?:\n\s*[A-Z][a-zA-Z\s]+[:\-]+|\Z))", body, re.DOTALL | re.I):
                stext = sm.group(1).strip()
                if len(stext) >= 15 and not re.match(r"^(?:topics?|difficulty|round)\b", stext, re.I):
                    solutions.append(stext)
            best_sol = solutions[-1] if solutions else ""

            scored_blocks.append((score, {
                "title": clean_title,
                "raw_title": raw_title,
                "body": body,
                "solution": best_sol,
                "title_matches": title_matches,
            }))

        if scored_blocks:
            scored_blocks.sort(key=lambda x: x[0], reverse=True)
            top_score, top_block = scored_blocks[0]

            if top_block["title_matches"] >= 1 or top_score >= 3.0:
                title = top_block["title"]
                sol = top_block["solution"]

                if sol:
                    clean_sol = cls._clean_passage(sol)
                    if not clean_sol.endswith((".", "!", "?")):
                        clean_sol += "."
                    if is_one_line:
                        return f"{title}: {clean_sol}"
                    return f"{title} (Solution):\n{clean_sol}"
                else:
                    body_clean = cls._clean_passage(top_block["body"])
                    cleaned_body = re.sub(r"(?:topics?|difficulty|generated\s+on)[^.]+?[.:\n]", "", body_clean, flags=re.I).strip()
                    if cleaned_body and len(cleaned_body) >= 20:
                        first_sent = re.split(r"(?<=[.!?])\s+", cleaned_body)[0].strip()
                        if not first_sent.endswith((".", "!", "?")):
                            first_sent += "."
                        if is_one_line or len(first_sent) > 80:
                            return f"{title}: {first_sent}"
                        return f"{title}:\n{cleaned_body}"

        # -----------------------------------------------------------------
        # 2. General Sentence & Passage Scoring (For Policies, Directives, Narrative Text)
        # -----------------------------------------------------------------
        norm_corpus = re.sub(r"([A-Za-z0-9_\s]{2,20}:)\s*\n+\s*", r"\1 ", corpus)
        raw_parts = re.split(r"(?<=[.!?])\s+|\n{2,}", norm_corpus)

        candidate_passages = []
        for p in raw_parts:
            p_clean = cls._clean_passage(p)
            if len(p_clean) < 15 or p_clean.isupper():
                continue
            if re.match(r"^(?:generated\s+on|page\s+\d+|aspect\s+details|field\s+details)\b", p_clean, re.I):
                continue
            candidate_passages.append(p_clean)

        scored_passages: List[Tuple[float, str]] = []
        for p in candidate_passages:
            p_tokens = set(re.findall(r"[a-zA-Z0-9]+", p.lower()))
            # Disqualify passage if it matches negative constraints
            if any(any(fuzzy_token_match(neg, pt) for pt in p_tokens) for neg in neg_tokens):
                continue

            matches = sum(1 for pw in pos_tokens if any(fuzzy_token_match(pw, pt) for pt in p_tokens))
            if matches == 0:
                continue

            phrase_bonus = 0.0
            for i in range(len(pos_tokens) - 1):
                bg = f"{pos_tokens[i]} {pos_tokens[i+1]}"
                if bg in p.lower():
                    phrase_bonus += 2.5

            len_weight = 1.0
            if len(p) < 35 and (p.endswith(":") or re.match(r"^\d+[\.\)]", p)):
                len_weight = 0.4
            elif 50 <= len(p) <= 220:
                len_weight = 1.2

            action_bonus = 0.3 * len(p_tokens & ACTION_KEYWORDS)
            total_score = (matches * 2.0 + phrase_bonus + action_bonus) * len_weight
            scored_passages.append((total_score, p))

        if not scored_passages:
            return None

        scored_passages.sort(key=lambda x: x[0], reverse=True)
        best_text = scored_passages[0][1]

        # If best_text is merely a header/title, attach the subsequent sentence from context
        if len(best_text) < 45 and (best_text.endswith(":") or re.match(r"^\d+[\.\)]", best_text)):
            try:
                idx = candidate_passages.index(best_text)
                if idx + 1 < len(candidate_passages):
                    next_text = candidate_passages[idx + 1]
                    best_text = f"{best_text} {next_text}"
            except ValueError:
                pass

        if not best_text.endswith((".", "!", "?")):
            best_text += "."

        return best_text

    @classmethod
    def _synthesize_concise_response(cls, query: str, corpus: str, intent: Optional[QueryIntent] = None) -> str:
        """Provide a concise grounded response matching query tokens."""
        is_one_line = intent.is_one_line if intent else cls._is_one_line_request(query)
        ans = cls._answer_question_from_context(query, corpus, is_one_line=is_one_line, intent=intent)
        if ans:
            return ans
        return cls._synthesize_one_line_summary(corpus)
