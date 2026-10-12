import os
import re
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY was not found. Please check your .env file."
    )

RESUME_DIR = BASE_DIR / "Resumes"
FALLBACK_ANSWER = (
    "I couldn't find sufficient information in the retrieved resumes."
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY,
)


# ============================================================
# 2. LOAD RESUMES AND SPLIT INTO CHUNKS
# ============================================================

print("\n" + "=" * 65)
print("RESUMIND AI - LOADING RESUMES")
print("=" * 65)

if not RESUME_DIR.exists():
    raise FileNotFoundError(f"Resume folder not found: {RESUME_DIR}")

# Load PDFs explicitly with PyPDFLoader. This avoids DirectoryLoader's default
# dependency on the optional `unstructured` package in Streamlit Cloud.
pdf_files = sorted(RESUME_DIR.rglob("*.pdf"))
documents = []
load_errors = []

for pdf_path in pdf_files:
    try:
        pdf_documents = PyPDFLoader(str(pdf_path)).load()
        # Keep paths relative to the project so local/cloud paths don't affect
        # candidate grouping or source display.
        for doc in pdf_documents:
            doc.metadata["source"] = str(pdf_path)
        documents.extend(pdf_documents)
    except Exception as exc:
        load_errors.append((pdf_path.name, str(exc)))
        print(f"Could not read {pdf_path.name}: {exc}")

if not documents:
    if not pdf_files:
        raise ValueError(f"No PDF resumes were found in {RESUME_DIR}")
    details = "; ".join(f"{name}: {error}" for name, error in load_errors[:3])
    raise ValueError(
        f"No readable PDF resumes found in {RESUME_DIR}. "
        f"Check that pypdf is installed and the PDFs are valid. {details}"
    )

if load_errors:
    print(f"Warning: {len(load_errors)} PDF file(s) could not be read.")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=120,
)
chunks = splitter.split_documents(documents)

print(f"Documents loaded: {len(documents)}")
print(f"Chunks created: {len(chunks)}")


# ============================================================
# 3. HELPERS AND RESUME GROUPING
# ============================================================

def normalize_text(text):
    """Normalize text for case-insensitive matching."""
    text = str(text or "").lower()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"[^a-z0-9+#.\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_source(doc):
    """Return the source path attached to a loaded document."""
    return str(doc.metadata.get("source", "Unknown resume"))


def source_key(source):
    return os.path.normcase(os.path.normpath(str(source)))


def get_candidate_tokens(source):
    """Filename tokens are useful for retrieval, not proof of identity."""
    filename = normalize_text(Path(source).stem)
    ignored_words = {
        "resume", "resumes", "cv", "updated", "final", "latest",
        "new", "profile", "document", "copy", "job", "application",
        "data", "science", "machine", "learning", "engineer",
        "developer", "analyst", "intern", "sample", "dummy",
        "project", "version",
    }
    return [
        token for token in filename.split()
        if token not in ignored_words and len(token) > 1
    ]


chunks_by_source = defaultdict(list)
for doc in chunks:
    chunks_by_source[get_source(doc)].append(doc)

# Full text is assembled per source to support deterministic skill checks.
resume_text_by_source = {}
for source, source_chunks in chunks_by_source.items():
    # Overlapping chunks can duplicate text. That is acceptable for keyword
    # evidence checks; unique chunks are used when returning evaluation context.
    resume_text_by_source[source] = "\n".join(
        doc.page_content for doc in source_chunks
    )

print(f"Current resume files: {len(chunks_by_source)}")


# ============================================================
# 4. CANDIDATE MATCHING
# ============================================================

QUERY_STOPWORDS = {
    "what", "which", "who", "where", "when", "why", "how", "does",
    "did", "has", "have", "is", "are", "the", "a", "an", "of", "for",
    "in", "on", "to", "from", "with", "and", "or", "me", "tell",
    "show", "find", "give", "list", "candidate", "candidates", "resume",
    "resumes", "skills", "skill", "projects", "project", "experience",
    "technical", "mentioned", "worked", "work", "about", "their", "his",
    "her", "they", "them", "using", "used", "demonstrate", "demonstrates",
    "relevant", "built", "developed", "all", "any", "applicants",
    "applicant", "who", "can", "which", "have", "has",
}


def find_candidate_matches(query):
    """Find possible candidate resumes from filename tokens."""
    query_tokens = set(normalize_text(query).split())
    possible_name_tokens = query_tokens - QUERY_STOPWORDS
    matches = []

    for source, source_chunks in chunks_by_source.items():
        name_tokens = set(get_candidate_tokens(source))
        if not name_tokens:
            continue

        # Support names written with spaces, underscores, or joined together
        # (for example, "Sri Chetan", "Sri_Chetan", and "Srichetan").
        compact_name = "".join(name_tokens)
        compact_query = "".join(query_tokens)
        joined_name_match = len(compact_name) >= 6 and compact_name in compact_query
        joined_partial_match = any(
            len(token) >= 6 and token in compact_query
            for token in name_tokens
        )

        overlap = name_tokens.intersection(possible_name_tokens)
        if joined_name_match or joined_partial_match:
            score = 120 + len(overlap)
        elif len(name_tokens) >= 2 and name_tokens.issubset(query_tokens):
            score = 100 + len(name_tokens)
        elif len(overlap) >= 2:
            score = 50 + len(overlap)
        elif len(overlap) == 1:
            token = next(iter(overlap))
            if len(token) >= 5:
                score = 20
            else:
                continue
        else:
            continue

        matches.append({
            "source": source,
            "chunks": source_chunks,
            "score": score,
            "matched_tokens": sorted(overlap),
        })

    matches.sort(key=lambda item: item["score"], reverse=True)

    unique = []
    seen = set()
    for match in matches:
        key = source_key(match["source"])
        if key not in seen:
            seen.add(key)
            unique.append(match)
    return unique


# ============================================================
# 5. QUESTION CLASSIFICATION AND SKILL EVIDENCE
# ============================================================

SECTION_KEYWORDS = {
    "skills": [
        "skill", "skills", "technologies", "programming languages",
        "technical knowledge", "tools",
    ],
    "projects": ["project", "projects", "developed", "built", "implemented"],
    "education": ["education", "degree", "college", "university", "graduation"],
    "experience": ["experience", "internship", "employment", "work experience"],
    "certifications": ["certification", "certifications", "certificate", "certificates"],
}

MULTI_CANDIDATE_PHRASES = [
    "which candidates", "what candidates", "list candidates",
    "all candidates", "find candidates", "candidates who",
    "candidates with", "which applicants", "list applicants",
    "all applicants", "which resumes", "which applicants have",
]

# Canonical skill names and common text variants. Word boundaries prevent
# e.g. "R" from matching inside unrelated words.
SKILL_ALIASES = {
    "python": [r"\bpython\b"],
    "sql": [r"\bsql\b", r"\bmysql\b", r"\bpostgres(?:ql)?\b", r"\bsql server\b"],
    "java": [r"\bjava\b"],
    "c++": [r"\bc\+\+\b"],
    "c#": [r"\bc#\b", r"\bc sharp\b"],
    "javascript": [r"\bjavascript\b", r"\bjs\b"],
    "typescript": [r"\btypescript\b"],
    "html": [r"\bhtml5?\b"],
    "css": [r"\bcss3?\b"],
    "pandas": [r"\bpandas\b"],
    "numpy": [r"\bnumpy\b"],
    "machine learning": [r"\bmachine learning\b"],
    "deep learning": [r"\bdeep learning\b"],
    "generative ai": [r"\bgenerative ai\b", r"\bgen ai\b", r"\bgenai\b"],
    "artificial intelligence": [r"\bartificial intelligence\b"],
    "nlp": [r"\bnlp\b", r"\bnatural language processing\b"],
    "computer vision": [r"\bcomputer vision\b"],
    "tensorflow": [r"\btensorflow\b"],
    "pytorch": [r"\bpytorch\b"],
    "scikit-learn": [r"\bscikit learn\b", r"\bsklearn\b"],
    "xgboost": [r"\bxgboost\b"],
    "power bi": [r"\bpower bi\b"],
    "tableau": [r"\btableau\b"],
    "excel": [r"\bexcel\b", r"\bmicrosoft excel\b"],
    "aws": [r"\baws\b", r"\bamazon web services\b"],
    "azure": [r"\bazure\b", r"\bmicrosoft azure\b"],
    "docker": [r"\bdocker\b"],
    "kubernetes": [r"\bkubernetes\b"],
    "git": [r"\bgit\b"],
    "linux": [r"\blinux\b"],
    "langchain": [r"\blangchain\b"],
    "langgraph": [r"\blanggraph\b"],
    "rag": [r"\brag\b", r"\bretrieval augmented generation\b"],
    "llm": [r"\bllms?\b", r"\blarge language models?\b"],
    "chroma db": [r"\bchroma(?:db)?\b"],
    "opencv": [r"\bopencv\b"],
    "yolo": [r"\byolo\b"],
}


def detect_requested_sections(query):
    normalized_query = normalize_text(query)
    requested = []
    for section, keywords in SECTION_KEYWORDS.items():
        if any(normalize_text(word) in normalized_query for word in keywords):
            requested.append(section)
    return requested


def is_multi_candidate_question(query):
    normalized = normalize_text(query)
    return any(phrase in normalized for phrase in MULTI_CANDIDATE_PHRASES)


def extract_requested_skills(query):
    """Return skills explicitly asked about, based on the alias dictionary."""
    normalized = normalize_text(query)
    found = []
    for skill, patterns in SKILL_ALIASES.items():
        if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns):
            found.append(skill)
    return found


def find_skill_evidence(text, skills):
    """
    Find explicit skill mentions and prefer concise snippets that include the
    skill plus nearby section labels such as Skills, Projects, or Experience.
    A textual mention does not prove professional proficiency.
    """
    raw_text = str(text or "")
    lines = [re.sub(r"\s+", " ", line).strip() for line in raw_text.splitlines()]
    lines = [line for line in lines if line]
    evidence = {}
    section_pattern = re.compile(
        r"\b(skills?|technical skills|projects?|experience|internship|"
        r"work experience|professional experience)\b", re.IGNORECASE
    )

    for skill in skills:
        patterns = SKILL_ALIASES.get(skill, [rf"\b{re.escape(skill)}\b"])
        candidates = []
        for index, line in enumerate(lines):
            if not any(re.search(pattern, line, re.IGNORECASE) for pattern in patterns):
                continue
            score = 0
            if re.search(r"\b(skills?|technical skills|programming|tools|technologies)\b", line, re.I):
                score += 3
            if re.search(r"\b(projects?|experience|internship|developed|implemented|used|built)\b", line, re.I):
                score += 2
            if len(line) < 260:
                score += 1
            snippet = line
            if index > 0 and section_pattern.search(lines[index - 1]) and len(lines[index - 1]) < 80:
                snippet = lines[index - 1] + ": " + line
                score += 1
            snippet = snippet[:260].strip()
            if snippet and all(existing[1] != snippet for existing in candidates):
                candidates.append((score, snippet))
        candidates.sort(key=lambda item: item[0], reverse=True)
        if candidates:
            evidence[skill] = [snippet for _, snippet in candidates[:2]]
    return evidence


def extract_resume_display_name(source, text):
    """Return a plausible name from the resume header, or a safe filename label."""
    raw_text = str(text or "")
    # Preserve actual line boundaries and normalize whitespace within each line.
    raw_lines = [re.sub(r"\s+", " ", line).strip() for line in raw_text.splitlines()]
    lines = [line.strip(" |•-:·") for line in raw_lines if line.strip()]

    ignored_exact = {
        "resume", "curriculum vitae", "profile", "contact", "skills",
        "education", "experience", "projects", "objective", "summary",
        "professional summary", "career objective", "objective summary",
        "technical skills", "key skills", "work experience",
        "academic projects", "professional experience", "contact details",
        "professional profile", "career summary", "programming languages",
        "technical expertise", "areas of expertise", "personal details",
        "certifications", "internship", "employment history", "gen ai",
        "generative ai", "data scientist", "software developer",
    }
    heading_or_role_words = {
        "python", "developer", "engineer", "analyst", "student", "machine",
        "learning", "data", "science", "software", "summary", "professional",
        "career", "objective", "technical", "skills", "experience", "education",
        "projects", "profile", "intern", "internship", "specialist", "consultant",
        "manager", "programming", "languages", "certifications", "contact",
        "email", "phone", "linkedin", "github", "portfolio", "aspiring",
        "seeking", "motivated", "results", "oriented", "information", "technology",
        "resume", "curriculum", "vitae", "qualification", "qualifications",
        "expertise", "tools", "database", "databases", "analytics", "artificial",
        "intelligence", "generative", "genai",
    }

    # Only inspect the resume header, where the candidate's name normally appears.
    for line in lines[:12]:
        cleaned = re.sub(r"^[|•\-:· ]+|[|•\-:· ]+$", "", line).strip()
        if not cleaned or len(cleaned) > 45 or "@" in cleaned:
            continue
        if re.search(r"\d", cleaned) or any(char in cleaned for char in "@:/\\"):
            continue
        if cleaned.lower().strip(".: ") in ignored_exact:
            continue

        words = cleaned.replace(".", " ").split()
        if not 2 <= len(words) <= 4:
            continue
        # Accept alphabetic name tokens, including apostrophes and hyphens.
        if not all(re.fullmatch(r"[A-Za-z][A-Za-z'’\-]*", word) for word in words):
            continue
        normalized_words = [word.lower() for word in words]
        if any(word in heading_or_role_words for word in normalized_words):
            continue
        if re.search(
            r"\b(summary|objective|skills|education|experience|projects?|languages|"
            r"certifications|profile|contact|programming)\b",
            cleaned,
            re.IGNORECASE,
        ):
            continue
        # All-caps headers are allowed only when they look like a short name.
        return cleaned.title() if cleaned.isupper() else cleaned

    # Filename-derived labels are useful for locating the file, but aren't
    # treated as verified identity. Prefer the portion after a separator when
    # filenames contain both a document label and a candidate name.
    filename_stem = Path(source).stem
    filename_candidate = re.split(r"\s+-\s+", filename_stem)[-1]
    filename_candidate = re.sub(r"[_]+", " ", filename_candidate)
    filename_candidate = re.sub(r"\b(FlowCV|Resume|CV|Updated|Final|Latest|Copy|\d{4}[-_]\d{2}[-_]\d{2})\b", " ", filename_candidate, flags=re.I)
    filename_candidate = re.sub(r"\s+", " ", filename_candidate).strip(" -_()")
    candidate_words = filename_candidate.split()
    if (2 <= len(candidate_words) <= 4
            and all(re.fullmatch(r"[A-Za-z][A-Za-z'’\-]*", word) for word in candidate_words)
            and not any(word.lower() in heading_or_role_words for word in candidate_words)):
        return f"{filename_candidate.title()} (name not verified)"

    return f"Resume source: {Path(source).name} (name not verified)"


# ============================================================
# 6. RETRIEVAL
# ============================================================

def retrieve_from_one_resume(query, source_chunks, max_chunks=8):
    """Retrieve relevant chunks from one resume only."""
    if not source_chunks:
        return []

    try:
        retriever = BM25Retriever.from_documents(source_chunks)
        retriever.k = min(max_chunks, len(source_chunks))
        results = retriever.invoke(query)
    except Exception as exc:
        print(f"BM25 retrieval warning: {exc}")
        results = []

    requested_sections = detect_requested_sections(query)
    for doc in source_chunks:
        content = normalize_text(doc.page_content)
        for section in requested_sections:
            if any(normalize_text(term) in content for term in SECTION_KEYWORDS[section]):
                results.append(doc)
                break

    unique_results = []
    seen_content = set()
    for doc in results:
        key = normalize_text(doc.page_content)
        if key not in seen_content:
            seen_content.add(key)
            unique_results.append(doc)
    return unique_results[:max_chunks]


def retrieve_general_context(query, top_k=12):
    """Retrieve a limited set of relevant chunks from all resumes."""
    if not chunks:
        return []

    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = min(top_k, len(chunks))
    retrieved_docs = retriever.invoke(query)

    unique_docs = []
    seen = set()
    for doc in retrieved_docs:
        key = (get_source(doc), normalize_text(doc.page_content))
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)
    return unique_docs


# ============================================================
# 7. GROUNDED ANSWER GENERATION
# ============================================================

def generate_answer(query, retrieved_docs):
    """Generate a concise answer grounded only in supplied resume evidence."""
    if not retrieved_docs:
        return FALLBACK_ANSWER

    context_parts = []
    for doc in retrieved_docs:
        source = Path(get_source(doc)).name
        context_parts.append(f"[Resume source: {source}]\n{doc.page_content}")

    context = "\n\n".join(context_parts)
    prompt = f"""
You are Resumind AI, a resume screening assistant.

QUESTION:
{query}

RESUME EVIDENCE:
{context}

Rules:
- Use only facts supported by the resume evidence above.
- Never invent names, skills, projects, education, certifications,
  employers, or experience.
- A filename identifies a source file; it does not prove the candidate's name.
- Do not combine facts from different candidates.
- A skill being listed does not automatically prove professional work experience.
  Say "lists/mentions this skill" unless the resume supports actual work experience.
- For multi-candidate questions, include only candidates whose supplied evidence
  supports the requested criterion. Mention the source filename if identity is unclear.
- If evidence is insufficient, respond exactly:
  "{FALLBACK_ANSWER}"
- Keep the answer concise and distinguish explicit evidence from inference.

ANSWER:
"""
    try:
        response = llm.invoke(prompt)
        answer = response.content
        if isinstance(answer, list):
            answer = "\n".join(str(item) for item in answer)
        answer = str(answer).strip()
        return answer if answer else FALLBACK_ANSWER
    except Exception as exc:
        print(f"Answer generation error: {exc}")
        return FALLBACK_ANSWER


# ============================================================
# 8. MULTI-CANDIDATE SEARCH WITH PRE-FILTERING
# ============================================================

def retrieve_multi_candidate_context(query, max_chunks_per_resume=3):
    """
    Search every resume independently. When the question names specific
    skills, include a resume only if every requested skill is explicitly found.
    """
    requested_skills = extract_requested_skills(query)
    candidate_evidence = []

    for source, source_chunks in chunks_by_source.items():
        full_text = resume_text_by_source.get(source, "")

        if requested_skills:
            evidence = find_skill_evidence(full_text, requested_skills)
            # Require every requested skill, not just one of them.
            if len(evidence) != len(requested_skills):
                continue

            # Return only chunks that contain at least one requested skill.
            matching_chunks = []
            for doc in source_chunks:
                doc_text = doc.page_content or ""
                doc_evidence = find_skill_evidence(doc_text, requested_skills)
                if doc_evidence:
                    matching_chunks.append(doc)

            if not matching_chunks:
                continue

            selected = matching_chunks[:max_chunks_per_resume]
            candidate_evidence.append({
                "source": source,
                "chunks": selected,
                "skills": requested_skills,
                "evidence": evidence,
                "display_name": extract_resume_display_name(source, full_text),
            })
        else:
            retrieved = retrieve_from_one_resume(
                query, source_chunks, max_chunks=max_chunks_per_resume
            )
            if retrieved:
                candidate_evidence.append({
                    "source": source,
                    "chunks": retrieved,
                    "skills": [],
                    "evidence": {},
                    "display_name": extract_resume_display_name(source, full_text),
                })

    return candidate_evidence


def format_multi_candidate_skill_answer(candidate_groups, requested_skills):
    """Format skill-match results directly from evidence; do not ask an LLM to invent summaries."""
    if not candidate_groups:
        return FALLBACK_ANSWER

    lines = [
        f"Found {len(candidate_groups)} resume(s) with explicit text evidence for all requested skills: "
        + ", ".join(requested_skills) + ".",
        "Note: a skill mention does not by itself prove professional work experience.",
        "",
    ]

    for index, group in enumerate(candidate_groups, start=1):
        lines.append(f"{index}. {group['display_name']}")
        lines.append(f"   Source: {Path(group['source']).name}")
        for skill in requested_skills:
            snippets = group["evidence"].get(skill, [])
            if snippets:
                # Keep snippets short and directly traceable to the source resume.
                snippet = re.sub(r"\s+", " ", snippets[0]).strip()
                if len(snippet) > 220:
                    snippet = snippet[:217].rstrip() + "..."
                lines.append(f"   - {skill} evidence: “{snippet}”")
        lines.append("")

    return "\n".join(lines).strip()


def answer_multi_candidate_question(query):
    candidate_groups = retrieve_multi_candidate_context(query)
    if not candidate_groups:
        return FALLBACK_ANSWER

    requested_skills = extract_requested_skills(query)
    if requested_skills:
        # Deterministic formatting prevents unsupported generic descriptions and
        # ensures every qualifying resume is listed, not only the first few.
        return format_multi_candidate_skill_answer(candidate_groups, requested_skills)

    all_docs = []
    for group in candidate_groups:
        all_docs.extend(group["chunks"])
    return generate_answer(query, all_docs)


# ============================================================
# 9. MAIN RAG API
# ============================================================

def RAG_SYSTEM(query):
    """Main RAG function used by the Streamlit application."""
    if not query or not query.strip():
        return FALLBACK_ANSWER

    query = query.strip()

    if is_multi_candidate_question(query):
        print("\nMulti-candidate retrieval")
        print(f"Resumes searched: {len(chunks_by_source)}")
        groups = retrieve_multi_candidate_context(query)
        print(f"Candidate resumes with evidence: {len(groups)}")
        if not groups:
            return FALLBACK_ANSWER
        return answer_multi_candidate_question(query)

    candidate_matches = find_candidate_matches(query)
    if candidate_matches:
        best_score = candidate_matches[0]["score"]
        best_matches = [
            match for match in candidate_matches
            if match["score"] == best_score
        ]
        retrieved_docs = []
        for match in best_matches:
            requested_sections = detect_requested_sections(query)
            if requested_sections:
                # For candidate-specific section questions, prioritize every
                # chunk containing the requested section, not only BM25 top hits.
                section_docs = []
                for doc in match["chunks"]:
                    content = normalize_text(doc.page_content)
                    if any(
                        normalize_text(term) in content
                        for section in requested_sections
                        for term in SECTION_KEYWORDS[section]
                    ):
                        section_docs.append(doc)
                selected_docs = section_docs or retrieve_from_one_resume(
                    query, match["chunks"], max_chunks=8
                )
            else:
                selected_docs = retrieve_from_one_resume(
                    query, match["chunks"], max_chunks=8
                )
            retrieved_docs.extend(selected_docs)
        if retrieved_docs:
            print("\nCandidate-specific retrieval:")
            for match in best_matches:
                print("Matched resume:", match["source"])
            print("Retrieved chunks:", len(retrieved_docs))
            return generate_answer(query, retrieved_docs)

    retrieved_docs = retrieve_general_context(query)
    print("\nGeneral resume retrieval")
    print("Retrieved chunks:", len(retrieved_docs))
    return generate_answer(query, retrieved_docs)


# ============================================================
# 10. EVALUATION COMPATIBILITY
# ============================================================

def RAG_SYSTEM_WITH_CONTEXT(query):
    """
    Return answer, contexts, and sources for the existing custom evaluator.
    Keep this public API stable for compatibility.
    """
    if not query or not query.strip():
        return {
            "question": query,
            "answer": FALLBACK_ANSWER,
            "contexts": [],
            "sources": [],
        }

    query = query.strip()

    if is_multi_candidate_question(query):
        groups = retrieve_multi_candidate_context(query)
        if extract_requested_skills(query):
            contexts = []
            sources = []

            for group in groups:
                block_lines = [
                    f"Candidate label: {group['display_name']}",
                    f"Source filename: {Path(group['source']).name}",
                    "Requested skills explicitly found: " + ", ".join(group["skills"]),
                ]
                for skill, snippets in group["evidence"].items():
                    block_lines.append(f"{skill} evidence: " + " | ".join(snippets[:2]))
                block = "\n".join(block_lines)
                contexts.append(block)
                sources.append(Path(group["source"]).name)
            answer = format_multi_candidate_skill_answer(
                groups, extract_requested_skills(query)
            )
            return {
                "question": query,
                "answer": answer,
                "contexts": contexts,
                "sources": sorted(set(sources)),
            }

        retrieved_docs = []
        for group in groups:
            retrieved_docs.extend(group["chunks"])
    else:
        candidate_matches = find_candidate_matches(query)
        if candidate_matches:
            best_score = candidate_matches[0]["score"]
            best_matches = [
                match for match in candidate_matches
                if match["score"] == best_score
            ]
            retrieved_docs = []
            for match in best_matches:
                retrieved_docs.extend(
                    retrieve_from_one_resume(query, match["chunks"], max_chunks=8)
                )
        else:
            retrieved_docs = retrieve_general_context(query)

    unique_docs = []
    seen = set()
    for doc in retrieved_docs:
        key = (get_source(doc), normalize_text(doc.page_content))
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    answer = generate_answer(query, unique_docs)
    return {
        "question": query,
        "answer": answer,
        "contexts": [doc.page_content for doc in unique_docs],
        "sources": sorted({Path(get_source(doc)).name for doc in unique_docs}),
    }


# ============================================================
# 11. LOCAL TESTS
# ============================================================

if __name__ == "__main__":
    test_questions = [
        "What technical skills are mentioned in Sri Chetan's resume?",
        "What projects did Chetan work on?",
        "Which candidates have experience with Python and SQL?",
        "Does the candidate have experience with quantum computing?",
    ]

    for question in test_questions:
        print("\n" + "=" * 65)
        print("QUESTION:", question)
        print("=" * 65)
        print(RAG_SYSTEM(question))
