import os
import re
from collections import defaultdict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


# ============================================================
# 1. ENVIRONMENT
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ============================================================
# 2. LLM
# ============================================================

model = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY
)


# ============================================================
# 3. CONTACT INFORMATION EXTRACTION
# ============================================================

def extract_email(text):
    """
    Extract the first valid email address from resume text.
    """

    if not text:
        return None

    email_pattern = (
        r"\b[A-Za-z0-9._%+-]+"
        r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    )

    matches = re.findall(
        email_pattern,
        text
    )

    if matches:
        return matches[0].strip()

    return None


def extract_phone(text):
    """
    Extract an Indian mobile number from resume text.

    Supports formats such as:

    +91 9876543210
    +91-9876543210
    +91 98765 43210
    9876543210
    98765-43210
    """

    if not text:
        return None

    patterns = [

        # +91 9876543210
        r"\+91[\s-]*[6-9]\d{9}",

        # +91 98765 43210
        r"\+91[\s-]*[6-9]\d{4}[\s-]\d{5}",

        # 98765 43210
        r"\b[6-9]\d{4}[\s-]\d{5}\b",

        # 9876543210
        r"\b[6-9]\d{9}\b"
    ]

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text
        )

        if matches:

            phone = matches[0].strip()

            # Normalize spaces
            phone = re.sub(
                r"\s+",
                " ",
                phone
            )

            return phone

    return None


def extract_profile_url(text, platform):
    """Extract the first LinkedIn or GitHub profile URL explicitly present in resume text."""
    if not text:
        return None

    if platform == "linkedin":
        pattern = r'(?i)(?:https?://)?(?:www\.)?linkedin\.com/[^\s<>()\[\]{}"\'`,;]+'
        domain = "linkedin.com"
    elif platform == "github":
        pattern = r'(?i)(?:https?://)?(?:www\.)?github\.com/[^\s<>()\[\]{}"\'`,;]+'
        domain = "github.com"
    else:
        return None

    matches = re.findall(pattern, text)

    for match in matches:
        url = match.strip().rstrip(".,;:!?)\"]}'")
        if domain not in url.lower():
            continue

        # Add a scheme when the resume contains a bare profile URL.
        if not url.lower().startswith(("http://", "https://")):
            url = "https://" + url

        # Avoid linking to just the platform homepage.
        path = re.sub(r"(?i)^https?://(?:www\.)?" + re.escape(domain) + r"/?", "", url)
        if path and path not in ("/",):
            return url

    return None


def extract_linkedin(text):
    """Extract a LinkedIn profile URL, if present."""
    return extract_profile_url(text, "linkedin")


def extract_github(text):
    """Extract a GitHub profile URL, if present."""
    return extract_profile_url(text, "github")


# ============================================================
# 4. GET ALL RESUME DOCUMENTS
# ============================================================

def get_all_resume_documents():
    """
    Load every PDF in the project's Resumes folder directly.

    The previous implementation imported `vector_store` from rag.py,
    but the current RAG module creates Chroma stores dynamically and does
    not expose a module-level vector_store. Reading PDFs directly keeps
    candidate search and job-description screening independent of that
    temporary vector store.
    """
    from pathlib import Path
    from langchain_community.document_loaders import PyPDFLoader

    resume_folder = Path(__file__).resolve().parent / "Resumes"
    resume_documents = []

    if not resume_folder.is_dir():
        print(f"[screening] Resume folder not found: {resume_folder}")
        return resume_documents

    pdf_files = sorted(
        path for path in resume_folder.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )

    if not pdf_files:
        print(f"[screening] No PDF resumes found in: {resume_folder}")
        return resume_documents

    for pdf_file in pdf_files:
        try:
            pages = PyPDFLoader(str(pdf_file)).load()

            for page in pages:
                text = (page.page_content or "").strip()
                if not text:
                    continue

                metadata = dict(page.metadata or {})
                metadata.update({
                    "source": str(pdf_file),
                    "candidate_name": pdf_file.stem,
                })

                resume_documents.append({
                    "text": text,
                    "metadata": metadata,
                })

        except Exception as error:
            # A damaged or unreadable PDF should not prevent the other
            # candidates from being screened.
            print(f"[screening] Could not read {pdf_file.name}: {error}")

    print(
        f"[screening] Loaded {len(resume_documents)} readable page(s) "
        f"from {len(pdf_files)} PDF resume(s)."
    )
    return resume_documents



# ============================================================
# 5. GROUP ALL RESUME CHUNKS BY CANDIDATE
# ============================================================

def get_all_candidates():

    documents = get_all_resume_documents()

    candidates = defaultdict(list)

    for document in documents:

        text = document["text"]

        metadata = document["metadata"]

        source = metadata.get(
            "source",
            ""
        )

        if not source:
            continue

        # Get resume filename
        candidate_name = os.path.basename(
            source
        )

        candidates[
            candidate_name
        ].append(
            text
        )

    return candidates


# ============================================================
# 6. EXTRACT SCORE
# ============================================================

def extract_score(text):

    if not text:
        return 0

    # SCORE: 85
    match = re.search(
        r"SCORE\s*:\s*(\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE
    )

    if match:

        score = float(
            match.group(1)
        )

        return max(
            0,
            min(
                100,
                score
            )
        )

    # 85/100
    match = re.search(
        r"\b(\d{1,3})\s*/\s*100\b",
        text
    )

    if match:

        score = float(
            match.group(1)
        )

        return max(
            0,
            min(
                100,
                score
            )
        )

    return 0


# ============================================================
# 7. EVALUATE CANDIDATE FOR SEARCH QUERY
# ============================================================

def evaluate_candidate_for_query(
    candidate_name,
    resume_text,
    query
):

    prompt = f"""
You are an AI Resume Screening Assistant.

Evaluate the candidate ONLY using the resume information
provided below.

CANDIDATE:
{candidate_name}

USER QUERY:
{query}

RESUME:
{resume_text}

Instructions:

1. Determine how relevant this candidate is to the query.
2. Give a relevance score from 0 to 100.
3. Use only information explicitly present in the resume.
4. Do not use outside knowledge.
5. Do not invent skills, experience, projects or qualifications.

Return ONLY:

SCORE: <number>

REASON:
<short explanation>
"""

    response = model.invoke(
        prompt
    )

    content = response.content

    score = extract_score(
        content
    )

    return {
        "candidate": candidate_name,
        "score": score,
        "reason": content
    }


# ============================================================
# 8. SEARCH CANDIDATES
# ============================================================

def search_candidates(
    query,
    candidates
):

    results = []

    for candidate_name, chunks in candidates.items():

        resume_text = "\n\n".join(
            chunks
        )

        result = evaluate_candidate_for_query(
            candidate_name,
            resume_text,
            query
        )

        results.append(
            result
        )

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results


# ============================================================
# 9. DETECT CANDIDATE SEARCH QUERY
# ============================================================

def is_candidate_search_query(query):

    query = query.lower()

    search_keywords = [

        "top candidates",
        "top candidate",
        "top applicants",
        "top applicant",

        "best candidates",
        "best candidate",

        "which candidates",
        "who has",
        "who knows",

        "find candidates",
        "find candidate",

        "candidates with",
        "candidate with",

        "profiles of",
        "top profiles",

        "applicants with",
        "applicant with"
    ]

    return any(
        keyword in query
        for keyword in search_keywords
    )


# ============================================================
# 10. CANDIDATE SEARCH
# ============================================================

def candidate_search(query):

    candidates = get_all_candidates()

    if not candidates:

        return []

    results = search_candidates(
        query,
        candidates
    )

    # --------------------------------------------------------
    # Keep candidates with a meaningful score
    # --------------------------------------------------------

    results = [
        result
        for result in results
        if result.get("score", 0) > 0
    ]

    # --------------------------------------------------------
    # Extract contact information
    # from COMPLETE candidate resume
    # --------------------------------------------------------

    for result in results:

        candidate_name = result[
            "candidate"
        ]

        candidate_chunks = candidates.get(
            candidate_name,
            []
        )

        complete_resume = "\n\n".join(
            candidate_chunks
        )

        result["email"] = extract_email(
            complete_resume
        )

        result["phone"] = extract_phone(
            complete_resume
        )

        result["linkedin"] = extract_linkedin(
            complete_resume
        )

        result["github"] = extract_github(
            complete_resume
        )

    return results


# ============================================================
# 11. EXTRACT JOB REQUIREMENTS
# ============================================================

def extract_job_requirements(
    job_description
):

    prompt = f"""
Extract the important requirements from this Job Description.

JOB DESCRIPTION:

{job_description}

Return the requirements in these categories:

Skills:
Experience:
Tools/Technologies:
Education:
Other Requirements:

Use only information from the Job Description.
"""

    response = model.invoke(
        prompt
    )

    return response.content


# ============================================================
# 12. EVALUATE CANDIDATE FOR JOB DESCRIPTION
# ============================================================

def evaluate_candidate(
    candidate_name,
    resume_text,
    job_description
):

    prompt = f"""
You are an AI Resume Screening System.

Compare the candidate's resume with the Job Description.

JOB DESCRIPTION:
{job_description}

CANDIDATE:
{candidate_name}

RESUME:
{resume_text}

Evaluate ONLY using the provided information.

Return:

SCORE: <0-100>

REASON:
<short explanation>

IMPORTANT:

- Do not invent information.
- Do not use outside knowledge.
- Give a higher score when the candidate clearly matches
  the required skills and experience.
"""

    response = model.invoke(
        prompt
    )

    content = response.content

    score = extract_score(
        content
    )

    return {
        "candidate": candidate_name,
        "score": score,
        "reason": content
    }


# ============================================================
# 13. SCREEN ALL CANDIDATES
# ============================================================

def screen_candidates(
    job_description
):

    requirements = extract_job_requirements(
        job_description
    )

    # --------------------------------------------------------
    # Get every candidate and every chunk
    # --------------------------------------------------------

    all_candidates = get_all_candidates()

    results = []

    # --------------------------------------------------------
    # Evaluate every candidate
    # --------------------------------------------------------

    for candidate_name, chunks in all_candidates.items():

        # Combine ALL chunks belonging to candidate
        resume_text = "\n\n".join(
            chunks
        )

        # ----------------------------------------------------
        # Evaluate candidate
        # ----------------------------------------------------

        result = evaluate_candidate(
            candidate_name,
            resume_text,
            job_description
        )

        # ----------------------------------------------------
        # Extract contact information
        # from complete resume
        # ----------------------------------------------------

        result["email"] = extract_email(
            resume_text
        )

        result["phone"] = extract_phone(
            resume_text
        )

        result["linkedin"] = extract_linkedin(
            resume_text
        )

        result["github"] = extract_github(
            resume_text
        )

        results.append(
            result
        )

    return (
        requirements,
        results
    )


# ============================================================
# 14. RANK CANDIDATES
# ============================================================

def rank_candidates(
    candidates
):

    ranked = sorted(
        candidates,
        key=lambda x: float(
            x.get(
                "score",
                0
            )
        ),
        reverse=True
    )

    return ranked