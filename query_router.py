"""
Routes user questions to the appropriate resume workflow.

Resume PDFs are loaded by rag_backup.py from the local Resumes folder.
The original Streamlit UI expects process_query() to return either:
    {"type": "resume_qa", "answer": "..."}
or:
    {"type": "candidate_search", "results": [...]}
"""

import re

from rag_backup import RAG_SYSTEM, answer_multi_candidate_question
from screening import is_candidate_search_query, candidate_search

FALLBACK_ANSWER = (
    "I couldn't find sufficient information in the retrieved resumes."
)


def _is_comparison_query(query: str) -> bool:
    """Recognize comparisons before candidate-ranking keywords."""
    q = re.sub(r"\s+", " ", (query or "").strip().lower())
    patterns = (
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bversus\b",
        r"\bvs\.?\b",
        r"\bdifference between\b",
        r"\bcompare .* and .*\b",
        r"\bbetween .+ and .+\b",
    )
    return any(re.search(pattern, q) for pattern in patterns)


def process_query(query):
    """Route a query while keeping the response shape expected by app.py."""
    if not query or not query.strip():
        return {"type": "resume_qa", "answer": FALLBACK_ANSWER}

    query = query.strip()

    # A comparison must be handled before broad candidate-search keywords.
    if _is_comparison_query(query):
        try:
            answer = answer_multi_candidate_question(query)
        except Exception as exc:
            print(f"[query_router] Candidate comparison failed: {exc}")
            answer = FALLBACK_ANSWER
        return {"type": "resume_qa", "answer": answer or FALLBACK_ANSWER}

    # Candidate-search requests return scored candidates for the UI charts.
    if is_candidate_search_query(query):
        try:
            results = candidate_search(query)
        except Exception as exc:
            print(f"[query_router] Candidate search failed: {exc}")
            results = []
        return {"type": "candidate_search", "results": results}

    # All other questions use the grounded RAG answer function.
    try:
        answer = RAG_SYSTEM(query)
    except Exception as exc:
        print(f"[query_router] RAG question failed: {exc}")
        answer = FALLBACK_ANSWER

    return {
        "type": "resume_qa",
        "answer": answer or FALLBACK_ANSWER,
    }
