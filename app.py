import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import re

from query_router import process_query
from screening import screen_candidates, rank_candidates


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Resumind AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0b0d12;
        color: #f8fafc;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 35px;
        padding-left: 36px;
        padding-right: 36px;
        padding-bottom: 60px;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -1px;
        margin-bottom: 5px;
    }

    .main-subtitle {
        color: #9ca3af;
        font-size: 15px;
        margin-bottom: 25px;
    }

    .stTextInput input {
        background-color: #20232d !important;
        color: #ffffff !important;
        border: 1px solid #343846 !important;
        border-radius: 10px !important;
        min-height: 48px;
    }

    .stTextInput input::placeholder {
        color: #9ca3af !important;
    }

    .stTextArea textarea {
        background-color: #20232d !important;
        color: #ffffff !important;
        border: 1px solid #343846 !important;
        border-radius: 10px !important;
        font-size: 14px !important;
    }

    .stTextArea textarea::placeholder {
        color: #8f96a3 !important;
    }

    .stButton > button {
        background-color: #171a22;
        color: #f8fafc;
        border: 1px solid #353a48;
        border-radius: 9px;
        min-height: 43px;
        font-weight: 600;
    }

    .stButton > button:hover {
        border-color: #6366f1;
        color: #ffffff;
    }

    div[data-testid="stMetric"] {
        background-color: #151820;
        border: 1px solid #292d38;
        border-radius: 12px;
        padding: 15px;
    }

    div[data-testid="stMetricLabel"] {
        color: #9ca3af;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc;
    }

    div[data-testid="stExpander"] {
        background-color: #151820;
        border: 1px solid #292d38;
        border-radius: 12px;
    }

    hr {
        border-color: #292d38;
    }

    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 11px;
        margin-top: 45px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# RANKING CATEGORY
# ============================================================

def get_rank_category(
    index,
    total
):

    # If only 3 or fewer candidates
    if total <= 3:

        return "green"

    # If 4-7 candidates
    if total <= 7:

        if index <= 3:
            return "green"

        return "red"

    # More than 7 candidates

    if index <= 3:
        return "green"

    if index > total - 4:
        return "red"

    return "yellow"


# ============================================================
# CATEGORY COLOR
# ============================================================

def get_category_color(
    category
):

    if category == "green":

        return "#22c55e"

    if category == "yellow":

        return "#eab308"

    return "#ef4444"


# ============================================================
# CLEAN DISPLAY NAME
# ============================================================

def get_display_name(
    candidate_name
):

    name = str(
        candidate_name
    ).strip()

    # Remove PDF / DOC / DOCX
    name = re.sub(
        r"\.(pdf|docx?)$",
        "",
        name,
        flags=re.IGNORECASE
    )

    # Remove resume-related words
    name = re.sub(
        r"\b(resume|cv|profile|curriculum vitae)\b",
        "",
        name,
        flags=re.IGNORECASE
    )

    # Remove (1), (2), etc.
    name = re.sub(
        r"\(\d+\)",
        "",
        name
    )

    # Remove standalone numbers
    name = re.sub(
        r"\b\d+\b",
        "",
        name
    )

    # Replace underscores / hyphens
    name = re.sub(
        r"[_\-]+",
        " ",
        name
    )

    # Remove duplicate spaces
    name = re.sub(
        r"\s+",
        " ",
        name
    )

    return name.strip().title()


# ============================================================
# CATEGORY COUNTS
# ============================================================

def calculate_category_counts(
    ranked_candidates
):

    total = len(
        ranked_candidates
    )

    strong_matches = 0
    potential_matches = 0
    low_matches = 0

    for index, candidate in enumerate(
        ranked_candidates,
        start=1
    ):

        category = get_rank_category(
            index,
            total
        )

        if category == "green":

            strong_matches += 1

        elif category == "yellow":

            potential_matches += 1

        else:

            low_matches += 1

    return (
        strong_matches,
        potential_matches,
        low_matches
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '📄 Resumind AI'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'AI-powered resume intelligence, candidate search and job matching.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# MAIN INPUT AREA
# ============================================================

left_col, right_col = st.columns(
    [1, 1],
    gap="large"
)


# ============================================================
# LEFT SIDE - RESUME QUERY
# ============================================================

with left_col:

    st.markdown(
        "## 💬 Ask About Resumes"
    )

    st.caption(
        "Ask questions about candidates, skills, "
        "projects, experience or find top applicants."
    )

    query = st.text_input(
        "Resume Query",
        placeholder=(
            "Example: Who has Python and SQL experience?"
        ),
        label_visibility="collapsed"
    )

    ask_button = st.button(
        "Ask",
        key="resume_query_button"
    )

    if ask_button:

        if not query.strip():

            st.warning(
                "Please enter a resume-related question."
            )

        else:

            with st.spinner(
                "Analyzing resumes..."
            ):

                try:

                    result = process_query(
                        query
                    )

                    # ==================================================
                    # NORMAL RESUME Q&A
                    # ==================================================

                    if result["type"] == "resume_qa":

                        st.markdown(
                            "### Answer"
                        )

                        st.success(
                            result["answer"]
                        )

                    # ==================================================
                    # CANDIDATE SEARCH
                    # ==================================================

                    elif result["type"] == "candidate_search":

                        results = result.get(
                            "results",
                            []
                        )

                        st.markdown(
                            "### 🏆 Top Applicants"
                        )

                        if not results:

                            st.warning(
                                "No matching candidates found."
                            )

                        else:

                            total = len(
                                results
                            )

                            scores = [
                                float(
                                    candidate.get(
                                        "score",
                                        0
                                    )
                                )
                                for candidate
                                in results
                            ]

                            average_score = (
                                sum(scores)
                                / total
                            )

                            highest_score = max(
                                scores
                            )

                            (
                                strong_matches,
                                potential_matches,
                                low_matches
                            ) = calculate_category_counts(
                                results
                            )

                            # ==================================================
                            # SEARCH KPIs
                            # ==================================================

                            search_kpi1, search_kpi2, search_kpi3, search_kpi4 = st.columns(4)

                            with search_kpi1:

                                st.metric(
                                    "Candidates",
                                    total
                                )

                            with search_kpi2:

                                st.metric(
                                    "Average Score",
                                    f"{average_score:.0f}%"
                                )

                            with search_kpi3:

                                st.metric(
                                    "Highest Score",
                                    f"{highest_score:.0f}%"
                                )

                            with search_kpi4:

                                st.metric(
                                    "Strong Matches",
                                    strong_matches
                                )

                            # Keep query results text-focused. Charts belong only
                            # in Job Description Matching, not in resume Q&A.
                            st.markdown("### Candidate Results")
                            for index, candidate in enumerate(results, start=1):
                                candidate_name = get_display_name(
                                    candidate.get("candidate", "Unknown")
                                )
                                candidate_score = float(candidate.get("score", 0))
                                with st.container(border=True):
                                    st.markdown(f"**{index}. {candidate_name}**")
                                    st.write(f"Match score: {candidate_score:.0f}%")
                                    reason = candidate.get("reason") or candidate.get("summary")
                                    if reason:
                                        st.write(reason)

                except Exception as e:

                    st.error(
                        f"Error while processing your query: {e}"
                    )


# ============================================================
# RIGHT SIDE - JOB DESCRIPTION
# ============================================================

with right_col:

    st.markdown(
        "## 📋 Job Description Matching"
    )

    st.caption(
        "Paste a Job Description to screen all available candidates."
    )

    job_description = st.text_area(
        "Job Description",

        placeholder="""
Paste the complete Job Description here...

Example:

We are looking for a Data Scientist with strong
Python, SQL and Machine Learning skills.

Requirements:

• Python
• SQL
• Pandas
• NumPy
• Machine Learning
• Power BI
• Data Analysis
• Problem Solving
• Good communication skills

Experience with Generative AI and RAG is preferred.
""",

        height=300,

        key="job_description_text"
    )

    find_button = st.button(
        "🔍 Find Candidates",
        key="find_candidates_button"
    )


# ============================================================
# PROCESS JOB DESCRIPTION
# ============================================================

if find_button:

    if not job_description.strip():

        st.warning(
            "Please enter a Job Description before finding candidates."
        )

    else:

        try:

            with st.spinner(
                "Screening candidates against the Job Description..."
            ):

                requirements, candidate_results = (
                    screen_candidates(
                        job_description
                    )
                )

                ranked_candidates = rank_candidates(
                    candidate_results
                )

            st.session_state[
                "ranked_candidates"
            ] = ranked_candidates

            st.session_state[
                "requirements"
            ] = requirements

            st.session_state[
                "job_description"
            ] = job_description

        except Exception as e:

            st.error(
                f"Error while screening candidates: {e}"
            )


# ============================================================
# DISPLAY SCREENING RESULTS
# ============================================================

if "ranked_candidates" in st.session_state:

    ranked_candidates = (
        st.session_state[
            "ranked_candidates"
        ]
    )

    if ranked_candidates:

        st.divider()

        st.markdown(
            "## 🏆 Candidate Screening Results"
        )

        st.caption(
            "Candidates ranked according to their match "
            "with the Job Description."
        )

        # ==================================================
        # SCORES
        # ==================================================

        scores = [
            float(
                candidate.get(
                    "score",
                    0
                )
            )
            for candidate
            in ranked_candidates
        ]

        total_candidates = len(
            ranked_candidates
        )

        average_score = (
            sum(scores)
            / total_candidates
        )

        highest_score = max(
            scores
        )

        (
            strong_matches,
            potential_matches,
            low_matches
        ) = calculate_category_counts(
            ranked_candidates
        )

        # ==================================================
        # KPI CARDS
        # ==================================================

        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

        with kpi1:

            st.metric(
                "👥 Candidates",
                total_candidates
            )

        with kpi2:

            st.metric(
                "📊 Average Match",
                f"{average_score:.0f}%"
            )

        with kpi3:

            st.metric(
                "🏆 Highest Match",
                f"{highest_score:.0f}%"
            )

        with kpi4:

            st.metric(
                "🟢 Strong Matches",
                strong_matches
            )

        with kpi5:

            st.metric(
                "🔴 Low Matches",
                low_matches
            )

        # ==================================================
        # SUMMARY + CHART
        # ==================================================

        ranking_col, visual_col = st.columns(
            [0.8, 1.7],
            gap="large"
        )

        # ==================================================
        # CANDIDATE SUMMARY
        # ==================================================

        with ranking_col:

            st.markdown(
                "### 👥 Candidate Summary"
            )

            st.caption(
                "Overview of the screening results."
            )

            st.metric(
                "Total Candidates",
                total_candidates
            )

            st.metric(
                "🟢 Top 3",
                strong_matches
            )

            st.metric(
                "🟡 Medium",
                potential_matches
            )

            st.metric(
                "🔴 Bottom 4",
                low_matches
            )

        # ==================================================
        # CANDIDATE VISUALIZATION
        # ==================================================

        with visual_col:

            st.markdown("### 📊 Candidate Match Visualization")
            st.caption("Green = Top 3 • Yellow = Medium • Red = Bottom 4")

            # Choose how many of the highest-ranked candidates to display.
            # The screening still evaluates all candidates; this only limits the chart.
            display_limit = st.number_input(
                "Number of candidates to display",
                min_value=5,
                max_value=max(5, total_candidates),
                value=min(10, max(5, total_candidates)),
                step=5,
                key="jd_display_limit"
            )

            displayed_candidates = ranked_candidates[:int(display_limit)]
            chart_data = []

            for index, candidate in enumerate(displayed_candidates, start=1):
                name = get_display_name(candidate.get("candidate", "Unknown"))
                score = float(candidate.get("score", 0))
                category = get_rank_category(index, total_candidates)
                chart_data.append({
                    "Candidate": name,
                    "Score": score,
                    "Rank": index,
                    "Category": category
                })

            chart_df = pd.DataFrame(chart_data).sort_values(
                "Score", ascending=True
            )
            bar_colors = [
                get_category_color(category)
                for category in chart_df["Category"]
            ]

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=chart_df["Score"],
                    y=chart_df["Candidate"],
                    orientation="h",
                    marker=dict(color=bar_colors),
                    text=[
                        f"#{rank} | {score:.0f}%"
                        for rank, score in zip(chart_df["Rank"], chart_df["Score"])
                    ],
                    textposition="outside",
                    cliponaxis=False,
                    customdata=chart_df[["Rank"]],
                    hovertemplate=(
                        "<b>Rank #%{customdata[0]}</b>"
                        "<br>Candidate: %{y}"
                        "<br>Match Score: %{x:.0f}%"
                        "<extra></extra>"
                    )
                )
            )

            fig.update_layout(
                # Give every candidate enough vertical space to avoid overlapping names.
                height=max(400, len(chart_df) * 38),
                xaxis=dict(
                    title="Match Score (%)",
                    range=[0, 110],
                    gridcolor="#292d38",
                    zeroline=False
                ),
                yaxis=dict(
                    title="Candidate",
                    automargin=True,
                    tickfont=dict(size=11)
                ),
                plot_bgcolor="#151820",
                paper_bgcolor="#151820",
                font=dict(color="#e5e7eb"),
                margin=dict(l=20, r=80, t=25, b=45),
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"Showing {len(displayed_candidates)} of "
                f"{total_candidates} screened candidates. "
                "Hover over a bar to see its rank, candidate name and match score."
            )

        # ==================================================
        # ANALYTICS
        # ==================================================

        st.divider()

        analytics1, analytics2 = st.columns(
            2,
            gap="large"
        )

        # ==================================================
        # SCORE DISTRIBUTION
        # ==================================================

        with analytics1:

            st.markdown(
                "### 📈 Match Score Distribution"
            )

            distribution_df = pd.DataFrame(
                {
                    "Score": scores
                }
            )

            fig_distribution = px.histogram(
                distribution_df,
                x="Score",
                nbins=10
            )

            fig_distribution.update_layout(

                height=320,

                plot_bgcolor="#151820",

                paper_bgcolor="#151820",

                font=dict(
                    color="#e5e7eb"
                ),

                xaxis=dict(
                    title="Match Score (%)",
                    range=[0, 100],
                    gridcolor="#292d38"
                ),

                yaxis=dict(
                    title="Number of Candidates",
                    gridcolor="#292d38"
                ),

                margin=dict(
                    l=10,
                    r=20,
                    t=20,
                    b=45
                ),

                showlegend=False
            )

            st.plotly_chart(
                fig_distribution,
                use_container_width=True
            )

        # ==================================================
        # MATCH SUMMARY
        # ==================================================

        with analytics2:

            st.markdown(
                "### 🎯 Candidate Match Summary"
            )

            category_df = pd.DataFrame(
                {
                    "Category": [
                        "Strong Match",
                        "Medium Match",
                        "Low Match"
                    ],

                    "Candidates": [
                        strong_matches,
                        potential_matches,
                        low_matches
                    ]
                }
            )

            fig_category = go.Figure()

            fig_category.add_trace(
                go.Pie(

                    labels=category_df[
                        "Category"
                    ],

                    values=category_df[
                        "Candidates"
                    ],

                    hole=0.58,

                    marker=dict(
                        colors=[
                            "#22c55e",
                            "#eab308",
                            "#ef4444"
                        ]
                    ),

                    textinfo="label+value",

                    hovertemplate=(
                        "<b>%{label}</b>"
                        "<br>Candidates: %{value}"
                        "<extra></extra>"
                    )
                )
            )

            fig_category.update_layout(

                height=320,

                paper_bgcolor="#151820",

                plot_bgcolor="#151820",

                font=dict(
                    color="#e5e7eb"
                ),

                margin=dict(
                    l=10,
                    r=10,
                    t=20,
                    b=20
                ),

                showlegend=False
            )

            st.plotly_chart(
                fig_category,
                use_container_width=True
            )

        # ==================================================
        # TOP CANDIDATE
        # ==================================================

        top_candidate_data = ranked_candidates[0]

        top_candidate = get_display_name(
            top_candidate_data.get(
                "candidate",
                "Unknown"
            )
        )

        top_candidate_score = float(
            top_candidate_data.get(
                "score",
                0
            )
        )

        top_candidate_email = (
            top_candidate_data.get(
                "email"
            )
        )

        top_candidate_phone = (
            top_candidate_data.get(
                "phone"
            )
        )

        st.divider()

        st.markdown(
            "### 🏆 Top Candidate"
        )

        top_col1, top_col2, top_col3 = st.columns(
            [3, 1, 1]
        )

        with top_col1:

            st.markdown(
                f"## 🥇 {top_candidate}"
            )

            st.caption(
                "Highest matching candidate for this Job Description."
            )

        with top_col2:

            st.metric(
                "Match Score",
                f"{top_candidate_score:.0f}%"
            )

        with top_col3:

            st.metric(
                "Position",
                "#1"
            )

        # ==================================================
        # CANDIDATE CONTACT DETAILS
        # ==================================================
        st.markdown("### 📞 Candidate Contact Details")
        st.caption("Contact details are extracted from each candidate's resume. Missing details are marked as not found.")

        with st.expander("View contact details for all ranked candidates", expanded=True):
            for rank, candidate_data in enumerate(ranked_candidates, start=1):
                candidate_label = get_display_name(candidate_data.get("candidate", "Unknown"))
                email = candidate_data.get("email") or "Not found in resume"
                phone = candidate_data.get("phone") or "Not found in resume"
                linkedin = candidate_data.get("linkedin") or "Not found in resume"
                github = candidate_data.get("github") or "Not found in resume"
                st.markdown(f"**#{rank} — {candidate_label}**")
                st.write(f"📧 Email: {email}")
                st.write(f"📱 Phone: {phone}")
                st.write(f"🔗 LinkedIn: {linkedin}")
                st.write(f"💻 GitHub: {github}")
                if rank < len(ranked_candidates):
                    st.divider()

        # ==================================================
        # EXTRACTED REQUIREMENTS
        # ==================================================

        if "requirements" in st.session_state:

            with st.expander(
                "📋 View Extracted Job Requirements"
            ):

                st.write(
                    st.session_state[
                        "requirements"
                    ]
                )

    else:

        st.info(
            "No candidate results were returned."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        AI Resume Screening & Candidate Ranking System
        • RAG • LLM • Hybrid Retrieval • Candidate Analytics
    </div>
    """,
    unsafe_allow_html=True
)