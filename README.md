# AI-Resume-Screening-and-Job-Matching-System

An AI-powered resume screening application that uses Retrieval-Augmented Generation (RAG) to analyze resumes, answer candidate-specific questions, and rank candidates based on a given Job Description (JD).

🚀 Features
1. Resume Q&A
Ask questions about candidates stored in the resume database.
Retrieve relevant resume information using hybrid retrieval.
Supports candidate-specific queries such as:
What are the skills of Srichetan Badugu?
What is the mobile number of Srichetan Badugu?
What is the educational qualification of the candidate?
Uses retrieved resume content as the source of information.
Avoids generating information that is not present in the retrieved resume.
2. Job Description Screening
Accepts a Job Description as input.
Extracts important requirements from the JD.
Retrieves relevant candidate resumes.
Evaluates candidates against the job requirements.
Generates:
Match Score
Matched Requirements
Missing Requirements
Candidate Strengths
Candidate Gaps
AI Evaluation Reason
3. Candidate Ranking
Candidates are ranked based on their match score.
Displays the Top-K candidates.
Provides an interactive candidate ranking table.
4. Interactive Dashboard

The Streamlit dashboard provides:

Total candidates
Average match score
Highest match score
Top candidate
Candidate match-score visualization
Match-score distribution
Candidate ranking
Individual candidate analysis
5. Hybrid Retrieval

The project combines multiple retrieval techniques:

ChromaDB Vector Search
        +
MMR Retrieval
        +
BM25 Keyword Retrieval
        ↓
   Hybrid Retrieval
        ↓
 Relevant Resume Evidence

This helps retrieve both semantically relevant and keyword-specific resume information.

🧠 RAG Architecture
Resume PDFs
     │
     ▼
Document Loading
     │
     ▼
Text Chunking
     │
     ▼
OpenAI Embeddings
     │
     ▼
ChromaDB
     │
     ├───────────────┐
     ▼               ▼
Vector Retrieval   BM25 Retrieval
     │               │
     └───────┬───────┘
             ▼
       Hybrid Retrieval
             │
             ▼
       Relevant Context
             │
             ▼
        GPT-4o-mini
             │
             ▼
        Final Answer

        
📊 Job Screening Workflow
Job Description
       │
       ▼
Requirement Extraction
       │
       ▼
Resume Retrieval
       │
       ▼
Candidate Grouping
       │
       ▼
Candidate Evaluation
       │
       ▼
Match Score
       │
       ▼
Candidate Ranking
       │
       ▼
Top-K Candidates
       │
       ▼
Streamlit Dashboard
🛠️ Technologies Used
Category	Technologies
Programming	Python
UI	Streamlit
LLM	OpenAI GPT-4o-mini
Embeddings	OpenAI Embeddings
Vector Database	ChromaDB
Retrieval	MMR, Similarity Search, BM25
Framework	LangChain
Data Processing	Pandas
Visualization	Plotly
Document Processing	Unstructured, PyPDF
Environment Management	python-dotenv

📁 Project Structure
Resume Screening RAG Project/
│
├── app.py
├── rag.py
├── screening.py
├── requirements.txt
├── Innomatics Logo.png
│
├── ResumeRAG/
│   └── ChromaDB files
│
├── Resumes/
│   └── Resume PDF files
│
├── screening_results.csv
│
├── .env
├── .gitignore
└── myvenv/

.env, myvenv/, and __pycache__/ should not be uploaded to GitHub.

⚙️ Installation
1. Clone the repository
git clone <your-github-repository-url>
cd Resume-Screening-RAG-Project
2. Create a virtual environment
python -m venv myvenv

Activate it on Windows:

myvenv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt


🔑 API Key Setup

For local development, create a .env file:

OPENAI_API_KEY=your_openai_api_key

Do not upload .env to GitHub.

For Streamlit Cloud deployment, add the API key through Streamlit Secrets:

OPENAI_API_KEY = "your_openai_api_key"


▶️ Run the Application

Run the Streamlit application:

streamlit run app.py

The application provides two modes:

Resume Q&A
     │
     └── Ask questions about candidates

Job Description Screening
     │
     └── Enter JD → Rank candidates

     
💬 Example Resume Q&A
What are the skills of Srichetan Badugu?

The system retrieves the relevant resume chunks and generates an answer based only on the retrieved resume information.

If the requested information cannot be found:

I couldn't find sufficient information in the retrieved resumes.


📋 Example Job Description
We are looking for a Data Scientist / Machine Learning Engineer.

Required Skills:
- Python
- SQL
- NumPy
- Pandas
- Machine Learning
- Scikit-learn
- Feature Engineering
- Statistics
- Data Visualization

Preferred Skills:
- Deep Learning
- NLP
- Generative AI
- RAG
- LangChain
- AI Agents

The system analyzes the JD and ranks the available resumes according to their match with the requirements.

📈 Dashboard

The dashboard displays:

Candidate Match Scores
Candidate Ranking
Score Distribution
Top Candidate
Average Match Score
Highest Match Score
Matched Requirements
Missing Requirements
Candidate Strengths
Candidate Gaps
AI Evaluation Reason


🔒 Grounded AI Responses

The Resume Q&A system is designed to reduce hallucination by instructing the LLM to use only the retrieved resume evidence.

If sufficient information is not available, the system returns:

I couldn't find sufficient information in the retrieved resumes.

This makes the application more suitable for resume screening where factual accuracy is important.

🚀 Deployment

The application can be deployed using Streamlit Community Cloud.

Deployment flow:

GitHub Repository
       │
       ▼
Streamlit Community Cloud
       │
       ▼
app.py
       │
       ▼
AI Resume Screening Dashboard

Required deployment files:

app.py
rag.py
screening.py
requirements.txt
Innomatics Logo.png
ResumeRAG/
Resumes/

The OpenAI API key should be configured using Streamlit Secrets rather than committing it to the repository.

🎯 Future Improvements
Resume upload directly through the Streamlit interface
Advanced semantic candidate matching
Skill-gap visualization
Resume-job similarity score
Candidate profile extraction
Downloadable candidate reports
Authentication for recruiters
Support for multiple job descriptions
Improved candidate scoring methodology
Recruiter analytics dashboard
