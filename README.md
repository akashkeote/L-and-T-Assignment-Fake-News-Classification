# Real-Time RAG Fake News Classification System

An automated, retrieval-augmented fact-checking pipeline built with LangChain, Groq LPUs, and real-time DuckDuckGo web search. The system takes arbitrary news articles or sensational claims, extracts the core assertions, verifies them against live authoritative journalism sources, and issues an objective, evidence-grounded verdict without hallucinations.

---

## 1. Introduction

Online misinformation spreads faster than manual fact-checking organizations can report on it. Traditional static classifiers fail when confronted with breaking events, while ungrounded LLMs frequently hallucinate plausible-sounding justifications.

This project implements an interactive Retrieval-Augmented Generation (RAG) system that couples high-throughput inference on Groq Cloud with zero-credential web retrieval via DuckDuckGo. Conversations are maintained with full context awareness using session-isolated SQLite databases.

---

## 2. Existing Modern Solutions

Modern approaches to automated fake news detection typically fall into three categories:

1. **Static Classification Models (Fine-Tuned BERT / RoBERTa):**
   * Classify text strictly from linguistic style, sentiment, and learned training patterns.
   * *Limitation:* They have zero awareness of real-world temporal facts and completely fail on zero-day events or new hoaxes not present in their training corpus.
2. **Vanilla LLMs (Zero-shot / Few-shot Prompting):**
   * Rely solely on internalized parametric memory.
   * *Limitation:* Prone to confident hallucinations, bounded by training cutoffs, and incapable of citing verifiable primary sources.
3. **Keyword-Matched Search Engines:**
   * Return ranked lists of links for user-provided keywords.
   * *Limitation:* Passing complete articles or long claims produces zero matches due to keyword over-specification, shifting the manual burden of reading and comparing sources back onto the human.

---

## 3. Problems Solved by This Project

* **Zero-Day Claim Verification:** Queries live internet indexes to analyze news published moments before the query was made.
* **Elimination of Hallucinations:** Constrains the final verdict strictly to external evidence retrieved during the pipeline run.
* **Over-Specification Search Failure:** Employs a dedicated extraction step so that entire long-form articles are distilled into compact, high-precision search keywords before executing web queries.
* **Cost & Infrastructure Barriers:** Eliminates paid vector databases and metered search APIs by pairing Groq's high-speed inference with headless DuckDuckGo scraping.

---

## 4. The Core Approach: How It Differs

Rather than feeding raw articles directly to a search engine or directly to an ungrounded LLM, this architecture implements a **Dual-Model RAG Assembly**:

[User Input Article / Claim]
│
▼
[1. Fast Extractor LLM (openai/gpt-oss-20b)] ──> Distills input into 5-7 keyword query
│
▼
[2. DuckDuckGo Engine (ddgs)]                ──> Pulls top 4 live fact-checks/reports
│
▼
[3. Context Injection (RunnablePassthrough)]  ──> Binds external snippets into prompt
│
▼

Note on Implementation:

The core pipeline logic was prototyped and tested in notebooks/rag.ipynb. That complete operational logic is ported to app/app.py wrapped inside an interactive Streamlit user interface. The UI provides live status trays and source inspection, and will continue to receive styling and multi-modal enhancements in future releases.
[4. Reasoning LLM (qwen/qwen3.8-27b) + DB]   ──> Evaluates evidence against SQLite history
│
▼
[Structured Verification Report]             ──> Claims, Verified Facts, Refutations, URLs

* **Step 1:** The extraction model analyzes the text, strips sensational filler, and isolates the core entity and claim.
* **Step 2:** DuckDuckGo queries the cleaned phrase, returning structured sources, titles, and snippets.
* **Step 3:** `RunnablePassthrough.assign` dynamically binds the retrieved evidence to `{context}` alongside the user input.
* **Step 4:** The primary reasoning model cross-references the claims against the retrieved text and outputs a structured Markdown report citing verifiable URLs.

---

## 5. Directory Structure

```bash
L-and-T-Assignment-Fake-News-Classification/
├── app/
│   ├── app.py                # Streamlit web user interface
│   ├── langserve.py          # FastAPI + LangServe REST API backend
│   └── chat_history.db       # Session-persistent conversation database
├── notebooks/
│   ├── rag.ipynb             # Core RAG experimentation & pipeline prototyping
│   ├── train.ipynb           # Model exploration and evaluation routines
│   ├── Untitled-1.py         # Exploratory scratchpad
│   └── chat_history.db       # Notebook-level conversation database
└── requirements.txt          # Production dependencies

Note on Implementation:

|The core pipeline logic was prototyped and tested in notebooks/rag.ipynb. 
|That complete operational logic is ported to app/app.py wrapped inside an interactive Streamlit user interface. 
|The UI provides live status trays and source inspection, and will continue to receive styling and multi-modal enhancements in future releases.
```
## 6. Recommended Environment
Recommended Python Version: Python 3.11.13


Compatibility Warning: Recent releases of langchain, pydantic v2, and langserve often encounter compatibility regressions and typing deprecations with newer Python versions (such as Python 3.12+). For maximum stability, use Python 3.11.x.

## 7. How to Run the Streamlit Frontend (app.py)
Prerequisites
* Create a .env file in your project root or app/ directory:

* Code snippet
```
GROQ_API_KEY=your_groq_api_key_here
```
# 1. Navigate to the repository root
```

cd L-and-T-Assignment-Fake-News-Classification
```
# 2. Create and activate a Python 3.11 virtual environment
```
python3.11 -m venv venv
source venv/bin/activate
```

# 3. Upgrade pip and install dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```

# 4. Run Streamlit from the app folder
```
cd app
streamlit run app.py
```
On Windows (PowerShell)
PowerShell
# 1. Navigate to the repository root
```
cd L-and-T-Assignment-Fake-News-Classification 
```

# 2. Create and activate a Python 3.11 virtual environment
```
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

# 3. Upgrade pip and install dependencies
```
python -m pip install --upgrade pip
pip install -r requirements.txt
```

# 4. Run Streamlit from the app folder
```
cd app
streamlit run app.py
Once running, access the web UI at http://localhost:8501.
```
## 8. Headless Backend for Custom Frontends (langserve.py)
If you want to build a custom frontend using HTML/CSS/JavaScript, React, Vue, or Next.js, use app/langserve.py. It runs a FastAPI server using LangServe that exposes the full RAG pipeline over standard HTTP endpoints with CORS enabled.

Starting the LangServe Server
Bash
cd app
python langserve.py
(Or via Uvicorn: uvicorn langserve:app --host 0.0.0.0 --port 8000 --reload)

Endpoints Available
GET /fact-check: Inspects the runnable schema and confirms server availability.

GET /fact-check/playground/: Interactive built-in web UI to test inputs and stream outputs directly in the browser.

POST /fact-check/invoke: Submits an article for fact-checking and returns the structured LLM response.

GET /docs: Auto-generated interactive OpenAPI / Swagger documentation.

Consuming via JavaScript (fetch)
JavaScript
async function verifyArticle(articleText, sessionId = "user_session_1") {
  const response = await fetch("http://localhost:8000/fact-check/invoke", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      input: {
        input: articleText
      },
      config: {
        configurable: {
          session_id: sessionId
        }
      }
    })
  });

  const data = await response.json();
  console.log("Verification Report:", data.output.content);
  return data.output.content;
}