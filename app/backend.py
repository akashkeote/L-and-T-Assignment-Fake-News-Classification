import os
import re
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

load_dotenv()

# 1. FastAPI Application Initialization
app = FastAPI(
    title="TruthPulse Fact-Checking API",
    version="1.0.0",
    description="LangServe backend providing real-time RAG fake news classification."
)

# 2. Enable CORS for External Frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Model Declarations (Configured with production-ready Groq models)
main_llm = ChatGroq(
    model=os.environ.get("GROQ_MAIN_MODEL", "llama-3.3-70b-versatile"),
    temperature=0.1
)

query_llm = ChatGroq(
    model=os.environ.get("GROQ_QUERY_MODEL", "llama-3.1-8b-instant"),
    temperature=0.0
)

# 4. Search Query Extraction Chain
search_prompts = ChatPromptTemplate.from_messages([
    ("system", """You are an AI research query generator.
Analyze the user's article text and extract the core claim and entities.
Output ONLY ONE concise search query (maximum 8 words) for a search engine to fact-check this claim.
Do not output quotes, explanations, or extra commentary."""),
    ("human", "{search_input}")
])

search_chain = search_prompts | query_llm | StrOutputParser()

# 5. Search Tool Helper
def clean_query_text(text: str) -> str:
    cleaned = re.sub(r'["\'\:\!\?\.\,\-\_]', ' ', text)
    return " ".join(cleaned.split()[:7])

def search_internet(user_input: str) -> str:
    try:
        raw_query = search_chain.invoke({"search_input": user_input[:1000]}).strip()
        search_query = clean_query_text(raw_query)
    except Exception:
        search_query = clean_query_text(user_input[:80])

    snippets = []
    # 1. Try DuckDuckGo
    try:
        with DDGS() as ddgs:
            q = f"{search_query} fact check"
            results = list(ddgs.text(q, max_results=4))
            for r in results:
                snippets.append(
                    f"Source: {r.get('href')}\nTitle: {r.get('title')}\nSnippet: {r.get('body')}"
                )
    except Exception as e:
        pass

    # 2. Resilient Fallback to Google News RSS if DDGS returns empty or rate-limits
    if len(snippets) < 2:
        try:
            encoded_q = urllib.parse.quote(f"{search_query} fact check")
            rss_url = f"https://news.google.com/rss/search?q={encoded_q}&hl=en-US&gl=US&ceid=US:en"
            req = urllib.request.Request(rss_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=4) as response:
                root = ET.fromstring(response.read())
                for item in root.findall(".//item")[:4]:
                    title_elem = item.find("title")
                    link_elem = item.find("link")
                    desc_elem = item.find("description")
                    title = title_elem.text if title_elem is not None else ""
                    link = link_elem.text if link_elem is not None else ""
                    desc = desc_elem.text if desc_elem is not None else ""
                    clean_desc = re.sub(r"<[^>]+>", " ", desc).strip()
                    if title and link:
                        snippets.append(f"Source: {link}\nTitle: {title}\nSnippet: {clean_desc}")
        except Exception:
            pass

    if not snippets:
        return "No external fact checks found."

    return "\n\n---\n\n".join(snippets)

# 6. Main Fact-Checking Prompt & RAG Chain
main_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a professional AI fact-checker. 
Analyze the user's article and compare it strictly against the verified external evidence provided in <context>.

Response format:
**Claims Made**
- [Claim 1]
- [Claim 2]

**FACTS & VERIFICATION**
[✓ Fact Right] - Accurate claims (if any) with source citation.
[✗ False / Unverified Claim] - False or misleading claims.
Explanation: 1 concise paragraph explaining why it is false using the context.

**Verdict Summary**
[1-2 sentences summarizing the verdict]

**Key Sources Cited**
- [List URLs from context used to verify]

<context>
{context}
</context>"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

rag_chain = (
    RunnablePassthrough.assign(
        context=lambda x: search_internet(x["input"])
    )
    | main_prompt
    | main_llm
)

# 7. SQLite Conversation History
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string="sqlite:///chat_history.db"
    )

chatbot = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# 8. Register LangServe Routes
add_routes(
    app,
    chatbot,
    path="/fact-check"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
