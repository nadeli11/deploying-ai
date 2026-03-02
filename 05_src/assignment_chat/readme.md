# NewsBot — AI Journalist Chat Assistant

## Description
NewsBot is a witty AI chat assistant that acts as a knowledgeable journalist.
It routes user questions to one of three specialized services depending on intent.

## How to Run
```bash
cd 05_src/assignment_chat
uv run python app.py
```
Then open http://127.0.0.1:7860 in your browser.

## Services

### Service 1: News API (NewsAPI.org)
- Fetches top US technology headlines using the NewsAPI REST API.
- The raw API output is never returned verbatim — it is passed to gpt-4o-mini
  to be rephrased naturally in NewsBot's journalist style.
- Trigger phrases: "news", "headlines", "latest", "top stories"

### Service 2: Semantic Search (ChromaDB)
- Answers AI and tech knowledge questions using a persistent ChromaDB 
  vector database stored in the ./db/ folder.
- Embeddings were generated using OpenAI's text-embedding-3-small model
  by running build_db.py once.
- The database contains 10 documents covering core AI concepts such as 
  RAG, embeddings, fine-tuning, prompt engineering, and agents.
- Do not re-run build_db.py — the database is already built and included.
- Trigger phrases: "what is", "explain", "how does", "define", "tell me about"

### Service 3: Web Search (OpenAI GPT)
- Uses gpt-4o-mini with a research-focused prompt to answer current 
  events and research questions with detailed, specific information.
- Trigger phrases: "search", "find", "current", "today", "recent", "who is"

## Guardrails
- The system prompt is protected: users cannot view, reveal, or modify it.
- Prompt injection attempts are detected and blocked with keyword matching.
- Restricted topics are refused: cats/dogs, horoscopes/zodiac signs, Taylor Swift.

## Memory Management
- Conversation history is maintained throughout the session via Gradio state.
- A sliding window keeps only the last 10 turns (20 messages) to prevent 
  exceeding the model's context window.

## Implementation Decisions
- Used gpt-4o-mini to minimize token costs while maintaining quality.
- Used text-embedding-3-small (cheapest OpenAI embedding model) for the 
  knowledge base — total embedding cost is under $0.001.
- Intent detection is rule-based (keyword matching) to avoid extra API calls.
- API keys are loaded from ../05_src/.secrets using python-dotenv.
- The course API Gateway is used instead of calling OpenAI directly.