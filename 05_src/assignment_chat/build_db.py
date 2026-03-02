# build_db.py
# Run this ONCE to create your ChromaDB database.
# After running, a db/ folder will be created - never delete it.


# Imports
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import os
os.environ["OPENAI_API_KEY"] = "any value"


# Load environment variables
load_dotenv(dotenv_path="../.secrets")

os.environ["OPENAI_API_KEY"] = "any_value"
client = OpenAI(
    default_headers={"x-api-key": os.getenv("API_GATEWAY_KEY")},
    base_url="https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1",
)

# --- Knowledge base: AI and tech concepts ---
documents = [
    "Retrieval Augmented Generation (RAG) is a technique that combines a retrieval system with a language model to produce grounded, accurate responses.",
    "ChromaDB is an open-source vector database used to store and query text embeddings for semantic search.",
    "Fine-tuning a language model involves training it further on a specific dataset to adapt its behavior to a new task.",
    "Prompt engineering is the practice of carefully crafting inputs to a language model to guide its outputs.",
    "Embeddings are numerical vector representations of text that capture semantic meaning, allowing similar texts to be close in vector space.",
    "OpenAI GPT models are transformer-based large language models trained on vast amounts of internet text.",
    "Hallucination in AI refers to when a model generates confident but factually incorrect information.",
    "Agents in AI systems are programs that can plan, use tools, and take actions to complete multi-step tasks.",
    "The context window of a language model is the maximum amount of text it can process in a single call.",
    "Semantic search finds results based on meaning rather than exact keyword matching.",
]

ids = [f"doc_{i}" for i in range(len(documents))]

# --- Create embeddings (text-embedding-3-small is the cheapest model) ---
print("Creating embeddings... this will take a few seconds.")
response = client.embeddings.create(
    input=documents,
    model="text-embedding-3-small"
)
embeddings = [item.embedding for item in response.data]

# --- Save to ChromaDB with file persistence ---
print("Saving to ChromaDB...")
chroma_client = chromadb.PersistentClient(path="./db")
collection = chroma_client.get_or_create_collection(name="ai_knowledge")
collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings
)

print(f"✅ Done! {len(documents)} documents stored in ./db")