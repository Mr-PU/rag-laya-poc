import os

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")

LAYA_URL = os.environ.get("LAYA_URL", "http://laya:8000")
LAYA_TIMEOUT_S = float(os.environ.get("LAYA_TIMEOUT_S", "15"))

# retrieval / rerank tuning
TOP_K_RETRIEVE = int(os.environ.get("TOP_K_RETRIEVE", "6"))   # candidates pulled from the vector store
TOP_N_CONTEXT = int(os.environ.get("TOP_N_CONTEXT", "3"))     # chunks actually sent to the LLM
LAYA_RELEVANCE_THRESHOLD = float(os.environ.get("LAYA_RELEVANCE_THRESHOLD", "0.35"))

DOCS_DIR = os.environ.get("DOCS_DIR", "/app/data/docs")
