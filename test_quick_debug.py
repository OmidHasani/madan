import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, ".")

from rag_engine import RAGEngine
from vector_store import VectorStore
from config import settings

vector_store = VectorStore(
    api_key=settings.openai_api_key,
    db_path=settings.vector_db_path,
    collection_name=settings.collection_name
)

rag = RAGEngine(
    vector_store=vector_store,
    api_key=settings.openai_api_key
)

history = [
    {"role": "user", "content": "ca135"},
    {"role": "assistant", "content": "..."}
]

question = "میشه بیشتر راجع بهش بگی؟"

print("Is followup?", rag._is_followup_question(question))
print()

retrieval_query = rag._build_retrieval_query(question, history)
print(f"Retrieval Query: '{retrieval_query}'")
print(f"Expected: 'ca135'")
print(f"Match: {retrieval_query == 'ca135'}")
