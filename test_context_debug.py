"""
Debug what context is being retrieved
"""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, ".")

from rag_engine import RAGEngine
from vector_store import VectorStore
from config import settings

# Initialize
vector_store = VectorStore(
    api_key=settings.openai_api_key,
    db_path=settings.vector_db_path,
    collection_name=settings.collection_name
)

rag = RAGEngine(
    vector_store=vector_store,
    api_key=settings.openai_api_key
)

print("Testing retrieval for follow-up question...")
print("="*70)

# Simulate conversation
history = [
    {"role": "user", "content": "ca135"},
    {"role": "assistant", "content": "کد خطای CA135..."}
]

question = "میشه بیشتر راجع بهش بگی؟"

# Test retrieval query building
print("\nIs followup?", rag._is_followup_question(question))
print("History:", history)

retrieval_query = rag._build_retrieval_query(question, history)
print(f"\nRetrieval Query: '{retrieval_query}'")
print("Expected: 'ca135'")
print("="*70)

# Check if retrieval query is correct
if retrieval_query == "ca135":
    print("\n✓ Retrieval query is CORRECT!")
else:
    print(f"\n✗ Retrieval query is WRONG! Got '{retrieval_query}' instead of 'ca135'")
