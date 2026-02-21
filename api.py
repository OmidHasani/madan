"""
FastAPI backend for PDF Chatbot system
"""
import os
import re
import logging
import uuid
from typing import List, Optional, Dict
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from config import settings, PROJECT_ROOT
from pdf_processor import PDFProcessor
from vector_store import VectorStore
from rag_engine import RAGEngine, IntentType
from troubleshooting_toc import get_section_page_range, get_code_for_page

# Change to project root directory
os.chdir(PROJECT_ROOT)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log which API key (ماسک‌شده) از config خوانده شده است
_key = settings.openai_api_key or ""
if _key:
    _masked = f"{_key[:8]}...{_key[-4:]}" if len(_key) > 12 else "***short_key***"
else:
    _masked = "<EMPTY>"
logger.info(f"Using OpenAI API key from settings: {_masked}")

# Global instances
vector_store: Optional[VectorStore] = None
rag_engine: Optional[RAGEngine] = None
is_initialized: bool = False

# حافظه مکالمات - {session_id: {"messages": [...], "created_at": datetime}}
conversation_sessions: Dict[str, Dict] = {}


def auto_initialize():
    """خودکار initialize کردن سیستم در startup"""
    global vector_store, rag_engine, is_initialized
    
    try:
        logger.info("Auto-initializing system...")
        
        # بررسی وجود PDF
        pdf_path = settings.pdf_file_path
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return False
        
        # Initialize vector store
        logger.info("Initializing vector store...")
        vector_store = VectorStore(
            api_key=settings.openai_api_key,
            db_path=settings.vector_db_path,
            collection_name=settings.collection_name,
            embedding_model=settings.embedding_model
        )
        
        # بررسی اینکه آیا قبلاً پردازش شده
        stats = vector_store.get_collection_stats()
        already_processed = stats['document_count'] > 0
        
        # Rebuild code index to ensure new regex patterns (like H-22) are indexed
        if already_processed:
            logger.info("Rebuilding code index to include new code patterns (H-22, etc.)...")
            vector_store.rebuild_code_index()
        
        if not already_processed:
            logger.info(f"Processing PDF: {pdf_path}")
            processor = PDFProcessor(pdf_path)
            
            chunks = processor.process(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap
            )

            # Add to vector store
            vector_store.add_documents(chunks)
            logger.info("PDF processing complete")
        else:
            logger.info(f"Using existing vector database with {stats['document_count']} documents")
            # Rebuild code index to ensure new regex patterns (like H-22) are indexed
            logger.info("Rebuilding code index to include new code patterns (H-22, etc.)...")
            vector_store.rebuild_code_index()
        
        # Initialize RAG engine
        logger.info("Initializing RAG engine...")
        rag_engine = RAGEngine(
            vector_store=vector_store,
            api_key=settings.openai_api_key,
            chat_model=settings.chat_model
        )
        
        is_initialized = True
        logger.info("✅ System auto-initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Auto-initialization error: {e}")
        is_initialized = False
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for the application"""
    # Startup
    logger.info("Starting PDF Chatbot API...")
    auto_initialize()
    yield
    # Shutdown
    logger.info("Shutting down PDF Chatbot API...")


# Initialize FastAPI app
app = FastAPI(
    title="PDF Chatbot API",
    description="Intelligent chatbot for querying PDF documents using RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class QuestionRequest(BaseModel):
    """Request model for asking questions"""
    question: str = Field(..., description="User's question", min_length=1)
    top_k: int = Field(default=5, description="Number of relevant chunks to retrieve", ge=1, le=30)
    use_reranking: bool = Field(default=True, description="Whether to use reranking")
    language: str = Field(default="persian", description="Response language (persian/english)")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation history")


class AnswerResponse(BaseModel):
    """Response model for answers"""
    answer: str
    sources: List[Dict]
    confidence: str
    num_sources: int
    prompt_sent: Optional[Dict] = Field(default=None, description="پرامپت دقیق ارسالی به مدل (system + user)")
    model: Optional[str] = None
    tokens_used: Optional[int] = None
    session_id: Optional[str] = None
    intent: Optional[str] = None
    needs_clarification: Optional[bool] = None


class StatusResponse(BaseModel):
    """Response model for system status"""
    initialized: bool
    pdf_file: Optional[str]
    collection_stats: Optional[Dict]
    error: Optional[str] = None
    faiss_enabled: Optional[bool] = None


class ChatMessage(BaseModel):
    """Chat message model"""
    role: str
    content: str


class ChatRequest(BaseModel):
    """Request model for chat"""
    messages: List[ChatMessage]
    retrieve_context: bool = Field(default=True, description="Whether to retrieve context")


# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface"""
    try:
        static_path = PROJECT_ROOT / "static" / "index.html"
        with open(static_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return """
        <html>
            <head><title>PDF Chatbot</title></head>
            <body>
                <h1>PDF Chatbot API</h1>
                <p>API is running. Please initialize the system first.</p>
                <p>Docs: <a href="/docs">/docs</a></p>
            </body>
        </html>
        """


@app.get("/api/health")
async def health_check():
    """Health check endpoint for Render and other platforms"""
    return {
        "status": "healthy", 
        "version": "1.0.0",
        "initialized": is_initialized
    }


@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Get system status"""
    global is_initialized, vector_store
    
    try:
        stats = None
        faiss_enabled = None
        if is_initialized and vector_store:
            stats = vector_store.get_collection_stats()
            faiss_enabled = vector_store.use_faiss
        
        return StatusResponse(
            initialized=is_initialized,
            pdf_file=settings.pdf_file_path if is_initialized else None,
            collection_stats=stats,
            faiss_enabled=faiss_enabled
        )
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return StatusResponse(
            initialized=False,
            pdf_file=None,
            collection_stats=None,
            error=str(e),
            faiss_enabled=None
        )




@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about the document با قابلیت حافظه مکالمه"""
    global rag_engine, is_initialized, conversation_sessions
    
    if not is_initialized or not rag_engine:
        raise HTTPException(
            status_code=400,
            detail="System not initialized. Please wait for auto-initialization or check logs."
        )
    
    try:
        logger.info(f"Processing question: {request.question}")
        
        # مدیریت session
        session_id = request.session_id
        if not session_id:
            # ایجاد session جدید
            session_id = str(uuid.uuid4())
            conversation_sessions[session_id] = {
                "messages": [],
                "created_at": datetime.now()
            }
            logger.info(f"Created new session: {session_id}")
        
        # گرفتن تاریخچه مکالمه
        conversation_history = []
        if session_id in conversation_sessions:
            conversation_history = conversation_sessions[session_id]["messages"]
        
        # پاسخ به سوال با در نظر گرفتن تاریخچه
        # برای کدهای خطا، تعداد بیشتری از اسناد را بگیر
        effective_top_k = request.top_k
        if any(keyword in request.question.upper() for keyword in ['H-', 'CA', 'E-', 'D-', 'S-']):
            effective_top_k = max(request.top_k, 30)  # حداقل 30 سند برای کدهای خطا
            logger.info(f"Error code detected, using top_k={effective_top_k}")
        
        result = rag_engine.answer_question(
            question=request.question,
            top_k=effective_top_k,
            use_reranking=request.use_reranking,
            language=request.language,
            conversation_history=conversation_history
        )

        # اضافه کردن session_id به نتیجه
        result["session_id"] = session_id
        result["num_sources"] = len(result.get("sources", []))

        # ذخیره در حافظه مکالمه
        if session_id in conversation_sessions:
            conversation_sessions[session_id]["messages"].append({
                "role": "user",
                "content": request.question
            })
            conversation_sessions[session_id]["messages"].append({
                "role": "assistant",
                "content": result["answer"]
            })
            
            # حذف پیام‌های قدیمی (نگهداری حداکثر 20 پیام = 10 تبادل)
            if len(conversation_sessions[session_id]["messages"]) > 20:
                conversation_sessions[session_id]["messages"] = conversation_sessions[session_id]["messages"][-20:]

        # پرامپت ارسالی به مدل را در پاسخ بگذار (برای نمایش به کاربر)
        prompt_sent = None
        if "prompt_system" in result and "prompt_user" in result:
            prompt_sent = {
                "system_prompt": result.pop("prompt_system", None),
                "user_prompt": result.pop("prompt_user", None),
            }
        result.pop("images", None)

        return AnswerResponse(prompt_sent=prompt_sent, **result)
        
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Continue a conversation"""
    global rag_engine, vector_store, is_initialized
    
    if not is_initialized or not rag_engine:
        raise HTTPException(
            status_code=400,
            detail="System not initialized. Please wait for auto-initialization or check logs."
        )
    
    try:
        # Convert Pydantic models to dicts
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Get context if requested
        context_docs = None
        if request.retrieve_context and len(messages) > 0:
            last_user_message = next(
                (msg['content'] for msg in reversed(messages) if msg['role'] == 'user'),
                None
            )
            if last_user_message:
                context_docs = vector_store.search(last_user_message, top_k=3)
        
        response = rag_engine.chat(messages, context_docs)
        
        return {"response": response}
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


@app.post("/api/search")
async def search_documents(query: str, top_k: int = 5):
    """Search for relevant document chunks"""
    global vector_store, is_initialized
    
    if not is_initialized or not vector_store:
        raise HTTPException(
            status_code=400,
            detail="System not initialized. Please wait for auto-initialization or check logs."
        )
    
    try:
        results = vector_store.search(query, top_k=top_k)
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Error searching: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")


@app.get("/api/session/{session_id}")
async def get_session_history(session_id: str):
    """دریافت تاریخچه یک session خاص"""
    global conversation_sessions
    
    if session_id not in conversation_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "messages": conversation_sessions[session_id]["messages"],
        "created_at": conversation_sessions[session_id]["created_at"].isoformat(),
        "message_count": len(conversation_sessions[session_id]["messages"])
    }


@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    """پاک کردن تاریخچه یک session"""
    global conversation_sessions
    
    if session_id in conversation_sessions:
        del conversation_sessions[session_id]
        return {"status": "success", "message": "Session cleared"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@app.get("/api/sessions")
async def list_sessions():
    """لیست تمام session های فعال"""
    global conversation_sessions
    
    sessions_info = []
    for sid, data in conversation_sessions.items():
        sessions_info.append({
            "session_id": sid,
            "created_at": data["created_at"].isoformat(),
            "message_count": len(data["messages"])
        })
    
    return {
        "total_sessions": len(sessions_info),
        "sessions": sessions_info
    }


@app.post("/api/rebuild-code-index")
async def rebuild_code_index():
    """
    Rebuild code index (useful after regex changes to detect new code patterns like H-22)
    """
    global vector_store, is_initialized
    
    if not is_initialized or not vector_store:
        raise HTTPException(
            status_code=400,
            detail="System not initialized. Please wait for auto-initialization or check logs."
        )
    
    try:
        vector_store.rebuild_code_index()
        stats = vector_store.get_collection_stats()
        
        return {
            "status": "success",
            "message": "Code index rebuilt successfully",
            "total_codes": len(vector_store._code_to_doc_idxs),
            "collection_stats": stats
        }
    except Exception as e:
        logger.error(f"Error rebuilding code index: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to rebuild code index: {str(e)}")


class DebugRequest(BaseModel):
    """Request model for debug endpoint"""
    question: str = Field(..., description="Question to debug")
    top_k: int = Field(default=5, description="Number of relevant chunks to retrieve", ge=1, le=30)
    use_reranking: bool = Field(default=True, description="Whether to use reranking")
    language: str = Field(default="persian", description="Response language (persian/english)")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation history")


@app.post("/api/debug")
async def debug_question(request: DebugRequest):
    """
    Debug endpoint to see exactly what is retrieved and sent to LLM
    Shows: search results, context, and full prompt
    """
    global rag_engine, is_initialized, conversation_sessions
    
    if not is_initialized or not rag_engine:
        raise HTTPException(
            status_code=400,
            detail="System not initialized. Please wait for auto-initialization or check logs."
        )
    
    try:
        logger.info(f"Debug request for question: {request.question}")
        
        # مدیریت session
        session_id = request.session_id
        conversation_history = []
        if session_id and session_id in conversation_sessions:
            conversation_history = conversation_sessions[session_id]["messages"]
        
        # 1. تشخیص intent
        intent_info = rag_engine.detect_intent(request.question)
        
        # 2. ساخت retrieval query
        retrieval_query = rag_engine._build_retrieval_query(request.question, conversation_history)
        
        # 3. جستجو (همان منطق بخش کامل عیب‌یابی مثل /api/ask: اول از ایندکس، بعد TOC)
        relevant_docs = []
        # کدهای تک‌رقمی مثل H-5، E-1 هم استخراج شوند
        codes_in_query = re.findall(r"\b[A-Za-z]{1,3}-?\d{1,6}\b", (request.question or "").strip(), re.IGNORECASE)
        for code in codes_in_query:
            page_range = rag_engine.vector_store.get_page_range_for_code(code, expand_adjacent=1)
            if page_range:
                start_page, end_page = page_range
                section_chunks = rag_engine.vector_store.get_chunks_by_page_range(
                    start_page, end_page, use_parent_context=True
                )
                if section_chunks:
                    relevant_docs = section_chunks
                    break
            info = get_section_page_range(code)
            if info:
                start_page, end_page, title = info
                section_chunks = rag_engine.vector_store.get_chunks_by_page_range(
                    start_page, end_page, use_parent_context=True
                )
                if section_chunks:
                    relevant_docs = section_chunks
                    break
        if not relevant_docs:
            if request.use_reranking:
                relevant_docs = rag_engine.vector_store.search_with_reranking(
                    query=retrieval_query,
                    top_k=request.top_k
                )
            else:
                relevant_docs = rag_engine.vector_store.search(
                    query=retrieval_query,
                    top_k=request.top_k
                )
            for doc in relevant_docs[:8]:
                page = doc.get("metadata", {}).get("page")
                if page is not None:
                    try:
                        p_int = int(page)
                        inferred_code = get_code_for_page(p_int)
                        if inferred_code:
                            info = get_section_page_range(inferred_code)
                            if info:
                                start_page, end_page, title = info
                                section_chunks = rag_engine.vector_store.get_chunks_by_page_range(
                                    start_page, end_page, use_parent_context=True
                                )
                                if section_chunks:
                                    relevant_docs = section_chunks
                                    break
                    except (TypeError, ValueError):
                        pass
        
        # 4. ساخت context
        context = rag_engine._build_context(relevant_docs, include_parent=True)
        
        # 5. ساخت prompt کامل
        system_prompt = rag_engine._get_system_prompt(request.language, intent_info)
        history_text = rag_engine._history_for_prompt(conversation_history, max_items=6)
        
        # ساخت user prompt (مثل _generate_answer)
        previous_topic_note = ""
        previous_answer_note = ""
        if conversation_history and rag_engine._is_followup_question(request.question):
            last_user_q = None
            last_assistant_a = None
            
            for i in range(len(conversation_history) - 1, -1, -1):
                msg = conversation_history[i]
                if msg.get("role") == "user" and not last_user_q:
                    user_q = str(msg.get("content", "")).strip()
                    if user_q and not rag_engine._is_followup_question(user_q):
                        last_user_q = user_q
                elif msg.get("role") == "assistant" and last_user_q and not last_assistant_a:
                    last_assistant_a = str(msg.get("content", "")).strip()[:500]
                    break
            
            if last_user_q:
                previous_topic_note = f"\n🔔 سوال قبلی کاربر: {last_user_q}\n"
                if last_assistant_a:
                    previous_answer_note = f"📝 خلاصه پاسخ قبلی: {last_assistant_a}...\n\n"
                previous_topic_note += f"⚠️ سوال جدید کاربر ('{request.question}') احتمالاً به همین موضوع اشاره دارد. باید پاسخ کامل‌تر و مفصل‌تری درباره همان موضوع بدهی.\n"
        
        if request.language == "english":
            user_prompt = f"""
HISTORY:
{history_text}

{previous_topic_note}
{previous_answer_note}
LAST_USER_MESSAGE:
{request.question}

DOCUMENT_CONTEXT:
{context if context.strip() else "(empty)"}
"""
        else:
            user_prompt = f"""
HISTORY:
{history_text}

{previous_topic_note}
{previous_answer_note}
LAST_USER_MESSAGE:
{request.question}

DOCUMENT_CONTEXT:
{context if context.strip() else "(خالی)"}
"""
        
        # فرمت نتایج جستجو برای نمایش
        formatted_docs = []
        for i, doc in enumerate(relevant_docs):
            formatted_docs.append({
                "index": i + 1,
                "id": doc.get("id"),
                "text": doc.get("text", "")[:500] + "..." if len(doc.get("text", "")) > 500 else doc.get("text", ""),
                "full_text": doc.get("text", ""),
                "metadata": doc.get("metadata", {}),
                "distance": doc.get("distance"),
                "similarity_score": 1.0 - doc.get("distance", 1.0) if doc.get("distance") else None,
                "parent_context": doc.get("metadata", {}).get("parent_context", "")[:500] + "..." if len(doc.get("metadata", {}).get("parent_context", "")) > 500 else doc.get("metadata", {}).get("parent_context", "")
            })
        
        return {
            "question": request.question,
            "retrieval_query": retrieval_query,
            "intent": {
                "type": intent_info.get("intent").value if isinstance(intent_info.get("intent"), IntentType) else str(intent_info.get("intent")),
                "confidence": intent_info.get("confidence"),
                "extracted_code": intent_info.get("extracted_code")
            },
            "search_results": {
                "total_found": len(relevant_docs),
                "documents": formatted_docs
            },
            "context": {
                "length": len(context),
                "content": context
            },
            "prompt": {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "full_messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            },
            "conversation_history": conversation_history
        }
        
    except Exception as e:
        logger.error(f"Error in debug endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}")


# Mount static files if directory exists
static_dir = PROJECT_ROOT / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting server on {settings.host}:{settings.port}")
    
    uvicorn.run(
        "api:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level="info"
    )

