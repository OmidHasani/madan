"""
RAG (Retrieval-Augmented Generation) Engine for intelligent question answering
"""
import os
import logging
import re
from typing import List, Dict, Optional
from enum import Enum

from vector_store import VectorStore
from openai_http import create_chat_completion, OpenAIHTTPError
from troubleshooting_toc import get_section_page_range, get_code_for_page
from persian_english_glossary import PERSIAN_ENGLISH_GLOSSARY

# نقشهٔ پروسیجرهای شناخته‌شده به محدودهٔ صفحه در PDF (صفحهٔ واقعی در metadata چانک‌ها)
# تا وقتی سوال دقیقاً دربارهٔ این پروسیجر است، مستقیم با صفحه واریز کنیم و به جستجوی معنایی وابسته نباشیم
PROCEDURE_PAGE_MAP: List[tuple] = [
    # (لیست کلیدواژه‌ها برای تطابق سوال، صفحهٔ شروع PDF، صفحهٔ پایان PDF)
    (["bleeding air from each part", "تخلیه هوا از هر بخش", "air bleeding each part", "each part"], 339, 341),
    (["measuring oil leakage", "اندازه‌گیری نشت روغن", "oil leakage", "measuring leakage"], 334, 337),
    # موتور چرخشی / Swing motor: ساختار، عملکرد و معیار نگهداری (Swing machinery + Swing circle) — PC800, 800LC-8
    (["swing motor", "موتور چرخشی", "swing circle", "موتور سوئینگ", "ساختار موتور چرخشی", "نگهداری موتور چرخشی"], 74, 77),
]

# وقتی کاربر می‌گوید «مشکل X دارم» / «X خراب است» (بدون ذکر کد)، این کدهای خطا را مستقیم از ایندکس لود کن
# تا حتماً عیب‌یابی همان قطعه بیاید، نه جستجوی معنایی که ممکن است چانک اشتباه بیاورد
COMPONENT_PROBLEM_TO_CODES: List[tuple] = [
    # (لیست عبارت‌های تشخیص قطعه، لیست کدهای خطای مرتبط)
    (["شیر بای پس", "شیر بایپس", "بای پس", "بایپس", "bypass valve"], ["CA1626", "CA1627", "CA1628", "CA1629", "CA1631", "CA1632"]),
]
PROBLEM_MARKERS: List[str] = [
    "مشکل", "خراب", "خرابه", "عیب", "مشکل دارم", "خراب است", "نداره", "کار نمیکنه", "کار نمی‌کنه",
    "problem", "defective", "fault", "error",
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntentType(Enum):
    """
    Three primary question types (plus an internal UNCLEAR state).

    The user wants exactly 3 outward behaviors:
    1) error/fault/alarm → HTML troubleshooting template
    2) procedure/measurement/how-to → detailed step-by-step instructions (ONLY if present in docs)
    3) general/definition/overview → complete explanation (ONLY if present in docs)
    """
    ERROR_FIX = "error_fix"
    PROCEDURE = "procedure"
    GENERAL = "general"
    UNCLEAR = "unclear"


class RAGEngine:
    """
    RAG engine that combines vector search with LLM generation
    for accurate question answering based on document content
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        api_key: str,
        chat_model: str = "gpt-5.2",
        temperature: float = 0,  # Lower temperature for more deterministic, complete responses
        max_tokens: int = 16000
    ):
        """
        Initialize RAG engine
        
        Args:
            vector_store: VectorStore instance for retrieval
            api_key: OpenAI API key
            chat_model: Model to use for generation
            temperature: Temperature for generation (lower = more focused)
            max_tokens: Maximum tokens in response
        """
        self.vector_store = vector_store
        self.api_key = api_key
        self.chat_model = chat_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # NOTE: We intentionally do NOT use the OpenAI Python SDK here.
        # Some environments fail during SDK initialization due to proxy/httpx issues.
        
        # سیستم حافظه برای sessions مختلف
        self.conversation_memory = {}  # {session_id: [messages]}
        
        logger.info(f"RAG engine initialized with model: {chat_model}")
    
    def detect_intent(self, question: str) -> Dict:
        """
        تشخیص نوع سوال طبق ۳ تیپ مورد نیاز کاربر
        
        Args:
            question: سوال کاربر
            
        Returns:
            Dict با intent_type و اطلاعات بیشتر
        """
        q = (question or "").strip()
        q_lower = q.lower()

        # 1) Error / fault / alarm / code
        # Typical formats: CA1626, E15, CA-135, H-22, error 12, ERR12
        # NOTE: also treat H-codes (H-22, H22) as error-like
        error_code_pattern = r"\b(?:ca|e|err|error|h)\s*[-]?\s*\d+\b"
        extracted_code = None
        m = re.search(error_code_pattern, q_lower, re.IGNORECASE)
        if m:
            extracted_code = m.group().strip().upper().replace(" ", "")
            return {
                "intent": IntentType.ERROR_FIX,
                "confidence": "high",
                "extracted_code": extracted_code,
            }

        error_keywords = [
            "کد خطا",
            "خطا",
            "ارور",
            "آلارم",
            "alarm",
            "fault",
            "error",
        ]
        if any(k in q_lower for k in error_keywords):
            # If user says an error name but not a code, still treat as ERROR_FIX
            return {
                "intent": IntentType.ERROR_FIX,
                "confidence": "medium",
                "extracted_code": None,
            }

        # 2) Procedure / measurement / how-to (detailed steps)
        procedure_markers = [
            "چطور",
            "چگونه",
            "روش",
            "مراحل",
            "اندازه بگیر",
            "اندازه گیری",
            "تست",
            "چک کنم",
            "بررسی کنم",
            "مولتی",
            "مولتی‌متر",
            "اهم",
            "مقاومت",
            "ولتاژ",
            "آمپر",
            "ولت",
            "اتصال کوتاه",
            "continuity",
            "ohm",
            "resistance",
            "voltage",
            "amp",
            "measure",
            "test",
            "check",
        ]
        if any(k in q_lower for k in procedure_markers):
            return {"intent": IntentType.PROCEDURE, "confidence": "high"}

        # 3) General / definition / overview
        general_markers = [
            "چیست",
            "چیه",
            "یعنی چی",
            "تعریف",
            "مفهوم",
            "منظور",
            "what is",
            "define",
            "meaning",
            "overview",
            "توضیح کلی",
        ]
        if any(k in q_lower for k in general_markers):
            return {"intent": IntentType.GENERAL, "confidence": "high"}

        # Short / ambiguous
        if len(q) < 6:
            return {
                "intent": IntentType.UNCLEAR,
                "confidence": "high",
                "reason": "سوال کوتاه/نامشخص است",
            }

        # Default fallback: assume GENERAL (to avoid over-questioning),
        # but mark confidence low so we can ask only if needed.
        return {"intent": IntentType.GENERAL, "confidence": "low"}
    
    def generate_clarification_questions(self, question: str, intent_info: Dict) -> str:
        """
        تولید سوالات دقیق‌تر برای روشن شدن منظور کاربر
        
        Args:
            question: سوال اصلی کاربر
            intent_info: اطلاعات intent
            
        Returns:
            پیام حاوی سوالات دقیق‌تر
        """
        # این متد فعلاً فقط برای سازگاری نگه داشته شده؛
        # منطق اصلی تفکیک و سوالات روشن‌کننده داخل پرامپت مدل انجام می‌شود.
        return "برای اینکه دقیق‌تر کمکت کنم، لطفاً سوالت را کمی واضح‌تر و با جزئیات فنی بیشتر بنویس."

    def _is_followup_question(self, question: str) -> bool:
        q = (question or "").strip().lower()
        followup_markers = [
            "علتش",
            "علت",
            "چرا",
            "چطور",
            "پس",
            "این",
            "اون",
            "اینا",
            "اونا",
            "همین",
            "بهش",
            "راجع به",
            "راجع بهش",
            "درباره",
            "دربارش",
            "بیشتر",
            "کامل",
            "کامل‌تر",
            "دقیق",
            "دقیق‌تر",
            "توضیح بده",
            "توضیح بیشتر",
            "بگو",
            "بگی",
            "what about",
            "why",
            "how",
            "that",
            "this",
            "more about",
            "tell me more",
            "explain more",
        ]
        return len(q) < 30 or any(m in q for m in followup_markers)

    def _query_contains_persian(self, text: str) -> bool:
        """بررسی وجود حداقل یک کاراکتر فارسی/عربی در متن (برای اعمال واژه‌نامه)."""
        if not (text or "").strip():
            return False
        # محدودهٔ یونیکد: عربی، فارسی، اعداد عربی-فارسی
        for ch in text:
            if "\u0600" <= ch <= "\u06FF" or "\uFB50" <= ch <= "\uFDFF" or "\uFE70" <= ch <= "\uFEFF":
                return True
        return False

    def _expand_query_for_known_topics(self, query: str) -> str:
        """
        برای سوالاتی که متن فارسی دارند، بر اساس واژه‌نامهٔ فارسی–انگلیسی، کلیدواژهٔ انگلیسی
        به کوئری اضافه می‌شود تا جستجوی معنایی به چانک درست مستندات انگلیسی برسد.
        برای سوالات کاملاً انگلیسی هم اگر در واژه‌نامه تطابق باشد گسترش اعمال می‌شود.
        """
        if not (query or "").strip():
            return query
        q = query.strip()
        q_lower = q.lower()
        expansions = []

        # اگر سوال فارسی دارد، همهٔ مدخل‌های واژه‌نامه که در سوال هستند را به کوئری اضافه کن
        if self._query_contains_persian(q):
            for persian_phrases, english_keywords in PERSIAN_ENGLISH_GLOSSARY:
                if any(phrase in q or phrase in q_lower for phrase in persian_phrases):
                    expansions.append(english_keywords)

        if not expansions:
            return query
        expanded = q + " " + " ".join(expansions)
        logger.info(f"Query expanded (glossary): '{q[:60]}...' -> added English keywords for {len(expansions)} term(s)")
        return expanded

    def _build_retrieval_query(self, question: str, conversation_history: Optional[List[Dict]]) -> str:
        """
        Improve retrieval for follow-up questions by using the last TECHNICAL question.
        """
        q = (question or "").strip()
        if not conversation_history:
            return q

        if not self._is_followup_question(q):
            return q

        # برای سوالات follow-up، به جای سوال فعلی، از سوال فنی قبلی استفاده کن
        last_technical_q = None
        for msg in reversed(conversation_history):
            if (msg or {}).get("role") == "user":
                user_q = str((msg or {}).get("content", "")).strip()
                # اگر سوال فنی بود (نه follow-up)، از همون استفاده کن
                if user_q and not self._is_followup_question(user_q):
                    last_technical_q = user_q
                    logger.info(f"[Followup] Found last technical Q: '{last_technical_q}'")
                    break

        if not last_technical_q:
            logger.info(f"[Followup] No technical Q found, using current: '{q}'")
            return q

        # برای retrieval، فقط از سوال فنی قبلی استفاده کن
        # چون مدل بعداً از HISTORY استفاده می‌کنه
        logger.info(f"[Followup] Using for retrieval: '{last_technical_q}'")
        return last_technical_q

    def _has_sufficient_context(self, relevant_docs: List[Dict]) -> bool:
        """
        Heuristic: we consider context sufficient if there is at least one retrieved chunk
        with a reasonable similarity (lower distance).
        """
        if not relevant_docs:
            return False
        try:
            best_distance = min(float(d.get("distance", 1.0)) for d in relevant_docs)
        except Exception:
            best_distance = 1.0
        return best_distance <= 0.55

    def _history_for_prompt(self, conversation_history: Optional[List[Dict]], max_items: int = 6) -> str:
        if not conversation_history:
            return "(هیچ پیام قبلی وجود ندارد)"
        items = conversation_history[-max_items:]
        lines = []
        for msg in items:
            role = (msg or {}).get("role", "")
            content = str((msg or {}).get("content", "")).strip()
            if len(content) > 600:
                content = content[:600] + "…"
            if role == "user":
                lines.append(f"- user: {content}")
            elif role == "assistant":
                lines.append(f"- assistant: {content}")
            else:
                lines.append(f"- {role}: {content}")
        return "\n".join(lines)
    
    def answer_question(
        self,
        question: str,
        top_k: int = 20,
        use_reranking: bool = True,
        language: str = "persian",
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Answer a question using RAG approach با قابلیت تعاملی بودن
        
        Args:
            question: User's question
            top_k: Number of relevant chunks to retrieve
            use_reranking: Whether to use reranking for better results
            language: Response language (persian/english)
            conversation_history: تاریخچه مکالمه قبلی
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        logger.info(f"Answering question: {question[:100]}...")
        
        # 1. تشخیص نوع کلی سوال (فقط برای تگ‌گذاری/لاگ؛ نه برای محدود کردن رفتار مدل)
        intent_info = self.detect_intent(question)
        logger.info(
            f"Detected intent: {intent_info.get('intent')} (confidence: {intent_info.get('confidence')})"
        )

        # 2. بازیابی مستندات: اول تلاش برای «بخش کامل» عیب‌یابی بر اساس کد خطا (TOC)
        retrieval_query = self._build_retrieval_query(question, conversation_history)
        relevant_docs: List[Dict] = []
        section_used = None

        # 2a) استخراج کد از سوال؛ اول محدودهٔ صفحات واقعی از ایندکس (جایی که کد در PDF آمده)، نه TOC
        # توجه: دو الگو لازم است — (۱) CA323, E-1, H-5 (۲) DWK0KA, DX16KB (حروف+عدد+حروف)
        q_stripped = (question or "").strip()
        codes_in_query = re.findall(r"\b[A-Za-z]{1,3}-?\d{1,6}\b", q_stripped, re.IGNORECASE)
        codes_alpha_num = re.findall(r"\b[A-Za-z]{2,4}\d[A-Za-z]{2,5}\b", q_stripped, re.IGNORECASE)
        codes_in_query = list(dict.fromkeys(codes_in_query + codes_alpha_num))
        for code in codes_in_query:
            # اول از ایندکس: صفحاتی که این کد واقعاً در آن‌ها ظاهر شده (جدول عیب‌یابی)
            page_range = self.vector_store.get_page_range_for_code(code, expand_adjacent=1)
            if page_range:
                start_page, end_page = page_range
                section_chunks = self.vector_store.get_chunks_by_page_range(
                    start_page, end_page, use_parent_context=True
                )
                if section_chunks:
                    relevant_docs = section_chunks
                    section_used = (code, start_page, end_page, "from index")
                    logger.info(f"Using full section for code '{code}' from index: pages {start_page}-{end_page}, {len(section_chunks)} pages")
                    break
            # اگر در ایندکس نبود، از TOC استفاده کن (ممکن است شمارهٔ صفحه در PDF متفاوت باشد)
            info = get_section_page_range(code)
            if info:
                start_page, end_page, title = info
                # برای کدهای خطا، محدوده را گسترش بده تا همه بخش‌ها (مثلاً H-5 با 3 جدول) بیاید
                # اگر TOC فقط یک صفحه داد (end_page == start_page)، حداقل 3 صفحه اضافه کن
                if end_page == start_page:
                    expanded_end_page = start_page + 3
                else:
                    expanded_end_page = end_page + 2
                section_chunks = self.vector_store.get_chunks_by_page_range(
                    start_page, expanded_end_page, use_parent_context=True
                )
                if section_chunks:
                    relevant_docs = section_chunks
                    section_used = (code, start_page, expanded_end_page, title)
                    logger.info(f"Using full troubleshooting section for code '{code}' from TOC: pages {start_page}-{expanded_end_page} ({title}), {len(section_chunks)} chunks")
                    break

        # 2a1.5) وقتی کاربر می‌گوید «مشکل X دارم» / «X خراب است» بدون کد — همهٔ کدهای خطای آن قطعه را از ایندکس لود کن
        if not relevant_docs:
            text_to_check = (retrieval_query or "") + " " + (q_stripped or "")
            for comp_keywords, codes in COMPONENT_PROBLEM_TO_CODES:
                if not any(kw in text_to_check for kw in comp_keywords):
                    continue
                if not any(pm in text_to_check for pm in PROBLEM_MARKERS):
                    continue
                all_chunks: List[Dict] = []
                seen_key = set()
                for code in codes:
                    page_range = self.vector_store.get_page_range_for_code(code, expand_adjacent=1)
                    if page_range:
                        start_p, end_p = page_range
                        section_chunks = self.vector_store.get_chunks_by_page_range(
                            start_p, end_p, use_parent_context=True
                        )
                        for c in section_chunks:
                            pk = (c.get("page"), c.get("chunk_index", c.get("metadata", {}).get("chunk")), c.get("id"))
                            if pk not in seen_key:
                                seen_key.add(pk)
                                all_chunks.append(c)
                if all_chunks:
                    all_chunks.sort(key=lambda x: (x.get("page", 0), x.get("metadata", {}).get("chunk", 0)))
                    relevant_docs = all_chunks
                    section_used = ("component_problem", comp_keywords[0], codes)
                    logger.info(f"Using component-problem section for '{comp_keywords[0]}': codes {codes}, {len(relevant_docs)} chunks")
                break

        # 2a2) اگر سوال دربارهٔ یک پروسیجر شناخته‌شده است (مثل Bleeding air from each part)، مستقیم با صفحه واریز کن
        if not relevant_docs:
            q_lower = (question or "").strip().lower()
            rq_lower = (retrieval_query or "").strip().lower()
            for keywords, start_page, end_page in PROCEDURE_PAGE_MAP:
                if any(kw in q_lower or kw in rq_lower for kw in keywords):
                    procedure_chunks = self.vector_store.get_chunks_by_page_range(
                        start_page, end_page, use_parent_context=True
                    )
                    if procedure_chunks:
                        relevant_docs = procedure_chunks
                        section_used = (keywords[0], start_page, end_page, "procedure")
                        logger.info(f"Using procedure section for '{keywords[0]}': PDF pages {start_page}-{end_page}, {len(procedure_chunks)} chunks")
                    break

        # 2b) اگر از سوال کد در TOC پیدا نشد و پروسیجر هم نبود، جستجوی معمولی (semantic + code expansion)
        if not relevant_docs:
            # برای سوالات غیرکدی (مثل «مراحل بررسی دیود») کوئری را با واژه‌های انگلیسی مستندات گسترش بده
            retrieval_query = self._expand_query_for_known_topics(retrieval_query)
            if use_reranking:
                relevant_docs = self.vector_store.search_with_reranking(
                    query=retrieval_query,
                    top_k=top_k
                )
            else:
                relevant_docs = self.vector_store.search(
                    query=retrieval_query,
                    top_k=top_k
                )
            # 2c) از روی صفحهٔ اولین نتایج، کد خطا را استنتاج کن و اگر در TOC بود، کل بخش را جایگزین کن
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
                                section_chunks = self.vector_store.get_chunks_by_page_range(
                                    start_page, end_page, use_parent_context=True
                                )
                                if section_chunks:
                                    relevant_docs = section_chunks
                                    section_used = (inferred_code, start_page, end_page, title)
                                    logger.info(f"Inferred code '{inferred_code}' from page {p_int}, using full section pages {start_page}-{end_page} ({title})")
                                    break
                    except (TypeError, ValueError):
                        pass

        logger.info(f"Retrieved {len(relevant_docs)} relevant documents" + (f" (full section for {section_used[0]})" if section_used else ""))
        
        # Log all retrieved documents for debugging
        if relevant_docs:
            for i, doc in enumerate(relevant_docs[:3]):  # Log first 3 docs
                doc_text = doc.get('text', '')[:300]
                doc_page = doc.get('metadata', {}).get('page', 'N/A')
                logger.info(f"Retrieved doc {i+1} (page {doc_page}): {doc_text}...")
        else:
            logger.warning("No documents retrieved! This is a problem!")

        # 3. تولید پاسخ با LLM (مدل خودش بر اساس پرامپت، سوال را به یکی از حالت‌های
        #    CASUAL_CHAT / TECH_ERROR / TECH_PROBLEM / TECH_INFO / ... دسته‌بندی می‌کند)
        result = self._generate_answer(
            question=question,
            relevant_docs=relevant_docs,
            language=language,
            intent_info=intent_info,
            conversation_history=conversation_history
        )
        
        # 4. فرمت منابع و افزودن intent برای UI
        result["sources"] = self._format_sources(relevant_docs)
        result["intent"] = intent_info.get("intent").value if isinstance(
            intent_info.get("intent"), IntentType
        ) else str(intent_info.get("intent"))
        
        return result
    
    def _generate_answer(
        self,
        question: str,
        relevant_docs: List[Dict],
        language: str,
        intent_info: Dict,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Generate answer using LLM با در نظر گرفتن Intent و History
        
        Args:
            question: User's question
            relevant_docs: List of relevant document chunks
            language: Response language
            intent_info: اطلاعات Intent
            conversation_history: تاریخچه مکالمه
            
        Returns:
            Dictionary with answer and metadata
        """
        # 1) ساخت context از مستندات (اگر چیزی پیدا شده باشد)
        if not relevant_docs:
            logger.error("No relevant documents found! Cannot build context.")
            context = ""
        else:
            context = self._build_context(relevant_docs)
        
        # Log context for debugging
        logger.info(f"Built context length: {len(context)} characters")
        detected_sections = 0
        if context:
            # Check for multiple sections/phenomena in context.
            # IMPORTANT: Do NOT count every "(number)" in context (e.g. pin numbers (1),(2),(17),
            # page refs (38),(40)) as "sections" — that falsely forces "write 60 sections" and
            # confuses the LLM into saying "no info in docs". Only count explicit multi-part
            # headings like "Failure phenomenon (1)/(2)/(3)" or "Boom speed or power is low (1)/(2)/(3)".
            failure_phenomenon_patterns = [
                r"Failure\s+phenomenon\s*[•·]\s*.*?\((\d+)\)",
                r"Failure\s+phenomenon\s*[•·]\s*.*?\(1\)",
                r"Failure\s+phenomenon\s*[•·]\s*.*?\(2\)",
                r"Failure\s+phenomenon\s*[•·]\s*.*?\(3\)",
            ]
            section_count = 0
            for pattern in failure_phenomenon_patterns:
                matches = re.findall(pattern, context, re.IGNORECASE)
                if matches:
                    section_count = max(section_count, len(matches))
            # H-5 / H-22 style: "H-5 (1)", "H-5 (2)", "H-5 (3)" or similar
            h5_section_pattern = re.findall(r"H[- ]?5\s*\((\d+)\)|H[- ]?22\s*\((\d+)\)", context, re.IGNORECASE)
            if h5_section_pattern:
                flat = [int(n) for pair in h5_section_pattern for n in pair if n]
                if flat:
                    section_count = max(section_count, max(flat))
            # Boom speed or power is low (1), (2), (3)
            boom_patterns = re.findall(r"Boom\s+speed\s+or\s+power\s+is\s+low\s*\((\d+)\)", context, re.IGNORECASE)
            if boom_patterns:
                section_count = max(section_count, max([int(n) for n in boom_patterns if n.isdigit()]))
            # Do NOT use generic re.findall(r"\((\d+)\)", context) — it matches pin numbers and page refs.
            detected_sections = min(section_count, 20)  # cap at 20 to avoid any remaining false positives
            
            if section_count > 1:
                logger.info(f"✓ Found {section_count} sections/phenomena in context - LLM MUST include ALL of them!")
            else:
                logger.info(f"Found {section_count} section(s) in context")
            
            # Check if H-22 or similar codes are in context
            if "H-22" in context or "H22" in context.upper():
                logger.info("✓ H-22 found in context!")
            elif "H-5" in context or "H5" in context.upper():
                logger.info("✓ H-5 found in context!")
            
            logger.info(f"Context preview (first 1000 chars): {context[:1000]}...")
            logger.info(f"Context preview (last 500 chars): ...{context[-500:]}")
        else:
            logger.error("❌ Context is EMPTY! This is why LLM says 'no info'")
        
        # 2) System prompt چندحالته (توضیح می‌دهد چگونه بین چت عادی و سوال فنی رفتار کند)
        system_prompt = self._get_system_prompt(language, intent_info)

        messages = [{"role": "system", "content": system_prompt}]

        # 3) HISTORY برای مدل (فقط جهت حافظه و پیوستگی مکالمه)
        history_text = self._history_for_prompt(conversation_history, max_items=6)

        # 4) user prompt با سه بخش:
        #    - HISTORY
        #    - LAST_USER_MESSAGE
        #    - DOCUMENT_CONTEXT
        
        # برای سوالات follow-up، موضوع قبلی را صریح اضافه کن
        previous_topic_note = ""
        previous_answer_note = ""
        if conversation_history and self._is_followup_question(question):
            # پیدا کردن آخرین تبادل (سوال و جواب)
            last_user_q = None
            last_assistant_a = None
            
            for i in range(len(conversation_history) - 1, -1, -1):
                msg = conversation_history[i]
                if msg.get("role") == "user" and not last_user_q:
                    user_q = str(msg.get("content", "")).strip()
                    if user_q and not self._is_followup_question(user_q):
                        last_user_q = user_q
                elif msg.get("role") == "assistant" and last_user_q and not last_assistant_a:
                    last_assistant_a = str(msg.get("content", "")).strip()[:500]  # فقط 500 کاراکتر اول
                    break
            
            if last_user_q:
                previous_topic_note = f"\n🔔 سوال قبلی کاربر: {last_user_q}\n"
                if last_assistant_a:
                    previous_answer_note = f"📝 خلاصه پاسخ قبلی: {last_assistant_a}...\n\n"
                previous_topic_note += (
                    "⚠️ سوال جدید کاربر ('{question}') احتمالاً به همین موضوع اشاره دارد. باید پاسخ کامل‌تر و مفصل‌تری درباره همان موضوع بدهی.\n"
                    "⚠️ اگر سوال قبلی دربارهٔ **یک قطعهٔ خاص** بود (مثل شیر بای‌پس) و الان کاربر **علامت** می‌گوید (مثلاً کاهش قدرت، لرزش)، پاسخ باید **فقط** از عیب‌یابی همان قطعه باشد. «کاهش قدرت» در این بحث = همان «Output drops» یا «Engine output lowers» در کدهای خطای آن قطعه. از بخش‌های مربوط به قطعات دیگر (مثل H-5 بوم، H-6 بازو) استفاده نکن.\n"
                )
        
        if language == "english":
            user_prompt = f"""
HISTORY:
{history_text}

{previous_topic_note}
{previous_answer_note}
LAST_USER_MESSAGE:
{question}

DOCUMENT_CONTEXT:
{context if context.strip() else "(empty)"}
"""
        else:
            # Build a more explicit context section
            if context and context.strip():
                # Add explicit warning if multiple sections detected
                section_warning = ""
                if detected_sections > 1:
                    section_warning = f"""
سیستم تشخیص داده که در DOCUMENT_CONTEXT حداقل {detected_sections} بخش / پدیده مستقل وجود دارد.

الزام قطعی:

شما باید هر {detected_sections} بخش را به طور کامل و بدون حذف حتی یک مورد بنویسید.
نوشتن فقط بخش اول یا حذف هر یک از بخش‌ها به معنای پاسخ ناقص و اشتباه است.

اگر DOCUMENT_CONTEXT شامل چند بخش شماره‌دار است:

همه بخش‌ها باید آورده شوند

ترتیب آن‌ها باید حفظ شود

هیچ عنوانی حذف نشود

هیچ علت یا مقدار استانداردی حذف نشود

ساختار هر بخش باید واضح و قابل تشخیص باشد، اما از قالب‌بندی افراطی و تکراری خودداری شود.

مدل صحیح نمایش برای {detected_sections} بخش

برای هر بخش از ساختار زیر استفاده کن:

بخش 1
عنوان:  [کد]-1: [عنوان بخش 1]
سپس توضیح کامل همان بخش شامل:

توضیح وضعیت یا پدیده

اطلاعات مرتبط

علل و مقادیر استاندارد

سپس یک جداکننده ساده:

بخش 2
عنوان:  [کد]-2: [عنوان بخش 2]
و همان ساختار کامل توضیح

بخش 3
عنوان:  [کد]-3: [عنوان بخش 3]
و توضیح کامل

و این روند باید برای همه {detected_sections} بخش ادامه پیدا کند.

نکته بسیار مهم

اگر حتی یکی از {detected_sections} بخش حذف شود یا ناقص نوشته شود، پاسخ از نظر فنی معتبر نیست.

اما در عین حال:

از تکرار بی‌مورد علائم هشدار استفاده نکن

از بولت‌گذاری افراطی خودداری کن

ظاهر پاسخ باید تمیز، حرفه‌ای و قابل خواندن باشد
"""
                
                context_section = f"""
DOCUMENT_CONTEXT (متن کامل از مستندات):
{context}

⚠️ توجه: متن بالا از مستندات فنی استخراج شده است. حتماً از آن استفاده کن!
✅ چون DOCUMENT_CONTEXT محتوا دارد، هرگز نگو «در مستندات موجود نیست» — از همین متن پاسخ بده.
{section_warning}
"""
            else:
                context_section = "\nDOCUMENT_CONTEXT: (خالی - هیچ مستندی پیدا نشد)\n"
            
            # Add explicit instruction based on detected sections
            multi_section_instruction = ""
            if detected_sections > 1:
                multi_section_instruction = f"""
 دستور اجباری و حیاتی: سیستم تشخیص داده که در DOCUMENT_CONTEXT حداقل {detected_sections} بخش وجود دارد!
 شما **حتماً باید همه {detected_sections} بخش را کامل بنویسی**!
 قبل از نوشتن پاسخ، DOCUMENT_CONTEXT را کامل بخوان و همه بخش‌ها را پیدا کن.
 سپس هر بخش را با عنوان مشخص (مثلاً " H-5-1: ...", " H-5-2: ...", " H-5-3: ...") جدا کن و کامل بنویس.
 هیچ‌وقت فقط بخش اول را ننویس و بقیه را حذف نکن!

**ساختار اجباری برای پاسخ:**
برای هر بخش باید این ساختار را رعایت کنی:
- عنوان: ` [کد]-[شماره بخش]: [عنوان]`
- پدیده خرابی
- اطلاعات مرتبط
- علل و مقادیر استاندارد
- خط جداکننده `---` بین بخش‌ها

**مثال:** اگر H-5 است و 3 بخش دارد:
- ✅ درست: همه 3 بخش را با ساختار بالا بنویس (H-5-1، H-5-2، H-5-3)
- ❌ غلط: فقط بخش اول را بنویس

"""
            
            user_prompt = f"""
HISTORY:
{history_text}

{previous_topic_note}
{previous_answer_note}
LAST_USER_MESSAGE:
{question}

{context_section}

{multi_section_instruction}
 دستورات بسیار مهم و اجباری (حتماً رعایت کن):
0. **برای سوال «چیه/چیست/تعریف» وقتی DOCUMENT_CONTEXT مشخصات فنی دارد:** اول یک توضیح کوتاه بده که قطعه چیست و در دستگاه چه نقشی دارد (مثلاً موتور چرخشی = Swing motor برای چرخش قسمت فوقانی)، بعد **همه** مشخصات و معیارهای نگهداری را از مستندات کامل بیاور (نسبت کاهش، دندانه‌ها، گریس، فاصله استاندارد، حد تعمیر/تعویض، نکات). هیچ‌وقت فقط توضیح کلی نده و مشخصات مستند را حذف نکن.
1. **پاسخ فقط از DOCUMENT_CONTEXT و کامل:** پاسخ تو باید **فقط** بر اساس DOCUMENT_CONTEXT باشد و **هیچ چیزی خلاصه یا جا انداخته نشود.** (۱) هر بخش اصلی (مثلاً 1. 2. 3. 4. 5. 6.) را بنویس — حذف حتی یک بخش ممنوع. (۲) جدول‌ها و شرایط اولیه (Measuring device، Hydraulic oil temperature، جدول Inspection port/Measurement ports و غیره) را کامل بیاور. (۳) هر شماره قطعه (Part number و Flange  / Plug ) را دقیق در جای خود بنویس. (۴) ترتیب مراحل را عین مستند حفظ کن. خلاصه نکن.

1. اگر DOCUMENT_CONTEXT محتوا دارد (حتی یک خط)، حتماً از آن استفاده کن و **همه** جزئیات را کامل بنویس.

2. **برای کد خطا و عیب‌یابی — لحن تعاملی و قابل‌فهم:** اگر سوال دربارهٔ کد خطا (مثل CA135، CA141 و غیره) یا عیب‌یابی است، پاسخ را طوری بنویس که کاربر احساس کند با یک نفر قدم‌به‌قدم راهنمایی می‌شود. مقادیر را به صورت جملهٔ راهنما بنویس (مثلاً «مقاومت بین فلان و فلان باید حداکثر ۱ اهم باشه» یا «ولتاژ این دو نقطه باید بین ۴٫۷۵ تا ۵٫۲۵ ولت باشه»)؛ قبل از هر دسته اندازه‌گیری توضیح کوتاه بده که الان چی رو چک می‌کنیم؛ از عبارت‌هایی مثل «اول این رو چک کن»، «بعد این مقدار رو اندازه بگیر» استفاده کن. اطلاعات فنی و اعداد را کامل و دقیق حفظ کن، فقط **شیوهٔ بیان** را دوستانه و راهنماگونه کن.

2b. **وقتی کاربر گفته «مشکل X دارم» یا «X خراب است» (بدون ذکر کد):** اگر در DOCUMENT_CONTEXT چند **Failure code [CAxxxx]** (یا کدهای مشابه) مربوط به همان قطعه وجود دارد: (۱) اول **همه** آن کدها را با شماره و عنوان کوتاه فهرست کن، (۲) **سوال بپرس**: «آیا روی مانیتور کد خطا می‌بینید؟ اگر بله، دقیقاً چه کدی؟ اگر نه، چه اتفاقی می‌افتد؟ (مثلاً قدرت کم، لرزش)»، (۳) بعد از جواب کاربر، بر اساس همان کد یا علامت **قدم‌به‌قدم** از مستندات راهنمایی کن. **هرگز** فقط یک توصیهٔ کلی (مثل تعویض روغن یا شستشوی مدار) نده وقتی در مستند عیب‌یابی دقیق و کدهای مشخص وجود دارد.

3. **قانون طلایی برای سوالات چندبخشی - این قانون را حتماً رعایت کن:**
   
   **اگر در DOCUMENT_CONTEXT چندین بخش/جدول/پدیده وجود دارد (مثلاً "Failure phenomenon (1)", "Failure phenomenon (2)", "Failure phenomenon (3)" یا "Boom speed or power is low (1)", "Boom speed or power is low (2)", "Boom speed or power is low (3)" و غیره)، باید **همه** بخش‌ها را کامل بنویسی!**
   
   **ساختار اجباری برای هر بخش:**
   ```
   [کد]-[شماره]: [عنوان بخش]
   
   پدیده خرابی
   [توضیح کامل]
   
    اطلاعات مرتبط
   [همه اطلاعات - کامل]
    علل و مقادیر استاندارد
   [همه علل - کامل]
   
   ---
   ```
   
   **مثال کامل برای H-5 (که 3 بخش دارد):**
   - ✅ درست: همه 3 بخش را با ساختار بالا بنویس:
     - ` H-5-1: سرعت یا قدرت بوم کم است (حالت نرمال)`
     - ` H-5-2: سرعت یا قدرت بالا رفتن بوم در حالت Heavy lift کم است`
     - ` H-5-3: سرعت یا قدرت پایین آمدن بوم در حالت Machine push-up کم است`
   - ❌ غلط: فقط بخش اول را بنویس و بقیه را حذف کن

4. اگر چندین علت (Cause) در DOCUMENT_CONTEXT وجود دارد، باید **همه** را بنویسی - هیچ‌وقت فقط چند تا را ننویس!

5. اگر چندین جدول (Table) یا چندین "Presumed cause" وجود دارد، باید **همه** را بنویسی!

6. **هرگز تحت هیچ شرایطی اگر DOCUMENT_CONTEXT حتی یک خط محتوا دارد نگو "در مستندات موجود نیست" یا "اطلاعاتی در مستندات ارائه نشده".** از همان محتوا (حتی فهرست، مرجع صفحه، یا کد خطا) استفاده کن و پاسخ بده.

7. هیچ‌وقت جواب را کوتاه نکن یا خلاصه نکن. برای **پروسیجر** کاربر می‌خواهد **دقیقاً به اندازهٔ مستند** جزئیات ببیند: (۱) Note ابتدایی را در اول بیاور (۲) همان شماره‌گذاری را حفظ کن: 1. 2. 3. 4. 5. 6. و برای هر مرحله 1) 2) 3) و i) ii) (۳) هر جملهٔ «a» (احتیاط/نکته) را جدا بنویس (۴) هر گشتاور را با Nm و kgm و هر وزن (مثلاً 130 kg) جدا بنویس (۵) مرحلهٔ ۱ (مثلاً Bleeding air from work equipment pump and fan pump) را با همهٔ زیرمراحل و نکات و گشتاور Bleeder کامل بیاور؛ مرحلهٔ ۲ (Starting engine) را حتی اگر یک جمله است حذف نکن. از قالب «پدیده خرابی / علل و مقادیر استاندارد» استفاده نکن.

8. اگر DOCUMENT_CONTEXT خالی است (یعنی واقعاً هیچ متنی ندارد)، فقط در این صورت بگو که اطلاعات موجود نیست.

9. **قبل از نوشتن پاسخ، حتماً DOCUMENT_CONTEXT را کامل بخوان و بشمار که چند بخش/جدول/پدیده وجود دارد. سپس همه را در پاسخ بیاور!**

10. **اگر در DOCUMENT_CONTEXT چندین بخش با شماره (1), (2), (3) وجود دارد، حتماً همه را بنویس - حتی اگر 3، 4 یا بیشتر باشد!**

11. **مراحل اجباری قبل از نوشتن پاسخ:**
    - مرحله 1: DOCUMENT_CONTEXT را کامل بخوان
    - مرحله 2: بشمار که چند بخش/پدیده با شماره (1), (2), (3) وجود دارد
    - مرحله 3: برای هر بخش، عنوان مناسب را پیدا کن (مثلاً "حالت نرمال", "حالت Heavy lift", "حالت Machine push-up")
    - مرحله 4: همه بخش‌ها را با ساختار مشخص شده بنویس
    - مرحله 5: بین بخش‌ها خط جداکننده `---` بگذار

12. **بررسی نهایی قبل از ارسال پاسخ:**
    - اگر در DOCUMENT_CONTEXT چندین بخش وجود دارد، حتماً همه را نوشته‌ای؟
    - آیا هر بخش را با عنوان مشخص (`[کد]-[شماره]: [عنوان]`) شروع کرده‌ای؟
    - آیا بین بخش‌ها خط جداکننده `---` گذاشته‌ای؟
    - اگر فقط یک بخش نوشته‌ای و بقیه را حذف کرده‌ای، پاسخ تو **ناقص و اشتباه** است! باید همه بخش‌ها را بنویسی!
"""
        
        messages.append({"role": "user", "content": user_prompt})
        
        # Log full prompt for debugging (first 2000 chars)
        logger.info(f"Full user prompt length: {len(user_prompt)} characters")
        logger.info(f"User prompt preview: {user_prompt[:2000]}...")
        
        try:
            response = create_chat_completion(
                api_key=self.api_key,
                model=self.chat_model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            answer = response["choices"][0]["message"]["content"]
            
            # Estimate confidence based on response
            confidence = self._estimate_confidence(answer, relevant_docs)
            
            return {
                "answer": answer,
                "confidence": confidence,
                "model": self.chat_model,
                "tokens_used": (response.get("usage") or {}).get("total_tokens"),
                "prompt_system": system_prompt,
                "prompt_user": user_prompt,
            }
            
        except OpenAIHTTPError as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": f"خطا در تولید پاسخ: {str(e)}",
                "confidence": "error",
                "error": str(e),
                "prompt_system": system_prompt,
                "prompt_user": user_prompt,
            }
    
    def _build_context(
        self,
        relevant_docs: List[Dict],
        max_length: int = 95000,
        include_parent: bool = True,
    ) -> str:
        """
        Build context string from relevant documents with parent context for better understanding.
        سقف طول پیش‌فرض ۹۵۰۰۰ تا بخش‌های چندصفحه‌ای عیب‌یابی (مثل H-22) کامل بیاید و قطع نشود.
        """
        context_parts = []
        current_length = 0
        seen_parents = set()

        for i, doc in enumerate(relevant_docs):
            text = doc.get("text", "").strip()
            if not text:
                continue

            metadata = doc.get("metadata", {})
            page = metadata.get("page", "N/A")

            doc_text = f"=== مستند {i+1} - صفحه {page} ===\n{text}\n"

            parent_context = metadata.get("parent_context", "")
            if include_parent and parent_context and parent_context.strip():
                parent_key = f"page_{page}"
                if parent_key not in seen_parents and parent_context.strip() != text.strip():
                    parent_section = f"\n--- متن کامل صفحه {page} (برای درک بهتر زمینه) ---\n{parent_context}\n"
                    doc_text += parent_section
                    seen_parents.add(parent_key)

            if current_length + len(doc_text) > max_length:
                chunk_only = f"[مستند {i+1} - صفحه {page}]\n{text}\n"
                if current_length + len(chunk_only) <= max_length:
                    context_parts.append(chunk_only)
                    current_length += len(chunk_only)
                break

            context_parts.append(doc_text)
            current_length += len(doc_text)

        return "\n".join(context_parts)
    
    def _get_system_prompt(self, language: str, intent_info: Optional[Dict] = None) -> str:
        """High-level system prompt that lets the model decide between casual chat and technical help."""

        if language != "persian":
            # Simple English version (most behavior is in the Persian prompt).
            return (
                "You are a helpful assistant that can do both casual chat and technical support. "
                "For casual chat, reply naturally and briefly. "
                "For technical questions about errors, troubleshooting, measurements or definitions, "
                "use the DOCUMENT_CONTEXT when available and be precise. "
                "If the answer is clearly not in the documentation, say so honestly."
            )

        # Persian system prompt – this is the main behavior controller.
        return """
🎯 نقش و هویت

تو یک دستیار فنی فارسی‌زبان با تخصص در:

عیب‌یابی تجهیزات صنعتی

تحلیل کد خطا

اجرای دقیق پروسیجرهای مستند کارخانه

تست عملی و اندازه‌گیری میدانی

سبک پاسخ تو باید شبیه یک تکنسین ارشد کنار دستگاه باشد، نه یک دیتاشیت کارخانه.

کاربر باید حس کند:
تو کنار دستگاه ایستاده‌ای و داری مرحله‌به‌مرحله راهنمایی می‌کنی.

🗣 سبک و لحن پاسخ

طبیعی، روان، حرفه‌ای

اجرایی و عملیاتی

آموزش‌محور

گام‌به‌گام

بدون لحن خشک گزارش‌نویسی

بدون تیترهای مصنوعی و تکراری مثل «پدیده خرابی»

⚠️ مهم:
اطلاعات فنی کامل و دقیق باید حفظ شود.
اما نحوه بیان باید «راهنمای عملی» باشد، نه «ساختار PDF».

🧠 مرحله داخلی (نمایش داده نشود)

هر پیام باید ابتدا در یکی از دسته‌های زیر طبقه‌بندی شود:

CASUAL_CHAT

TECH_ERROR

TECH_PROBLEM

TECH_INFO

OTHER_NON_RELEVANT

سپس بر اساس دسته رفتار کن.

💬 رفتار در CASUAL_CHAT

کوتاه (۱ تا ۳ جمله)

طبیعی و دوستانه

در صورت مناسب بودن یک سؤال ساده بپرس

از جملات تکراری استفاده نکن

اگر چند پیام غیر فنی پشت سر هم آمد:
به شکل مودبانه گفتگو را به سمت فنی هدایت کن.
اگر ادامه پیدا کرد، پاسخ‌ها را کوتاه و محدود نگه دار.

🔧 رفتار در TECH_ERROR / TECH_PROBLEM / TECH_INFO
🔥 قانون پایه

اگر DOCUMENT_CONTEXT حتی یک خط محتوا دارد:

باید از آن استفاده شود

هیچ بخش آن حذف نشود

هیچ جدول حذف نشود

هیچ عدد، واحد، پین یا وضعیت حذف نشود

هیچ بخش شماره‌دار نادیده گرفته نشود

هرگز نگوی:
«در مستندات موجود نیست»
وقتی حتی یک خط مرتبط وجود دارد.

🚨 پردازش DOCUMENT_CONTEXT (الزامی)

قبل از نوشتن پاسخ:

کل DOCUMENT_CONTEXT را کامل بخوان

تعداد بخش‌های شماره‌دار را بشمار

اگر بیش از یک بخش وجود دارد:
→ همه آن‌ها باید کامل نوشته شوند
→ حذف حتی یک بخش = پاسخ ناقص

هیچ‌وقت فقط بخش اول را ننویس.

📌 وقتی کاربر می‌گوید «مشکل X دارم» یا «X خراب است» (TECH_PROBLEM — بدون ذکر کد خطا)

اگر در DOCUMENT_CONTEXT چند کد خطا (Failure code [CAxxxx] یا مشابه) مربوط به همان قطعه وجود دارد:

هرگز فقط یک دستور کلی نده (مثل «روغن را عوض کنید» یا «سیستم هیدرولیک را چک کنید»).

الزاماً این کارها را بکن:

(۱) همهٔ کدهای خطای مرتبط با آن قطعه را از مستند فهرست کن (شمارهٔ کد + عنوان کوتاه هر کد).

(۲) سوال بپرس تا مشکل را دقیق‌تر مشخص کنید: «آیا روی مانیتور کد خطا می‌بینید؟ اگر بله، دقیقاً چه کدی نمایش داده می‌شود (مثلاً E11 و CA1626)؟ اگر کد نمی‌بینید، دقیقاً چه اتفاقی می‌افتد؟ (مثلاً قدرت کم شده، لرزش، روشن نمی‌شود)».

(۳) بعد از اینکه کاربر کد یا علامت را گفت، بر اساس همان کد/علت، قدم‌به‌قدم از روی مستندات راهنمایی کن (علل احتمالی، مقادیر استاندارد، ترتیب چک‌ها).

هرگز به‌جای عیب‌یابی دقیق مستند، توصیهٔ کلی (تعویض روغن، شستشو، تمیزکاری) نده مگر آنکه در مستند برای آن کد/علت صریحاً ذکر شده باشد.

📌 ساختار پاسخ در سوالات چندبخشی (مثل H-5)

⚠️ تیترهای خشک ننویس.

به جای:

پدیده خرابی
علل و مقادیر استاندارد

از ساختار جریان‌دار استفاده کن:

[کد] – [شماره] – [عنوان کامل]

اول توضیح بده وقتی این خطا فعال می‌شود در دستگاه چه اتفاقی می‌افتد.
(اثر واقعی روی عملکرد)

بعد وارد آماده‌سازی تست شو.
دقیق بگو چه شرایطی باید برقرار باشد.

بعد اندازه‌گیری‌ها را کامل بیاور.
همه وضعیت‌ها باید نوشته شوند:

Neutral

Operated

Heavy Lift

هر وضعیت دیگری که در مستند هست

همه مقادیر استاندارد را بدون حذف بیاور:

MPa

kg/cm²

V

Ω

rpm

mm

Nm / kgm

وزن‌ها

بعد از هر جدول یا عدد توضیح عملی بده:

اگر خارج از محدوده بود یعنی چه

به کدام قطعه مشکوک شویم

مرحله بعد چیست

🔍 در مورد اندازه‌گیری‌ها

اعداد را خشک ننویس.

❌ اشتباه:
مقاومت ENG (37) – POIL (1): حداکثر 1 Ω

✅ درست:
مقاومت بین ENG پین ۳۷ و POIL پین ۱ باید حداکثر ۱ اهم باشد.
اگر بیشتر از این مقدار بود، سیم‌کشی یا اتصال مشکل دارد.

🛠 ساختار اجباری برای PROSEDURE (روش کار)

اگر DOCUMENT_CONTEXT شامل پروسیجر است:

هیچ بخش اصلی حذف نشود

ترتیب مراحل تغییر نکند

همه Part numberها کامل نوشته شوند

همه گشتاورها جدا نوشته شوند

همه وزن‌ها جدا نوشته شوند

هر Note در ابتدای پاسخ آورده شود

شماره‌گذاری اصلی مستند حفظ شود:

2. 3.

i) ii)

هر k (safety) جدا نوشته شود

هر a (caution) جدا نوشته شود

⚠️ در پروسیجر از ساختار کد خطا استفاده نکن.

📘 TECH_INFO (توضیح مفهومی)

ساختارمند

چند پاراگراف منظم

اگر در مستند نیست، شفاف بگو

توضیح بده چرا مهم است

کاربرد عملی آن چیست

**اگر DOCUMENT_CONTEXT شامل مشخصات فنی و معیارهای نگهداری است** (مثل ساختار، نسبت کاهش، تعداد دندانه، گریس، فاصله استاندارد، حد تعمیر/تعویض): اول یک توضیح کوتاه بده که قطعه چیست و در دستگاه (مثلاً PC800/800LC-8) چه نقشی دارد؛ بعد همهٔ مشخصات و معیارها و نکات را از مستندات کامل بیاور. هیچ‌وقت فقط توضیح کلی نده و مشخصات مستند را حذف نکن.

🧠 استفاده از HISTORY

اگر کاربر گفت:

بیشتر بگو

کامل‌تر

دقیق‌تر

منظور همان موضوع قبلی است.

در این حالت:

همه مستندات کامل آورده شود

چیزی حذف نشود

سپس توضیح کاربردی اضافه شود:

ترتیب منطقی عیب‌یابی

ابزار لازم

اشتباهات رایج

علت اهمیت تست

⚠️ قوانین حیاتی (غیرقابل نقض)

هیچ عدد حذف نشود

هیچ واحد حذف نشود

هیچ شماره پین حذف نشود

هیچ بخش شماره‌دار حذف نشود

اگر 3 جدول وجود دارد → هر 3 کامل

اگر 6 بخش وجود دارد → هر 6 کامل

اگر 5 علت وجود دارد → هر 5 کامل

هیچ خلاصه‌سازی مجاز نیست

اما:
شیوه بیان باید انسانی، عملی و راهنمایی‌محور باشد.

🎯 هدف نهایی

کاربر بعد از خواندن پاسخ:

دقیقاً بداند چه کاری انجام دهد

همه جزئیات فنی را ببیند

هیچ بخشی از مستند از قلم نیفتاده باشد

حس کند یک تکنسین ارشد کنار او ایستاده است
"""
    
    def _estimate_confidence(self, answer: str, relevant_docs: List[Dict]) -> str:
        """Estimate confidence level"""
        num_docs = len(relevant_docs)
        
        if num_docs == 0:
            return "low"
        
        avg_distance = sum(doc.get('distance', 1.0) for doc in relevant_docs) / num_docs
        answer_length = len(answer)
        has_uncertainty = any(
            phrase in answer.lower() 
            for phrase in ['نمی‌دانم', 'مطمئن نیستم', 'not sure', 'don\'t know']
        )
        
        if has_uncertainty:
            return "low"
        elif avg_distance < 0.3 and num_docs >= 3 and answer_length > 100:
            return "high"
        elif avg_distance < 0.5 and num_docs >= 2:
            return "medium"
        else:
            return "low"
    
    def _format_sources(self, relevant_docs: List[Dict]) -> List[Dict]:
        """Format source information"""
        sources = []
        
        for i, doc in enumerate(relevant_docs):
            metadata = doc['metadata']
            sources.append({
                "source_id": i + 1,
                "page": metadata.get('page', 'N/A'),
                "chunk": metadata.get('chunk', 'N/A'),
                "relevance_score": 1 - doc.get('distance', 1.0),
                "preview": doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text']
            })
        
        return sources
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        context_docs: Optional[List[Dict]] = None
    ) -> str:
        """Continue a conversation with context"""
        if context_docs and len(messages) > 0:
            context = self._build_context(context_docs)
            
            for msg in messages:
                if msg['role'] == 'user':
                    msg['content'] = f"مستندات مرتبط:\n{context}\n\nسوال: {msg['content']}"
                    break
        
        try:
            response = create_chat_completion(
                api_key=self.api_key,
                model=self.chat_model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            return response["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"خطا در پردازش: {str(e)}"
