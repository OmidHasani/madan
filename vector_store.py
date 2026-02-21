"""
Vector store implementation using Faiss for fast similarity search.

Why:
- In some Windows/conda setups, installing `chromadb` is painful.
- You also hit OpenAI SDK proxy/httpx issues.
- Faiss provides much faster similarity search than NumPy for large datasets.

This module provides a simple, production-acceptable local persistent store:
- embeddings are stored in `vector_db/<collection>.embeddings.npy`
- documents+metadata are stored in `vector_db/<collection>.docs.jsonl`
- Faiss index is stored in `vector_db/<collection>.faiss.index`

Search is cosine similarity using Faiss for fast retrieval.
"""

from __future__ import annotations

import json
import os
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("Faiss not available, falling back to NumPy. Install with: pip install faiss-cpu")

from pdf_processor import DocumentChunk
from openai_http import create_embeddings, OpenAIHTTPError

import re

# برای غنی‌سازی متادیتا با section_code/section_title (عیب‌یابی)
try:
    from troubleshooting_toc import get_code_for_page, get_section_page_range
except ImportError:
    get_code_for_page = None
    get_section_page_range = None


# Match codes like CA1626, E11, H-22, H22, H-5, E-1, CA-135, etc. (\d{1,6} برای کدهای تک‌رقمی مثل H-5)
_CODE_RE = re.compile(r"\b[A-Za-z]{1,3}-?\d{1,6}\b")


def _extract_codes(text: str) -> List[str]:
    if not text:
        return []
    return [m.group(0).upper() for m in _CODE_RE.finditer(text)]


def _normalize_query(q: str) -> str:
    return (q or "").strip()


@dataclass
class _StoredDoc:
    id: str
    text: str
    metadata: Dict


class VectorStore:
    """
    Local persistent vector store using Faiss for fast cosine similarity search.
    Falls back to NumPy if Faiss is not available.
    """

    def __init__(
        self,
        api_key: str,
        db_path: str = "vector_db",
        collection_name: str = "documents",
        embedding_model: str = "text-embedding-3-small",
    ):
        self.api_key = api_key
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.use_faiss = FAISS_AVAILABLE

        os.makedirs(self.db_path, exist_ok=True)

        safe_name = "".join(c for c in collection_name if c.isalnum() or c in ("-", "_")).strip() or "collection"
        self._emb_path = os.path.join(self.db_path, f"{safe_name}.embeddings.npy")
        self._docs_path = os.path.join(self.db_path, f"{safe_name}.docs.jsonl")
        self._norm_path = os.path.join(self.db_path, f"{safe_name}.embeddings_norm.npy")
        self._codes_path = os.path.join(self.db_path, f"{safe_name}.codes.json")
        self._faiss_path = os.path.join(self.db_path, f"{safe_name}.faiss.index")

        self._embeddings: Optional[np.ndarray] = None  # shape (N, D)
        self._embeddings_norm: Optional[np.ndarray] = None  # shape (N, D)
        self._faiss_index: Optional[faiss.IndexFlatIP] = None  # Faiss index for fast search
        self._docs: List[_StoredDoc] = []
        self._code_to_doc_idxs: Dict[str, List[int]] = {}
        self._embedding_dim: Optional[int] = None  # Dimension of embeddings

        self._load()
        logger.info(f"Vector store initialized (local) collection={self.collection_name}, count={len(self._docs)}, using Faiss={self.use_faiss}")

    def _load(self) -> None:
        if os.path.exists(self._docs_path):
            self._docs = []
            with open(self._docs_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    self._docs.append(_StoredDoc(id=obj["id"], text=obj["text"], metadata=obj.get("metadata") or {}))

        if os.path.exists(self._emb_path):
            self._embeddings = np.load(self._emb_path)
            if self._embeddings.size > 0:
                self._embedding_dim = self._embeddings.shape[1] if len(self._embeddings.shape) > 1 else None

        if os.path.exists(self._norm_path):
            self._embeddings_norm = np.load(self._norm_path)
        else:
            if self._embeddings is not None:
                self._embeddings_norm = self._normalize(self._embeddings)

        # Load Faiss index if available
        if self.use_faiss and os.path.exists(self._faiss_path) and self._embeddings_norm is not None:
            try:
                self._faiss_index = faiss.read_index(self._faiss_path)
                logger.info(f"Loaded Faiss index with {self._faiss_index.ntotal} vectors")
            except Exception as e:
                logger.warning(f"Failed to load Faiss index: {e}, will rebuild")
                self._faiss_index = None

        # Rebuild Faiss index if needed
        if self.use_faiss and self._faiss_index is None and self._embeddings_norm is not None:
            self._build_faiss_index()

        # Load or rebuild code index
        if os.path.exists(self._codes_path):
            try:
                with open(self._codes_path, "r", encoding="utf-8") as f:
                    self._code_to_doc_idxs = json.load(f)
                # normalize values to int lists
                self._code_to_doc_idxs = {
                    str(k).upper(): [int(i) for i in (v or [])] for k, v in (self._code_to_doc_idxs or {}).items()
                }
            except Exception:
                self._code_to_doc_idxs = {}

        if not self._code_to_doc_idxs and self._docs:
            self._rebuild_code_index()

    @staticmethod
    def _normalize(mat: np.ndarray) -> np.ndarray:
        # Normalize rows to unit vectors
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return mat / norms

    def _build_faiss_index(self) -> None:
        """Build Faiss index from normalized embeddings"""
        if not self.use_faiss or self._embeddings_norm is None:
            return
        
        if self._embeddings_norm.size == 0:
            return
        
        dim = self._embeddings_norm.shape[1]
        # Use IndexFlatIP (Inner Product) for cosine similarity (since vectors are normalized)
        self._faiss_index = faiss.IndexFlatIP(dim)
        
        # Convert to float32 for Faiss
        embeddings_f32 = self._embeddings_norm.astype(np.float32)
        self._faiss_index.add(embeddings_f32)
        
        logger.info(f"Built Faiss index with {self._faiss_index.ntotal} vectors of dimension {dim}")
        
        # Save the index
        try:
            faiss.write_index(self._faiss_index, self._faiss_path)
        except Exception as e:
            logger.warning(f"Failed to save Faiss index: {e}")

    def _rebuild_code_index(self) -> None:
        """
        Build mapping CODE -> [doc indices] by scanning stored docs.
        This makes short queries like 'CA1626' extremely accurate (exact match),
        instead of relying solely on embeddings.
        """
        mapping: Dict[str, List[int]] = {}
        for i, d in enumerate(self._docs):
            for code in _extract_codes(d.text):
                mapping.setdefault(code, []).append(i)

        # de-dup while preserving order
        for k, idxs in mapping.items():
            seen = set()
            deduped = []
            for x in idxs:
                if x in seen:
                    continue
                seen.add(x)
                deduped.append(x)
            mapping[k] = deduped

        self._code_to_doc_idxs = mapping

        try:
            with open(self._codes_path, "w", encoding="utf-8") as f:
                json.dump(self._code_to_doc_idxs, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def get_embedding(self, text: str) -> List[float]:
        try:
            vectors = create_embeddings(api_key=self.api_key, model=self.embedding_model, inputs=text)
            return vectors[0]
        except OpenAIHTTPError as e:
            logger.error(f"Error getting embedding (HTTP): {e}")
            raise

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        all_embeddings: List[List[float]] = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
            batch = texts[i : i + batch_size]
            try:
                batch_embeddings = create_embeddings(api_key=self.api_key, model=self.embedding_model, inputs=batch)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error processing batch {i}: {e}")
                # Retry individual items
                for text in batch:
                    try:
                        all_embeddings.append(self.get_embedding(text))
                    except Exception:
                        # Fallback: zero vector (dimension for text-embedding-3-small is 1536)
                        all_embeddings.append([0.0] * 1536)
        return all_embeddings

    def _persist(self) -> None:
        if self._embeddings is None:
            return
        np.save(self._emb_path, self._embeddings.astype(np.float32))
        if self._embeddings_norm is not None:
            np.save(self._norm_path, self._embeddings_norm.astype(np.float32))

        # Persist Faiss index
        if self.use_faiss and self._faiss_index is not None:
            try:
                faiss.write_index(self._faiss_index, self._faiss_path)
            except Exception as e:
                logger.warning(f"Failed to persist Faiss index: {e}")

        with open(self._docs_path, "w", encoding="utf-8") as f:
            for d in self._docs:
                f.write(json.dumps({"id": d.id, "text": d.text, "metadata": d.metadata}, ensure_ascii=False) + "\n")

        # Persist code index
        try:
            with open(self._codes_path, "w", encoding="utf-8") as f:
                json.dump(self._code_to_doc_idxs, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _enrich_metadata_with_section(self, metadata: Dict) -> Dict:
        """غنی‌سازی متادیتا با section_code و section_title از TOC عیب‌یابی برای بهبود RAG."""
        if get_code_for_page is None or get_section_page_range is None:
            return metadata
        page = metadata.get("page")
        if page is None:
            return metadata
        try:
            p_int = int(page)
        except (TypeError, ValueError):
            return metadata
        section_code = get_code_for_page(p_int)
        if not section_code:
            return metadata
        out = dict(metadata)
        out["section_code"] = section_code
        info = get_section_page_range(section_code)
        if info:
            out["section_title"] = info[2]
        return out

    def add_documents(self, chunks: List[DocumentChunk], batch_size: int = 100) -> None:
        logger.info(f"Adding {len(chunks)} documents to local vector store...")

        texts = [c.text for c in chunks]
        ids = [f"chunk_{c.chunk_index}" for c in chunks]
        metadatas = [self._enrich_metadata_with_section(c.metadata) for c in chunks]

        embeddings = self.get_embeddings_batch(texts, batch_size=batch_size)
        emb_mat = np.array(embeddings, dtype=np.float32)

        new_docs = [_StoredDoc(id=ids[i], text=texts[i], metadata=metadatas[i]) for i in range(len(chunks))]

        if self._embeddings is None:
            self._embeddings = emb_mat
            self._docs = new_docs
            self._embedding_dim = emb_mat.shape[1] if len(emb_mat.shape) > 1 else None
        else:
            self._embeddings = np.vstack([self._embeddings, emb_mat])
            self._docs.extend(new_docs)

        self._embeddings_norm = self._normalize(self._embeddings)

        # Update Faiss index with new embeddings
        if self.use_faiss:
            if self._faiss_index is None:
                self._build_faiss_index()
            else:
                # Add new embeddings to existing index
                new_embeddings_norm = self._normalize(emb_mat)
                new_embeddings_f32 = new_embeddings_norm.astype(np.float32)
                self._faiss_index.add(new_embeddings_f32)
                logger.info(f"Added {len(new_docs)} vectors to Faiss index")

        # Update code index incrementally for new docs
        start_idx = len(self._docs) - len(new_docs)
        for offset, d in enumerate(new_docs):
            doc_idx = start_idx + offset
            for code in _extract_codes(d.text):
                self._code_to_doc_idxs.setdefault(code, []).append(doc_idx)

        # de-dup lists
        for code, idxs in list(self._code_to_doc_idxs.items()):
            seen = set()
            deduped = []
            for x in idxs:
                if x in seen:
                    continue
                seen.add(x)
                deduped.append(x)
            self._code_to_doc_idxs[code] = deduped

        self._persist()

        logger.info(f"Successfully added. Total documents now: {len(self._docs)}")

    def search(self, query: str, top_k: int = 5, filter_metadata: Optional[Dict] = None) -> List[Dict]:
        query_norm = _normalize_query(query)
        if not self._docs:
            return []

        # 1) If query contains a code like CA1626, do exact lookup first (high precision).
        codes = _extract_codes(query_norm)
        logger.info(f"Extracted codes from query '{query_norm}': {codes}")
        exact_results: List[Dict] = []
        if codes:
            # Gather candidates for all codes (cap to keep it fast)
            cand_idxs: List[int] = []
            for code in codes:
                idxs = self._code_to_doc_idxs.get(code) or []
                logger.info(f"Code '{code}' found in index: {len(idxs)} documents")
                if not idxs:
                    # Fallback: search in text directly
                    idxs = [i for i, d in enumerate(self._docs) if code in (d.text or "").upper()]
                    logger.info(f"Code '{code}' found in text search: {len(idxs)} documents")
                cand_idxs.extend(idxs[:300])

            # de-dup
            seen = set()
            cand_idxs = [i for i in cand_idxs if not (i in seen or seen.add(i))]

            # Expand: include all chunks from the same page(s) as code hits (and adjacent pages)
            if cand_idxs:
                pages_hit = set()
                toc_pages_with_page_refs = {}  # {code: page_number_from_toc}
                
                for idx in cand_idxs:
                    if 0 <= idx < len(self._docs):
                        p = self._docs[idx].metadata.get("page")
                        if p is not None:
                            try:
                                pages_hit.add(int(p))
                            except Exception:
                                continue
                        
                        # Check if this is a TOC entry and extract page number reference
                        doc_text = self._docs[idx].text or ""
                        for code in codes:
                            # Look for pattern like "H-22 ... 24" or "H-22 Upper structure ... 24"
                            # Pattern: CODE followed by text, dots, and a number at the end of line
                            # Try multiple patterns
                            patterns = [
                                rf"{re.escape(code)}\s+[^\d]*?(\d{{1,4}})\s*$",  # End of line
                                rf"{re.escape(code)}\s+[^\d]*?\.{{3,}}\s*(\d{{1,4}})\b",  # With dots
                                rf"{re.escape(code)}\s+[^\d]*?(\d{{1,4}})\s*PC800",  # Before PC800
                            ]
                            for pattern in patterns:
                                match = re.search(pattern, doc_text, re.IGNORECASE | re.MULTILINE)
                                if match:
                                    ref_page = int(match.group(1))
                                    if 1 <= ref_page <= 1000:  # Reasonable page range
                                        toc_pages_with_page_refs[code] = ref_page
                                        pages_hit.add(ref_page)
                                        logger.info(f"✓ Found page reference in TOC: {code} -> page {ref_page}")
                                        # Also add adjacent pages (troubleshooting tables often span multiple pages)
                                        pages_hit.add(ref_page - 1)
                                        pages_hit.add(ref_page + 1)
                                        break
                
                logger.info(f"Pages hit by code search: {sorted(pages_hit)}")
                
                # For troubleshooting codes (H-xx, CAxxxx, etc.), also search for the actual troubleshooting page
                # Look for pages that contain the code AND troubleshooting keywords
                if codes:
                    for code in codes:
                        for i, d in enumerate(self._docs):
                            text = (d.text or "").upper()
                            # Check if this page contains the code AND troubleshooting keywords
                            if code.upper() in text:
                                p = d.metadata.get("page")
                                if p:
                                    try:
                                        p_int = int(p)
                                        # Check if this looks like a troubleshooting page (not just TOC)
                                        has_troubleshooting_keywords = any(
                                            keyword in text for keyword in [
                                                "FAILURE PHENOMENON",
                                                "PRESUMED CAUSE",
                                                "STANDARD VALUE",
                                                "TROUBLESHOOTING",
                                                "MALFUNCTION"
                                            ]
                                        )
                                        if has_troubleshooting_keywords:
                                            pages_hit.add(p_int)
                                            logger.info(f"Found troubleshooting page {p_int} for code {code}")
                                    except Exception:
                                        continue
                
                # adjacent pages help when a troubleshooting table spans multiple pages
                pages_hit |= {p - 1 for p in list(pages_hit)} | {p + 1 for p in list(pages_hit)}
                pages_hit = {p for p in pages_hit if p >= 1}
                
                logger.info(f"Expanded pages to search: {sorted(pages_hit)}")
                
                if pages_hit:
                    for i, d in enumerate(self._docs):
                        p = d.metadata.get("page")
                        try:
                            p_int = int(p)
                        except Exception:
                            continue
                        if p_int in pages_hit:
                            cand_idxs.append(i)

            # Validate indices: ensure they're within bounds of both docs and embeddings
            max_valid_idx = min(len(self._docs), len(self._embeddings_norm) if self._embeddings_norm is not None else len(self._docs))
            cand_idxs = [i for i in cand_idxs if 0 <= i < max_valid_idx]

            # Canonical troubleshooting chunks: same chunk contains the code AND full table (Failure phenomenon / Presumed cause).
            # Put these at the very top so the full error table is always returned for queries like "h-22".
            _TROUBLESHOOTING_MARKERS = ("FAILURE PHENOMENON", "PRESUMED CAUSE", "STANDARD VALUE", "MALFUNCTION")
            canonical_idxs: List[int] = []
            for idx in cand_idxs:
                if idx >= len(self._docs):
                    continue
                t = (self._docs[idx].text or "").upper()
                if not any(m in t for m in _TROUBLESHOOTING_MARKERS):
                    continue
                if not any(c.upper() in t for c in codes):
                    continue
                canonical_idxs.append(idx)
            # Also include the next chunk if it's the same topic (e.g. H-22 part 2 "does not swing in only 1 direction")
            for idx in list(canonical_idxs):
                next_idx = idx + 1
                if next_idx < len(self._docs):
                    t_next = (self._docs[next_idx].text or "").upper()
                    if any(m in t_next for m in _TROUBLESHOOTING_MARKERS) and next_idx not in canonical_idxs:
                        canonical_idxs.append(next_idx)
            canonical_idxs = list(dict.fromkeys(canonical_idxs))  # preserve order, dedup
            if canonical_idxs:
                logger.info(f"Canonical troubleshooting chunks for codes {codes}: doc indices {canonical_idxs}")

            # If we have embeddings, rank candidates by semantic similarity to an expanded query.
            # This pushes "troubleshooting/details" chunks above TOC-style lists.
            # Note: For code matching with limited candidates, NumPy is fast enough.
            # Faiss is used for the main semantic search where we search all vectors.
            if self._embeddings_norm is not None and cand_idxs:
                # Generic troubleshooting query around the extracted codes (works for CAxxxx, H-xx, etc.)
                expanded = " ".join([f"{c} troubleshooting details" for c in codes])
                q = np.array(self.get_embedding(expanded), dtype=np.float32)
                q_norm = q / (np.linalg.norm(q) or 1.0)
                # Double-check bounds before indexing - only use valid indices
                valid_idxs = np.array([i for i in cand_idxs if 0 <= i < len(self._embeddings_norm) and 0 <= i < len(self._docs)], dtype=np.int64)
                if len(valid_idxs) > 0:
                    # Use NumPy for code matching (limited candidates, fast enough)
                    mat = self._embeddings_norm[valid_idxs]
                    sims = mat @ q_norm
                    
                    # Heuristic penalty: TOC/list chunks often contain many codes and dotted leaders.
                    penalties = np.zeros_like(sims)
                    for k, idx in enumerate(valid_idxs):
                        idx_int = int(idx)
                        if idx_int >= len(self._docs):
                            continue
                        t = (self._docs[idx_int].text or "")
                        code_count = len(_extract_codes(t))
                        if code_count > 3:
                            penalties[k] += min(0.30, 0.05 * (code_count - 3))
                        if "...." in t:
                            penalties[k] += 0.10
                        if "Failure code" in t and code_count > 8:
                            penalties[k] += 0.15

                    adj = sims - penalties
                    order = np.argsort(-adj)
                    canonical_set = set(canonical_idxs)

                    # 1) Add canonical troubleshooting chunks first (full error table for this code)
                    for idx in canonical_idxs:
                        doc = self._docs[idx]
                        if filter_metadata:
                            ok = all(doc.metadata.get(k) == v for k, v in filter_metadata.items())
                            if not ok:
                                continue
                        exact_results.append({
                            "text": doc.text,
                            "metadata": doc.metadata,
                            "distance": 0.0,
                            "id": doc.id,
                        })

                    # 2) Fill with remaining candidates by similarity (skip already-added canonical)
                    results_to_check = min(len(order), max(top_k * 3, 50))
                    seen_ids = {r["id"] for r in exact_results}
                    for j in order[:results_to_check]:
                        idx_int = int(valid_idxs[int(j)])
                        if idx_int >= len(self._docs) or idx_int in canonical_set:
                            continue
                        doc = self._docs[idx_int]
                        if doc.id in seen_ids:
                            continue
                        if filter_metadata:
                            ok = all(doc.metadata.get(k) == v for k, v in filter_metadata.items())
                            if not ok:
                                continue
                        seen_ids.add(doc.id)
                        sim_val = float(sims[int(j)]) if int(j) < len(sims) else 0.0
                        exact_results.append({
                            "text": doc.text,
                            "metadata": doc.metadata,
                            "distance": float(1.0 - sim_val),
                            "id": doc.id,
                        })
                        if len(exact_results) >= top_k * 2:
                            break

                    # Return top_k results (canonical full tables first, then by similarity)
                    return exact_results[:top_k]
            else:
                # Fallback: return first hits (still better than pure embeddings for short codes)
                for idx in cand_idxs[:top_k]:
                    doc = self._docs[int(idx)]
                    exact_results.append({"text": doc.text, "metadata": doc.metadata, "distance": 0.0, "id": doc.id})
                if exact_results:
                    return exact_results[:top_k]

        # 2) Semantic fallback (embeddings) if needed.
        if self._embeddings_norm is None:
            return exact_results[:top_k]

        q = np.array(self.get_embedding(query), dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) or 1.0)

        # Use Faiss for fast search if available, otherwise fall back to NumPy
        if self.use_faiss and self._faiss_index is not None:
            # Faiss search (much faster)
            q_f32 = q_norm.astype(np.float32).reshape(1, -1)
            k = min(top_k * 2, self._faiss_index.ntotal)  # Get more candidates for filtering
            distances, indices = self._faiss_index.search(q_f32, k)
            
            results: List[Dict] = []
            seen_ids = set()
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self._docs):
                    continue
                doc = self._docs[int(idx)]
                
                # Skip if already in exact_results
                if doc.id in seen_ids:
                    continue
                seen_ids.add(doc.id)
                
                if filter_metadata:
                    ok = True
                    for k, v in filter_metadata.items():
                        if doc.metadata.get(k) != v:
                            ok = False
                            break
                    if not ok:
                        continue

                # Faiss returns inner product (cosine similarity since vectors are normalized)
                sim = float(dist)
                results.append(
                    {
                        "text": doc.text,
                        "metadata": doc.metadata,
                        "distance": 1.0 - sim,  # keep old contract: lower is better
                        "id": doc.id,
                    }
                )
                if len(results) >= top_k:
                    break
        else:
            # NumPy fallback (slower but works without Faiss)
            sims = self._embeddings_norm @ q_norm  # shape (N,)

            # Optional metadata filter (simple exact match)
            indices = np.argsort(-sims)  # descending
            results: List[Dict] = []
            for idx in indices:
                doc = self._docs[int(idx)]
                if filter_metadata:
                    ok = True
                    for k, v in filter_metadata.items():
                        if doc.metadata.get(k) != v:
                            ok = False
                            break
                    if not ok:
                        continue

                sim = float(sims[int(idx)])
                results.append(
                    {
                        "text": doc.text,
                        "metadata": doc.metadata,
                        "distance": 1.0 - sim,  # keep old contract: lower is better
                        "id": doc.id,
                    }
                )
                if len(results) >= top_k:
                    break

        # Merge: keep exact hits first, then semantic (dedup by id)
        if exact_results:
            seen_ids = {r["id"] for r in exact_results}
            for r in results:
                if r["id"] in seen_ids:
                    continue
                exact_results.append(r)
                if len(exact_results) >= top_k:
                    break
            return exact_results[:top_k]

        # تقویت با متادیتا: اگر سوال حاوی کد خطا بود، چانک‌هایی که section_code مطابق دارند بالاتر بروند
        if results and codes:
            codes_set = {c.upper().replace("-", "") for c in codes}
            codes_set |= {c.upper() for c in codes}
            for r in results:
                sc = (r.get("metadata") or {}).get("section_code") or ""
                if sc and (sc.upper() in codes_set or sc.upper().replace("-", "") in codes_set):
                    r["distance"] = max(0.0, (r.get("distance") or 1.0) - 0.15)
            results.sort(key=lambda x: x.get("distance", 1.0))

        return results

    def search_with_reranking(
        self,
        query: str,
        top_k: int = 5,
        initial_k: int = 40,
        use_mmr: bool = True,
        mmr_lambda: float = 0.7,
    ) -> List[Dict]:
        """
        جستجو با کاندید بیشتر + MMR برای تنوع (کاهش تکرار و افزایش پوشش).
        mmr_lambda نزدیک ۱ = تأکید بیشتر روی شباهت به سوال؛ نزدیک ۰ = تنوع بیشتر.
        """
        initial_k = max(initial_k, top_k * 2)
        candidates = self.search(query, top_k=initial_k)
        candidates.sort(key=lambda x: x.get("distance", 1.0))

        if len(candidates) <= top_k or not use_mmr or self._embeddings_norm is None:
            return candidates[:top_k]

        # MMR: Maximal Marginal Relevance
        id_to_idx = {d.id: i for i, d in enumerate(self._docs)}
        q = np.array(self.get_embedding(query), dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) or 1.0)

        # فقط کاندیدهایی که ایندکس معتبر دارند
        valid = [
            (c, id_to_idx.get(c["id"]))
            for c in candidates
            if c["id"] in id_to_idx and 0 <= id_to_idx[c["id"]] < len(self._embeddings_norm)
        ]
        if len(valid) <= top_k:
            return candidates[:top_k]

        selected: List[Dict] = []
        selected_emb: List[np.ndarray] = []
        remaining = [c for c, _ in valid]
        embs = [self._embeddings_norm[idx] for _, idx in valid]

        for _ in range(top_k):
            if not remaining:
                break
            best_score = -np.inf
            best_i = 0
            for i, (c, emb) in enumerate(zip(remaining, embs)):
                sim_q_d = float(np.dot(q_norm, emb))
                if not selected_emb:
                    mmr_score = sim_q_d
                else:
                    max_sim_sel = max(float(np.dot(emb, s)) for s in selected_emb)
                    mmr_score = mmr_lambda * sim_q_d - (1.0 - mmr_lambda) * max_sim_sel
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_i = i
            selected.append(remaining[best_i])
            selected_emb.append(embs[best_i])
            remaining.pop(best_i)
            embs.pop(best_i)

        return selected

    def _is_toc_or_index_chunk(self, text: str) -> bool:
        """چانک شبیه فهرست/ایندکس است نه جدول عیب‌یابی واقعی."""
        if not (text or "").strip():
            return True
        t = (text or "")[:4000]
        if "INDEX AND FOREWORD" in t.upper() or "00 INDEX" in t.upper():
            return True
        toc_line_pattern = re.compile(r"Failure\s+code\s+\[[^\]]+\][^\d]*\d{2,4}\s*$", re.IGNORECASE | re.MULTILINE)
        if len(toc_line_pattern.findall(t)) >= 3:
            return True
        if t.count(".") > 50 and "Failure code" in t:
            return True
        return False

    def _is_reference_table_chunk(self, text: str) -> bool:
        """
        چانک‌های «جدول‌های راهنما/ارجاع» که کدها را فقط لیست می‌کنند (نه خودِ بدنهٔ عیب‌یابی).
        مثال‌ها: Classification and troubleshooting steps، Failure-looking phenomenon table،
        connection table for connector pin numbers، T-boxes/T-adapters، و ... .
        """
        if not (text or "").strip():
            return True
        t = (text or "")[:6000].upper()
        markers = (
            "CLASSIFICATION AND TROUBLESHOOTING STEPS",
            "FAILURE-LOOKING PHENOMENON",
            "FAILURE-LOOKING PHENOMENON AND TROUBLESHOOTING",
            "PHENOMENA LOOKING LIKE TROUBLES",
            "TROUBLESHOOTING NO.",
            "FAILURE CODES TABLE",
            "TESTING AND ADJUSTING",
            "SERVICE MENU",
            "CONNECTION TABLE FOR CONNECTOR PIN NUMBERS",
            "T-BOXES AND T-ADAPTERS TABLE",
            "T-ADAPTER KIT",
            "HD30 SERIES CONNECTOR",
            "DT SERIES CONNECTOR",
            "SWP TYPE CONNECTOR",
            "MIC TYPE CONNECTOR",
            "AMP070 TYPE CONNECTOR",
            "AMP040 TYPE CONNECTOR",
        )
        if any(m in t for m in markers):
            return True
        # اگر تعداد زیادی "Troubleshooting No." یا جدول پین/کانکتور داشته باشد، احتمالاً رفرنس است
        if t.count("TROUBLESHOOTING") > 10 and t.count("PC800") > 0 and t.count("UEN") > 0:
            return True
        return False

    def get_page_range_for_code(
        self, code: str, expand_adjacent: int = 1
    ) -> Optional[tuple]:
        """
        محدودهٔ صفحات واقعیِ ظاهر شدن این کد در PDF را از ایندکس برمی‌گرداند
        (نه از TOC، تا با شماره‌گذاری نسبی دفترچه‌های مختلف اشتباه نشود).
        ترجیح با چانک‌هایی است که جدول عیب‌یابی دارند (Cause, Standard value, ...).
        Returns:
            (min_page, max_page) یا None اگر هیچ چانکی حاوی این کد نبود.
        """
        if not self._docs or not code:
            return None
        code_upper = (code or "").strip().upper().replace("-", "").replace(" ", "")
        # هم CA135 و هم CA-135 در ایندکس ممکن است ذخیره شده باشند
        code_variants = [code_upper]
        m = re.match(r"^([A-Za-z]+)(\d+)$", code_upper)
        if m:
            code_variants.append(m.group(1) + "-" + m.group(2))
        cand_idxs = []
        for v in code_variants:
            cand_idxs.extend(self._code_to_doc_idxs.get(v) or [])
        cand_idxs = list(dict.fromkeys(cand_idxs))
        if not cand_idxs:
            cand_idxs = [
                i
                for i, d in enumerate(self._docs)
                if any(v in (d.text or "").upper() for v in code_variants)
            ]
        if not cand_idxs:
            return None

        # برای کدهای E-xx, S-xx, H-xx: فقط چانک‌هایی که کد به‌صورت «کد خطا» آمده (مثلاً S-15)
        # نه به‌عنوان نام کانکتور (مثل S15 (female) در E-31)
        is_ehs_code = bool(
            re.match(r"^[EHS]\d+$", code_upper)
            or re.match(r"^[EHS]-\d+$", (code or "").strip().upper().replace(" ", ""))
        )
        if is_ehs_code:
            # در متن مستند کد خطا معمولاً با خط تیره است: S-15 نه S15
            code_with_hyphen = code_upper[0] + "-" + code_upper[1:] if len(code_upper) >= 2 and code_upper[0].isalpha() and code_upper[1:].isdigit() else code_upper
            pattern = re.compile(r"\b" + re.escape(code_with_hyphen) + r"\b", re.IGNORECASE)
            filtered = [idx for idx in cand_idxs if 0 <= idx < len(self._docs) and pattern.search(self._docs[idx].text or "")]
            if filtered:
                cand_idxs = filtered

        # انتخاب «بدنهٔ عیب‌یابی» باید به خانوادهٔ کد حساس باشد تا صفحات مرجع قاطی نشوند.
        is_ca_family = bool(re.match(r"^CA\d{2,6}$", code_upper))

        ca_body_keywords = (
            "CONTENTS OF TROUBLE",
            "ACTION OF CONTROLLER",
            "PROBLEM THAT",
            "RELATED INFORMATION",
            "POSSIBLE CAUSES",
            "STANDARD VALUE",
            "WIRING HARNESS",
            "RESISTANCE",
            "VOLTAGE",
            "POWER SUPPLY",
            "CHASSIS GROUND",
            "ENGINE CONTROLLER SYSTEM",
            "POIL",
        )
        ehs_body_keywords = (
            "FAILURE PHENOMENON",
            "RELATIVE INFORMATION",
            "PRESUMED CAUSE",
            "STANDARD VALUE",
            "CAUSE",
        )
        s_body_keywords = (
            "ABNORMAL NOISE",
            "GENERAL CAUSES",
            "QUESTIONS",
            "CHECK ITEMS",
            "TROUBLESHOOTING",
            "REMEDY",
            "TURBOCHARGER",
            "EGR",
            "VALVE CLEARANCE",
            "COMPRESSION PRESSURE",
            "BLOW-BY",
        )
        pages_from_table = []
        toc_page_ref = None
        for idx in cand_idxs:
            if 0 <= idx < len(self._docs):
                d = self._docs[idx]
                p = d.metadata.get("page")
                try:
                    p_int = int(p)
                except (TypeError, ValueError):
                    continue
                text = d.text or ""
                # فهرست/ایندکس یا جداول ارجاعی (مثل Failure-looking phenomenon) را در انتخاب بخش اصلی دخیل نکن
                if self._is_toc_or_index_chunk(text):
                    # استخراج شمارهٔ صفحه از خط فهرست مثل "Failure code [CA135] ... 38"
                    ref_m = re.search(
                        rf"\[{re.escape(code_upper)}\][^\d]*(\d{{2,4}})\s*$",
                        text, re.IGNORECASE | re.MULTILINE,
                    )
                    if not ref_m:
                        ref_m = re.search(
                            rf"Failure\s+code\s+\[{re.escape(code_upper)}\].*?(\d{{2,4}})\s*$",
                            text, re.IGNORECASE | re.MULTILINE | re.DOTALL,
                        )
                    if ref_m:
                        toc_page_ref = int(ref_m.group(1))
                    continue
                if self._is_reference_table_chunk(text):
                    continue
                t_up = text.upper()
                if is_ca_family:
                    # برای CA باید عنوان اصلی به صورت Failure code [CAxxxx] باشد؛ جدول‌های لیست (Failure codes table) را رد کن
                    if f"FAILURE CODE [{code_upper}]" in t_up and any(k in t_up for k in ca_body_keywords):
                        pages_from_table.append(p_int)
                    continue
                if is_ehs_code:
                    # S-mode بدنه‌اش Flow/Table متفاوت دارد
                    if code_upper.startswith("S") and any(k in t_up for k in s_body_keywords):
                        pages_from_table.append(p_int)
                    elif any(k in t_up for k in ehs_body_keywords):
                        pages_from_table.append(p_int)
                    continue
                # سایر کدها: محافظه‌کارانه
                if any(k in t_up for k in ("CONTENTS OF TROUBLE", "ACTION OF CONTROLLER", "FAILURE PHENOMENON", "STANDARD VALUE", "PRESUMED CAUSE")):
                    pages_from_table.append(p_int)
        if pages_from_table:
            min_p = max(1, min(pages_from_table) - expand_adjacent)
            max_p = max(pages_from_table) + expand_adjacent
            logger.info(f"Code '{code}' from table chunks on pages {sorted(set(pages_from_table))} -> range [{min_p}, {max_p}]")
            return (min_p, max_p)
        if toc_page_ref is not None:
            min_p = max(1, toc_page_ref - expand_adjacent)
            max_p = toc_page_ref + 2
            logger.info(f"Code '{code}' page from TOC reference: {toc_page_ref} -> range [{min_p}, {max_p}]")
            return (min_p, max_p)
        # Fallback برای E/S/H: اگر چیزی از بدنه پیدا نشد، یک جستجوی embedding-only انجام بده و رفرنس‌ها را حذف کن
        if is_ehs_code and self._embeddings_norm is not None and get_section_page_range is not None:
            try:
                code_with_hyphen = code_upper[0] + "-" + code_upper[1:] if len(code_upper) >= 2 and code_upper[0].isalpha() and code_upper[1:].isdigit() else code_upper
                info = get_section_page_range(code_with_hyphen) or get_section_page_range(code_upper)
                title = info[2] if info else ""
                query = f"{code_with_hyphen} {title}".strip()
                if query:
                    q = np.array(self.get_embedding(query), dtype=np.float32)
                    q_norm = q / (np.linalg.norm(q) or 1.0)
                    # سریع: faiss اگر موجود است
                    candidate_pages: List[int] = []
                    if self.use_faiss and self._faiss_index is not None:
                        q_f32 = q_norm.astype(np.float32).reshape(1, -1)
                        k = min(80, self._faiss_index.ntotal)
                        distances, indices = self._faiss_index.search(q_f32, k)
                        for idx2 in indices[0]:
                            if idx2 < 0 or idx2 >= len(self._docs):
                                continue
                            doc = self._docs[int(idx2)]
                            txt = doc.text or ""
                            if self._is_toc_or_index_chunk(txt) or self._is_reference_table_chunk(txt):
                                continue
                            # برای S-15/S-16 حداقل باید "ABNORMAL NOISE" یا "VIBRATION" و ... باشد
                            if "ABNORMAL NOISE" in query.upper() and "ABNORMAL NOISE" not in txt.upper():
                                continue
                            p2 = doc.metadata.get("page")
                            try:
                                candidate_pages.append(int(p2))
                            except Exception:
                                continue
                    else:
                        sims = self._embeddings_norm @ q_norm
                        top = np.argsort(-sims)[:200]
                        for idx2 in top:
                            doc = self._docs[int(idx2)]
                            txt = doc.text or ""
                            if self._is_toc_or_index_chunk(txt) or self._is_reference_table_chunk(txt):
                                continue
                            if "ABNORMAL NOISE" in query.upper() and "ABNORMAL NOISE" not in txt.upper():
                                continue
                            p2 = doc.metadata.get("page")
                            try:
                                candidate_pages.append(int(p2))
                            except Exception:
                                continue
                    if candidate_pages:
                        min_p = max(1, min(candidate_pages) - expand_adjacent)
                        max_p = max(candidate_pages) + expand_adjacent
                        logger.info(f"Code '{code}' fallback semantic pages {sorted(set(candidate_pages))} -> range [{min_p}, {max_p}]")
                        return (min_p, max_p)
            except Exception:
                pass
        return None

    def get_chunks_by_page_range(
        self, start_page: int, end_page: int, use_parent_context: bool = False
    ) -> List[Dict]:
        """
        همهٔ چانک‌هایی که شماره صفحه‌شان در [start_page, end_page] است را
        به ترتیب (صفحه، چانک) برمی‌گرداند. برای ارسال «بخش کامل» عیب‌یابی به LLM.
        اگر use_parent_context=True باشد برای هر صفحه فقط یک بار parent_context (متن کامل صفحه) برگردانده می‌شود.
        """
        if not self._docs:
            return []
        out: List[Dict] = []
        for i, d in enumerate(self._docs):
            p = d.metadata.get("page")
            try:
                p_int = int(p)
            except (TypeError, ValueError):
                continue
            if start_page <= p_int <= end_page:
                if use_parent_context:
                    text = (d.metadata.get("parent_context") or d.text or "").strip()
                else:
                    text = (d.text or "").strip()
                if not text:
                    continue
                out.append({
                    "text": text,
                    "metadata": {**d.metadata, "page": p_int},
                    "id": d.id,
                    "page": p_int,
                    "chunk_index": d.metadata.get("chunk", i),
                })
        # حذف تکرار parent_context برای یک صفحه (هر صفحه یک بار)
        if use_parent_context and out:
            by_page: Dict[int, List[Dict]] = {}
            for r in out:
                pg = r.get("page") or r["metadata"].get("page")
                by_page.setdefault(pg, []).append(r)
            out = []
            for pg in sorted(by_page.keys()):
                out.append(by_page[pg][0])
        out.sort(key=lambda x: (x.get("page", 0), x.get("chunk_index", 0)))
        return out

    def get_collection_stats(self) -> Dict:
        return {
            "collection_name": self.collection_name,
            "document_count": len(self._docs),
            "embedding_model": self.embedding_model,
            "db_path": self.db_path,
        }

    def rebuild_code_index(self) -> None:
        """
        Public method to rebuild code index (useful after regex changes)
        """
        logger.info("Rebuilding code index...")
        self._rebuild_code_index()
        logger.info(f"Code index rebuilt. Found {len(self._code_to_doc_idxs)} unique codes")
    
    def reset_collection(self) -> None:
        logger.warning(f"Resetting collection: {self.collection_name}")
        self._docs = []
        self._embeddings = None
        self._embeddings_norm = None
        self._code_to_doc_idxs = {}
        for p in (self._docs_path, self._emb_path, self._norm_path, self._codes_path, self._faiss_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        logger.info("Collection reset complete")

