"""
Utilities for loading and selecting extracted images for answers.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Iterable, Optional, Tuple
import re

import numpy as np

from openai_http import create_embeddings, OpenAIHTTPError


_CODE_RE = re.compile(r"\b[A-Za-z]{1,3}\d{2,6}\b")


def extract_codes(text: str) -> List[str]:
    if not text:
        return []
    return [m.group(0).upper() for m in _CODE_RE.finditer(text)]


def load_images_manifest(manifest_path: str) -> Dict[str, Any]:
    if not os.path.exists(manifest_path):
        return {"pages": {}}
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _image_key(it: Dict[str, Any]) -> str:
    page = it.get("page")
    file = it.get("file") or it.get("url") or ""
    return f"{page}|{file}"


def _all_images_from_manifest(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    pages_map = (manifest.get("pages") or {})
    out: List[Dict[str, Any]] = []
    for p in sorted(pages_map.keys(), key=lambda x: int(x)):
        out.extend(pages_map.get(str(p)) or [])
    return out


_IMAGE_EMB_CACHE: Dict[str, Any] = {
    "fingerprint": None,
    "embeddings": {},  # key -> np.array
    "items": [],       # list of items
}


def _build_image_embedding_cache(
    manifest: Dict[str, Any],
    api_key: Optional[str],
    embedding_model: str = "text-embedding-3-small",
    batch_size: int = 64,
) -> Dict[str, Any]:
    if not api_key:
        return _IMAGE_EMB_CACHE

    items = _all_images_from_manifest(manifest)
    fingerprint = f"{manifest.get('source_pdf','')}|{len(items)}"

    if _IMAGE_EMB_CACHE.get("fingerprint") == fingerprint and _IMAGE_EMB_CACHE.get("embeddings"):
        return _IMAGE_EMB_CACHE

    texts: List[str] = []
    keys: List[str] = []
    for it in items:
        caption = (it.get("caption") or "").strip()
        context = (it.get("context") or "").strip()
        text = " ".join(t for t in [caption, context] if t)
        if not text:
            text = str(it.get("file") or "")
        keys.append(_image_key(it))
        texts.append(text)

    embeddings: Dict[str, np.ndarray] = {}
    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vecs = create_embeddings(api_key=api_key, model=embedding_model, inputs=batch)
            for j, v in enumerate(vecs):
                embeddings[keys[i + j]] = np.array(v, dtype=np.float32)
    except OpenAIHTTPError:
        # Fallback: no embeddings, keep empty cache
        embeddings = {}

    _IMAGE_EMB_CACHE["fingerprint"] = fingerprint
    _IMAGE_EMB_CACHE["embeddings"] = embeddings
    _IMAGE_EMB_CACHE["items"] = items
    return _IMAGE_EMB_CACHE


def images_for_pages(manifest: Dict[str, Any], pages: Iterable[int], limit: int = 6) -> List[Dict[str, Any]]:
    """
    Pick images for a set of pages. Keeps order by page number.
    """
    pages_map = (manifest.get("pages") or {})
    out: List[Dict[str, Any]] = []

    for p in sorted(set(int(x) for x in pages)):
        items = pages_map.get(str(p)) or []
        for it in items:
            out.append(it)
            if len(out) >= limit:
                return out
    return out


def images_for_query(
    manifest: Dict[str, Any],
    query: str,
    pages_hint: Iterable[int],
    limit: int = 12,
    include_adjacent_pages: bool = True,
    api_key: Optional[str] = None,
    embedding_model: str = "text-embedding-3-small",
    use_embeddings: bool = True,
) -> List[Dict[str, Any]]:
    """
    High-precision image selection:
    - If query contains codes (e.g. CA1626) => return ALL images from pages_hint (+/-1 optionally)
      and prefer images whose manifest.codes includes that code.
    - Otherwise => rank by token overlap with caption/context, limited to pages_hint (+/-1).
    """
    pages_map = (manifest.get("pages") or {})
    q = (query or "").strip()
    codes = extract_codes(q)

    pages_set = set(int(p) for p in pages_hint if p is not None)
    if include_adjacent_pages:
        pages_set |= {p - 1 for p in list(pages_set)} | {p + 1 for p in list(pages_set)}
    pages_set = {p for p in pages_set if p >= 1}

    candidates: List[Dict[str, Any]] = []
    # If no pages hint, search across all pages
    if not pages_set:
        pages_set = set(int(p) for p in (pages_map.keys() or []))

    for p in sorted(pages_set):
        for it in (pages_map.get(str(p)) or []):
            candidates.append(it)

    # Fallback: if no images on hinted pages, search all pages for non-code queries
    if not candidates and not codes:
        for p in sorted(pages_map.keys(), key=lambda x: int(x)):
            candidates.extend(pages_map.get(str(p)) or [])

    if not candidates:
        return []

    if codes:
        # score: code match first, then keep stable ordering by page
        def score(it: Dict[str, Any]) -> int:
            it_codes = set((it.get("codes") or []))
            return 10 if any(c in it_codes for c in codes) else 0

        candidates.sort(key=lambda it: (-(score(it)), int(it.get("page") or 0)))
        # For code queries, user asked "تمام تصاویر" => return up to limit but typically many.
        return candidates[:limit]

    # Non-code query: prefer embedding similarity (if available), otherwise token overlap
    tokens = [t for t in re.split(r"\W+", q.lower()) if len(t) >= 3]
    token_set = set(tokens)

    def overlap_score(it: Dict[str, Any]) -> int:
        text = ((it.get("caption") or "") + " " + (it.get("context") or "")).lower()
        score = 0
        for t in token_set:
            if t in text:
                score += 1
        return score

    if use_embeddings and api_key:
        cache = _build_image_embedding_cache(manifest, api_key, embedding_model=embedding_model)
        emb_map: Dict[str, np.ndarray] = cache.get("embeddings") or {}
        if emb_map:
            try:
                q_vec = create_embeddings(api_key=api_key, model=embedding_model, inputs=[q])[0]
                q_vec = np.array(q_vec, dtype=np.float32)
                q_norm = q_vec / (np.linalg.norm(q_vec) or 1.0)
                scored: List[Tuple[float, Dict[str, Any]]] = []
                for it in candidates:
                    key = _image_key(it)
                    v = emb_map.get(key)
                    if v is None:
                        # fallback to token overlap if embedding missing
                        score = float(overlap_score(it))
                    else:
                        v_norm = v / (np.linalg.norm(v) or 1.0)
                        score = float(np.dot(v_norm, q_norm))
                    scored.append((score, it))
                scored.sort(key=lambda x: (-(x[0]), int(x[1].get("page") or 0)))
                return [it for _, it in scored[:limit]]
            except OpenAIHTTPError:
                pass

    candidates.sort(key=lambda it: (-(overlap_score(it)), int(it.get("page") or 0)))
    return candidates[:limit]


