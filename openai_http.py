"""
Minimal OpenAI HTTP client (no SDK) to avoid proxy/httpx incompatibilities.

Uses urllib with proxies disabled, so environments that break the OpenAI SDK
due to proxy/httpx signature mismatches can still call the API.
"""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional
import logging


logger = logging.getLogger(__name__)


class OpenAIHTTPError(RuntimeError):
    pass


def _opener_no_proxy() -> urllib.request.OpenerDirector:
    # Disable proxies entirely (ignores HTTP_PROXY/HTTPS_PROXY env vars)
    proxy_handler = urllib.request.ProxyHandler({})
    return urllib.request.build_opener(proxy_handler)


def _request_json(
    url: str,
    api_key: str,
    payload: Dict[str, Any],
    timeout_s: int = 60,
    max_retries: int = 5,
) -> Dict[str, Any]:
    opener = _opener_no_proxy()
    data = json.dumps(payload).encode("utf-8")

    # Masked key for logging / debugging (نمایش بخشی از کلید برای اطمینان)
    if api_key:
        masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***short_key***"
    else:
        masked_key = "<EMPTY>"

    logger.info(f"OpenAI HTTP call to {url} using key={masked_key}")

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        req = urllib.request.Request(
            url=url,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with opener.open(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
            # Retry on transient errors / rate limits
            if e.code == 429:
                logger.error(f"OpenAI 429 (insufficient_quota or rate limit) with key={masked_key}: {raw}")
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                # exponential backoff
                time.sleep(min(2 ** attempt, 10))
                last_err = OpenAIHTTPError(f"HTTP {e.code}: {raw}")
                continue
            raise OpenAIHTTPError(f"HTTP {e.code}: {raw}") from e
        except Exception as e:
            # Retry a few times on network hiccups
            if attempt < max_retries - 1:
                time.sleep(min(2 ** attempt, 10))
                last_err = e
                continue
            raise OpenAIHTTPError(str(e)) from e

    raise OpenAIHTTPError(str(last_err) if last_err else "Unknown OpenAI HTTP error")


def create_embeddings(
    *,
    api_key: str,
    model: str,
    inputs: List[str] | str,
    timeout_s: int = 60,
) -> List[List[float]]:
    """
    Returns a list of embedding vectors, one per input.
    """
    url = "https://api.openai.com/v1/embeddings"
    payload = {"model": model, "input": inputs}
    data = _request_json(url, api_key, payload, timeout_s=timeout_s)

    if "data" not in data:
        raise OpenAIHTTPError(f"Unexpected embeddings response: {data}")

    # sort by index to preserve ordering (API may return ordered, but be safe)
    items = sorted(data["data"], key=lambda x: x.get("index", 0))
    return [item["embedding"] for item in items]


def create_chat_completion(
    *,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.1,
    max_tokens: int = 2000,
    timeout_s: int = 120,
) -> Dict[str, Any]:
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    return _request_json(url, api_key, payload, timeout_s=timeout_s)




