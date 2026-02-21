"""
Tavily search and content extraction module
"""
import os
import logging
import requests
from typing import List, Dict, Optional
from urllib.parse import urljoin
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    logger.warning("tavily-python package not available, using HTTP API directly")


class TavilySearch:
    """Tavily search client for web search and content extraction"""
    
    def __init__(self, api_key: str):
        """
        Initialize Tavily search client
        
        Args:
            api_key: Tavily API key
        """
        self.api_key = api_key
        
        if TAVILY_AVAILABLE:
            self.client = TavilyClient(api_key=api_key)
            self.use_package = True
        else:
            self.base_url = "https://api.tavily.com"
            self.use_package = False
        
    def search(self, query: str, search_depth: str = "advanced", max_results: int = 8) -> Dict:
        """
        Search the web using Tavily API
        
        Args:
            query: Search query
            search_depth: Search depth ("basic" or "advanced")
            max_results: Maximum number of results
            
        Returns:
            Dictionary with search results
        """
        try:
            if self.use_package:
                # Use tavily-python package
                response = self.client.search(
                    query=query,
                    search_depth=search_depth,
                    max_results=max_results,
                    include_answer=False
                )
                return response
            else:
                # Fallback to HTTP API
                url = f"{self.base_url}/search"
                headers = {
                    "Content-Type": "application/json",
                }
                payload = {
                    "api_key": self.api_key,
                    "query": query,
                    "search_depth": search_depth,
                    "max_results": max_results,
                    "include_answer": False
                }
                
                response = requests.post(url, json=payload, headers=headers, timeout=30)
                response.raise_for_status()
                
                return response.json()
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return {"results": [], "error": str(e)}
    
    def extract_text(self, url: str, max_length: int = 5000) -> str:
        """
        Extract text content from a URL
        
        Args:
            url: URL to extract content from
            max_length: Maximum length of extracted text
            
        Returns:
            Extracted text content
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            r = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")
            
            # Remove script, style, nav, footer, header tags
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            
            # Extract text from paragraphs
            text = "\n".join(
                p.get_text(strip=True)
                for p in soup.find_all("p")
                if len(p.get_text(strip=True)) > 40
            )
            
            # Limit length
            if len(text) > max_length:
                text = text[:max_length] + "..."
                
            return text
        except Exception as e:
            logger.warning(f"Failed to extract text from {url}: {e}")
            return ""
    
    def extract_images(self, url: str, max_images: int = 5) -> List[str]:
        """
        Extract image URLs from a webpage
        
        Args:
            url: URL to extract images from
            max_images: Maximum number of images to extract
            
        Returns:
            List of image URLs
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            r = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")
            
            images = []
            for img in soup.find_all("img"):
                src = img.get("src") or img.get("data-src")
                if not src:
                    continue
                
                img_url = urljoin(url, src)
                
                if img_url.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):
                    images.append(img_url)
                
                if len(images) >= max_images:
                    break
            
            return images
        except Exception as e:
            logger.warning(f"Failed to extract images from {url}: {e}")
            return []
    
    def search_and_extract(self, query: str, max_results: int = 8) -> List[Dict]:
        """
        Search and extract content from results
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of dictionaries with URL, text content, and images
        """
        logger.info(f"Searching Tavily for: {query}")
        
        # Perform search
        search_response = self.search(query, max_results=max_results)
        
        if "error" in search_response:
            logger.error(f"Tavily search error: {search_response['error']}")
            return []
        
        results = search_response.get("results", [])
        extracted_content = []
        
        for i, result in enumerate(results, 1):
            url = result.get("url", "")
            if not url:
                continue
            
            logger.info(f"Extracting content from source {i}: {url}")
            
            # Extract text
            text_content = self.extract_text(url)
            
            # Extract images
            images = self.extract_images(url)
            
            extracted_content.append({
                "url": url,
                "title": result.get("title", ""),
                "content": result.get("content", ""),  # Tavily's summary
                "extracted_text": text_content,
                "images": images,
                "score": result.get("score", 0.0)
            })
        
        logger.info(f"Extracted content from {len(extracted_content)} sources")
        return extracted_content
