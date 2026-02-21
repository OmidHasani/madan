"""
Advanced PDF processing module with multiple extraction methods
"""
import os
import json
import logging
import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None

try:
    import pdfplumber
except Exception:  # pragma: no cover
    pdfplumber = None

try:
    import PyPDF2
except Exception:  # pragma: no cover
    PyPDF2 = None


@dataclass
class DocumentChunk:
    """Represents a chunk of text from the document"""
    text: str
    page_number: int
    chunk_index: int
    metadata: Dict


class PDFProcessor:
    """
    Advanced PDF processor with multiple extraction strategies
    for maximum accuracy and content preservation
    """
    
    def __init__(self, pdf_path: str):
        """
        Initialize PDF processor
        
        Args:
            pdf_path: Path to the PDF file
        """
        self.pdf_path = pdf_path
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Initialized PDF processor for: {pdf_path}")
    
    def extract_text_pymupdf(self) -> List[Tuple[int, str]]:
        """
        Extract text using PyMuPDF (best for preserving layout)
        
        Returns:
            List of tuples (page_number, text_content)
        """
        logger.info("Extracting text using PyMuPDF...")
        if fitz is None:
            raise RuntimeError("PyMuPDF (fitz) is not installed. Install with: pip install pymupdf")
        pages_content = []
        
        try:
            doc = fitz.open(self.pdf_path)
            
            for page_num in tqdm(range(len(doc)), desc="Processing pages"):
                page = doc[page_num]
                text = page.get_text("text", sort=True)
                
                # Clean and normalize text
                text = self._clean_text(text)
                
                if text.strip():
                    pages_content.append((page_num + 1, text))
            
            doc.close()
            logger.info(f"Extracted {len(pages_content)} pages using PyMuPDF")
            
        except Exception as e:
            logger.error(f"Error extracting with PyMuPDF: {e}")
            raise
        
        return pages_content
    
    def extract_text_pdfplumber(self) -> List[Tuple[int, str]]:
        """
        Extract text using pdfplumber (best for tables and structured content)
        
        Returns:
            List of tuples (page_number, text_content)
        """
        logger.info("Extracting text using pdfplumber...")
        if pdfplumber is None:
            raise RuntimeError("pdfplumber is not installed. Install with: pip install pdfplumber")
        pages_content = []
        
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(tqdm(pdf.pages, desc="Processing pages")):
                    text = page.extract_text()
                    
                    # Try to extract tables if present
                    tables = page.extract_tables()
                    if tables:
                        table_text = self._format_tables(tables)
                        text = f"{text}\n\n{table_text}" if text else table_text
                    
                    text = self._clean_text(text)
                    
                    if text.strip():
                        pages_content.append((page_num + 1, text))
            
            logger.info(f"Extracted {len(pages_content)} pages using pdfplumber")
            
        except Exception as e:
            logger.error(f"Error extracting with pdfplumber: {e}")
            raise
        
        return pages_content

    def extract_text_pypdf2(self) -> List[Tuple[int, str]]:
        """
        Extract text using PyPDF2 (lightweight fallback).
        """
        logger.info("Extracting text using PyPDF2...")
        if PyPDF2 is None:
            raise RuntimeError("PyPDF2 is not installed. Install with: pip install pypdf2")

        pages_content: List[Tuple[int, str]] = []
        try:
            with open(self.pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page_num in tqdm(range(len(reader.pages)), desc="Processing pages"):
                    page = reader.pages[page_num]
                    text = page.extract_text() or ""
                    text = self._clean_text(text)
                    if text.strip():
                        pages_content.append((page_num + 1, text))
        except Exception as e:
            logger.error(f"Error extracting with PyPDF2: {e}")
            raise

        return pages_content
    
    def extract_text_hybrid(self) -> List[Tuple[int, str]]:
        """
        Use hybrid approach: combine both methods for best results
        
        Returns:
            List of tuples (page_number, text_content)
        """
        logger.info("Using hybrid extraction approach...")

        # Prefer PyMuPDF -> pdfplumber -> PyPDF2
        extractors = []
        if fitz is not None:
            extractors.append(("pymupdf", self.extract_text_pymupdf))
        if pdfplumber is not None:
            extractors.append(("pdfplumber", self.extract_text_pdfplumber))
        if PyPDF2 is not None:
            extractors.append(("pypdf2", self.extract_text_pypdf2))

        last_err: Exception | None = None
        for name, fn in extractors:
            try:
                pages = fn()
                total_chars = sum(len(text) for _, text in pages)
                if total_chars < 100:
                    logger.warning(f"{name} extraction seems poor ({total_chars} chars), trying next extractor...")
                    continue
                return pages
            except Exception as e:
                logger.warning(f"{name} extraction failed, trying next extractor: {e}")
                last_err = e

        raise RuntimeError(
            "No PDF text extractor is available. Install one of: pymupdf, pdfplumber, pypdf2"
        ) from last_err

    def extract_images_to_static(
        self,
        output_dir: str,
        manifest_path: str,
        image_format: str = "png",
        max_images_per_page: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Extract images from PDF pages (requires PyMuPDF) and write a manifest JSON.

        Output is intended to be served via FastAPI static mount, e.g.:
          output_dir = <project>/static/extracted_images
          manifest_path = <project>/static/images_manifest.json

        Manifest schema:
        {
          "source_pdf": "...",
          "output_dir": "static/extracted_images",
          "pages": {
             "1": [{"file": "...", "url": "...", "caption": "..."}],
             ...
          }
        }
        """
        if fitz is None:
            logger.warning("PyMuPDF (fitz) not installed; skipping image extraction.")
            manifest = {
                "source_pdf": self.pdf_path,
                "output_dir": output_dir,
                "pages": {},
                "note": "PyMuPDF not installed; no images extracted.",
            }
            os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            return manifest

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

        doc = fitz.open(self.pdf_path)
        pages: Dict[str, List[Dict[str, Any]]] = {}

        try:
            for page_index in tqdm(range(len(doc)), desc="Extracting images"):
                page = doc[page_index]
                page_num = page_index + 1

                # Text blocks for caption heuristics + context extraction
                blocks = page.get_text("blocks")  # (x0, y0, x1, y1, "text", block_no, block_type)
                page_text = self._clean_text(page.get_text("text", sort=True))
                page_codes = self._extract_codes(page_text)

                images = page.get_images(full=True)
                if not images:
                    continue

                page_items: List[Dict[str, Any]] = []
                for img_i, img in enumerate(images):
                    if max_images_per_page is not None and img_i >= max_images_per_page:
                        break

                    xref = img[0]
                    try:
                        pix = fitz.Pixmap(doc, xref)
                        if pix.n >= 5:  # CMYK: convert to RGB
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                    except Exception:
                        continue

                    filename = f"p{page_num:04d}_img{img_i:02d}.{image_format}"
                    out_path = os.path.join(output_dir, filename)
                    try:
                        pix.save(out_path)
                    except Exception:
                        try:
                            pix.tobytes(output=image_format)
                            with open(out_path, "wb") as f:
                                f.write(pix.tobytes(output=image_format))
                        except Exception:
                            continue
                    finally:
                        try:
                            pix = None
                        except Exception:
                            pass

                    # Caption heuristic: find nearest text block below the image rect.
                    caption = self._guess_caption_for_image(page, xref, blocks)
                    context_text = self._context_text_for_image(page, xref, blocks)
                    codes = sorted(set(self._extract_codes(caption) + self._extract_codes(context_text) + page_codes))

                    # URL path assuming output_dir is inside ./static
                    url = "/static/extracted_images/" + filename
                    page_items.append(
                        {
                            "file": filename,
                            "url": url,
                            "page": page_num,
                            "xref": xref,
                            "caption": caption,
                            "context": context_text,
                            "codes": codes,
                        }
                    )

                if page_items:
                    pages[str(page_num)] = page_items

        finally:
            doc.close()

        manifest = {
            "source_pdf": self.pdf_path,
            "output_dir": output_dir,
            "pages": pages,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        logger.info(f"Extracted images manifest written to: {manifest_path}")
        return manifest

    def _guess_caption_for_image(self, page, xref: int, blocks) -> str:
        """
        Heuristic caption extractor: pick the closest text block below the image.
        """
        if fitz is None:
            return ""
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            rects = []
        if not rects:
            return ""

        img_rect = rects[0]
        best_text = ""
        best_score = float("inf")

        for b in blocks or []:
            try:
                x0, y0, x1, y1, text, *_rest = b
            except Exception:
                continue
            if not text:
                continue
            # Only consider blocks below (or slightly overlapping) the image
            if y0 < img_rect.y1 - 5:
                continue

            # distance from image bottom to text top
            dy = y0 - img_rect.y1
            # horizontal overlap preference
            overlap = max(0.0, min(x1, img_rect.x1) - max(x0, img_rect.x0))
            overlap_penalty = 0.0 if overlap > 10 else 50.0

            score = dy + overlap_penalty
            if score < best_score:
                best_score = score
                best_text = str(text).strip()

        best_text = self._clean_text(best_text)
        # trim to reasonable caption length
        if len(best_text) > 240:
            best_text = best_text[:240] + "..."
        return best_text

    def _context_text_for_image(self, page, xref: int, blocks, max_chars: int = 600) -> str:
        """
        Extract a short context snippet around an image:
        - nearest text block below + nearest above (if any)
        This helps linking images to troubleshooting sections and error codes.
        """
        if fitz is None:
            return ""
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            rects = []
        if not rects:
            return ""
        img_rect = rects[0]

        above = []
        below = []
        for b in blocks or []:
            try:
                x0, y0, x1, y1, text, *_rest = b
            except Exception:
                continue
            if not text:
                continue
            text = str(text).strip()
            if not text:
                continue

            # prefer horizontally overlapping blocks
            overlap = max(0.0, min(x1, img_rect.x1) - max(x0, img_rect.x0))
            if overlap < 5:
                continue

            if y1 <= img_rect.y0 + 5:
                above.append((img_rect.y0 - y1, text))
            elif y0 >= img_rect.y1 - 5:
                below.append((y0 - img_rect.y1, text))

        above.sort(key=lambda x: x[0])
        below.sort(key=lambda x: x[0])

        parts = []
        if above:
            parts.append(above[0][1])
        if below:
            parts.append(below[0][1])

        ctx = self._clean_text("\n".join(parts))
        if len(ctx) > max_chars:
            ctx = ctx[:max_chars] + "..."
        return ctx

    def _extract_codes(self, text: str) -> List[str]:
        """
        Extract error-like codes such as CA1626, E11, etc.
        """
        if not text:
            return []
        # Common patterns in manuals: CA1626, E11, etc.
        codes = re.findall(r"\b[A-Za-z]{1,3}\d{2,6}\b", text)
        return [c.upper() for c in codes]
    
    def create_chunks(
        self, 
        pages_content: List[Tuple[int, str]], 
        chunk_size: int = 2500,
        chunk_overlap: int = 500
    ) -> List[DocumentChunk]:
        """
        Create overlapping chunks from pages with smart splitting that preserves tables.
        Includes parent context (full page) in metadata for better retrieval.
        
        Args:
            pages_content: List of (page_number, text) tuples
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
            
        Returns:
            List of DocumentChunk objects
        """
        logger.info(f"Creating chunks (size={chunk_size}, overlap={chunk_overlap})...")

        chunks: List[DocumentChunk] = []
        chunk_index = 0
        
        # Build page-to-text mapping for parent context
        page_text_map = {page_num: text for page_num, text in pages_content}

        for page_num, page_text in pages_content:
            # Preserve structure: split by blank lines (paragraphs / table blocks)
            blocks = [b.strip() for b in re.split(r"\n\s*\n", page_text) if b.strip()]

            current = ""
            for block in blocks:
                candidate = (current + "\n\n" + block).strip() if current else block
                
                # If this block looks like a table (has multiple lines with consistent structure),
                # try to keep it together even if it exceeds chunk_size slightly
                is_table = len(block.split('\n')) > 3 and any(
                    keyword in block.lower() 
                    for keyword in ['cause', 'standard value', 'resistance', 'voltage', 'wiring', 'trouble']
                )
                
                # Allow tables to be up to 1.5x chunk_size to keep them intact
                max_size = int(chunk_size * 1.5) if is_table else chunk_size
                
                if len(candidate) <= max_size:
                    current = candidate
                    continue

                # Save current chunk if it exists
                if current.strip():
                    # Get parent context (full page text)
                    parent_context = page_text.strip()
                    
                    chunks.append(
                        DocumentChunk(
                            text=current.strip(),
                            page_number=page_num,
                            chunk_index=chunk_index,
                            metadata={
                                "source": os.path.basename(self.pdf_path),
                                "page": page_num,
                                "chunk": chunk_index,
                                "char_count": len(current),
                                "parent_context": parent_context,  # Full page as parent
                            },
                        )
                    )
                    chunk_index += 1

                # Start new chunk with overlap
                if chunk_overlap > 0 and current:
                    overlap = current[-chunk_overlap:]
                    current = (overlap + "\n\n" + block).strip()
                else:
                    current = block

                # If still too big after allowing table size, hard split by length
                while len(current) > chunk_size * 2:  # Only split if REALLY big
                    part = current[:chunk_size]
                    # Get parent context (full page text)
                    parent_context = page_text.strip()
                    
                    chunks.append(
                        DocumentChunk(
                            text=part.strip(),
                            page_number=page_num,
                            chunk_index=chunk_index,
                            metadata={
                                "source": os.path.basename(self.pdf_path),
                                "page": page_num,
                                "chunk": chunk_index,
                                "char_count": len(part),
                                "parent_context": parent_context,  # Full page as parent
                            },
                        )
                    )
                    chunk_index += 1
                    current = current[chunk_size - chunk_overlap :] if chunk_overlap > 0 else current[chunk_size:]

            # Save remaining text
            if current.strip():
                # Get parent context (full page text)
                parent_context = page_text.strip()
                
                chunks.append(
                    DocumentChunk(
                        text=current.strip(),
                        page_number=page_num,
                        chunk_index=chunk_index,
                        metadata={
                            "source": os.path.basename(self.pdf_path),
                            "page": page_num,
                            "chunk": chunk_index,
                            "char_count": len(current),
                            "parent_context": parent_context,  # Full page as parent
                        },
                    )
                )
                chunk_index += 1

        logger.info(f"Created {len(chunks)} chunks from {len(pages_content)} pages")
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text WITHOUT destroying structure.
        - preserve newlines (important for troubleshooting tables)
        - normalize per-line spacing
        """
        if not text:
            return ""

        text = text.replace("\x00", "")
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        cleaned_lines: List[str] = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                cleaned_lines.append("")
                continue
            line = re.sub(r"[ \t]{2,}", " ", line)
            cleaned_lines.append(line)

        out_lines: List[str] = []
        blank_run = 0
        for ln in cleaned_lines:
            if ln == "":
                blank_run += 1
                if blank_run <= 1:
                    out_lines.append("")
            else:
                blank_run = 0
                out_lines.append(ln)

        return "\n".join(out_lines).strip()
    
    def _format_tables(self, tables: List) -> str:
        """Format extracted tables into readable text"""
        formatted = []
        
        for table in tables:
            for row in table:
                if row:
                    row_text = " | ".join(str(cell) if cell else "" for cell in row)
                    formatted.append(row_text)
            formatted.append("")  # Empty line between tables
        
        return "\n".join(formatted)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences with smart handling of abbreviations
        """
        import re
        
        # Simple sentence splitter (can be enhanced with NLTK if needed)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_text(self, sentences: List[str], target_overlap: int) -> str:
        """
        Get overlap text from the end of previous chunk
        """
        overlap = ""
        for sentence in reversed(sentences):
            if len(overlap) + len(sentence) <= target_overlap:
                overlap = sentence + " " + overlap
            else:
                break
        return overlap.strip()
    
    def process(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[DocumentChunk]:
        """
        Main processing method: extract and chunk the PDF
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of DocumentChunk objects
        """
        logger.info(f"Starting PDF processing: {self.pdf_path}")
        
        # Extract text
        pages_content = self.extract_text_hybrid()
        
        # Create chunks
        chunks = self.create_chunks(pages_content, chunk_size, chunk_overlap)
        
        logger.info(f"PDF processing complete: {len(chunks)} chunks created")
        
        return chunks

