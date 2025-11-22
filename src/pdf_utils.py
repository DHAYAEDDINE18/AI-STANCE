"""
pdf_utils.py - PDF extraction and text utilities with header/footer removal
"""
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import fitz  # PyMuPDF


@dataclass
class PageText:
    """Container for page number and extracted text."""
    page_num: int  # 1-based
    text: str


def extract_text_from_pdf(
    path: str,
    header_clip_height: int = 50,
    footer_clip_height: int = 50
) -> List[PageText]:
    """
    Extract text from PDF with optional header/footer clipping.
    
    Args:
        path: Path to PDF file
        header_clip_height: Pixels to clip from top (0 = no clipping)
        footer_clip_height: Pixels to clip from bottom (0 = no clipping)
        
    Returns:
        List of PageText objects with extracted text
    """
    from .analysis import get_logger
    logger = get_logger()
    
    logger.info(f"Extracting text from PDF: {path}")
    logger.debug(f"Clipping - Header: {header_clip_height}px, Footer: {footer_clip_height}px")
    
    doc = fitz.open(path)
    pages = []
    
    for i, page in enumerate(doc):
        rect = page.rect
        clip = fitz.Rect(
            0,
            header_clip_height,
            rect.width,
            rect.height - footer_clip_height
        )
        text = page.get_text(clip=clip)
        pages.append(PageText(page_num=i + 1, text=text))
    
    doc.close()
    logger.info(f"✓ Extracted {len(pages)} pages ({sum(len(p.text) for p in pages)} chars)")
    return pages


def detect_repeated_headers_footers(
    pages: List[PageText],
    top_n_lines: int = 2,
    bottom_n_lines: int = 2
) -> Dict[str, str]:
    """
    Detect repeated header and footer patterns across pages.
    
    Args:
        pages: List of PageText objects
        top_n_lines: Number of top lines to check for headers
        bottom_n_lines: Number of bottom lines to check for footers
        
    Returns:
        Dictionary with 'header' and 'footer' regex patterns (escaped)
    """
    from .analysis import get_logger
    logger = get_logger()
    
    logger.debug("Detecting repeated headers and footers...")
    
    top_counter, bot_counter = Counter(), Counter()
    
    for p in pages:
        lines = [ln.strip() for ln in p.text.splitlines() if ln.strip()]
        if not lines:
            continue
        
        # Top lines (header)
        top = " ".join(lines[:top_n_lines])
        # Bottom lines (footer)
        bot = " ".join(lines[-bottom_n_lines:]) if len(lines) >= bottom_n_lines else ""
        
        if top:
            top_counter[top] += 1
        if bot:
            bot_counter[bot] += 1
    
    # Most common patterns
    header_candidate = next(iter(top_counter.most_common(1)), ("", 0))[0]
    footer_candidate = next(iter(bot_counter.most_common(1)), ("", 0))[0]
    
    # Only use if repeated on multiple pages (threshold: 3+ pages)
    header_count = top_counter.get(header_candidate, 0)
    footer_count = bot_counter.get(footer_candidate, 0)
    
    result = {
        "header": re.escape(header_candidate) if header_count >= 3 else "",
        "footer": re.escape(footer_candidate) if footer_count >= 3 else ""
    }
    
    if result["header"]:
        logger.info(f"✓ Detected header pattern (appears {header_count}x): {header_candidate[:50]}...")
    if result["footer"]:
        logger.info(f"✓ Detected footer pattern (appears {footer_count}x): {footer_candidate[:50]}...")
    
    return result


def remove_headers_footers_and_numbers(
    pages: List[PageText],
    patterns: Dict[str, str]
) -> List[PageText]:
    """
    Remove detected headers, footers, and standalone page numbers.
    
    Args:
        pages: List of PageText objects
        patterns: Dictionary with 'header' and 'footer' regex patterns
        
    Returns:
        List of cleaned PageText objects
    """
    from .analysis import get_logger
    logger = get_logger()
    
    logger.debug("Removing headers, footers, and page numbers...")
    
    cleaned = []
    header_re = re.compile(patterns.get("header", ""), re.IGNORECASE) if patterns.get("header") else None
    footer_re = re.compile(patterns.get("footer", ""), re.IGNORECASE) if patterns.get("footer") else None
    page_num_re = re.compile(r"^\s*(\d+|[ivxlcdmIVXLCDM]+)\s*$")
    
    total_lines_removed = 0
    
    for p in pages:
        new_lines = []
        lines_removed_this_page = 0
        
        for ln in p.text.splitlines():
            s = ln.strip()
            
            if not s:
                continue
            
            # Check patterns
            if header_re and header_re.search(s):
                lines_removed_this_page += 1
                continue
            if footer_re and footer_re.search(s):
                lines_removed_this_page += 1
                continue
            if page_num_re.match(s):
                lines_removed_this_page += 1
                continue
            
            new_lines.append(ln)
        
        total_lines_removed += lines_removed_this_page
        cleaned.append(PageText(page_num=p.page_num, text="\n".join(new_lines)))
    
    logger.info(f"✓ Removed {total_lines_removed} header/footer/number lines")
    return cleaned


def combine_pages(pages: List[PageText]) -> str:
    """
    Combine page texts with <<PAGE N>> anchors.
    
    Args:
        pages: List of PageText objects
        
    Returns:
        Combined text with page markers
    """
    from .analysis import get_logger
    logger = get_logger()
    
    parts = []
    for p in pages:
        parts.append(f"<<PAGE {p.page_num}>>\n{p.text}")
    
    combined = "\n\n".join(parts)
    logger.debug(f"Combined {len(pages)} pages into {len(combined)} characters")
    return combined


def chunk_text_for_model(
    text: str,
    target_chars: int = 24000,
    overlap_chars: int = 1200
) -> List[str]:
    """
    Split text into overlapping chunks for model processing.
    
    Args:
        text: Input text to chunk
        target_chars: Target size per chunk (default: 24000)
        overlap_chars: Overlap between chunks (default: 1200)
        
    Returns:
        List of text chunks
    """
    from .analysis import get_logger
    logger = get_logger()
    
    chunks = []
    start = 0
    n = len(text)
    
    while start < n:
        end = min(start + target_chars, n)
        
        # Try to find paragraph boundary
        boundary = text.rfind("\n\n", start, end)
        if boundary == -1 or boundary <= start + int(0.3 * target_chars):
            boundary = end
        
        chunk = text[start:boundary]
        chunks.append(chunk)
        
        if boundary == n:
            break
        
        # Move start with overlap
        start = max(0, boundary - overlap_chars)
    
    logger.debug(f"Split text into {len(chunks)} chunks (target: {target_chars} chars, overlap: {overlap_chars})")
    return chunks


def extract_text_pipeline(pdf_path: str) -> Tuple[List[PageText], List[PageText], str]:
    """
    Complete extraction pipeline: extract → clean → combine.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Tuple of (raw_pages, cleaned_pages, combined_text)
    """
    from .analysis import get_logger
    logger = get_logger()
    
    logger.info(f"Starting extraction pipeline for: {pdf_path}")
    
    # Step 1: Extract raw text
    raw_pages = extract_text_from_pdf(pdf_path)
    
    # Step 2: Detect patterns
    patterns = detect_repeated_headers_footers(raw_pages)
    
    # Step 3: Clean
    cleaned_pages = remove_headers_footers_and_numbers(raw_pages, patterns)
    
    # Step 4: Combine
    combined_text = combine_pages(cleaned_pages)
    
    logger.info(f"✓ Pipeline complete. Combined text: {len(combined_text)} characters")
    return raw_pages, cleaned_pages, combined_text


def save_pdf_as_text(
    pdf_path: str,
    output_dir: str = "converted_PDFs"
) -> str:
    """
    Convert PDF to text file using the full extraction pipeline.
    
    Args:
        pdf_path: Path to source PDF
        output_dir: Directory to save text files (default: converted_PDFs)
        
    Returns:
        Path to the created text file
    """
    from .analysis import get_logger
    logger = get_logger()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate text filename
    pdf_name = Path(pdf_path).stem
    text_file = output_path / f"{pdf_name}.txt"
    
    logger.info(f"Converting PDF to text: {pdf_path}")
    
    # Use full pipeline (cleans headers/footers)
    _, cleaned_pages, combined_text = extract_text_pipeline(pdf_path)
    
    # Save to file
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(combined_text)
    
    logger.info(f"✓ Saved cleaned text file: {text_file} ({len(combined_text)} chars)")
    logger.info(f"  Original pages: {len(cleaned_pages)}, cleaned lines included")
    
    return str(text_file)


def extract_text_with_pages(pdf_path: str) -> str:
    """
    Simple extraction with page markers (backward compatibility).
    Uses the full cleaning pipeline.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Text with <<PAGE N>> markers
    """
    _, _, combined_text = extract_text_pipeline(pdf_path)
    return combined_text


def _is_aux_page(text: str) -> bool:
    """
    Detect if page is likely TOC, References, or auxiliary content.
    
    Args:
        text: Page text content
        
    Returns:
        True if page appears to be auxiliary
    """
    text_lower = text.lower()
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    
    # TOC indicators
    if any(kw in text_lower for kw in ["table of contents", "contents", "chapter"]):
        dots_or_numbers = sum(1 for ln in lines if "...." in ln or re.search(r"\d+\s*$", ln))
        if dots_or_numbers > 3:
            return True
    
    # References indicators
    if any(kw in text_lower for kw in ["references", "bibliography", "works cited"]):
        return True
    
    return False
