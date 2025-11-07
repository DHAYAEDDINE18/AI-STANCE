from dataclasses import dataclass
from typing import List, Tuple, Dict
import re
import fitz  # PyMuPDF

@dataclass
class PageText:
    page_num: int  # 1-based
    text: str

def extract_text_from_pdf(path: str, header_clip_height: int = 50, footer_clip_height: int = 50) -> List[PageText]:
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        rect = page.rect
        clip = fitz.Rect(0, header_clip_height, rect.width, rect.height - footer_clip_height)
        text = page.get_text(clip=clip)
        pages.append(PageText(page_num=i + 1, text=text))
    doc.close()
    return pages

def detect_repeated_headers_footers(pages: List[PageText], top_n_lines: int = 2, bottom_n_lines: int = 2) -> Dict[str, str]:
    from collections import Counter
    top_counter, bot_counter = Counter(), Counter()
    for p in pages:
        lines = [ln.strip() for ln in p.text.splitlines() if ln.strip()]
        if not lines:
            continue
        top = " ".join(lines[:top_n_lines])
        bot = " ".join(lines[-bottom_n_lines:]) if len(lines) >= bottom_n_lines else ""
        if top: top_counter[top] += 1
        if bot: bot_counter[bot] += 1
    header_candidate = next(iter(top_counter.most_common(1)), ("", 0))[0]
    footer_candidate = next(iter(bot_counter.most_common(1)), ("", 0))[0]
    return {"header": re.escape(header_candidate) if header_candidate else "", "footer": re.escape(footer_candidate) if footer_candidate else ""}

def remove_headers_footers_and_numbers(pages: List[PageText], patterns: Dict[str, str]) -> List[PageText]:
    cleaned = []
    header_re = re.compile(patterns.get("header", ""), re.IGNORECASE) if patterns.get("header") else None
    footer_re = re.compile(patterns.get("footer", ""), re.IGNORECASE) if patterns.get("footer") else None
    page_num_re = re.compile(r"^\s*(\d+|[ivxlcdmIVXLCDM]+)\s*$")
    for p in pages:
        new_lines = []
        for ln in p.text.splitlines():
            s = ln.strip()
            if not s:
                continue
            if header_re and header_re.search(s):
                continue
            if footer_re and footer_re.search(s):
                continue
            if page_num_re.match(s):
                continue
            new_lines.append(ln)
        cleaned.append(PageText(page_num=p.page_num, text="\n".join(new_lines)))
    return cleaned

def combine_pages(pages: List[PageText]) -> str:
    parts = []
    for p in pages:
        parts.append(f"<<PAGE {p.page_num}>>\n{p.text}")
    return "\n\n".join(parts)

def chunk_text_for_model(text: str, target_chars: int = 24000, overlap_chars: int = 1200):
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + target_chars, n)
        boundary = text.rfind("\n\n", start, end)
        if boundary == -1 or boundary <= start + int(0.3 * target_chars):
            boundary = end
        chunk = text[start:boundary]
        chunks.append(chunk)
        if boundary == n:
            break
        start = max(0, boundary - overlap_chars)
    return chunks

def extract_text_pipeline(pdf_path: str) -> Tuple[List[PageText], List[PageText], str]:
    raw_pages = extract_text_from_pdf(pdf_path)
    patterns = detect_repeated_headers_footers(raw_pages)
    cleaned_pages = remove_headers_footers_and_numbers(raw_pages, patterns)
    combined_text = combine_pages(cleaned_pages)
    return raw_pages, cleaned_pages, combined_text
