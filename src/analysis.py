"""
analysis.py - Core analysis functions with logging and error handling
"""
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .prompts import (
    STANCE_SYSTEM, STANCE_USER_TEMPLATE
)
from .ai_clients import GeminiClient

from .html_report import generate_html_report


# --- Logging Configuration ---
_logger = None

def setup_logger(output_folder: str = "outputs") -> logging.Logger:
    """
    Initialize file and console logging.
    
    Args:
        output_folder: Directory to store log files
        
    Returns:
        Configured logger instance
    """
    global _logger
    
    if _logger is not None:
        return _logger
    
    # Create output folder if needed
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(output_folder) / f"analysis_log_{timestamp}.txt"
    
    # Configure logger
    logger = logging.getLogger("StanceAnalyzer")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # Remove any existing handlers
    
    # Console handler - important messages only
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.CRITICAL + 1)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(console_handler)
    
    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    """Get the configured logger instance."""
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger


# --- Rate Limiting Configuration ---
MAX_RETRIES = 5
BASE_DELAY = 2.0  # seconds
MAX_DELAY = 60.0  # seconds
REQUEST_DELAY = 0.5  # delay between successful requests


def _retry_with_backoff(func, *args, **kwargs):
    """
    Execute a function with exponential backoff on rate limit errors.
    
    Args:
        func: Function to execute
        *args, **kwargs: Arguments to pass to function
        
    Returns:
        Function result
        
    Raises:
        Exception: If all retries exhausted or non-retryable error
    """
    logger = get_logger()
    
    for attempt in range(MAX_RETRIES):
        try:
            result = func(*args, **kwargs)
            if attempt > 0:
                logger.info(f"✓ Retry successful after {attempt} attempt(s)")
            return result
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check for resource exhaustion / rate limit errors
            is_rate_limit = any(
                keyword in error_msg 
                for keyword in ["429", "resource", "exhausted", "quota", "rate limit"]
            )
            
            if is_rate_limit:
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"✗ Max retries ({MAX_RETRIES}) exceeded. Error: {e}")
                    raise
                
                # Exponential backoff
                delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                logger.warning(
                    f"⚠ Rate limit hit (attempt {attempt + 1}/{MAX_RETRIES}). "
                    f"Retrying in {delay:.1f}s... Error: {e}"
                )
                time.sleep(delay)
            else:
                # Non-rate-limit error, don't retry
                logger.error(f"✗ Non-retryable error: {e}")
                raise
    
    raise Exception("Max retries exceeded")


# --- JSON Parsing ---
def parse_json_str(s: str) -> Any:
    """
    Parse JSON that may be wrapped in Markdown code fences.
    
    Args:
        s: JSON string, possibly with `````` wrappers
        
    Returns:
        Parsed JSON object
    """
    logger = get_logger()
    s = s.strip()
    
    # Try to extract from code fence
    m = re.search(r"``````", s, flags=re.DOTALL)
    candidate = m.group(1) if m else s

    # Balanced brace/bracket extraction
    start = None
    depth = 0
    in_str = False
    esc = False
    
    for i, ch in enumerate(candidate):
        if start is None:
            if ch in "[{":
                start = i
                depth = 1
        else:
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch in "[{":
                    depth += 1
                elif ch in "]}":
                    depth -= 1
                    if depth == 0:
                        fragment = candidate[start : i + 1]
                        try:
                            return json.loads(fragment)
                        except json.JSONDecodeError as e:
                            logger.warning(f"JSON parse failed on extracted fragment: {e}")
                            # Fall through to try full candidate
    
    # Fallback: parse entire candidate
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}")
        logger.debug(f"Problematic JSON string: {s}")
        raise


# --- Semantic Text Sectioning ---
SECTION_HEADINGS = {
    "introduction": r"^\s*introduction\s*$",
    "literature_review": r"^\s*literature\s+review\s*$",
    "methodology": r"^\s*(methodology|methods)\s*$",
    "results": r"^\s*(results|findings)\s*$",
    "discussion": r"^\s*discussion\s*$",
    "conclusion": r"^\s*(conclusion|conclusions)\s*$",
    "general_conclusion": r"^\s*general\s+conclusion\s*$",
    "references": r"^\s*(references|bibliography)\s*$",
    "appendices": r"^\s*(appendices|appendix)\s*$",
}

def split_text_into_sections(
    full_text: str,
    chunk_size: int = 80000,
    overlap: int = 1000
) -> list[dict[str, Any]]:
    """
    Splits text into sections based on detected headings, falling back to chunking.

    Args:
        full_text: The entire document text.
        chunk_size: Target size for chunks if no sections are found.
        overlap: Overlap for character-based chunking.

    Returns:
        A list of dictionaries, each representing a section or chunk.
    """
    logger = get_logger()
    logger.info("Attempting to split text by semantic sections...")

    # Find all potential headings
    found_headings = []
    # Use re.finditer to get all matches with their positions
    for name, pattern in SECTION_HEADINGS.items():
        for match in re.finditer(pattern, full_text, re.IGNORECASE | re.MULTILINE):
            # Check if the matched line is short enough to be a heading
            line = match.group(0).strip()
            if len(line) < 50: # A reasonable max length for a heading
                 found_headings.append({
                    "title": name.replace("_", " ").title(),
                    "text": line,
                    "start_pos": match.start(),
                })

    # Sort headings by their position in the text
    found_headings.sort(key=lambda x: x["start_pos"])

    # Remove overlapping/duplicate headings
    unique_headings = []
    last_pos = -1
    for heading in found_headings:
        if heading["start_pos"] > last_pos:
            unique_headings.append(heading)
            last_pos = heading["start_pos"]

    if len(unique_headings) > 1:
        logger.info(f"Found {len(unique_headings)} section headings. Splitting text...")
        sections = []
        for i, heading in enumerate(unique_headings):
            start_pos = heading["start_pos"]
            # Determine end position
            if i + 1 < len(unique_headings):
                end_pos = unique_headings[i+1]["start_pos"]
            else:
                end_pos = len(full_text)
            
            section_text = full_text[start_pos:end_pos].strip()
            
            # Page number calculation for the section
            page_pattern = re.compile(r"<<PAGE\s+(\d+)>>")
            section_pages = [int(m.group(1)) for m in page_pattern.finditer(section_text)]
            start_page = min(section_pages) if section_pages else 0
            end_page = max(section_pages) if section_pages else 0

            sections.append({
                "title": heading["title"],
                "start_page": start_page,
                "end_page": end_page,
                "text": section_text,
                "char_count": len(section_text),
                "chunk_number": i + 1,
            })
            logger.info(f"  ✓ Created section: '{heading['title']}' ({len(section_text):,} chars)")
        return sections
    else:
        logger.warning("No clear section headings found. Falling back to fixed-size chunking.")
        # Fallback to original chunking logic if no sections are detected
        return split_text_into_chunks(full_text, chunk_size, overlap)


# --- Simple Text Chunking (No AI Segmentation) ---
def split_text_into_chunks(
    full_text: str,
    chunk_size: int = 80000,
    overlap: int = 1000
) -> list[dict[str, Any]]:
    """
    Split text into fixed-size chunks without AI segmentation.
    Creates simple sequential sections based on character count.
    
    Args:
        full_text: Page-anchored text to split
        chunk_size: Maximum characters per chunk (default: 80000)
        overlap: Character overlap between chunks for context (default: 1000)
        
    Returns:
        List of chunk dictionaries with text and metadata
    """
    logger = get_logger()
    logger.info("Splitting text into fixed-size chunks...")
    logger.info(f"  Chunk size: {chunk_size:,} chars, Overlap: {overlap} chars")
    
    # Extract page information
    page_pattern = re.compile(r"<<PAGE\s+(\d+)>>")
    pages_in_text = [int(m.group(1)) for m in page_pattern.finditer(full_text)]
    
    if not pages_in_text:
        logger.warning("⚠ No page anchors found in text")
        first_page, last_page = 1, 1
    else:
        first_page = min(pages_in_text)
        last_page = max(pages_in_text)
    
    logger.info(f"Document spans pages {first_page}-{last_page}")
    
    # Split into chunks
    chunks = []
    text_length = len(full_text)
    start_pos = 0
    chunk_num = 1
    
    while start_pos < text_length:
        # Calculate end position
        end_pos = min(start_pos + chunk_size, text_length)
        
        # Try to break at paragraph boundary
        if end_pos < text_length:
            # Look for paragraph break within last 20% of chunk
            search_start = int(start_pos + chunk_size * 0.8)
            paragraph_break = full_text.rfind("\n\n", search_start, end_pos)
            
            if paragraph_break != -1 and paragraph_break > start_pos:
                end_pos = paragraph_break
        
        # Extract chunk
        chunk_text = full_text[start_pos:end_pos].strip()
        
        # Find page range for this chunk
        chunk_pages = [int(m.group(1)) for m in page_pattern.finditer(chunk_text)]
        if chunk_pages:
            chunk_start_page = min(chunk_pages)
            chunk_end_page = max(chunk_pages)
        else:
            # Estimate based on position
            progress = start_pos / text_length
            chunk_start_page = int(first_page + progress * (last_page - first_page))
            chunk_end_page = chunk_start_page
        
        # Create chunk entry
        chunk = {
            "title": f"Part {chunk_num}",
            "start_page": chunk_start_page,
            "end_page": chunk_end_page,
            "text": chunk_text,
            "char_count": len(chunk_text),
            "chunk_number": chunk_num
        }
        
        chunks.append(chunk)
        logger.info(
            f"✓ Chunk {chunk_num}: pages {chunk_start_page}-{chunk_end_page}, "
            f"{len(chunk_text):,} chars"
        )
        
        # Move to next chunk with overlap
        start_pos = end_pos - overlap if end_pos < text_length else text_length
        chunk_num += 1
    
    logger.info(f"✓ Created {len(chunks)} chunks from {text_length:,} characters")
    return chunks


def save_chunks_to_files(
    chunks: list[dict[str, Any]],
    pdf_name: str,
    base_dir: str = "."
) -> list[str]:
    """
    Save each text chunk as a separate file.
    
    Args:
        chunks: List of chunk dictionaries with text and metadata
        pdf_name: Name of source PDF (without extension)
        base_dir: Base directory for output
        
    Returns:
        List of created file paths
    """
    logger = get_logger()
    
    # Create section directory
    section_dir = Path(base_dir) / f"{pdf_name}_sections"
    section_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving {len(chunks)} chunks to: {section_dir}")
    
    created_files: list[str] = []
    skipped_empty = 0
    
    for chunk in chunks:
        chunk_num = chunk.get("chunk_number", 1)
        title = chunk.get("title", f"Part {chunk_num}")
        start_page = chunk.get("start_page", 1)
        end_page = chunk.get("end_page", start_page)
        text = chunk.get("text", "")
        char_count = chunk.get("char_count", len(text))
        
        # Skip empty chunks
        if len(text.strip()) < 100:
            logger.warning(f"⚠ Skipping empty chunk #{chunk_num}")
            skipped_empty += 1
            continue
        
        # Create filename
        filename = f"{chunk_num:03d}_{title.replace(' ', '_')}.txt"
        file_path = section_dir / filename
        
        # Write file with metadata header
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"SECTION: {title}\n")
                f.write(f"CHUNK: {chunk_num}\n")
                f.write(f"PAGES: {start_page}-{end_page} (estimated)\n")
                f.write(f"LENGTH: {char_count:,} characters\n")
                f.write("=" * 70 + "\n\n")
                f.write(text)
            
            logger.info(f"✓ Saved: {filename} ({char_count:,} chars)")
            created_files.append(str(file_path))
            
        except Exception as e:
            logger.error(f"✗ Failed to save chunk #{chunk_num}: {e}")
    
    if skipped_empty > 0:
        logger.info(f"⚠ Skipped {skipped_empty} empty chunks")
    
    logger.info(f"✓ Created {len(created_files)} chunk files in: {section_dir}")
    return created_files


# --- Complete PDF Processing Pipeline (No AI Segmentation) ---
def process_pdf_to_chunks(
    pdf_path: str,
    output_base: str = ".",
    chunk_size: int = 80000,
    overlap: int = 1000
) -> dict[str, Any]:
    """
    Simple pipeline: PDF → Text → Fixed-size Chunks → Files.
    Organizes outputs into a directory named after the PDF.
    
    Args:
        pdf_path: Path to source PDF file
        output_base: Base directory for all outputs
        chunk_size: Maximum characters per chunk (default: 80000)
        overlap: Character overlap between chunks (default: 1000)
        
    Returns:
        Dictionary with paths and metadata
    """
    logger = get_logger()
    logger.info("=" * 70)
    logger.info(f"Starting PDF chunking pipeline: {pdf_path}")
    
    from .pdf_utils import save_pdf_as_text
    
    # Create a dedicated directory for this PDF's output
    pdf_name = Path(pdf_path).stem
    pdf_output_dir = Path(output_base) / pdf_name
    pdf_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output for this PDF will be saved in: {pdf_output_dir}")
    
    logger.info(f"Chunk size: {chunk_size:,} chars, Overlap: {overlap} chars")
    logger.info("=" * 70)
    
    # Step 1: Convert PDF to text
    logger.info("STEP 1: Converting PDF to text file...")
    text_file = save_pdf_as_text(pdf_path, output_dir=str(pdf_output_dir))
    
    # Load the text
    with open(text_file, "r", encoding="utf-8") as f:
        full_text = f.read()
    
    logger.info(f"✓ Loaded text: {len(full_text):,} characters")
    
    # Step 2: Split into sections/chunks
    logger.info("STEP 2: Splitting text into sections...")
    chunks = split_text_into_sections(full_text, chunk_size=chunk_size, overlap=overlap)
    logger.info(f"✓ Created {len(chunks)} sections/chunks")
    
    # Step 3: Save each chunk to individual file
    logger.info("STEP 3: Saving chunks as individual text files...")
    chunk_files = save_chunks_to_files(
        chunks=chunks,
        pdf_name=pdf_name,
        base_dir=str(pdf_output_dir)
    )
    
    # Summary
    logger.info("=" * 70)
    logger.info("Chunking complete for this PDF!")
    logger.info(f"  Text file: {text_file}")
    logger.info(f"  Chunks: {len(chunk_files)} files")
    logger.info(f"  Chunk directory: {pdf_output_dir / f'{pdf_name}_sections'}")
    logger.info("=" * 70)
    
    return {
        "pdf_path": pdf_path,
        "text_file": text_file,
        "chunks": chunks,
        "chunk_files": chunk_files,
        "chunk_directory": str(pdf_output_dir / f"{pdf_name}_sections"),
        "total_chunks": len(chunk_files)
    }


# --- Backward Compatibility Alias ---
def process_pdf_to_sections(
    pdf_path: str,
    output_base: str = ".",
    model_name: str | None = None,
    chunk_size: int = 80000,
    overlap: int = 1000
) -> dict[str, Any]:
    """
    Alias for process_pdf_to_chunks for backward compatibility.
    Ignores model_name parameter (no AI used).
    
    Args:
        pdf_path: Path to source PDF file
        output_base: Base directory for all outputs
        model_name: Ignored (kept for API compatibility)
        chunk_size: Maximum characters per chunk
        overlap: Character overlap between chunks
        
    Returns:
        Dictionary with paths and metadata
    """
    logger = get_logger()
    if model_name:
        logger.info("ℹ model_name parameter ignored (no AI segmentation used)")
    
    result = process_pdf_to_chunks(pdf_path, output_base, chunk_size, overlap)
    
    # Rename fields for backward compatibility
    result["sections"] = result.pop("chunks", [])
    result["section_files"] = result.pop("chunk_files", [])
    result["section_directory"] = result.pop("chunk_directory", "")
    
    return result


# --- Stance Analysis ---
def ai_analyse_stance(
    section_title: str,
    section_text: str,
    model_name: str | None = None
) -> dict[str, Any]:
    """
    Analyze Hyland stance markers in text section.
    Skips analysis if text is empty or too short.
    
    Args:
        section_title: Section name for logging
        section_text: Text to analyze
        model_name: Optional model override
        
    Returns:
        Dictionary with stance categories and markers
    """
    logger = get_logger()
    logger.info(f"Analyzing stance for: {section_title}")
    logger.debug(f"Section text length: {len(section_text)} chars")
    
    # Skip empty or very short sections
    if len(section_text.strip()) < 50:
        logger.warning(f"⚠ Skipping '{section_title}' - insufficient text ({len(section_text)} chars)")
        return {
            "section": section_title,
            "hedges": [],
            "boosters": [],
            "attitude_markers": [],
            "self_mentions": [],
            "summary": "Skipped: insufficient text",
        }
    
    client = GeminiClient(model_name=model_name)
    
    # Aggressive truncation to avoid exhaustion
    max_text_length = 80000
    truncated = section_text[:max_text_length]
    
    if len(section_text) > max_text_length:
        logger.warning(
            f"⚠ Text truncated from {len(section_text):,} to {max_text_length:,} chars"
        )
    
    prompt = STANCE_USER_TEMPLATE.format(
        section_title=section_title,
        text=truncated
    )
    
    def _generate():
        return client.generate_json(
            prompt,
            system_instruction=STANCE_SYSTEM,
            temperature=0.2,
            max_output_tokens=16000,
        )
    
    try:
        out = _retry_with_backoff(_generate)
        time.sleep(REQUEST_DELAY)
        
        data = parse_json_str(out)
        
        # Validate structure
        for k in ["hedges", "boosters", "attitude_markers", "self_mentions"]:
            if k not in data or not isinstance(data[k], list):
                logger.warning(f"Missing or invalid stance category: {k}")
                data[k] = []
        
        data.setdefault("section", section_title)
        data.setdefault("summary", "")
        
        # Count markers
        total_markers = sum(len(data[k]) for k in ["hedges", "boosters", "attitude_markers", "self_mentions"])
        logger.info(f"✓ Stance analysis complete. Found {total_markers} markers")
        logger.debug(
            f"  Hedges: {len(data['hedges'])}, "
            f"Boosters: {len(data['boosters'])}, "
            f"Attitude: {len(data['attitude_markers'])}, "
            f"Self-mentions: {len(data['self_mentions'])}"
        )
        
        return data
        
    except Exception as e:
        logger.error(f"✗ Stance analysis failed for '{section_title}': {e}")
        return {
            "section": section_title,
            "hedges": [],
            "boosters": [],
            "attitude_markers": [],
            "self_mentions": [],
            "summary": f"Analysis failed: {str(e)[:100]}",
        }


def analyze_chunks_for_stance(
    chunk_files: list[str],
    output_dir: str,
    model_name: str | None = None
) -> dict[str, Any]:
    """
    Analyze each chunk file for Hyland stance markers.
    
    Args:
        chunk_files: List of paths to chunk text files
        output_dir: Directory to save stance results
        model_name: Optional Gemini model name
        
    Returns:
        Dictionary with results and statistics
    """
    logger = get_logger()
    logger.info("=" * 70)
    logger.info(f"Starting stance analysis on {len(chunk_files)} chunks")
    logger.info("=" * 70)
    
    all_stance_results = []
    successful = 0
    failed = 0
    
    for idx, chunk_file in enumerate(chunk_files, 1):
        chunk_path = Path(chunk_file)
        logger.info(f"Analyzing chunk {idx}/{len(chunk_files)}: {chunk_path.name}")
        
        try:
            # Read chunk file
            with open(chunk_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Extract metadata and text (skip header)
            lines = content.split("\n")
            text_start = 0
            for i, line in enumerate(lines):
                if line.startswith("=" * 70):
                    text_start = i + 1
                    break
            
            chunk_text = "\n".join(lines[text_start:]).strip()
            
            # Extract chunk number from filename
            chunk_num_match = re.match(r"(\d+)_", chunk_path.name)
            chunk_num = int(chunk_num_match.group(1)) if chunk_num_match else idx
            
            # Analyze stance
            stance_result = ai_analyse_stance(
                section_title=f"Chunk {chunk_num}",
                section_text=chunk_text,
                model_name=model_name
            )
            
            # Add metadata
            stance_result["chunk_file"] = chunk_path.name
            stance_result["chunk_number"] = chunk_num
            stance_result["word_count"] = len(chunk_text.split())
            
            all_stance_results.append(stance_result)
            successful += 1
            
        except Exception as e:
            logger.error(f"✗ Failed to analyze {chunk_path.name}: {e}")
            failed += 1
            continue
    
    logger.info(f"Analysis complete. Success: {successful}/{len(chunk_files)}, Failed: {failed}")
    
    # Clean up empty 'sentence' fields before saving
    for result in all_stance_results:
        for category in ["hedges", "boosters", "attitude_markers", "self_mentions"]:
            if category in result:
                for marker_data in result[category]:
                    if "sentence" in marker_data and not marker_data["sentence"]:
                        del marker_data["sentence"]

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # JSON output
    json_file = output_path / "stance_results.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(all_stance_results, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Saved JSON: {json_file}")
    
    # Excel output (flattened)
    excel_file = output_path / "stance_results.xlsx"
    save_stance_to_excel(all_stance_results, str(excel_file))
    logger.info(f"✓ Saved Excel: {excel_file}")
    
    # Summary
    total_markers = sum(
        len(r.get("hedges", [])) + 
        len(r.get("boosters", [])) + 
        len(r.get("attitude_markers", [])) + 
        len(r.get("self_mentions", []))
        for r in all_stance_results
    )
    
    logger.info("=" * 70)
    logger.info("Stance analysis summary:")
    logger.info(f"  Total chunks analyzed: {successful}")
    logger.info(f"  Total stance markers: {total_markers}")
    logger.info(f"  Results saved to: {output_dir}")
    logger.info("=" * 70)
    
    return {
        "results": all_stance_results,
        "json_file": str(json_file),
        "excel_file": str(excel_file),
        "total_chunks": successful,
        "total_markers": total_markers,
        "failed_chunks": failed
    }


def save_stance_to_excel(results: list[dict[str, Any]], excel_path: str):
    """
    Save stance results to Excel with a flattened structure.
    
    Args:
        results: List of stance result dictionaries.
        excel_path: Output Excel file path.
    """
    logger = get_logger()
    
    rows = []
    for result in results:
        # Flatten each category
        for category in ["hedges", "boosters", "attitude_markers", "self_mentions"]:
            markers = result.get(category, [])
            
            for marker in markers:
                row_data = {
                    "hyland_category": category,
                    "marker": "",
                    "context": ""
                }
                
                if isinstance(marker, dict):
                    row_data.update({
                        "marker": marker.get("marker", marker.get("text", "")),
                        "context": marker.get("context", "")
                    })
                else:
                    # Handle simple string markers
                    row_data["marker"] = str(marker)
                
                rows.append(row_data)
    
    if not rows:
        logger.warning("⚠ No stance markers to save. Creating an empty Excel file with headers.")
        df = pd.DataFrame(columns=[
            "hyland_category", "marker", "context"
        ])
    else:
        df = pd.DataFrame(rows)
    
    try:
        df.to_excel(excel_path, index=False, engine='openpyxl')
        logger.info(f"✓ Saved {len(df)} stance markers to Excel: {excel_path}")
    except Exception as e:
        logger.error(f"✗ Failed to save Excel: {e}")





def process_pdf_with_stance_analysis(
    pdf_path: str,
    output_base: str = ".",
    chunk_size: int = 80000,
    overlap: int = 1000,
    model_name: str | None = None
) -> dict[str, Any]:
    """
    Complete pipeline: PDF → Chunks → Stance Analysis → Optional Diagnostic Analysis
    
    Args:
        pdf_path: Path to source PDF file
        output_base: Base directory for outputs
        chunk_size: Maximum characters per chunk
        overlap: Character overlap between chunks
        model_name: Gemini model name for stance analysis
        do_diagnostics: Whether to perform diagnostic analysis
        
    Returns:
        Dictionary with all results and file paths
    """
    logger = get_logger()
    logger.info("=" * 70)
    logger.info("Starting complete PDF → Chunks → Stance pipeline")
    logger.info("=" * 70)
    
    # Step 1: Chunk the PDF
    chunk_result = process_pdf_to_chunks(
        pdf_path=pdf_path,
        output_base=output_base,
        chunk_size=chunk_size,
        overlap=overlap
    )
    
    # The output directory is now inside the chunk_result, but let's create it here
    # to be explicit.
    pdf_name = Path(pdf_path).stem
    pdf_output_dir = Path(output_base) / pdf_name
    pdf_output_dir.mkdir(parents=True, exist_ok=True)


    # Step 2: Analyze stance on chunks
    stance_result = analyze_chunks_for_stance(
        chunk_files=chunk_result["chunk_files"],
        output_dir=str(pdf_output_dir),
        model_name=model_name
    )
    
    # Combine results
    final_result = {
        **chunk_result,
        "results": stance_result["results"],
        "stance_json": stance_result["json_file"],
        "stance_excel": stance_result["excel_file"],
        "total_markers": stance_result["total_markers"],
        "failed_chunks": stance_result["failed_chunks"]
    }
    


    # Rename fields for backward compatibility, ensuring consistent output
    final_result["sections"] = final_result.pop("chunks", [])
    final_result["section_files"] = final_result.pop("chunk_files", [])
    final_result["section_directory"] = final_result.pop("chunk_directory", "")
    
    # Save the final result with diagnostics to the JSON file
    with open(stance_result["json_file"], "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)
        
    # Generate HTML report
    html_report_path = str(Path(stance_result["json_file"]).with_suffix(".html"))
    generate_html_report(stance_result["json_file"], html_report_path)
    final_result["html_report"] = html_report_path
    
    logger.info("=" * 70)
    logger.info("Complete pipeline finished!")
    logger.info(f"HTML report generated at: {html_report_path}")
    logger.info("=" * 70)
    
    return final_result
