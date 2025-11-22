"""
analysis.py - Core analysis functions with logging and error handling
"""
import csv
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .prompts import (
    SEGMENT_SYSTEM, SEGMENT_USER_TEMPLATE, STANCE_SYSTEM, STANCE_USER_TEMPLATE,
    QUERY_SYSTEM, QUERY_USER_TEMPLATE
)
from .ai_clients import GeminiClient
from .pdf_utils import chunk_text_for_model


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
    
    # File handler - detailed logs
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler - important messages only
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info("=" * 70)
    logger.info("PDF Text Chunking & Stance Analysis Session Started")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 70)
    
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
        logger.debug(f"Problematic JSON string: {s[:500]}...")
        raise


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
    logger.info(f"Splitting text into fixed-size chunks...")
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
    Simple pipeline: PDF → Text → Fixed-size Chunks → Files
    No AI segmentation - just split by character count.
    
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
    logger.info(f"Chunk size: {chunk_size:,} chars, Overlap: {overlap} chars")
    logger.info("=" * 70)
    
    from .pdf_utils import save_pdf_as_text
    
    # Get PDF name
    pdf_name = Path(pdf_path).stem
    
    # Step 1: Convert PDF to text
    logger.info("STEP 1: Converting PDF to text file...")
    text_file = save_pdf_as_text(pdf_path, output_dir=str(Path(output_base) / "converted_PDFs"))
    
    # Load the text
    with open(text_file, "r", encoding="utf-8") as f:
        full_text = f.read()
    
    logger.info(f"✓ Loaded text: {len(full_text):,} characters")
    
    # Step 2: Split into fixed-size chunks
    logger.info("STEP 2: Splitting text into chunks (no AI segmentation)...")
    chunks = split_text_into_chunks(full_text, chunk_size=chunk_size, overlap=overlap)
    logger.info(f"✓ Created {len(chunks)} chunks")
    
    # Step 3: Save each chunk to individual file
    logger.info("STEP 3: Saving chunks as individual text files...")
    chunk_files = save_chunks_to_files(
        chunks=chunks,
        pdf_name=pdf_name,
        base_dir=output_base
    )
    
    # Summary
    logger.info("=" * 70)
    logger.info("Pipeline complete!")
    logger.info(f"  Text file: {text_file}")
    logger.info(f"  Chunks: {len(chunk_files)} files")
    logger.info(f"  Chunk directory: {Path(output_base) / f'{pdf_name}_sections'}")
    logger.info("=" * 70)
    
    return {
        "pdf_path": pdf_path,
        "text_file": text_file,
        "chunks": chunks,
        "chunk_files": chunk_files,
        "chunk_directory": str(Path(output_base) / f"{pdf_name}_sections"),
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
        logger.info(f"ℹ model_name parameter ignored (no AI segmentation used)")
    
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
            max_output_tokens=12000,
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
            
            all_stance_results.append(stance_result)
            successful += 1
            
        except Exception as e:
            logger.error(f"✗ Failed to analyze {chunk_path.name}: {e}")
            failed += 1
            continue
    
    logger.info(f"Analysis complete. Success: {successful}/{len(chunk_files)}, Failed: {failed}")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # JSON output
    json_file = output_path / "stance_results.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(all_stance_results, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Saved JSON: {json_file}")
    
    # CSV output (flattened)
    csv_file = output_path / "stance_results.csv"
    save_stance_to_csv(all_stance_results, str(csv_file))
    logger.info(f"✓ Saved CSV: {csv_file}")
    
    # Summary
    total_markers = sum(
        len(r.get("hedges", [])) + 
        len(r.get("boosters", [])) + 
        len(r.get("attitude_markers", [])) + 
        len(r.get("self_mentions", []))
        for r in all_stance_results
    )
    
    logger.info("=" * 70)
    logger.info(f"Stance analysis summary:")
    logger.info(f"  Total chunks analyzed: {successful}")
    logger.info(f"  Total stance markers: {total_markers}")
    logger.info(f"  Results saved to: {output_dir}")
    logger.info("=" * 70)
    
    return {
        "results": all_stance_results,
        "json_file": str(json_file),
        "csv_file": str(csv_file),
        "total_chunks": successful,
        "total_markers": total_markers,
        "failed_chunks": failed
    }


def save_stance_to_csv(results: list[dict[str, Any]], csv_path: str):
    """
    Save stance results to CSV with flattened structure.
    
    Args:
        results: List of stance result dictionaries
        csv_path: Output CSV file path
    """
    logger = get_logger()
    
    rows = []
    for result in results:
        chunk_num = result.get("chunk_number", 0)
        chunk_file = result.get("chunk_file", "")
        section = result.get("section", "")
        
        # Flatten each category
        for category in ["hedges", "boosters", "attitude_markers", "self_mentions"]:
            markers = result.get(category, [])
            for marker in markers:
                if isinstance(marker, dict):
                    rows.append({
                        "chunk_number": chunk_num,
                        "chunk_file": chunk_file,
                        "section": section,
                        "category": category,
                        "marker": marker.get("marker", marker.get("text", "")),
                        "sentence": marker.get("sentence", ""),
                        "context": marker.get("context", "")
                    })
                else:
                    # Handle simple string markers
                    rows.append({
                        "chunk_number": chunk_num,
                        "chunk_file": chunk_file,
                        "section": section,
                        "category": category,
                        "marker": str(marker),
                        "sentence": "",
                        "context": ""
                    })
    
    if not rows:
        logger.warning("⚠ No stance markers to save to CSV")
        # Create empty CSV with headers
        rows = [{
            "chunk_number": 0,
            "chunk_file": "",
            "section": "",
            "category": "",
            "marker": "",
            "sentence": "",
            "context": ""
        }]
    
    fieldnames = ["chunk_number", "chunk_file", "section", "category", "marker", "sentence", "context"]
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    logger.info(f"✓ Saved {len(rows)} stance markers to CSV")


def process_pdf_with_stance_analysis(
    pdf_path: str,
    output_base: str = ".",
    chunk_size: int = 80000,
    overlap: int = 1000,
    model_name: str | None = None
) -> dict[str, Any]:
    """
    Complete pipeline: PDF → Chunks → Stance Analysis
    
    Args:
        pdf_path: Path to source PDF file
        output_base: Base directory for outputs
        chunk_size: Maximum characters per chunk
        overlap: Character overlap between chunks
        model_name: Gemini model name for stance analysis
        
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
    
    # Step 2: Analyze stance on chunks
    stance_result = analyze_chunks_for_stance(
        chunk_files=chunk_result["chunk_files"],
        output_dir=output_base,
        model_name=model_name
    )
    
    # Combine results
    final_result = {
        **chunk_result,
        "stance_json": stance_result["json_file"],
        "stance_csv": stance_result["csv_file"],
        "total_markers": stance_result["total_markers"],
        "failed_chunks": stance_result["failed_chunks"]
    }
    
    logger.info("=" * 70)
    logger.info("Complete pipeline finished!")
    logger.info("=" * 70)
    
    return final_result
