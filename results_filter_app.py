"highlight_stance.py - Filter and highlight stance markers in PDF"
""
import os
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
import threading
import time

import fitz  # PyMuPDF
import pandas as pd


# Column name aliases for flexible CSV/Excel formats
CATEGORY_COL_ALIASES = [
    "hyland_category", "category", "type", "stance", "class", "stance_type", 
    "stance_category", "marker_type"
]
TEXT_COL_ALIASES = [
    "context", "sentence", "text", "snippet", "full_text", 
    "sentence_text", "original_text", "sentence_context"
]
PAGE_COL_ALIASES = [
    "page", "page_num", "page_number", "start_page"
]
CHUNK_COL_ALIASES = [
    "chunk_number", "chunk_num", "chunk", "section", "chunk_id"
]
MARKER_COL_ALIASES = ["marker", "word", "stance_word", "marker_text"]

# Priority order for highlighting preference
HIGHLIGHT_PREFERENCE = ["marker", "context", "sentence", "text", "snippet"]

# RGB colors (0..1 range for PyMuPDF)
CATEGORY_COLORS = {
    "hedges": (1.0, 1.0, 0.0),           # yellow
    "boosters": (1.0, 0.65, 0.0),         # orange
    "attitude_markers": (0.5, 0.9, 0.5),  # light green
    "self_mentions": (0.6, 0.8, 1.0),     # light blue
}
DEFAULT_COLOR = (1.0, 0.8, 0.2)  # amber


def norm_category(value: str) -> str:
    """Normalize category names to standard format."""
    if not value:
        return ""
    v = value.strip().lower().replace("-", "_").replace(" ", "_")
    
    if v in ("attitude", "attitude_marker", "attitude_markers", "attitude_mark"):
        return "attitude_markers"
    if v in ("self_mention", "self_mentions", "self-mentions", "self", "i_mentions"):
        return "self_mentions"
    if v in ("hedge", "hedges"):
        return "hedges"
    if v in ("booster", "boosters"):
        return "boosters"
    
    # Handle common misspellings
    if "attitud" in v:
        return "attitude_markers"
    if "self" in v:
        return "self_mentions"
    
    return v


def pick_color(cat: str):
    """Get color for category."""
    return CATEGORY_COLORS.get(cat, DEFAULT_COLOR)


def get_highlight_text(
    row,
    preference_order: list = None
) -> str:
    """
    Get the best text to highlight from row.
    Prioritizes context field, then sentence, then marker.
    """
    if preference_order is None:
        preference_order = HIGHLIGHT_PREFERENCE
    
    for col in preference_order:
        try:
            if col in row.index and pd.notna(row[col]):
                text = str(row[col]).strip()
                if text:
                    return text
        except Exception:
            continue
    
    return ""


def highlight_text_on_page(
    page: fitz.Page,
    text: str,
    color=(1, 1, 0),
    note: str = None,
    max_rects: int = 10
) -> int:
    """Highlight all occurrences of text on a page."""
    if not text or not text.strip():
        return 0
    
    clean_text = text.strip()
    is_single_word = len(clean_text.split()) == 1
    
    count = 0
    
    if is_single_word:
        # Use strict word matching for single words to avoid partial matches (e.g. "us" in "use")
        # page.get_text("words") returns list of (x0, y0, x1, y1, "word", block_no, line_no, word_no)
        words = page.get_text("words")
        target_lower = clean_text.lower()
        # Regex to match the word exactly, allowing for attached punctuation if strictly necessary,
        # but PyMuPDF 'words' usually isolates words well.
        # We'll check if the target is in the extracted word token, bounded by non-alphanumeric if needed.
        # Simple equality check is safest for "us", "we", "I".
        # But sometimes "us." comes out as "us.".
        
        for w in words:
            # w[4] is the text content of the word
            # Check for exact match or match with trailing punctuation
            word_text = w[4].lower()
            
            # Use regex to check if the target word exists as a distinct word within the token
            # This handles cases like "us." or "(us)"
            if re.search(r'\b' + re.escape(target_lower) + r'\b', word_text):
                rect = fitz.Rect(w[:4])
                try:
                    annot = page.add_highlight_annot(rect)
                    if annot:
                        annot.set_colors(stroke=color, fill=color)
                        if note:
                            try:
                                annot.set_info(content=note[:100])
                            except Exception:
                                pass
                        annot.update()
                        count += 1
                        if count >= max_rects:
                            break
                except Exception:
                    continue
    else:
        # Phrase/Sentence matching - standard search_for is appropriate
        search_text = re.sub(r'\s+', ' ', clean_text)
        try:
            rects = page.search_for(search_text, quads=False)
        except Exception:
            return 0
        
        if not rects:
            return 0
        
        if len(rects) > max_rects:
            rects = rects[:max_rects]
        
        rects.sort(key=lambda r: (r.y0, r.x0))
        
        for rect in rects:
            try:
                annot = page.add_highlight_annot(rect)
                if annot:
                    annot.set_colors(stroke=color, fill=color)
                    if note:
                        try:
                            annot.set_info(content=note[:100])
                        except Exception:
                            pass
                    annot.update()
                    count += 1
            except Exception:
                continue
    
    return count


def filter_data_against_pdf(
    pdf_path: str,
    data_path: str,
    out_dir: str,
    save_pdf: bool = True,
    progress_callback=None
) -> dict:
    """
    Remove rows whose text cannot be found in the PDF.
    Supports CSV and Excel.
    """
    def log(msg):
        if progress_callback:
            progress_callback(msg)
    
    log("Starting filtering process...")
    
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")

    os.makedirs(out_dir, exist_ok=True)
    filt_dir = os.path.join(out_dir, "filtered_results")
    os.makedirs(filt_dir, exist_ok=True)

    is_excel = data_path.lower().endswith(('.xlsx', '.xls'))
    file_type = "Excel" if is_excel else "CSV"
    
    log(f"Loading {file_type}: {Path(data_path).name}")
    try:
        if is_excel:
            df = pd.read_excel(data_path, keep_default_na=False)
        else:
            df = pd.read_csv(data_path, encoding="utf-8", keep_default_na=False)
    except Exception as e:
        raise ValueError(f"Failed to read {file_type}: {e}")
    
    if df.empty:
        log(f"⚠ {file_type} is empty")
        empty_df = pd.DataFrame(columns=df.columns if not df.empty else ["category", "context", "marker"])
        out_name = Path(data_path).name
        out_path = os.path.join(filt_dir, out_name)
        json_out = os.path.join(filt_dir, Path(data_path).stem + ".json")
        
        if is_excel:
            empty_df.to_excel(out_path, index=False)
        else:
            empty_df.to_csv(out_path, index=False, encoding="utf-8")
            
        return {
            "input_rows": 0,
            "kept_rows": 0,
            "dropped_rows": 0,
            "filter_rate": "0%",
            "data_out": out_path,
            "json_out": json_out,
            "pdf_out": None,
            "highlighted_field": "context"
        }

    # Find best text column
    text_col = None
    for col in HIGHLIGHT_PREFERENCE:
        if col in df.columns:
            text_col = col
            break
    
    if not text_col:
        raise ValueError(f"File must contain one of: {', '.join(HIGHLIGHT_PREFERENCE)}\nFound: {list(df.columns)}")
    
    col_category = next((c for c in CATEGORY_COL_ALIASES if c in df.columns), None)
    col_page = next((c for c in PAGE_COL_ALIASES if c in df.columns), None)

    log(f"Using text column: '{text_col}'")
    log(f"Opening PDF: {Path(pdf_path).name}")
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise ValueError(f"Failed to open PDF: {e}")
    
    keep_rows = []
    dropped_rows = 0
    total_rows = len(df)

    log(f"Validating {total_rows} rows...")

    for idx, row in df.iterrows():
        if (idx + 1) % 50 == 0:
            log(f"  Validated {idx + 1}/{total_rows} rows...")
        
        text_to_validate = str(row.get(text_col, "")).strip()
        if not text_to_validate:
            dropped_rows += 1
            continue

        ok = False
        if col_page:
            try:
                page_num = int(row.get(col_page))
                if 1 <= page_num <= len(doc):
                    page = doc[page_num - 1]
                    if page.search_for(text_to_validate):
                        ok = True
            except (ValueError, TypeError):
                pass
        
        if not ok:
            for p in doc:
                if p.search_for(text_to_validate):
                    ok = True
                    break

        if ok:
            keep_rows.append(row)
        else:
            dropped_rows += 1

    if keep_rows:
        fdf = pd.DataFrame(keep_rows)
    else:
        fdf = df.head(0)

    base_name = Path(data_path).name
    base_pdf = Path(pdf_path).name
    data_out = os.path.join(filt_dir, base_name)
    json_out = os.path.join(filt_dir, Path(base_name).stem + ".json")
    pdf_out = os.path.join(filt_dir, Path(base_pdf).stem + "_highlighted.pdf")

    log(f"Saving filtered {file_type} and JSON...")
    try:
        if is_excel:
            fdf.to_excel(data_out, index=False)
        else:
            fdf.to_csv(data_out, index=False, encoding="utf-8")
        fdf.to_json(json_out, orient="records", force_ascii=False, indent=2)
        log(f"✓ Saved: {len(fdf)} rows")
    except Exception as e:
        raise ValueError(f"Failed to save outputs: {e}")

    pdf_saved = False
    if save_pdf and not fdf.empty:
        log("Creating highlighted PDF...")
        total_highlights = 0
        try:
            doc = fitz.open(pdf_path)
            if doc:
                for idx, row in fdf.iterrows():
                    highlight_text = get_highlight_text(row, HIGHLIGHT_PREFERENCE)
                    if not highlight_text: continue
                    
                    cat_raw = str(row.get(col_category, "")).strip() if col_category else ""
                    cat = norm_category(cat_raw)
                    color = pick_color(cat)
                    marker = row.get("marker", "")
                    note = f"Hyland: {cat}\nMarker: {marker}"
                    
                    added = 0
                    if col_page:
                        try:
                            page_num = int(row.get(col_page))
                            if 1 <= page_num <= len(doc):
                                page = doc[page_num - 1]
                                added = highlight_text_on_page(page, highlight_text, color=color, note=note)
                        except: pass
                    
                    if added == 0:
                        for p in doc:
                            added = highlight_text_on_page(p, highlight_text, color=color, note=note)
                            if added > 0: break
                    total_highlights += added
                
                doc.save(pdf_out, incremental=False)
                doc.close()
                pdf_saved = True
                log(f"✓ PDF saved ({total_highlights} highlights)")
        except Exception as e:
            log(f"⚠ PDF save failed: {e}")

    log("Filtering complete!")
    return {
        "input_rows": int(total_rows),
        "kept_rows": int(len(keep_rows)),
        "dropped_rows": int(dropped_rows),
        "filter_rate": f"{(len(keep_rows)/total_rows*100):.1f}%" if total_rows > 0 else "0%",
        "highlighted_field": text_col,
        "data_out": data_out,
        "json_out": json_out,
        "pdf_out": pdf_out if pdf_saved else None
    }


class HighlighterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PDF Stance Highlighter & Filter")
        self.geometry("900x650")
        self.resizable(True, True)

        self.pdf_path = tk.StringVar()
        self.data_path = tk.StringVar()
        self.out_dir = tk.StringVar(value="outputs")
        self.use_filtered = tk.BooleanVar(value=False)
        self.auto_open = tk.BooleanVar(value=False)
        self.show_details = tk.BooleanVar(value=False)

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 8}
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        title_label = ttk.Label(main_frame, text="PDF Stance Highlighter & Filter", font=("Arial", 14, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 15))

        input_frame = ttk.LabelFrame(main_frame, text="Input Files", padding=10)
        input_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=5)
        
        ttk.Label(input_frame, text="PDF:", font=("Arial", 10)).grid(row=0, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.pdf_path, width=55).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(input_frame, text="Browse", command=self.browse_pdf).grid(row=0, column=2, sticky="e")
        
        ttk.Label(input_frame, text="Stance File (CSV/Excel):", font=("Arial", 10)).grid(row=1, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.data_path, width=55).grid(row=1, column=1, sticky="ew", padx=5)
        ttk.Button(input_frame, text="Browse", command=self.browse_data).grid(row=1, column=2, sticky="e")
        input_frame.grid_columnconfigure(1, weight=1)

        output_frame = ttk.LabelFrame(main_frame, text="Output Settings", padding=10)
        output_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5)
        ttk.Label(output_frame, text="Output folder:", font=("Arial", 10)).grid(row=0, column=0, sticky="w")
        ttk.Entry(output_frame, textvariable=self.out_dir, width=55).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(output_frame, text="Browse", command=self.browse_outdir).grid(row=0, column=2, sticky="e")
        output_frame.grid_columnconfigure(1, weight=1)

        options_frame = ttk.LabelFrame(main_frame, text="Options", padding=10)
        options_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=5)
        ttk.Checkbutton(options_frame, text="Use filtered results", variable=self.use_filtered).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(options_frame, text="Auto-open PDF", variable=self.auto_open).grid(row=0, column=1, sticky="w", padx=20)
        ttk.Checkbutton(options_frame, text="Show details", variable=self.show_details).grid(row=1, column=0, sticky="w")

        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=4, column=0, columnspan=3, pady=15)
        self.btn_filter = ttk.Button(btn_frame, text="🔍 Filter Data", command=self.run_filter, width=25)
        self.btn_filter.pack(side=tk.LEFT, padx=5)
        self.run_btn = ttk.Button(btn_frame, text="🖍️ Highlight PDF", command=self.run_highlight, width=25)
        self.run_btn.pack(side=tk.LEFT, padx=5)

        self.pb = ttk.Progressbar(main_frame, orient="horizontal", mode="determinate", length=600)
        self.pb.grid(row=5, column=0, columnspan=3, sticky="ew", pady=5)
        self.status = scrolledtext.ScrolledText(main_frame, height=12, width=80, wrap="word", font=("Consolas", 9))
        self.status.grid(row=6, column=0, columnspan=3, sticky="nsew", pady=5)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(6, weight=1)

    def log(self, msg: str):
        self.status.insert("end", f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        self.status.see("end")
        self.update_idletasks()

    def browse_pdf(self):
        fn = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")])
        if fn: self.pdf_path.set(fn)

    def browse_data(self):
        fn = filedialog.askopenfilename(filetypes=[("Data files", "*.csv *.xlsx *.xls"), ("All files", "*.*")])
        if fn: self.data_path.set(fn)

    def browse_outdir(self):
        d = filedialog.askdirectory()
        if d: self.out_dir.set(d)

    def run_filter(self):
        pdf = self.pdf_path.get().strip()
        data_path = self.data_path.get().strip()
        outdir = self.out_dir.get().strip() or "outputs"
        if not pdf or not data_path: return
        
        self.btn_filter.config(state="disabled")
        self.run_btn.config(state="disabled")
        
        def worker():
            try:
                res = filter_data_against_pdf(pdf, data_path, outdir, progress_callback=lambda m: self.after(0, self.log, m))
                self.after(0, self.log, f"Done! Kept {res['kept_rows']}/{res['input_rows']} rows.")
                self.after(0, lambda: messagebox.showinfo("Complete", "Filtering complete!"))
            except Exception as e:
                self.after(0, self.log, f"Error: {e}")
            finally:
                self.after(0, lambda: self.btn_filter.config(state="normal"))
                self.after(0, lambda: self.run_btn.config(state="normal"))
        threading.Thread(target=worker, daemon=True).start()

    def run_highlight(self):
        pdf = self.pdf_path.get().strip()
        data_path = self.data_path.get().strip()
        outdir = self.out_dir.get().strip() or "outputs"
        
        if self.use_filtered.get():
            base = Path(data_path).name
            data_path = os.path.join(outdir, "filtered_results", base)
            
        if not pdf or not data_path: 
            return

        self.btn_filter.config(state="disabled")
        self.run_btn.config(state="disabled")
        self.pb["value"] = 0

        def worker():
            try:
                self.after(0, self.log, "Starting highlighting...")
                
                # Load data
                is_excel = data_path.lower().endswith(('.xlsx', '.xls'))
                try:
                    df = pd.read_excel(data_path) if is_excel else pd.read_csv(data_path)
                except Exception as e:
                    raise ValueError(f"Failed to read data file: {e}")

                text_col = next((c for c in HIGHLIGHT_PREFERENCE if c in df.columns), None)
                if not text_col: 
                    raise ValueError(f"No text column found (checked: {HIGHLIGHT_PREFERENCE})")
                
                # Pre-process markers into a efficient structure
                # List of dicts: {'text': str, 'color': tuple, 'note': str, 'is_single_word': bool}
                markers = []
                for _, row in df.iterrows():
                    text = str(row.get(text_col, "")).strip()
                    if not text: continue
                    
                    cat = norm_category(str(row.get("hyland_category", "")))
                    color = pick_color(cat)
                    marker = row.get("marker", "")
                    note = f"Hyland: {cat}\nMarker: {marker}"
                    
                    markers.append({
                        'text': text,
                        'clean_text': text.lower(),
                        'color': color,
                        'note': note,
                        'is_single_word': len(text.split()) == 1,
                        'found': False # Track if we found it (optional, logic below highlights all occurrences)
                    })

                self.after(0, self.log, f"Loaded {len(markers)} markers. Processing PDF...")

                doc = fitz.open(pdf)
                total_pages = len(doc)
                total_highlights = 0
                
                # optimization: Iterate pages ONCE
                for page_idx, page in enumerate(doc):
                    # Update progress every page
                    progress = int((page_idx / total_pages) * 100)
                    self.after(0, lambda v=progress: self.pb.configure(value=v))
                    if page_idx % 5 == 0:
                         self.after(0, self.log, f"Processing page {page_idx+1}/{total_pages}...")

                    # 1. Get words once for single-word matching
                    page_words = page.get_text("words") # (x0, y0, x1, y1, "word", ...)
                    
                    # 2. Iterate all markers
                    for m in markers:
                        count_on_page = 0
                        
                        if m['is_single_word']:
                            target = m['clean_text']
                            # Check against page words
                            for w in page_words:
                                # Strict match check
                                word_text = w[4].lower()
                                if re.search(r'\b' + re.escape(target) + r'\b', word_text):
                                    rect = fitz.Rect(w[:4])
                                    try:
                                        annot = page.add_highlight_annot(rect)
                                        if annot:
                                            annot.set_colors(stroke=m['color'], fill=m['color'])
                                            if m['note']:
                                                annot.set_info(content=m['note'][:100])
                                            annot.update()
                                            count_on_page += 1
                                    except: pass
                        else:
                            # Phrase matching - fallback to search_for
                            # This is still potentially slow if many phrases, but search_for is optimized C++
                            try:
                                rects = page.search_for(m['text'], quads=False)
                                if rects:
                                    for rect in rects[:10]: # Limit max highlights per phrase per page
                                        annot = page.add_highlight_annot(rect)
                                        if annot:
                                            annot.set_colors(stroke=m['color'], fill=m['color'])
                                            if m['note']:
                                                annot.set_info(content=m['note'][:100])
                                            annot.update()
                                            count_on_page += 1
                            except: pass
                        
                        if count_on_page > 0:
                            total_highlights += count_on_page
                            # Optional: if you only want to highlight the FIRST occurrence in the document
                            # you could mark m['found'] = True and skip in future pages.
                            # But usually, we want all occurrences.

                out_path = os.path.join(outdir, f"{Path(pdf).stem}_highlighted.pdf")
                doc.save(out_path)
                doc.close()
                
                self.after(0, self.pb.configure, {"value": 100})
                self.after(0, self.log, f"✓ Saved: {out_path}")
                self.after(0, self.log, f"✓ Total highlights created: {total_highlights}")
                self.after(0, lambda: messagebox.showinfo("Success", f"PDF saved with {total_highlights} highlights!"))

            except Exception as e:
                self.after(0, self.log, f"Error: {e}")
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.after(0, lambda: self.btn_filter.config(state="normal"))
                self.after(0, lambda: self.run_btn.config(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

if __name__ == "__main__":
    HighlighterApp().mainloop()