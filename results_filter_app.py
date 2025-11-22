"""
highlight_stance.py - Filter and highlight stance markers in PDF
"""
import os
import json
import re  # ADDED: Missing import
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
import threading
import time

import fitz  # PyMuPDF
import pandas as pd


# Column name aliases for flexible CSV formats
CATEGORY_COL_ALIASES = [
    "category", "type", "stance", "class", "stance_type", 
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
HIGHLIGHT_PREFERENCE = ["context", "sentence", "text", "snippet"]

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
    preference_order: list = None,
    max_length: int = 100
) -> str:
    """
    Get the best text to highlight from CSV row.
    Prioritizes context field, then sentence, then marker.
    
    Args:
        row: DataFrame row from CSV
        preference_order: List of preferred column names
        max_length: Maximum length of highlighted text
        
    Returns:
        The text to highlight, or empty string if none found
    """
    if preference_order is None:
        preference_order = HIGHLIGHT_PREFERENCE
    
    for col in preference_order:
        try:
            if col in row.index and pd.notna(row[col]):
                text = str(row[col]).strip()
                if text:
                    # Truncate if too long
                    if len(text) > max_length:
                        text = text[:max_length] + "..."
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
    """
    Highlight all occurrences of text on a page.
    
    Args:
        page: PyMuPDF page
        text: Text to search for and highlight
        color: RGB tuple (0-1 range)
        note: Annotation note
        max_rects: Maximum rectangles to process per page
        
    Returns:
        Number of highlights added
    """
    if not text or not text.strip():
        return 0
    
    # Clean text for search
    search_text = re.sub(r'\s+', ' ', text.strip())
    
    # Search for the text
    try:
        rects = page.search_for(search_text, quads=False)
    except Exception:
        return 0
    
    if not rects:
        return 0
    
    # Limit to avoid performance issues
    if len(rects) > max_rects:
        rects = rects[:max_rects]
    
    # Sort by position
    rects.sort(key=lambda r: (r.y0, r.x0))
    
    count = 0
    
    # Create highlights
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


def filter_csv_against_pdf(
    pdf_path: str,
    csv_path: str,
    out_dir: str,
    save_pdf: bool = True,
    progress_callback=None
) -> dict:
    """
    Remove rows whose text cannot be found in the PDF.
    Uses context field for validation and highlighting.
    """
    def log(msg):
        if progress_callback:
            progress_callback(msg)
    
    log("Starting filtering process...")
    
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    os.makedirs(out_dir, exist_ok=True)
    filt_dir = os.path.join(out_dir, "filtered_results")
    os.makedirs(filt_dir, exist_ok=True)

    log(f"Loading CSV: {Path(csv_path).name}")
    try:
        df = pd.read_csv(csv_path, encoding="utf-8", keep_default_na=False)
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {e}")
    
    if df.empty:
        log("⚠ CSV is empty")
        empty_df = pd.DataFrame(columns=df.columns if not df.empty else ["category", "context", "marker"])
        csv_out = os.path.join(filt_dir, Path(csv_path).name)
        json_out = os.path.join(filt_dir, Path(csv_path).stem + ".json")
        empty_df.to_csv(csv_out, index=False, encoding="utf-8")
        return {
            "input_rows": 0,
            "kept_rows": 0,
            "dropped_rows": 0,
            "filter_rate": "0%",
            "csv_out": csv_out,
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
        raise ValueError(f"CSV must contain one of: {', '.join(HIGHLIGHT_PREFERENCE)}\nFound: {list(df.columns)}")
    
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

        # Try to find text in PDF
        ok = False
        
        # Try page-specific search first
        if col_page:
            try:
                page_num = int(row.get(col_page))
                if 1 <= page_num <= len(doc):
                    page = doc[page_num - 1]
                    if page.search_for(text_to_validate):
                        ok = True
            except (ValueError, TypeError):
                pass
        
        # Document-wide search if page-specific fails
        if not ok:
            for p in doc:
                if p.search_for(text_to_validate):
                    ok = True
                    break

        if ok:
            keep_rows.append(row)
        else:
            dropped_rows += 1

    # Build filtered DataFrame
    if keep_rows:
        fdf = pd.DataFrame(keep_rows)
    else:
        fdf = df.head(0)

    # Generate output paths
    base_csv = Path(csv_path).name
    base_pdf = Path(pdf_path).name
    csv_out = os.path.join(filt_dir, base_csv)
    json_out = os.path.join(filt_dir, Path(base_csv).stem + ".json")
    pdf_out = os.path.join(filt_dir, Path(base_pdf).stem + "_highlighted.pdf")

    # Save filtered CSV and JSON
    log("Saving filtered CSV and JSON...")
    try:
        fdf.to_csv(csv_out, index=False, encoding="utf-8")
        fdf.to_json(json_out, orient="records", force_ascii=False, indent=2)
        log(f"✓ Saved: {len(fdf)} rows")
    except Exception as e:
        raise ValueError(f"Failed to save outputs: {e}")

    # Create highlighted PDF
    pdf_saved = False
    if save_pdf and not fdf.empty:
        log("Creating highlighted PDF...")
        total_highlights = 0
        
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            log(f"⚠ Could not reopen PDF: {e}")
            doc = None
        
        if doc:
            for idx, row in fdf.iterrows():
                highlight_text = get_highlight_text(row, HIGHLIGHT_PREFERENCE, max_length=100)
                if not highlight_text:
                    continue
                
                cat_raw = str(row.get(col_category, "")).strip() if col_category else ""
                cat = norm_category(cat_raw)
                color = pick_color(cat)
                marker = row.get("marker", "")
                note = f"{cat}: {marker}" if cat else f"Stance: {marker}"
                
                # Try page-specific highlighting
                added = 0
                if col_page:
                    try:
                        page_num = int(row.get(col_page))
                        if 1 <= page_num <= len(doc):
                            page = doc[page_num - 1]
                            added = highlight_text_on_page(page, highlight_text, color=color, note=note)
                    except (ValueError, TypeError):
                        pass
                
                # Document-wide if page-specific fails
                if added == 0:
                    for p in doc:
                        added += highlight_text_on_page(p, highlight_text, color=color, note=note)
                        if added > 0:
                            break
                
                total_highlights += added
            
            try:
                log(f"Saving PDF ({total_highlights} highlights)...")
                doc.save(pdf_out, incremental=False)
                doc.close()
                pdf_saved = True
            except Exception as e:
                log(f"⚠ PDF save failed: {e}")
                if doc:
                    doc.close()

    log("Filtering complete!")
    
    return {
        "input_rows": int(total_rows),
        "kept_rows": int(len(keep_rows)),
        "dropped_rows": int(dropped_rows),
        "filter_rate": f"{(len(keep_rows)/total_rows*100):.1f}%" if total_rows > 0 else "0%",
        "highlighted_field": text_col,
        "csv_out": csv_out,
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
        self.csv_path = tk.StringVar()
        self.out_dir = tk.StringVar(value="outputs")
        self.use_filtered = tk.BooleanVar(value=False)
        self.auto_open = tk.BooleanVar(value=False)
        self.show_details = tk.BooleanVar(value=False)

        self._build_ui()

    def _build_ui(self):
        """Build the GUI layout."""
        pad = {"padx": 10, "pady": 8}
        
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="PDF Stance Highlighter & Filter",
            font=("Arial", 14, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 15))

        # Input files
        input_frame = ttk.LabelFrame(main_frame, text="Input Files", padding=10)
        input_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=5)
        
        # PDF
        ttk.Label(input_frame, text="PDF:", font=("Arial", 10)).grid(
            row=0, column=0, sticky="w", padx=(0, 5), pady=2
        )
        ttk.Entry(input_frame, textvariable=self.pdf_path, width=55).grid(
            row=0, column=1, sticky="ew", padx=5, pady=2
        )
        ttk.Button(input_frame, text="Browse", command=self.browse_pdf).grid(
            row=0, column=2, sticky="e", padx=5, pady=2
        )
        
        # CSV
        ttk.Label(input_frame, text="Stance CSV:", font=("Arial", 10)).grid(
            row=1, column=0, sticky="w", padx=(0, 5), pady=2
        )
        ttk.Entry(input_frame, textvariable=self.csv_path, width=55).grid(
            row=1, column=1, sticky="ew", padx=5, pady=2
        )
        ttk.Button(input_frame, text="Browse", command=self.browse_csv).grid(
            row=1, column=2, sticky="e", padx=5, pady=2
        )
        
        input_frame.grid_columnconfigure(1, weight=1)

        # Output settings
        output_frame = ttk.LabelFrame(main_frame, text="Output Settings", padding=10)
        output_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5)
        
        # Output directory
        ttk.Label(output_frame, text="Output folder:", font=("Arial", 10)).grid(
            row=0, column=0, sticky="w", padx=(0, 5), pady=2
        )
        ttk.Entry(output_frame, textvariable=self.out_dir, width=55).grid(
            row=0, column=1, sticky="ew", padx=5, pady=2
        )
        ttk.Button(output_frame, text="Browse", command=self.browse_outdir).grid(
            row=0, column=2, sticky="e", padx=5, pady=2
        )
        
        output_frame.grid_columnconfigure(1, weight=1)

        # Options
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding=10)
        options_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=5)
        
        ttk.Checkbutton(
            options_frame,
            text="Use filtered CSV (run filter first)",
            variable=self.use_filtered
        ).grid(row=0, column=0, sticky="w", pady=2)
        
        ttk.Checkbutton(
            options_frame,
            text="Auto-open PDF after completion",
            variable=self.auto_open
        ).grid(row=0, column=1, sticky="w", pady=2, padx=20)
        
        ttk.Checkbutton(
            options_frame,
            text="Show detailed log",
            variable=self.show_details
        ).grid(row=1, column=0, sticky="w", pady=2)

        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=4, column=0, columnspan=3, pady=15)
        
        self.btn_filter = ttk.Button(
            btn_frame,
            text="🔍 Filter CSV Against PDF",
            command=self.run_filter,
            width=25
        )
        self.btn_filter.pack(side=tk.LEFT, padx=5)

        self.run_btn = ttk.Button(
            btn_frame,
            text="🖍️ Highlight Context",
            command=self.run_highlight,
            width=25
        )
        self.run_btn.pack(side=tk.LEFT, padx=5)

        # Progress bar
        self.pb = ttk.Progressbar(main_frame, orient="horizontal", mode="determinate", length=600)
        self.pb.grid(row=5, column=0, columnspan=3, sticky="ew", pady=5)
        
        self.progress_label = ttk.Label(main_frame, text="Ready", foreground="blue")
        self.progress_label.grid(row=6, column=0, columnspan=3, pady=(0, 5))

        # Color legend
        legend_frame = ttk.LabelFrame(main_frame, text="Highlighting Legend", padding=5)
        legend_frame.grid(row=7, column=0, columnspan=3, sticky="ew", pady=5)
        
        legend_text = "Yellow=Hedges | Orange=Boosters | Green=Attitude | Blue=Self-mentions"
        ttk.Label(legend_frame, text=legend_text, font=("Arial", 9)).pack()

        # Status log
        log_frame = ttk.LabelFrame(main_frame, text="Status Log", padding=5)
        log_frame.grid(row=8, column=0, columnspan=3, sticky="nsew", pady=5)
        
        self.status = scrolledtext.ScrolledText(
            log_frame,
            height=12,
            width=80,
            wrap="word",
            font=("Consolas", 9)
        )
        self.status.pack(fill="both", expand=True)
        
        # Configure grid weights
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(8, weight=1)

    def log(self, msg: str):
        """Add message to status log."""
        timestamp = time.strftime("%H:%M:%S")
        self.status.insert("end", f"[{timestamp}] {msg}\n")
        self.status.see("end")
        self.update_idletasks()

    def set_progress(self, value: int, text: str = ""):
        """Update progress bar and label."""
        self.pb["value"] = value
        if text:
            self.progress_label.config(text=text)
        self.update_idletasks()

    def browse_pdf(self):
        """Browse for PDF file."""
        fn = filedialog.askopenfilename(
            title="Select PDF",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if fn:
            self.pdf_path.set(fn)
            self.log(f"Selected PDF: {Path(fn).name}")

    def browse_csv(self):
        """Browse for CSV file."""
        fn = filedialog.askopenfilename(
            title="Select stance CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if fn:
            self.csv_path.set(fn)
            self.log(f"Selected CSV: {Path(fn).name}")

    def browse_outdir(self):
        """Browse for output directory."""
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self.out_dir.set(d)
            self.log(f"Output folder: {d}")

    def run_filter(self):
        """Run filtering in background thread."""
        pdf = self.pdf_path.get().strip()
        csv_path = self.csv_path.get().strip()
        outdir = self.out_dir.get().strip() or "outputs"

        if not os.path.isfile(pdf):
            messagebox.showerror("Missing PDF", "Please select a valid PDF.")
            return
        if not os.path.isfile(csv_path):
            messagebox.showerror("Missing CSV", "Please select a valid CSV.")
            return

        # Disable buttons
        self.btn_filter.config(state="disabled")
        self.run_btn.config(state="disabled")
        self.status.delete(1.0, tk.END)
        self.set_progress(0, "Starting...")
        self.log("=" * 70)
        self.log("Starting CSV filtering...")
        self.log("=" * 70)

        def worker():
            try:
                res = filter_csv_against_pdf(
                    pdf,
                    csv_path,
                    outdir,
                    save_pdf=True,
                    progress_callback=lambda msg: self.after(0, self.log, msg)
                )
                
                self.after(0, self.set_progress, 100, "Complete!")
                self.after(0, self.log, "\n" + "=" * 70)
                self.after(0, self.log, "FILTERING COMPLETE!")
                self.after(0, self.log, f"Input: {res['input_rows']} rows")
                self.after(0, self.log, f"Kept: {res['kept_rows']} ({res['filter_rate']})")
                self.after(0, self.log, f"Dropped: {res['dropped_rows']}")
                self.after(0, self.log, f"\nHighlighted field: '{res['highlighted_field']}'")
                self.after(0, self.log, f"CSV: {res['csv_out']}")
                self.after(0, self.log, f"JSON: {res['json_out']}")
                if res.get("pdf_out"):
                    self.after(0, self.log, f"PDF: {res['pdf_out']}")
                self.after(0, self.log, "=" * 70)
                
                self.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Filter Complete",
                        f"Filtering complete!\n\n"
                        f"Kept: {res['kept_rows']}/{res['input_rows']} ({res['filter_rate']})\n"
                        f"Field: '{res['highlighted_field']}'"
                    )
                )
                
            except Exception as e:
                self.after(0, self.set_progress, 0, "Failed")
                self.after(0, self.log, f"\n✗ ERROR: {str(e)}")
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
            
            finally:
                self.after(0, lambda: self.btn_filter.config(state="normal"))
                self.after(0, lambda: self.run_btn.config(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    def run_highlight(self):
        """Run highlighting in background thread."""
        pdf = self.pdf_path.get().strip()
        csv_path = self.csv_path.get().strip()
        outdir = self.out_dir.get().strip() or "outputs"

        # Use filtered CSV if checkbox is selected
        if self.use_filtered.get():
            base_csv = Path(csv_path).name if csv_path else "stance_results.csv"
            csv_path = os.path.join(outdir, "filtered_results", base_csv)

        if not os.path.isfile(pdf):
            messagebox.showerror("Missing PDF", "Please select a valid PDF.")
            return
        if not os.path.isfile(csv_path):
            messagebox.showerror("Missing CSV", f"CSV not found: {csv_path}")
            return

        # Disable buttons
        self.btn_filter.config(state="disabled")
        self.run_btn.config(state="disabled")
        self.status.delete(1.0, tk.END)
        self.set_progress(0, "Starting...")
        self.log("=" * 70)
        self.log("Starting highlighting...")
        self.log("=" * 70)

        def worker():
            try:
                # Load CSV
                self.after(0, self.log, "Loading CSV...")
                self.after(0, self.set_progress, 10, "Loading...")
                df = pd.read_csv(csv_path, encoding="utf-8", keep_default_na=False)
                
                total_rows = len(df)
                if total_rows == 0:
                    self.after(0, self.log, "⚠ CSV is empty")
                    self.after(0, lambda: messagebox.showwarning("Empty CSV", "CSV is empty."))
                    return
                
                # Find text column
                text_col = None
                for pref in HIGHLIGHT_PREFERENCE:
                    if pref in df.columns:
                        text_col = pref
                        break
                
                if not text_col:
                    raise ValueError(f"CSV must contain one of: {', '.join(HIGHLIGHT_PREFERENCE)}")
                
                col_category = next((c for c in CATEGORY_COL_ALIASES if c in df.columns), None)
                col_page = next((c for c in PAGE_COL_ALIASES if c in df.columns), None)

                self.after(0, self.log, f"Processing {total_rows} rows (field: '{text_col}')...")
                self.after(0, self.set_progress, 20, "Opening PDF...")
                doc = fitz.open(pdf)
                
                total_highlights = 0
                skipped = 0

                self.after(0, self.set_progress, 25, "Highlighting...")

                for idx, row in df.iterrows():
                    if (idx + 1) % 20 == 0:
                        progress = 25 + (70 * (idx + 1) / total_rows)
                        self.after(0, self.set_progress, int(progress), f"Highlighting {idx + 1}/{total_rows}...")
                    
                    highlight_text = get_highlight_text(row, HIGHLIGHT_PREFERENCE, max_length=100)
                    if not highlight_text:
                        skipped += 1
                        continue

                    cat_raw = str(row.get(col_category, "")).strip() if col_category else ""
                    cat = norm_category(cat_raw)
                    color = pick_color(cat)
                    marker = row.get("marker", "")
                    note = f"{cat}: {marker}" if cat else f"Stance: {marker}"

                    # Try page-specific first
                    added = 0
                    if col_page:
                        try:
                            page_num = int(row.get(col_page))
                            if 1 <= page_num <= len(doc):
                                page = doc[page_num - 1]
                                added = highlight_text_on_page(page, highlight_text, color=color, note=note)
                        except (ValueError, TypeError):
                            pass
                    
                    # Document-wide if page-specific fails
                    if added == 0:
                        for p in doc:
                            added = highlight_text_on_page(p, highlight_text, color=color, note=note)
                            if added > 0:
                                break
                    
                    if added > 0:
                        total_highlights += added
                        if self.show_details.get():
                            self.after(0, self.log, f"✓ Row {idx + 1}: {added} highlight(s)")
                    else:
                        skipped += 1

                # Save highlighted PDF
                base = Path(pdf).stem
                out_path = os.path.join(outdir, f"{base}_highlighted.pdf")
                
                self.after(0, self.log, f"\nSaving PDF...")
                self.after(0, self.set_progress, 95, "Saving...")
                doc.save(out_path, incremental=False)
                doc.close()

                self.after(0, self.set_progress, 100, "Complete!")
                self.after(0, self.log, "\n" + "=" * 70)
                self.after(0, self.log, "HIGHLIGHTING COMPLETE!")
                self.after(0, self.log, f"Processed: {total_rows} rows")
                self.after(0, self.log, f"Highlights: {total_highlights}")
                self.after(0, self.log, f"Skipped: {skipped}")
                self.after(0, self.log, f"\nSaved: {out_path}")
                self.after(0, self.log, "=" * 70)
                
                if self.auto_open.get():
                    try:
                        if os.name == 'nt':
                            os.startfile(out_path)
                        else:
                            import subprocess
                            subprocess.call(["xdg-open", out_path])
                    except Exception:
                        pass
                
                self.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Complete",
                        f"Highlighting complete!\n\n"
                        f"Highlights: {total_highlights}\n"
                        f"File: {out_path}"
                    )
                )

            except Exception as e:
                self.after(0, self.set_progress, 0, "Failed")
                self.after(0, self.log, f"\n✗ ERROR: {str(e)}")
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
            
            finally:
                self.after(0, lambda: self.btn_filter.config(state="normal"))
                self.after(0, lambda: self.run_btn.config(state="normal"))

        threading.Thread(target=worker, daemon=True).start()


def main():
    """Entry point."""
    app = HighlighterApp()
    app.mainloop()


if __name__ == "__main__":
    main()
