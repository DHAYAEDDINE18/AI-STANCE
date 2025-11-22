"""
app.py - GUI for PDF chunking and optional stance analysis
"""
import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
from pathlib import Path

from src.analysis import (
    setup_logger, 
    process_pdf_to_sections, 
    process_pdf_with_stance_analysis,
    get_logger
)

KEY_FILE = "gemini_key.txt"


def read_api_key_from_file(path: str = KEY_FILE) -> str:
    """Read API key from file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""
    except Exception:
        return ""


def write_api_key_to_file(key: str, path: str = KEY_FILE):
    """Write API key to file with secure permissions."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(key.strip())
        try:
            os.chmod(path, 0o600)  # Read/write for owner only
        except Exception:
            pass  # Windows doesn't support chmod
    except Exception as e:
        raise RuntimeError(f"Failed to write API key file: {e}")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PDF Chunking + Stance Analysis")
        self.geometry("900x800")
        self.resizable(True, True)

        # Initialize from key file
        initial_key = read_api_key_from_file()

        # Variables
        self.pdf_path = tk.StringVar()
        self.out_dir = tk.StringVar(value="outputs")
        self.chunk_size = tk.IntVar(value=80000)
        self.overlap = tk.IntVar(value=1000)
        self.analyze_stance = tk.BooleanVar(value=False)
        self.model_name = tk.StringVar(value="gemini-3-pro-preview")
        self.api_key = tk.StringVar(value=initial_key)
        self.api_show = tk.BooleanVar(value=False)

        # Build UI
        self._build_ui()
        
        # Initialize logger
        setup_logger(self.out_dir.get())

    def _build_ui(self):
        """Build the GUI layout."""
        pad = {"padx": 10, "pady": 8}
        
        # Main frame
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="PDF Chunking + Stance Analysis",
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # Subtitle
        subtitle_label = ttk.Label(
            main_frame,
            text="Split PDF into chunks, optionally analyze for Hyland stance markers",
            font=("Arial", 9),
            foreground="gray"
        )
        subtitle_label.grid(row=1, column=0, columnspan=3, pady=(0, 15))

        # PDF file selection
        ttk.Label(main_frame, text="PDF file:", font=("Arial", 10)).grid(
            row=2, column=0, sticky="w", **pad
        )
        ttk.Entry(main_frame, textvariable=self.pdf_path, width=60).grid(
            row=2, column=1, sticky="we", **pad
        )
        ttk.Button(main_frame, text="Browse...", command=self.browse_pdf).grid(
            row=2, column=2, sticky="e", **pad
        )

        # Output folder
        ttk.Label(main_frame, text="Output folder:", font=("Arial", 10)).grid(
            row=3, column=0, sticky="w", **pad
        )
        ttk.Entry(main_frame, textvariable=self.out_dir, width=60).grid(
            row=3, column=1, sticky="we", **pad
        )
        ttk.Button(main_frame, text="Choose...", command=self.browse_outdir).grid(
            row=3, column=2, sticky="e", **pad
        )

        # Separator
        ttk.Separator(main_frame, orient="horizontal").grid(
            row=4, column=0, columnspan=3, sticky="we", pady=15
        )

        # Chunking parameters
        ttk.Label(main_frame, text="Chunk Settings:", font=("Arial", 10, "bold")).grid(
            row=5, column=0, sticky="w", **pad
        )

        # Chunk size and overlap
        chunk_frame = ttk.Frame(main_frame)
        chunk_frame.grid(row=6, column=0, columnspan=3, sticky="w", **pad)
        
        ttk.Label(chunk_frame, text="Chunk size (chars):").pack(side=tk.LEFT)
        ttk.Spinbox(
            chunk_frame,
            from_=10000,
            to=200000,
            increment=10000,
            textvariable=self.chunk_size,
            width=15
        ).pack(side=tk.LEFT, padx=(10, 20))
        
        ttk.Label(chunk_frame, text="Overlap (chars):").pack(side=tk.LEFT)
        ttk.Spinbox(
            chunk_frame,
            from_=0,
            to=5000,
            increment=100,
            textvariable=self.overlap,
            width=15
        ).pack(side=tk.LEFT, padx=(10, 0))

        # Info label
        info_label = ttk.Label(
            main_frame,
            text="💡 Default: 80,000 chars (~40-50 pages), 1,000 overlap",
            font=("Arial", 8),
            foreground="blue"
        )
        info_label.grid(row=7, column=0, columnspan=3, sticky="w", padx=(10, 0))

        # Separator
        ttk.Separator(main_frame, orient="horizontal").grid(
            row=8, column=0, columnspan=3, sticky="we", pady=15
        )

        # Stance analysis option
        ttk.Label(main_frame, text="Analysis Options:", font=("Arial", 10, "bold")).grid(
            row=9, column=0, sticky="w", **pad
        )

        stance_check = ttk.Checkbutton(
            main_frame,
            text="🔍 Analyze chunks for Hyland stance markers (requires API key)",
            variable=self.analyze_stance,
            command=self.toggle_stance_options
        )
        stance_check.grid(row=10, column=0, columnspan=3, sticky="w", **pad)

        # Model selection (initially disabled)
        model_frame = ttk.Frame(main_frame)
        model_frame.grid(row=11, column=0, columnspan=3, sticky="w", **pad)
        
        self.model_label = ttk.Label(model_frame, text="Gemini model:", state="disabled")
        self.model_label.pack(side=tk.LEFT)
        
        self.model_entry = ttk.Entry(
            model_frame,
            textvariable=self.model_name,
            width=30,
            state="disabled"
        )
        self.model_entry.pack(side=tk.LEFT, padx=(10, 0))

        # Separator
        ttk.Separator(main_frame, orient="horizontal").grid(
            row=12, column=0, columnspan=3, sticky="we", pady=15
        )

        # API Key section
        ttk.Label(
            main_frame,
            text="Gemini API Key:",
            font=("Arial", 10, "bold")
        ).grid(row=13, column=0, columnspan=3, sticky="w", **pad)
        
        key_subframe = ttk.Frame(main_frame)
        key_subframe.grid(row=14, column=0, columnspan=3, sticky="w", **pad)
        
        self.api_entry = ttk.Entry(key_subframe, textvariable=self.api_key, width=40, show="•")
        self.api_entry.pack(side=tk.LEFT)
        
        ttk.Checkbutton(
            key_subframe,
            text="Show",
            variable=self.api_show,
            command=self.toggle_show
        ).pack(side=tk.LEFT, padx=(10, 10))
        
        ttk.Button(
            key_subframe,
            text="💾 Save",
            command=self.save_key_to_file,
            width=8
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            key_subframe,
            text="🔄 Reload",
            command=self.reload_key_from_file,
            width=8
        ).pack(side=tk.LEFT)

        # Note about API key
        api_note = ttk.Label(
            main_frame,
            text="ℹ️ API key only required for stance analysis",
            font=("Arial", 8),
            foreground="gray"
        )
        api_note.grid(row=15, column=0, columnspan=3, sticky="w", padx=(10, 0))

        # Separator
        ttk.Separator(main_frame, orient="horizontal").grid(
            row=16, column=0, columnspan=3, sticky="we", pady=15
        )

        # Run button
        self.run_btn = ttk.Button(
            main_frame,
            text="▶ Process PDF",
            command=self.start_run,
            style="Accent.TButton"
        )
        self.run_btn.grid(row=17, column=0, columnspan=3, pady=10, sticky="we")

        # Progress section
        ttk.Label(main_frame, text="Progress:", font=("Arial", 10)).grid(
            row=18, column=0, sticky="w", **pad
        )
        self.progress_label = ttk.Label(main_frame, text="Ready", foreground="blue")
        self.progress_label.grid(row=18, column=1, sticky="w")
        
        self.pb = ttk.Progressbar(
            main_frame,
            orient="horizontal",
            mode="determinate",
            length=600,
            maximum=100
        )
        self.pb.grid(row=19, column=0, columnspan=3, sticky="we", **pad)

        # Status log
        ttk.Label(main_frame, text="Status Log:", font=("Arial", 10)).grid(
            row=20, column=0, sticky="nw", **pad
        )
        self.status = scrolledtext.ScrolledText(
            main_frame,
            width=80,
            height=10,
            wrap="word",
            font=("Consolas", 9)
        )
        self.status.grid(row=21, column=0, columnspan=3, sticky="nswe", **pad)

        # Configure grid weights for resizing
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(21, weight=1)

    def toggle_stance_options(self):
        """Enable/disable stance analysis options."""
        state = "normal" if self.analyze_stance.get() else "disabled"
        self.model_label.config(state=state)
        self.model_entry.config(state=state)

    def toggle_show(self):
        """Toggle API key visibility."""
        self.api_entry.config(show="" if self.api_show.get() else "•")

    def save_key_to_file(self):
        """Save API key to file."""
        key = self.api_key.get().strip()
        if not key:
            messagebox.showerror("No key", "Enter an API key first.")
            return
        try:
            write_api_key_to_file(key)
            messagebox.showinfo("Saved", f"✓ API key saved to {KEY_FILE}")
            self.log_status(f"✓ API key saved to {KEY_FILE}")
        except Exception as e:
            messagebox.showerror("Error saving key", str(e))
            self.log_status(f"✗ Failed to save key: {e}")

    def reload_key_from_file(self):
        """Reload API key from file."""
        key = read_api_key_from_file()
        if key:
            self.api_key.set(key)
            messagebox.showinfo("Reloaded", f"✓ API key loaded from {KEY_FILE}")
            self.log_status(f"✓ API key loaded from {KEY_FILE}")
        else:
            messagebox.showwarning("No key found", f"No API key found in {KEY_FILE}")
            self.log_status(f"⚠ No key found in {KEY_FILE}")

    def browse_pdf(self):
        """Browse for PDF file."""
        filename = filedialog.askopenfilename(
            title="Select PDF document",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if filename:
            self.pdf_path.set(filename)
            self.log_status(f"Selected PDF: {Path(filename).name}")

    def browse_outdir(self):
        """Browse for output directory."""
        directory = filedialog.askdirectory(title="Select output folder")
        if directory:
            self.out_dir.set(directory)
            self.log_status(f"Output folder: {directory}")

    def set_progress(self, val: int, label: str = ""):
        """Update progress bar and label."""
        self.pb["value"] = max(0, min(100, val))
        if label:
            self.progress_label.config(text=label)
        self.update_idletasks()

    def log_status(self, msg: str):
        """Add message to status log (sanitize API key)."""
        key = self.api_key.get().strip()
        safe_msg = msg.replace(key, "****") if key else msg
        self.status.insert("end", safe_msg + "\n")
        self.status.see("end")
        self.update_idletasks()

    def start_run(self):
        """Start the processing pipeline."""
        pdf = self.pdf_path.get().strip()
        outdir = self.out_dir.get().strip() or "outputs"
        chunk_size = self.chunk_size.get()
        overlap = self.overlap.get()
        do_stance = self.analyze_stance.get()
        model = self.model_name.get().strip() if do_stance else None

        # Validation
        if not pdf or not os.path.isfile(pdf):
            messagebox.showerror("Missing PDF", "Please select a valid PDF file.")
            return

        if chunk_size < 1000:
            messagebox.showerror("Invalid chunk size", "Chunk size must be at least 1,000 characters.")
            return

        if overlap >= chunk_size:
            messagebox.showerror("Invalid overlap", "Overlap must be smaller than chunk size.")
            return

        # API key validation for stance analysis
        key = read_api_key_from_file()
        if do_stance and not key:
            messagebox.showerror(
                "Missing API key",
                f"Stance analysis requires an API key.\nPlease save your Gemini API key first."
            )
            return

        if key:
            os.environ["GEMINI_API_KEY"] = key

        # Reinitialize logger
        setup_logger(outdir)

        # Disable UI during processing
        self.run_btn.config(state="disabled")
        self.set_progress(0, "Starting...")
        self.status.delete(1.0, tk.END)
        self.log_status("=" * 70)
        self.log_status(f"PDF: {Path(pdf).name}")
        self.log_status(f"Output: {outdir}")
        self.log_status(f"Chunk size: {chunk_size:,} chars, Overlap: {overlap:,} chars")
        self.log_status(f"Stance analysis: {'✓ Enabled' if do_stance else '✗ Disabled'}")
        if do_stance:
            self.log_status(f"Model: {model}")
        self.log_status("=" * 70)

        def worker():
            """Background worker thread."""
            try:
                if do_stance:
                    # Full pipeline with stance analysis
                    self.after(0, self.set_progress, 20, "Extracting PDF...")
                    self.after(0, self.log_status, "▶ Step 1: Extracting PDF text...")

                    self.after(0, self.set_progress, 40, "Chunking text...")
                    self.after(0, self.log_status, "▶ Step 2: Creating chunks...")
                    
                    self.after(0, self.set_progress, 60, "Analyzing stance...")
                    self.after(0, self.log_status, "▶ Step 3: Analyzing stance markers...")
                    
                    result = process_pdf_with_stance_analysis(
                        pdf_path=pdf,
                        output_base=outdir,
                        chunk_size=chunk_size,
                        overlap=overlap,
                        model_name=model
                    )
                    
                    # Stance-specific results
                    self.after(0, self.log_status, f"\n✓ Stance markers found: {result.get('total_markers', 0):,}")
                    self.after(0, self.log_status, f"✓ Stance JSON: {result.get('stance_json', 'N/A')}")
                    self.after(0, self.log_status, f"✓ Stance CSV: {result.get('stance_csv', 'N/A')}")
                    
                else:
                    # Chunking only
                    self.after(0, self.set_progress, 30, "Extracting PDF...")
                    self.after(0, self.log_status, "▶ Step 1: Extracting PDF text...")

                    self.after(0, self.set_progress, 70, "Chunking text...")
                    self.after(0, self.log_status, "▶ Step 2: Creating chunks...")
                    
                    result = process_pdf_to_sections(
                        pdf_path=pdf,
                        output_base=outdir,
                        model_name=None,
                        chunk_size=chunk_size,
                        overlap=overlap
                    )

                # Common results
                self.after(0, self.set_progress, 100, "Complete!")
                
                self.after(0, self.log_status, "\n" + "=" * 70)
                self.after(0, self.log_status, "✓ PROCESSING COMPLETE!")
                self.after(0, self.log_status, f"✓ Text file: {result['text_file']}")
                self.after(0, self.log_status, f"✓ Chunks: {result.get('total_chunks', len(result.get('section_files', [])))}")
                self.after(0, self.log_status, f"✓ Directory: {result['section_directory']}")
                self.after(0, self.log_status, "=" * 70)

                # Success message
                num_chunks = result.get('total_chunks', len(result.get('section_files', [])))
                msg = f"✓ Processing complete!\n\nChunks: {num_chunks}\nLocation: {result['section_directory']}"
                
                if do_stance:
                    msg += f"\n\nStance markers found: {result.get('total_markers', 0):,}"
                    msg += f"\nResults: stance_results.csv"
                
                self.after(0, lambda: messagebox.showinfo("Success", msg))

            except Exception as e:
                logger = get_logger()
                logger.error(f"Pipeline failed: {e}", exc_info=True)
                
                self.after(0, self.log_status, f"\n✗ ERROR: {str(e)}")
                self.after(0, self.set_progress, 0, "Failed")
                self.after(0, lambda: messagebox.showerror("Error", f"Processing failed:\n\n{str(e)}"))

            finally:
                self.after(0, lambda: self.run_btn.config(state="normal"))

        # Start worker thread
        threading.Thread(target=worker, daemon=True).start()


def main():
    """Entry point."""
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
