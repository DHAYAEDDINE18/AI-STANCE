"""
app.py - GUI for PDF chunking and optional stance analysis
"""
import os
import threading
import webbrowser
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
from pathlib import Path

from src.web_server import start_server
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
        self.pdf_paths = []
        self.out_dir = tk.StringVar(value="outputs")
        self.chunk_size = tk.IntVar(value=80000)
        self.overlap = tk.IntVar(value=1000)
        self.analyze_stance = tk.BooleanVar(value=False)
        self.do_process_all = tk.BooleanVar(value=False)
        self.model_name = tk.StringVar(value="gemini-pro") # Changed default model
        self.api_key = tk.StringVar(value=initial_key)
        self.api_show = tk.BooleanVar(value=False)
        self.report_path = None

        # Build UI
        self._build_ui()
        
        # Initialize logger
        setup_logger(self.out_dir.get())

    def _build_ui(self):
        """Build the GUI layout."""
        pad = {"padx": 10, "pady": 8}
        
        # Create a scrollable main frame
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        main_frame = ttk.Frame(canvas, padding=(10, 10))

        main_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=main_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

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

        # PDF file selection frame
        pdf_frame = ttk.LabelFrame(main_frame, text="PDF Files", padding=10)
        pdf_frame.grid(row=2, column=0, columnspan=3, sticky="ew", **pad)
        pdf_frame.columnconfigure(0, weight=1)

        self.pdf_listbox = tk.Listbox(pdf_frame, height=5)
        self.pdf_listbox.grid(row=0, column=0, columnspan=3, sticky="nsew", pady=5)
        pdf_frame.rowconfigure(0, weight=1)


        pdf_button_frame = ttk.Frame(pdf_frame)
        pdf_button_frame.grid(row=1, column=0, columnspan=3, sticky="w")

        ttk.Button(pdf_button_frame, text="Add PDFs...", command=self.add_pdfs).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(pdf_button_frame, text="Remove Selected", command=self.remove_selected_pdfs).pack(side=tk.LEFT, padx=5)
        ttk.Button(pdf_button_frame, text="Clear All", command=self.clear_all_pdfs).pack(side=tk.LEFT, padx=5)

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
        model_frame.grid(row=12, column=0, columnspan=3, sticky="w", **pad)
        
        self.model_label = ttk.Label(model_frame, text="Gemini model:", state="disabled")
        self.model_label.pack(side=tk.LEFT)
        
        self.model_entry = ttk.Combobox(
            model_frame,
            textvariable=self.model_name,
            width=28, # Adjusted width for Combobox
            state="disabled",
            values=['gemini-pro', 'gemini-ultra', 'gemini-3-pro-preview', 'gemini-3-flash-preview', 'gemini-2.5-flash', 'gemini-2.5-flash-lite', 'gemini-2.0-flash'] # Updated model list
        )
        self.model_entry.pack(side=tk.LEFT, padx=(10, 0))

        # Separator
        ttk.Separator(main_frame, orient="horizontal").grid(
            row=13, column=0, columnspan=3, sticky="we", pady=15
        )

        # API Key section
        ttk.Label(
            main_frame,
            text="Gemini API Key:",
            font=("Arial", 10, "bold")
        ).grid(row=14, column=0, columnspan=3, sticky="w", **pad)
        
        key_subframe = ttk.Frame(main_frame)
        key_subframe.grid(row=15, column=0, columnspan=3, sticky="w", **pad)
        
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
        api_note.grid(row=16, column=0, columnspan=3, sticky="w", padx=(10, 0))

        # Separator
        ttk.Separator(main_frame, orient="horizontal").grid(
            row=17, column=0, columnspan=3, sticky="we", pady=15
        )

        # Run and View buttons
        action_frame = ttk.Frame(main_frame)
        action_frame.grid(row=18, column=0, columnspan=3, pady=10, sticky="we")
        action_frame.columnconfigure(0, weight=1)
        action_frame.columnconfigure(1, weight=1)

        self.run_btn = ttk.Button(
            action_frame,
            text="▶ Process PDF",
            command=self.start_run,
            style="Accent.TButton"
        )
        self.run_btn.grid(row=0, column=0, sticky="we", padx=(0, 5))

        self.view_report_btn = ttk.Button(
            action_frame,
            text="📄 View Report",
            command=self.view_report,
            state="disabled"
        )
        self.view_report_btn.grid(row=0, column=1, sticky="we", padx=(5, 0))

        # Progress section
        ttk.Label(main_frame, text="Progress:", font=("Arial", 10)).grid(
            row=19, column=0, sticky="w", **pad
        )
        self.progress_label = ttk.Label(main_frame, text="Ready", foreground="blue")
        self.progress_label.grid(row=19, column=1, sticky="w")
        
        self.pb = ttk.Progressbar(
            main_frame,
            orient="horizontal",
            mode="determinate",
            length=600,
            maximum=100
        )
        self.pb.grid(row=20, column=0, columnspan=3, sticky="we", **pad)

        # Status log
        ttk.Label(main_frame, text="Status Log:", font=("Arial", 10)).grid(
            row=21, column=0, sticky="nw", **pad
        )
        self.status = scrolledtext.ScrolledText(
            main_frame,
            width=80,
            height=10,
            wrap="word",
            font=("Consolas", 9)
        )
        self.status.grid(row=22, column=0, columnspan=3, sticky="nswe", **pad)

        # Configure grid weights for resizing
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(22, weight=1)

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

    def add_pdfs(self):
        """Browse for multiple PDF files and add them to the list."""
        filenames = filedialog.askopenfilenames(
            title="Select PDF documents",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if filenames:
            for fn in filenames:
                if fn not in self.pdf_paths:
                    self.pdf_paths.append(fn)
                    self.pdf_listbox.insert(tk.END, Path(fn).name)
            self.log_status(f"Added {len(filenames)} PDF(s). Total: {len(self.pdf_paths)}")

    def remove_selected_pdfs(self):
        """Remove selected PDFs from the list."""
        selected_indices = self.pdf_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No selection", "Please select PDFs to remove.")
            return
        
        for i in reversed(selected_indices):
            del self.pdf_paths[i]
            self.pdf_listbox.delete(i)
        self.log_status(f"Removed {len(selected_indices)} PDF(s).")

    def clear_all_pdfs(self):
        """Clear all PDFs from the list."""
        if self.pdf_paths and messagebox.askyesno("Confirm", "Are you sure you want to remove all PDFs?"):
            self.pdf_paths.clear()
            self.pdf_listbox.delete(0, tk.END)
            self.log_status("Cleared all PDFs.")

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
        """Start the processing pipeline for all selected PDFs."""
        pdfs = self.pdf_paths
        outdir = self.out_dir.get().strip() or "outputs"
        chunk_size = self.chunk_size.get()
        overlap = self.overlap.get()
        do_stance = self.analyze_stance.get()
        model = self.model_name.get().strip() if do_stance else None

        # Validation
        if not pdfs:
            messagebox.showerror("Missing PDF", "Please add at least one PDF file to the list.")
            return

        if chunk_size < 1000:
            messagebox.showerror("Invalid chunk size", "Chunk size must be at least 1,000 characters.")
            return

        if overlap >= chunk_size:
            messagebox.showerror("Invalid overlap", "Overlap must be smaller than chunk size.")
            return

        key = read_api_key_from_file()
        if do_stance and not key:
            messagebox.showerror(
                "Missing API key",
                "Stance analysis requires an API key.\nPlease save your Gemini API key first."
            )
            return

        if key:
            os.environ["GEMINI_API_KEY"] = key

        setup_logger(outdir)

        self.run_btn.config(state="disabled")
        self.set_progress(0, "Starting...")
        self.status.delete(1.0, tk.END)
        self.log_status("=" * 80)
        self.log_status(f"Starting batch processing for {len(pdfs)} PDF(s)...")
        self.log_status(f"Output: {outdir}")
        self.log_status(f"Chunk size: {chunk_size:,} chars, Overlap: {overlap:,} chars")
        self.log_status(f"Stance analysis: {'✓ Enabled' if do_stance else '✗ Disabled'}")
        if do_stance:
            self.log_status(f"  - Model: {model}")

        self.log_status("=" * 80)

        def worker():
            """Background worker thread to process multiple PDFs."""
            total_files = len(pdfs)
            completed_files = 0
            total_errors = 0
            summary_messages = []

            for i, pdf_path in enumerate(pdfs):
                file_progress_start = (i / total_files) * 100
                file_progress_end = ((i + 1) / total_files) * 100
                
                try:
                    self.after(0, self.set_progress, int(file_progress_start), f"Starting {Path(pdf_path).name} ({i+1}/{total_files})")
                    self.after(0, self.log_status, f"\n--- Processing file {i+1}/{total_files}: {Path(pdf_path).name} ---")

                    if do_stance:
                        self.after(0, self.log_status, "▶ Step 1: Extracting, chunking, and analyzing stance...")
                        result = process_pdf_with_stance_analysis(
                            pdf_path=pdf_path,
                            output_base=outdir,
                            chunk_size=chunk_size,
                            overlap=overlap,
                                            model_name=model                        )
                        self.after(0, self.log_status, f"  ✓ Stance markers found: {result.get('total_markers', 0):,}")
                    else:
                        self.after(0, self.log_status, "▶ Step 1: Extracting and chunking text...")
                        result = process_pdf_to_sections(
                            pdf_path=pdf_path,
                            output_base=outdir,
                            model_name=None,
                            chunk_size=chunk_size,
                            overlap=overlap
                        )

                    self.after(0, self.log_status, f"  ✓ Text file: {result['text_file']}")
                    self.after(0, self.log_status, f"  ✓ Chunks created: {result.get('total_chunks', len(result.get('section_files', [])))}")
                    self.after(0, self.log_status, f"  ✓ Output directory: {result['section_directory']}")
                    
                    num_chunks = result.get('total_chunks', len(result.get('section_files', [])))
                    msg = f"✓ {Path(pdf_path).name}: {num_chunks} chunks created."
                    if do_stance:
                        msg += f" Found {result.get('total_markers', 0):,} stance markers."
                        if result.get("html_report"):
                            self.report_path = result["html_report"]
                            self.after(0, lambda: self.view_report_btn.config(state="normal"))
                    summary_messages.append(msg)
                    completed_files += 1

                except Exception as e:
                    total_errors += 1
                    logger = get_logger()
                    logger.error(f"Pipeline failed for {pdf_path}: {e}", exc_info=True)
                    self.after(0, self.log_status, f"\n✗ ERROR processing {Path(pdf_path).name}: {e}")
                    summary_messages.append(f"✗ {Path(pdf_path).name}: FAILED - {e}")
                
                self.after(0, self.set_progress, int(file_progress_end))

            # Final summary
            self.after(0, self.set_progress, 100, "Batch complete!")
            self.after(0, self.log_status, "\n" + "=" * 80)
            self.after(0, self.log_status, "BATCH PROCESSING COMPLETE!")
            self.after(0, self.log_status, f"  Successfully processed: {completed_files}/{total_files}")
            self.after(0, self.log_status, f"  Errors: {total_errors}")
            self.after(0, self.log_status, "=" * 80)
            
            final_summary = f"Batch processing finished.\n\nSuccess: {completed_files}/{total_files}\nErrors: {total_errors}\n\n"
            final_summary += "Summary:\n" + "\n".join(summary_messages)
            
            self.after(0, lambda: messagebox.showinfo("Batch Complete", final_summary))

            # Re-enable the button
            self.after(0, lambda: self.run_btn.config(state="normal"))

        # Start the worker thread
        threading.Thread(target=worker, daemon=True).start()

    def view_report(self):
        """Start a web server and open the latest HTML report."""
        if not self.report_path or not os.path.exists(self.report_path):
            messagebox.showerror("No Report", "No report found. Please run the analysis first.")
            return

        port = 8123
        report_dir = os.path.dirname(self.report_path)
        report_filename = os.path.basename(self.report_path)
        
        # Run server in a separate thread
        server_thread = threading.Thread(
            target=start_server,
            args=(report_dir, port),
            daemon=True
        )
        server_thread.start()
        
        url = f"http://localhost:{port}/{report_filename}"
        self.log_status(f"Opening report at: {url}")
        
        # Give server a moment to start
        self.after(1000, lambda: webbrowser.open_new_tab(url))


def main():
    """Entry point."""
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()