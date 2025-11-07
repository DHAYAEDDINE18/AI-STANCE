import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from src.pdf_utils import extract_text_pipeline
from src.analysis import ai_segment_text, slice_pages_text, ai_analyse_stance
from src.io_utils import save_json, save_csv_stance


def run_pipeline(
    pdf_path: str,
    out_dir: str = "outputs",
    model_name: str | None = None,
    analyze_only_conclusion: bool = False,
    progress_cb=None,
    status_cb=None,
):
    """
    Workflow:
      1) Extract/clean PDF text locally and add <<PAGE N>> anchors.
      2) Segment sections via AI on anchored text.
      3) Run stance analysis per section.
    """
    try:
        os.makedirs(out_dir, exist_ok=True)

        # Stage 0: extract and clean
        if status_cb:
            status_cb("Extracting and cleaning PDF text...")
        if progress_cb:
            progress_cb(10)
        _, _, combined = extract_text_pipeline(pdf_path)

        # Stage 1: segment with anchored text
        if status_cb:
            status_cb("Segmenting sections with AI (local text with anchors)...")
        if progress_cb:
            progress_cb(40)
        segmentation = ai_segment_text(combined, model_name=model_name)

        seg_path = os.path.join(out_dir, "segmentation.json")
        save_json(segmentation, seg_path)

        # Stage 2: stance analysis
        if status_cb:
            status_cb("Analyzing stance markers per section...")
        stance_results = []
        total = max(len(segmentation), 1)
        for idx, seg in enumerate(segmentation, start=1):
            title = seg["title"]
            if analyze_only_conclusion and "conclusion" not in title.lower():
                if progress_cb:
                    progress_cb(int(60 + 35 * (idx / total)))
                continue
            text = slice_pages_text(combined, seg["start_page"], seg["end_page"])
            result = ai_analyse_stance(title, text, model_name=model_name)
            stance_results.append(result)
            if progress_cb:
                progress_cb(int(60 + 35 * (idx / total)))

        stance_json_path = os.path.join(out_dir, "stance.json")
        stance_csv_path = os.path.join(out_dir, "stance.csv")
        save_json(stance_results, stance_json_path)
        save_csv_stance(stance_results, stance_csv_path)

        if progress_cb:
            progress_cb(100)
        if status_cb:
            status_cb(f"Done.\nSaved:\n{seg_path}\n{stance_json_path}\n{stance_csv_path}")
    except Exception as e:
        if status_cb:
            status_cb(f"Error: {e}")
        raise


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Thesis Sectioning + Hyland Stance Analysis")
        self.geometry("760x520")
        self.resizable(False, False)

        self.pdf_path = tk.StringVar()
        self.out_dir = tk.StringVar(value="outputs")
        self.model_name = tk.StringVar(value=os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"))
        self.conclusion_only = tk.BooleanVar(value=False)

        self.api_key = tk.StringVar(value=os.getenv("GEMINI_API_KEY", ""))
        self.api_show = tk.BooleanVar(value=False)
        self.save_env = tk.BooleanVar(value=False)

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 8}
        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True, **pad)

        ttk.Label(frm, text="PDF file:").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.pdf_path, width=60).grid(row=0, column=1, sticky="we")
        ttk.Button(frm, text="Browse...", command=self.browse_pdf).grid(row=0, column=2, sticky="e")

        ttk.Label(frm, text="Output folder:").grid(row=1, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.out_dir, width=60).grid(row=1, column=1, sticky="we")
        ttk.Button(frm, text="Choose...", command=self.browse_outdir).grid(row=1, column=2, sticky="e")

        ttk.Label(frm, text="Gemini model:").grid(row=2, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.model_name, width=40).grid(row=2, column=1, sticky="w")

        ttk.Checkbutton(
            frm, text="Analyze only General Conclusion", variable=self.conclusion_only
        ).grid(row=3, column=1, sticky="w")

        ttk.Label(frm, text="Gemini API key:").grid(row=4, column=0, sticky="w")
        self.api_entry = ttk.Entry(frm, textvariable=self.api_key, width=40, show="•")
        self.api_entry.grid(row=4, column=1, sticky="w")
        ttk.Checkbutton(frm, text="Show", variable=self.api_show, command=self.toggle_show).grid(
            row=4, column=2, sticky="w"
        )

        ttk.Checkbutton(frm, text="Save to .env", variable=self.save_env).grid(
            row=5, column=1, sticky="w"
        )
        ttk.Button(frm, text="Save key", command=self.save_key_to_env).grid(
            row=5, column=2, sticky="w"
        )

        self.run_btn = ttk.Button(frm, text="Run Analysis", command=self.start_run)
        self.run_btn.grid(row=6, column=1, sticky="w", pady=10)

        ttk.Label(frm, text="Progress:").grid(row=7, column=0, sticky="w")
        self.pb = ttk.Progressbar(frm, orient="horizontal", mode="determinate", length=560, maximum=100)
        self.pb.grid(row=7, column=1, columnspan=2, sticky="we")

        ttk.Label(frm, text="Status:").grid(row=8, column=0, sticky="nw")
        self.status = tk.Text(frm, width=80, height=12, wrap="word")
        self.status.grid(row=8, column=1, columnspan=2, sticky="we")

        for i in range(3):
            frm.grid_columnconfigure(i, weight=1)

    def toggle_show(self):
        self.api_entry.config(show="" if self.api_show.get() else "•")

    def save_key_to_env(self):
        key = self.api_key.get().strip()
        if not key:
            messagebox.showerror("No key", "Enter a Gemini API key first.")
            return
        try:
            env_path = os.path.join(os.getcwd(), ".env")
            existing = {}
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.rstrip("\n")
                        if "=" in line and not line.strip().startswith("#"):
                            k, v = line.split("=", 1)
                            existing[k] = v
            existing["GEMINI_API_KEY"] = key
            model = self.model_name.get().strip()
            if model:
                existing["GEMINI_MODEL"] = model
            with open(env_path, "w", encoding="utf-8") as f:
                for k, v in existing.items():
                    f.write(f"{k}={v}\n")
            messagebox.showinfo("Saved", f"Saved to {env_path}")
        except Exception as e:
            messagebox.showerror("Error saving key", str(e))

    def browse_pdf(self):
        filename = filedialog.askopenfilename(
            title="Select PhD thesis PDF", filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if filename:
            self.pdf_path.set(filename)

    def browse_outdir(self):
        directory = filedialog.askdirectory(title="Select output folder")
        if directory:
            self.out_dir.set(directory)

    def set_progress(self, val: int):
        self.pb["value"] = max(0, min(100, val))
        self.update_idletasks()

    def set_status(self, msg: str):
        safe_msg = msg.replace(self.api_key.get().strip(), "****") if self.api_key.get() else msg
        self.status.insert("end", safe_msg + "\n")
        self.status.see("end")
        self.update_idletasks()

    def start_run(self):
        pdf = self.pdf_path.get().strip()
        outdir = self.out_dir.get().strip() or "outputs"
        model = self.model_name.get().strip() or None
        conc_only = self.conclusion_only.get()
        key = self.api_key.get().strip()

        if not pdf or not os.path.isfile(pdf):
            messagebox.showerror("Missing PDF", "Please select a valid PDF file.")
            return

        if key:
            os.environ["GEMINI_API_KEY"] = key

        if not os.getenv("GEMINI_API_KEY"):
            proceed = messagebox.askyesno("Missing GEMINI_API_KEY", "GEMINI_API_KEY is not set. Continue?")
            if not proceed:
                return

        self.run_btn.config(state="disabled")
        self.set_progress(0)
        self.set_status("Starting...")

        def worker():
            try:
                run_pipeline(
                    pdf_path=pdf,
                    out_dir=outdir,
                    model_name=model,
                    analyze_only_conclusion=conc_only,
                    progress_cb=lambda p: self.after(0, self.set_progress, p),
                    status_cb=lambda s: self.after(0, self.set_status, s),
                )
                self.after(0, lambda: messagebox.showinfo("Completed", "Analysis completed. Files saved."))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.after(0, lambda: self.run_btn.config(state="normal"))

        threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    app = App()
    app.mainloop()
