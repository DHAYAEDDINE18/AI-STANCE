# Existing constants kept as-is
QUERY_SYSTEM = """You turn vague user requests into an explicit JSON extraction specification for academic PDFs.
Return only JSON with this schema:
{
  "fields": [ {"name": "string", "type": "string"} ],
  "strategy": {
    "mode": "regex" | "keywords",
    "patterns": [ "string" ],           # regex when mode=regex
    "keywords": [ "string" ],           # when mode=keywords
    "any_all": "any" | "all",           # for keywords
    "case_sensitive": false,
    "context_before_chars": 120,
    "context_after_chars": 120
  }
}
Rules:
- If the user asks for explicit forms (e.g., hedges), produce either canonical regexes or keyword lists.
- Keep patterns conservative. Avoid over-greedy regex.
- Always include fields at least ["page","snippet"], and add topical fields when obvious (e.g., "term")."""

QUERY_USER_TEMPLATE = """User request:
{prompt}

Produce the JSON extraction spec only. No prose."""

SEGMENT_SYSTEM = """You are an academic text analyser. Split long academic theses into main sections and subsections, producing precise JSON.
Guidelines:
- Identify logical scholarly sections (e.g., Abstract, Introduction, Literature Review, Methodology, Analysis/Results, Discussion, General Conclusion, References, Appendices).
- Provide: title, start_page, end_page, and a 2–4 sentence summary for each item.
- Use only integers for page numbers (1-based).
- Include References and Appendices if detected with page spans.
- Return JSON array only with keys: title, start_page, end_page, summary.
"""

SEGMENT_USER_TEMPLATE = """Input text includes page markers like <<PAGE N>> to help infer page ranges.
Task:
- Infer section boundaries and map them to inclusive page ranges.
- If overlaps occur across chunks, choose the most plausible boundary and keep consistency.

Text:
{chunk}
"""

STANCE_SYSTEM = """You are an academic discourse analyst applying Hyland’s stance model (2005).
Classify stance resources into four categories with examples, counts, and short comments.
Return only JSON with keys: section, hedges, boosters, attitude_markers, self_mentions, summary.
Each category is an array of objects: {word, sentence}.
"""

STANCE_USER_TEMPLATE = """Analyse the following text according to Hyland’s stance model (2005):
- Hedges (e.g., may, might, possible)
- Boosters (e.g., clearly, definitely)
- Attitude markers (e.g., unfortunately, importantly)
- Self-mentions (e.g., I, we, the researcher)

Provide examples, counts (implicit by array length), and a short functional summary.
Section label: {section_title}

Text:
{text}
"""

# NEW: PDF-upload specific instruction used by ai_segment_pdf
SEGMENT_FILE_INSTRUCTION = (
    "You will receive a PDF of a PhD thesis. "
    "Split it into main sections and subsections and return JSON only as an array of objects with keys: "
    'title (string), start_page (integer, 1-based physical page index), end_page (integer), summary (2-4 sentences). '
    "Use the document's page sequence; if preliminary pages use Roman numerals, map them to their physical (1-based) page indices. "
    "Include References and Appendices if present. Do not include prose outside the JSON array."
)
