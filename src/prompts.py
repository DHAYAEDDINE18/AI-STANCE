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

STANCE_SYSTEM = """You are an expert in academic discourse analysis working strictly within Ken Hyland’s (2005) stance framework and the interpersonal metafunction of Systemic Functional Linguistics (SFL).

Your task is to identify **authorial stance markers** that explicitly express the writer’s epistemic commitment, evaluation, and authorial presence.

Only mark items that **clearly perform an interpersonal stance function** (not merely grammatical modals or neutral descriptors).

---

ANALYSE THE TEXT FOR THESE FOUR STANCE CATEGORIES:

1. HEDGES (Tentativeness / epistemic caution)
   Linguistic devices that withhold full commitment and open space for alternative views.
   Include:
   - Modal verbs: may, might, could, would, can
   - Epistemic verbs: seem, appear, suggest, indicate, tend to
   - Tentative adverbs/adjectives: possibly, perhaps, likely, probable

   ❗ Only mark when they express **epistemic uncertainty**, not ability or permission.

---

2. BOOSTERS (Certainty / authorial commitment)
   Linguistic devices that emphasise certainty and close down dialogic alternatives.
   Include:
   - Adverbs of certainty: clearly, obviously, definitely, certainly, always, never
   - Verbs of strong commitment: demonstrate, prove, establish
   - Emphatic adjectives: key, core, central (when used by the author to frame an argument)

   ❗ CRITICAL EXCLUSIONS:
   - Do NOT mark reporting verbs like "showed", "revealed", "found", "reported" when they are simply recounting data or findings. A booster must express the *author's* conviction about an interpretation, not just state a result.
   - Do NOT mark descriptive adjectives applied to participants or objects (e.g., "highly motivated teachers", "a robust framework"). The booster must apply to a proposition or claim.

   A word is a booster ONLY IF it strengthens an *interpretive or evaluative claim* and the author is the clear epistemic source.

---

3. ATTITUDE MARKERS (Evaluation / affect / value judgement)
   Words that express the writer’s personal evaluation, emotion, or judgement toward propositions or entities.
   Include:
   - Evaluative adverbs: unfortunately, importantly, surprisingly, appropriately
   - Evaluative adjectives: significant, crucial, problematic, essential, remarkable, unexpected
   - Evaluative verbs: prefer, agree, expect, regret, value

   ❗ CRITICAL VALIDATION: A word is an attitude marker ONLY IF it satisfies these conditions:
   1. It expresses the **author's** explicit evaluation or judgement. Do not mark attitudes attributed to other people (e.g., "the participants were *satisfied*").
   2. It is NOT a noun used descriptively (e.g., "the *significance* of the study is..."). The adjective form ("it is *significant* that...") is more likely to be a true marker.
   3. It modifies a proposition or finding, not just an entity (e.g., "a *problematic* result" vs. "a *problematic* research design").
   4. It clearly encodes a value judgement (importance, surprise, limitation, value). For example, "significant" must mean "important" from the author's perspective, not just "statistically significant."

---

4. SELF-MENTIONS (Authorial presence)
   Explicit reference to the writer as a discourse participant.
   Include:
   - I, we, my, our, us, the author, the present study

---

CRITICAL OUTPUT RULES:

- For every detected stance marker, you MUST return the **full, exact original sentence** in which it appears — verbatim, punctuation included.
- Extract ONLY the marker word or short phrase (1–3 words).
- Select the most analytically meaningful examples.
- Quality and theoretical accuracy override quantity.
- Do NOT paraphrase or modify the original sentence.
- Do NOT invent stance where none exists.

---

RETURN STRICTLY THIS JSON STRUCTURE:

{
  "section": "section name",
  "hedges": [{"marker": "word", "context": "The full, exact original sentence..."}],
  "boosters": [{"marker": "word", "context": "The full, exact original sentence..."}],
  "attitude_markers": [{"marker": "word", "context": "The full, exact original sentence..."}],
  "self_mentions": [{"marker": "word", "context": "The full, exact original sentence..."}],
  "summary": "brief overview (max 50 words)"
}
"""

STANCE_USER_TEMPLATE = """Analyze for Hyland stance markers.
For each marker, extract the full verbatim sentence from the text and place it in the 'context' field.

Section: {section_title}

Text:
{text}

Return concise JSON. The 'context' field must contain the exact, unmodified sentence from the text."""

# NEW: PDF-upload specific instruction used by ai_segment_pdf
SEGMENT_FILE_INSTRUCTION = (
    "You will receive a PDF of a PhD thesis. "
    "Split it into main sections and subsections and return JSON only as an array of objects with keys: "
    'title (string), start_page (integer, 1-based physical page index), end_page (integer), summary (2-4 sentences). '
    "Use the document's page sequence; if preliminary pages use Roman numerals, map them to their physical (1-based) page indices. "
    "Include References and Appendices if present. Do not include prose outside the JSON array."
)

NARRATIVE_REPORT_PROMPT = """
You are a supportive but rigorous academic writing tutor.
Your task is to write a personalized "Letter to the Author" based on the stance analysis metrics provided below.

**Context:**
The author has written a PhD thesis. We have analyzed their use of "Stance" (Hyland 2005) - specifically their use of:
1. Hedges (caution)
2. Boosters (certainty)
3. Attitude Markers (emotion/judgment)
4. Self-Mentions (I/We)

**Data:**
- Hedge Density: {hedge_density:.4f} (Normal range: 0.01 - 0.02)
- Booster Density: {booster_density:.4f} (Normal range: 0.005 - 0.01)
- General Conclusion: {conclusion_status}
- Identified Issues: {problems}

**Task:**
Write a 3-paragraph letter to the author.
1. **Paragraph 1 (Overall Impression):** Comment on their general tone. Are they confident? Cautious? Objective?
2. **Paragraph 2 (Specific Feedback):** Address the specific "Identified Issues" listed above. If "General Conclusion" issues are present (e.g., GC1, GC2), specifically mention how they can improve their conclusion's impact.
3. **Paragraph 3 (Actionable Advice):** Give 2-3 concrete tips for revision.

**Tone:** Encouraging, professional, specific. Avoid generic praise.
**Format:** Plain text, no markdown headers. Start with "Dear Author,".
"""
