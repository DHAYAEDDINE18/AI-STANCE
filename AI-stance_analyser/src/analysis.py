import json
import re
from typing import Any

from .prompts import (
    SEGMENT_SYSTEM, SEGMENT_USER_TEMPLATE, STANCE_SYSTEM, STANCE_USER_TEMPLATE,
    QUERY_SYSTEM, QUERY_USER_TEMPLATE
)

from .ai_clients import GeminiClient
from .prompts import (
    SEGMENT_SYSTEM,
    SEGMENT_USER_TEMPLATE,
    STANCE_SYSTEM,
    STANCE_USER_TEMPLATE,
)
from .pdf_utils import chunk_text_for_model

def ai_refine_query(user_prompt: str, model_name: str | None = None) -> dict[str, Any]:
    client = GeminiClient(model_name=model_name)
    out = client.generate_json(
        QUERY_USER_TEMPLATE.format(prompt=user_prompt),
        system_instruction=QUERY_SYSTEM,
        temperature=0.1,
        max_output_tokens=2000,
    )
    spec = parse_json_str(out)
    if not isinstance(spec, dict):
        raise ValueError("Query spec is not a JSON object")
    # Minimal validation and sane defaults
    spec.setdefault("fields", [{"name": "page", "type": "string"}, {"name": "snippet", "type": "string"}])
    strategy = spec.setdefault("strategy", {})
    strategy.setdefault("mode", "keywords")
    strategy.setdefault("keywords", [])
    strategy.setdefault("patterns", [])
    strategy.setdefault("any_all", "any")
    strategy.setdefault("case_sensitive", False)
    strategy.setdefault("context_before_chars", 120)
    strategy.setdefault("context_after_chars", 120)
    return spec

def run_query_on_text(full_text: str, spec: dict[str, Any]) -> list[dict[str, Any]]:
    # Split by page anchors
    segments = re.split(r"<<PAGE\s+(\d+)>>", full_text)
    results: list[dict[str, Any]] = []

    mode = spec["strategy"]["mode"]
    case = 0 if spec["strategy"]["case_sensitive"] else re.IGNORECASE
    before = int(spec["strategy"]["context_before_chars"])
    after = int(spec["strategy"]["context_after_chars"])

    keyword_list = [str(k) for k in spec["strategy"].get("keywords", [])]
    any_all = spec["strategy"].get("any_all", "any")
    patterns = [re.compile(p, case) for p in spec["strategy"].get("patterns", [])]

    def collect(page: int, text: str, term: str, start: int, end: int):
        s = max(0, start - before)
        e = min(len(text), end + after)
        snippet = text[s:e].replace("\n", " ").strip()
        row = {"page": page, "snippet": snippet}
        # Include topical term if spec defines one
        want_term = any(f.get("name") == "term" for f in spec.get("fields", []))
        if want_term:
            row["term"] = term
        results.append(row)

    for i in range(1, len(segments), 2):
        try:
            page = int(segments[i])
        except Exception:
            continue
        text = segments[i + 1] if i + 1 < len(segments) else ""

        if mode == "regex" and patterns:
            for rx in patterns:
                for m in rx.finditer(text):
                    collect(page, text, rx.pattern, m.start(), m.end())
        else:
            # keywords
            if not keyword_list:
                continue
            # Simple scan: find all occurrences of each keyword
            hits = []
            for kw in keyword_list:
                # Find all case-insensitive occurrences
                pos = 0
                rx = re.compile(re.escape(kw), case)
                for m in rx.finditer(text):
                    hits.append((kw, m.start(), m.end()))
            if any_all == "all":
                present = {kw for kw, _, _ in hits}
                if not all(kw.lower() in {p.lower() for p in present} for kw in keyword_list):
                    continue
            for kw, s0, e0 in hits:
                collect(page, text, kw, s0, e0)

    return results

def save_query_results(items: list[dict[str, Any]], json_path: str, csv_path: str):
    # JSON
    from .io_utils import save_json as _save_json
    _save_json(items, json_path)
    # CSV
    if items:
        fieldnames = sorted(items[0].keys())
    else:
        fieldnames = ["page", "snippet", "term"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(items)

def parse_json_str(s: str) -> Any:
    """
    Parse JSON that may be wrapped in Markdown code fences.
    Uses a balanced-brace/Bracket scan; no fragile backtick-anchored regex.
    """
    s = s.strip()
    m = re.search(r"``````", s, flags=re.DOTALL)
    candidate = m.group(1) if m else s

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
                        return json.loads(fragment)
    return json.loads(candidate)


def _reconcile_sections(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for it in items or []:
        if not isinstance(it, dict):
            continue
        title = str(it.get("title", "")).strip()
        if not title:
            continue
        try:
            sp = int(it.get("start_page", 1))
        except Exception:
            sp = 1
        try:
            ep = int(it.get("end_page", sp))
        except Exception:
            ep = sp
        summary = str(it.get("summary", "")).strip()
        key = re.sub(r"\s+", " ", title.lower())
        sp = max(1, sp)
        ep = max(sp, ep)
        if key not in groups:
            groups[key] = {"title": title, "start_page": sp, "end_page": ep, "summaries": [summary] if summary else []}
        else:
            groups[key]["start_page"] = min(groups[key]["start_page"], sp)
            groups[key]["end_page"] = max(groups[key]["end_page"], ep)
            if summary:
                groups[key]["summaries"].append(summary)

    merged: list[dict[str, Any]] = []
    for v in groups.values():
        merged.append(
            {
                "title": v["title"],
                "start_page": v["start_page"],
                "end_page": v["end_page"],
                "summary": " ".join(v["summaries"])[:1200],
            }
        )

    merged.sort(key=lambda x: (x["start_page"], x["end_page"]))
    for i in range(1, len(merged)):
        prev, cur = merged[i - 1], merged[i]
        if cur["start_page"] <= prev["end_page"]:
            cur["start_page"] = prev["end_page"] + 1
            if cur["start_page"] > cur["end_page"]:
                cur["end_page"] = cur["start_page"]
    return merged


def ai_segment_text(full_text: str, model_name: str | None = None) -> list[dict[str, Any]]:
    """
    Segment using locally extracted text that includes <<PAGE N>> anchors.
    """
    client = GeminiClient(model_name=model_name)
    chunks = chunk_text_for_model(full_text)
    collected: list[dict[str, Any]] = []
    for ch in chunks:
        prompt = SEGMENT_USER_TEMPLATE.format(chunk=ch)
        out = client.generate_json(
            prompt,
            system_instruction=SEGMENT_SYSTEM,
            temperature=0.1,
            max_output_tokens=20000,
        )
        try:
            arr = parse_json_str(out)
            if isinstance(arr, dict):
                arr = [arr]
            for it in arr:
                if isinstance(it, dict):
                    collected.append(it)
        except Exception:
            continue
    return _reconcile_sections(collected)


def slice_pages_text(full_text: str, start_page: int, end_page: int) -> str:
    parts = re.split(r"<<PAGE\s+(\d+)>>", full_text)
    chunks: list[str] = []
    for i in range(1, len(parts), 2):
        try:
            pnum = int(parts[i])
        except Exception:
            continue
        ptext = parts[i + 1] if i + 1 < len(parts) else ""
        if start_page <= pnum <= end_page:
            chunks.append(ptext.strip())
    return "\n\n".join(chunks)


def ai_analyse_stance(section_title: str, section_text: str, model_name: str | None = None) -> dict[str, Any]:
    client = GeminiClient(model_name=model_name)
    prompt = STANCE_USER_TEMPLATE.format(section_title=section_title, text=section_text[:200000])
    out = client.generate_json(
        prompt,
        system_instruction=STANCE_SYSTEM,
        temperature=0.2,
        max_output_tokens=20000,
    )
    try:
        data = parse_json_str(out)
        for k in ["hedges", "boosters", "attitude_markers", "self_mentions"]:
            if k not in data or not isinstance(data[k], list):
                data[k] = []
        data.setdefault("section", section_title)
        data.setdefault("summary", "")
        return data
    except Exception:
        return {
            "section": section_title,
            "hedges": [],
            "boosters": [],
            "attitude_markers": [],
            "self_mentions": [],
            "summary": "Parsing failed",
        }
