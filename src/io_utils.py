import orjson
import csv
from pathlib import Path

def get_sfl_interpretation(hyland_category: str) -> tuple[str, str]:
    """
    Maps a Hyland stance category to its Systemic Functional Linguistics (SFL)
    interpretation.
    """
    if hyland_category == "hedges":
        return "Engagement", "Heterogloss"
    if hyland_category == "boosters":
        return "Engagement", "Monogloss"
    if hyland_category == "attitude_markers":
        return "Attitude", "Appreciation"
    if hyland_category == "self_mentions":
        return "Interpersonal metafunction", "Authorial presence"
    return "", ""

def save_json(data, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))

def save_csv_stance(items, path: str):
    rows = []
    for item in items:
        section = item.get("section", "")
        for cat in ["hedges", "boosters", "attitude_markers", "self_mentions"]:
            sfl_system, sfl_sub_system = get_sfl_interpretation(cat)
            for ex in item.get(cat, []):
                rows.append({
                    "section": section,
                    "hyland_category": cat,
                    "sfl_system": sfl_system,
                    "sfl_sub_system": sfl_sub_system,
                    "word": ex.get("word", ""),
                    "context": ex.get("context", ""),
                })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    if not rows:
        fieldnames = ["section", "hyland_category", "sfl_system", "sfl_sub_system", "word", "context"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)