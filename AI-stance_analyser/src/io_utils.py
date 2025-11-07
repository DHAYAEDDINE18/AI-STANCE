import orjson
import csv
from pathlib import Path

def save_json(data, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))

def save_csv_stance(items, path: str):
    rows = []
    for item in items:
        section = item.get("section", "")
        for cat in ["hedges", "boosters", "attitude_markers", "self_mentions"]:
            for ex in item.get(cat, []):
                rows.append({
                    "section": section,
                    "category": cat,
                    "word": ex.get("word", ""),
                    "sentence": ex.get("sentence", ""),
                })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["section", "category", "word", "sentence"])
        w.writeheader()
        w.writerows(rows)
