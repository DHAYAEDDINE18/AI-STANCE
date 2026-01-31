"""
diagnostic_analysis.py - Third layer of analysis for stance results.
"""
import json
from collections import Counter

def compute_diagnostic_metrics(stance_data, total_words):
    """
    Computes diagnostic metrics from stance analysis data.
    
    Args:
        stance_data (list): A list of dictionaries, where each dictionary represents a chunk of text
                            and contains the stance markers found in that chunk.
        total_words (int): The total number of words in the document.
        
    Returns:
        dict: A dictionary containing the computed diagnostic metrics.
    """
    
    metrics = {
        "hedge_density": 0,
        "booster_density": 0,
        "repetition_counts": {},
        "stacked_hedges": [],
        "self_mention_absence": False,
        "attitude_marker_distribution": {},
        "hedge_distribution": {},
        "section_metrics": {},
        "stance_scope_balance": {"authorial": 0, "reported": 0},
    }
    
    all_hedges = []
    all_boosters = []
    all_attitude_markers = []
    all_self_mentions = []
    
    authorial_count = 0
    reported_count = 0

    for chunk in stance_data:
        # Filter markers by stance_scope, defaulting to "authorial" if not present
        authorial_hedges = [m for m in chunk.get("hedges", []) if m.get("stance_scope", "authorial") == "authorial"]
        authorial_boosters = [m for m in chunk.get("boosters", []) if m.get("stance_scope", "authorial") == "authorial"]
        authorial_attitude_markers = [m for m in chunk.get("attitude_markers", []) if m.get("stance_scope", "authorial") == "authorial"]
        
        all_hedges.extend(authorial_hedges)
        all_boosters.extend(authorial_boosters)
        all_attitude_markers.extend(authorial_attitude_markers)
        # Self-mentions are considered inherently authorial for this analysis
        all_self_mentions.extend(chunk.get("self_mentions", [])) 
        
        # Count authorial vs reported for balance metric
        for category in ["hedges", "boosters", "attitude_markers", "self_mentions"]:
            for marker in chunk.get(category, []):
                if marker.get("stance_scope") == "reported":
                    reported_count += 1
                else: # Handles authorial and cases where scope is not specified
                    authorial_count += 1
        
        # Section-level metrics
        section = chunk.get("section") or chunk.get("title") or "unknown"
        if section not in metrics["section_metrics"]:
            metrics["section_metrics"][section] = {
                "word_count": 0,
                "hedges": 0,
                "boosters": 0,
                "attitude_markers": 0,
                "hedge_density": 0,
                "booster_density": 0,
                "attitude_density": 0
            }
        
        chunk_word_count = chunk.get("word_count", 0)
        metrics["section_metrics"][section]["word_count"] += chunk_word_count
        metrics["section_metrics"][section]["hedges"] += len(authorial_hedges)
        metrics["section_metrics"][section]["boosters"] += len(authorial_boosters)
        metrics["section_metrics"][section]["attitude_markers"] += len(authorial_attitude_markers)

        # For backward compatibility with attitude_marker_distribution/hedge_distribution
        if section not in metrics["attitude_marker_distribution"]:
            metrics["attitude_marker_distribution"][section] = 0
            metrics["hedge_distribution"][section] = 0
        metrics["attitude_marker_distribution"][section] += len(authorial_attitude_markers)
        metrics["hedge_distribution"][section] += len(authorial_hedges)
        
        # Stacked hedge detection (using authorial markers only)
        for i in range(len(authorial_hedges) - 1):
            hedge1 = authorial_hedges[i]
            hedge2 = authorial_hedges[i+1]
            if hedge1.get("context") == hedge2.get("context"):
                metrics["stacked_hedges"].append(hedge1["context"])

    # Compute densities for each section
    for section, s_metrics in metrics["section_metrics"].items():
        if s_metrics["word_count"] > 0:
            s_metrics["hedge_density"] = s_metrics["hedges"] / s_metrics["word_count"]
            s_metrics["booster_density"] = s_metrics["boosters"] / s_metrics["word_count"]
            s_metrics["attitude_density"] = s_metrics["attitude_markers"] / s_metrics["word_count"]

    if total_words > 0:
        metrics["hedge_density"] = len(all_hedges) / total_words if total_words else 0
        metrics["booster_density"] = len(all_boosters) / total_words if total_words else 0

    # Repetition counts (from authorial markers)
    all_authorial_markers = all_hedges + all_boosters + all_attitude_markers + all_self_mentions
    marker_texts = [marker.get("marker", "").lower() for marker in all_authorial_markers if marker.get("marker")]
    metrics["repetition_counts"] = dict(Counter(marker_texts))
    
    # Self-mention functional role distribution
    metrics["self_mention_functional_distribution"] = {
        "procedural": 0,
        "interpretive": 0,
        "claiming": 0,
        "unknown": 0,
    }
    for sm in all_self_mentions:
        role = sm.get("functional_role", "unknown")
        if role in metrics["self_mention_functional_distribution"]:
            metrics["self_mention_functional_distribution"][role] += 1
        else:
            metrics["self_mention_functional_distribution"]["unknown"] += 1

    # Self-mention absence (S1)
    if not all_self_mentions:
        metrics["self_mention_absence"] = True
    else:
        metrics["self_mention_absence"] = False

    metrics["stance_scope_balance"]["authorial"] = authorial_count
    metrics["stance_scope_balance"]["reported"] = reported_count

    # --- General Conclusion Analysis ---
    metrics["general_conclusion"] = {
        "found": False,
        "hedge_density": 0,
        "booster_density": 0,
        "text_length": 0
    }
    
    conclusion_text = ""
    conclusion_markers = {"hedges": [], "boosters": []}
    header_found = False

    # New Heuristic: Look for a chunk/section titled "General Conclusion" or "Conclusion"
    for chunk in stance_data:
        chunk_title = (chunk.get("title") or chunk.get("section") or "").lower()
        if "general conclusion" in chunk_title or "conclusion" in chunk_title:
             header_found = True
             metrics["general_conclusion"]["found"] = True
             conclusion_text = chunk.get("text", "")
             # If text is not in chunk (it might not be if we only saved results), 
             # we use word_count if available
             conclusion_word_count = chunk.get("word_count", 0)
             # Only consider authorial markers for conclusion metrics
             conclusion_markers["hedges"] = [m for m in chunk.get("hedges", []) if m.get("stance_scope", "authorial") == "authorial"]
             conclusion_markers["boosters"] = [m for m in chunk.get("boosters", []) if m.get("stance_scope", "authorial") == "authorial"]
             
             if not conclusion_text and conclusion_word_count > 0:
                 metrics["general_conclusion"]["hedge_density"] = len(conclusion_markers["hedges"]) / conclusion_word_count
                 metrics["general_conclusion"]["booster_density"] = len(conclusion_markers["boosters"]) / conclusion_word_count
                 metrics["general_conclusion"]["text_length"] = conclusion_word_count
             break 
    
    # Fallback: Use the last chunk if no specific conclusion section was found
    if not header_found and stance_data:
        last_chunk = stance_data[-1]
        metrics["general_conclusion"]["found"] = False
        metrics["general_conclusion"]["is_fallback"] = True
        conclusion_text = last_chunk.get("text", "")
        conclusion_markers["hedges"] = [m for m in last_chunk.get("hedges", []) if m.get("stance_scope", "authorial") == "authorial"]
        conclusion_markers["boosters"] = [m for m in last_chunk.get("boosters", []) if m.get("stance_scope", "authorial") == "authorial"]

    if conclusion_text:
        conclusion_words = len(conclusion_text.split())
        if conclusion_words > 0:
            metrics["general_conclusion"]["hedge_density"] = len(conclusion_markers["hedges"]) / conclusion_words
            metrics["general_conclusion"]["booster_density"] = len(conclusion_markers["boosters"]) / conclusion_words
            metrics["general_conclusion"]["text_length"] = conclusion_words

    return {
        "metrics": metrics,
        "aggregated_markers": {
            "hedging": all_hedges,
            "boosting": all_boosters,
            "attitude": all_attitude_markers,
            "self_mention": all_self_mentions
        }
    }


def classify_writing_problems(metrics):
    """
    Classifies writing problems based on computed metrics.
    """
    
    problems = []
    
    # Define thresholds (these can be fine-tuned)
    HEDGE_DENSITY_LOW = 0.005
    HEDGE_DENSITY_HIGH = 0.02
    BOOSTER_DENSITY_HIGH = 0.01
    REPETITION_THRESHOLD = 5
    
    # H1: Under-hedging
    if metrics["hedge_density"] < HEDGE_DENSITY_LOW:
        problems.append("H1")
        
    # H2: Over-hedging
    if metrics["hedge_density"] > HEDGE_DENSITY_HIGH:
        problems.append("H2")
        
    # H3: Limited hedge repertoire
    hedge_markers = [marker for marker, count in metrics["repetition_counts"].items() if "hedge" in marker]
    if len(hedge_markers) < 3 and len(hedge_markers) > 0:
        problems.append("H3")
        
    # H4: Hedge stacking
    if metrics["stacked_hedges"]:
        problems.append("H4")

    # H5: Hedging restricted to limitations section
    hedge_dist = metrics.get("hedge_distribution", {})
    if hedge_dist:
        non_limitation_hedges = False
        limitation_hedges_found = False
        for section, count in hedge_dist.items():
            if "limitation" in section.lower():
                if count > 0:
                    limitation_hedges_found = True
            elif count > 0:
                non_limitation_hedges = True
                break
        if limitation_hedges_found and not non_limitation_hedges:
            problems.append("H5")
        
    # B1: Over-assertiveness
    if metrics["booster_density"] > BOOSTER_DENSITY_HIGH:
        problems.append("B1")
        
    # B2: Booster redundancy
    booster_markers = [marker for marker, count in metrics["repetition_counts"].items() if "booster" in marker and count > REPETITION_THRESHOLD]
    if booster_markers:
        problems.append("B2")
        
    # S1: Authorial invisibility
    if metrics["self_mention_absence"]:
        problems.append("S1")
    # S2: Restricted authorial presence
    elif not metrics["self_mention_absence"]:
        distribution = metrics.get("self_mention_functional_distribution", {})
        total_mentions = sum(distribution.values())
        if total_mentions > 0 and distribution.get("procedural") == total_mentions:
            problems.append("S2")
    
    # --- General Conclusion Problems ---
    gc_metrics = metrics.get("general_conclusion", {})
    if gc_metrics.get("text_length", 0) > 100: # Only analyze if we have enough text
        # GC1: General Conclusions - Lack of Confidence (High Hedging)
        # Conclusions are often expected to be more assertive than the rest of the text.
        if gc_metrics["hedge_density"] > HEDGE_DENSITY_HIGH:
            problems.append("GC1")
            
        # GC2: General Conclusions - Lack of Conviction (Low Boosting)
        # Conclusions should highlight the significance.
        if gc_metrics["booster_density"] < 0.001: # Very low booster usage
            problems.append("GC2")

    return problems

def generate_pedagogical_feedback(problems):
    """
    Generates pedagogical feedback for identified writing problems.
    """
    
    feedback = {}
    
    feedback_map = {
        "H1": "The text may be under-hedged. Consider using more cautious language to express uncertainty.",
        "H2": "The text may be over-hedged. Consider reducing the number of hedges for a more confident tone.",
        "H3": "The variety of hedges is limited. Consider using a wider range of hedging words and phrases.",
        "H4": "There are instances of stacked hedges. Avoid using multiple hedges in the same sentence.",
        "H5": "Hedging seems to be restricted to a 'Limitations' section. While appropriate there, consider whether cautious language is needed in other parts of your text, such as when interpreting results or discussing implications.",
        "B1": "The text may be over-assertive. Consider reducing the number of boosters for a more nuanced tone.",
        "B2": "There are instances of booster redundancy. Avoid overusing the same booster words.",
        "S1": "The author is not explicitly present in the text. Consider using self-mentions to guide the reader.",
        "S2": "Authorial presence is restricted to procedural self-mentions. Consider using 'I' or 'we' to make claims and interpretations, not just to describe your research process.",
        "GC1": "The General Conclusion appears over-hedged. Conclusions should generally summarize findings with more confidence and clarity than the discussion section.",
        "GC2": "The General Conclusion lacks boosters. Consider using more emphatic language to highlight the significance and contribution of your research.",
    }
    
    for problem in problems:
        if problem in feedback_map:
            feedback[problem] = feedback_map[problem]
            
    return feedback

def run_diagnostic_analysis(stance_results_path, text_file_path):
    """
    Runs the diagnostic analysis on the given stance analysis results.
    """
    
    with open(stance_results_path, "r", encoding="utf-8") as f:
        stance_data = json.load(f)
        
    with open(text_file_path, "r", encoding="utf-8") as f:
        total_words = len(f.read().split())
        
    result = compute_diagnostic_metrics(stance_data, total_words)
    metrics = result["metrics"]
    aggregated_markers = result["aggregated_markers"]

    problems = classify_writing_problems(metrics)
    feedback = generate_pedagogical_feedback(problems)
    
    return {
        "diagnostics": {
            "metrics": metrics,
            "problems": problems,
            "feedback": feedback,
            "hedging": aggregated_markers["hedging"],
            "boosting": aggregated_markers["boosting"],
            "attitude": aggregated_markers["attitude"],
            "self_mention": aggregated_markers["self_mention"],
            "stance_scope_balance": metrics["stance_scope_balance"]
        }
    }
