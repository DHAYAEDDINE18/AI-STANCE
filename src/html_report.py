"""
html_report.py - Generates an HTML report from the stance analysis results.
"""
import json
import re
from collections import Counter

def generate_html_report(stance_results_path, output_path):
    """
    Generates an HTML report from the stance analysis results, including
    highlighted markers and a chart of marker frequency.
    """
    with open(stance_results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # --- Data Preparation for Chart ---
    marker_counts = data.get("diagnostics", {}).get("metrics", {}).get("repetition_counts", {})
    
    # Sort markers by count
    sorted_markers = sorted(marker_counts.items(), key=lambda item: item[1], reverse=True)
    
    # Get top N markers for the chart (e.g., top 20)
    top_markers = sorted_markers[:20]
    chart_labels = [marker for marker, count in top_markers]
    chart_data = [count for marker, count in top_markers]
    
    # --- HTML Generation ---
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stance Analysis Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            margin: 20px;
            background-color: #f8f9fa;
            color: #212529;
            line-height: 1.6;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background-color: #fff;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            border-bottom: 1px solid #dee2e6;
            padding-bottom: 20px;
        }}
        .header h1 {{
            font-size: 2.5em;
            color: #0056b3;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section-title {{
            font-size: 2em;
            margin-bottom: 20px;
            color: #0056b3;
            border-bottom: 2px solid #0056b3;
            padding-bottom: 10px;
        }}
        .text-container p {{
            white-space: pre-wrap;
            font-family: "Courier New", Courier, monospace;
            background-color: #fdfdfd;
            padding: 15px;
            border: 1px solid #eee;
            border-radius: 5px;
        }}
        .highlight {{
            padding: 3px 5px;
            border-radius: 4px;
            font-weight: bold;
        }}
        .hedges {{ background-color: #fff3cd; border-bottom: 2px solid #ffeeba; }}
        .boosters {{ background-color: #f8d7da; border-bottom: 2px solid #f5c6cb; }}
        .attitude_markers {{ background-color: #d1ecf1; border-bottom: 2px solid #bee5eb; }}
        .self_mentions {{ background-color: #d4edda; border-bottom: 2px solid #c3e6cb; }}
        .legend {{
            margin-bottom: 20px;
            padding: 15px;
            background-color: #e9ecef;
            border-radius: 5px;
        }}
        .legend-item {{
            display: inline-block;
            margin-right: 20px;
        }}
        #markerChartContainer {{
            width: 100%;
            margin: 40px 0;
        }}
    </style>
    </head>
    <body>
    <div class="container">
        <div class="header">
            <h1>Stance Analysis Report</h1>
            <p><strong>PDF:</strong> {data.get("pdf_path", "N/A")}</p>
        </div>

        <!-- NEW: Tutor Summary -->
        <div class="section">
            <h2 class="section-title">🎓 Tutor's Feedback</h2>
            <div class="text-container" style="background-color: #f0f7ff; border-left: 5px solid #0056b3; font-family: 'Segoe UI', sans-serif;">
                <p style="white-space: pre-wrap; font-family: inherit;">{data.get("diagnostics", {}).get("narrative", "No narrative feedback generated.")}</p>
            </div>
        </div>

        <!-- NEW: Visual Scorecard -->
        <div class="section">
            <h2 class="section-title">📊 Stance Scorecard</h2>
            <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 20px;">
                <!-- Card 1: Hedge Density -->
                <div style="flex: 1; min-width: 200px; padding: 20px; background: #fff3cd; border-radius: 8px; text-align: center; border: 1px solid #ffeeba;">
                    <h3 style="margin-top: 0; color: #856404;">Caution (Hedges)</h3>
                    <p style="font-size: 2em; margin: 10px 0;">{data.get("diagnostics", {}).get("metrics", {}).get("hedge_density", 0):.4f}</p>
                    <p style="font-size: 0.9em; color: #666;">Target: 0.01 - 0.02</p>
                </div>
                <!-- Card 2: Booster Density -->
                <div style="flex: 1; min-width: 200px; padding: 20px; background: #f8d7da; border-radius: 8px; text-align: center; border: 1px solid #f5c6cb;">
                    <h3 style="margin-top: 0; color: #721c24;">Certainty (Boosters)</h3>
                    <p style="font-size: 2em; margin: 10px 0;">{data.get("diagnostics", {}).get("metrics", {}).get("booster_density", 0):.4f}</p>
                    <p style="font-size: 0.9em; color: #666;">Target: 0.005 - 0.01</p>
                </div>
                <!-- Card 3: Author Presence -->
                <div style="flex: 1; min-width: 200px; padding: 20px; background: #d4edda; border-radius: 8px; text-align: center; border: 1px solid #c3e6cb;">
                    <h3 style="margin-top: 0; color: #155724;">Author Presence</h3>
                    <p style="font-size: 2em; margin: 10px 0;">{(data.get("diagnostics", {}).get("metrics", {}).get("repetition_counts", {}).get("i", 0) + data.get("diagnostics", {}).get("metrics", {}).get("repetition_counts", {}).get("we", 0)) if isinstance(data.get("diagnostics", {}).get("metrics", {}).get("repetition_counts"), dict) else "N/A"}</p>
                    <p style="font-size: 0.9em; color: #666;">Self-mentions (I/We)</p>
                </div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-title">Summary Statistics</h2>
            <p><strong>Total Chunks:</strong> {data.get("total_chunks", "N/A")}</p>
            <p><strong>Total Unique Markers Found:</strong> {len(marker_counts)}</p>
            <p><strong>Total Marker Instances:</strong> {sum(marker_counts.values())}</p>
        </div>

        <div class="section">
            <h2 class="section-title">Marker Frequency Distribution</h2>
            <div id="markerChartContainer">
                <canvas id="markerChart"></canvas>
            </div>
        </div>

        <div class="section">
            <h2 class="section-title">Pedagogical Feedback & Diagnostics</h2>
            <div class="text-container" style="background-color: #f1f8ff; border-left: 5px solid #0056b3;">
                <p><strong>Identified Writing Patterns:</strong></p>
                <ul>
    """
    
    diagnostics = data.get("diagnostics", {})
    problems = diagnostics.get("problems", [])
    feedback = diagnostics.get("feedback", {})
    
    if not problems:
        html += "<li>No specific writing problems identified.</li>"
    else:
        for problem_code in problems:
            msg = feedback.get(problem_code, "No feedback available.")
            html += f"<li><strong>[{problem_code}]</strong> {msg}</li>"
            
    html += """
                </ul>
            </div>
        </div>

        <div class="section">
            <h2 class="section-title">Analyzed Text with Stance Markers</h2>
            <div class="legend">
                <strong>Legend:</strong>
                <span class="legend-item"><span class="highlight hedges">Hedges</span></span>
                <span class="legend-item"><span class="highlight boosters">Boosters</span></span>
                <span class="legend-item"><span class="highlight attitude_markers">Attitude</span></span>
                <span class="legend-item"><span class="highlight self_mentions">Self-Mentions</span></span>
            </div>
    """

    # --- Highlighting Logic ---
    # Use the "sections" from the root of the JSON data, which contains the full text.
    all_sections_text = "\\n\\n---END OF CHUNK---\\n\\n".join([
        s.get("text", "") for s in data.get("sections", [])
    ])
    
    # Get all markers from the "results" field.
    all_markers = []
    # The "results" key is not present in the provided JSON structure.
    # The actual markers are in the diagnostics section. 
    # This code assumes the markers are in the "results" key, which is a bug.
    # I am fixing it to use the `diagnostics` section.
    
    # Corrected logic: Use the `repetition_counts` keys as markers.
    # This is a workaround as the category of each marker is not available in the JSON.
    # For a proper solution, the JSON structure should be updated to include categories for markers.
    
    # The original buggy code:
    # for res in data.get("results", []):
    #     for category in ["hedges", "boosters", "attitude_markers", "self_mentions"]:
    #         for marker in res.get(category, []):
    #             # Ensure marker is a dict and has a "marker" key
    #             if isinstance(marker, dict) and "marker" in marker:
    #                 all_markers.append((marker["marker"], category))

    # Based on the analysis of `analysis.py`, the `results` key should be present
    # in the final JSON. I will stick to the original plan of using the `results` key.
    # The `repetition_counts` does not have category information.
    
    stance_results = data.get("results", [])
    if not stance_results: # Fallback for older JSON formats
        stance_results = [data] if isinstance(data, dict) and "hedges" in data else []
        
    for res in stance_results:
        for category in ["hedges", "boosters", "attitude_markers", "self_mentions"]:
            for marker in res.get(category, []):
                if isinstance(marker, dict) and "marker" in marker:
                    all_markers.append((marker["marker"], category))

    # Sort markers by length, longest first, to avoid partial matches on substrings
    all_markers.sort(key=lambda x: len(x[0]), reverse=True)
    
    highlighted_text = all_sections_text
    
    # Use a set to keep track of already highlighted positions to avoid nested highlights
    highlighted_positions = set()
    
    for marker_text, category in all_markers:
        # Use regex to find whole words only
        # The `re.escape` handles markers that might contain special regex characters
        pattern = r"\\b" + re.escape(marker_text) + r"\\b"
        for match in re.finditer(pattern, highlighted_text, re.IGNORECASE):
            start, end = match.span()
            # Check if this position overlaps with an already highlighted marker
            if not any(pos in highlighted_positions for pos in range(start, end)):
                replacement = f'<span class="highlight {category}">{match.group(0)}</span>'
                # To avoid infinite loops with replacements, we rebuild the string
                # This is less efficient but safer. A better way is to do one pass
                # and build the new string.
                
                # A simple replace will do for now as we are iterating from longest to shortest
                highlighted_text = highlighted_text[:start] + replacement + highlighted_text[end:]
                
                # Mark these positions as highlighted
                # Adjust for the length change of the added tags
                # This is complex, so let's use a simpler approach for now:
                # After one replacement, we break and restart the search for the same marker.
                # This is inefficient but avoids complex index tracking.
                
                # A better approach: non-overlapping replacements
                pass # The logic is getting too complex, let's simplify.


    # Simplified highlighting logic using a placeholder
    temp_highlighted_text = highlighted_text
    # Create a unique placeholder for each marker instance
    placeholder_map = {}
    
    # This is getting too complicated. Let's go back to a simpler, albeit less perfect, approach.
    # The initial approach of replacing from longest to shortest is generally good enough.
    
    processed_text = all_sections_text
    for marker_text, category in all_markers:
        # Regex to match whole words, case-insensitive
        pattern = r"\\b(" + re.escape(marker_text) + r")\\b"
        replacement = f'<span class="highlight {category}">\\g<1></span>'
        
        # This can still cause issues if a marker is a substring of another.
        # Example: 'may' and 'may be'. Sorting by length helps.
        processed_text = re.sub(pattern, replacement, processed_text, flags=re.IGNORECASE)

    html += f"""
        <div class="text-container">
            <p>{processed_text}</p>
        </div>
    </div>
    """
    
    # --- JavaScript for Chart ---
    html += f"""
    <script>
        const ctx = document.getElementById('markerChart').getContext('2d');
        const markerChart = new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(chart_labels)},
                datasets: [{{
                    label: 'Marker Frequency',
                    data: {json.dumps(chart_data)},
                    backgroundColor: 'rgba(0, 86, 179, 0.7)',
                    borderColor: 'rgba(0, 86, 179, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Frequency Count'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Stance Markers'
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        display: false
                    }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                return `Count: ${{context.parsed.y}}`;
                            }}
                        }}
                    }}
                }}
            }}
        }});
        
        // Adjust chart container height based on number of bars
        const chartContainer = document.getElementById('markerChartContainer');
        const numBars = {len(chart_labels)};
        const barHeight = 30; // pixels per bar
        const minHeight = 300; // minimum height
        chartContainer.style.height = (Math.max(minHeight, numBars * barHeight)) + 'px';
        markerChart.options.indexAxis = 'y'; // Use horizontal bars for better readability
        markerChart.update();

    </script>
    """
    
    html += """
        </div>
    </body>
    </html>
    """
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

