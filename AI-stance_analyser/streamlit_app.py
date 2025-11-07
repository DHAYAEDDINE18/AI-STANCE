import streamlit as st
import tempfile
from pdf_utils import extract_text_pipeline
from analysis import ai_segment_text, slice_pages_text, ai_analyse_stance
from io_utils import save_json, save_csv_stance

st.title("Thesis Sectioning + Hyland Stance Analysis")

uploaded = st.file_uploader("Upload PhD thesis PDF", type=["pdf"])
model = st.text_input("Gemini model name", value="gemini-2.0-flash")
conclusion_only = st.checkbox("Analyse only General Conclusion")

if uploaded and st.button("Run"):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    raw_pages, cleaned_pages, combined = extract_text_pipeline(tmp_path)
    st.success("Extracted and cleaned text.")

    segmentation = ai_segment_text(combined, model_name=model)
    st.json(segmentation)

    stance_results = []
    for seg in segmentation:
        title = seg["title"]
        if conclusion_only and "conclusion" not in title.lower():
            continue
        text = slice_pages_text(combined, seg["start_page"], seg["end_page"])
        res = ai_analyse_stance(title, text, model_name=model)
        stance_results.append(res)

    st.subheader("Stance Results")
    st.json(stance_results)

    if st.button("Download JSON"):
        save_json(segmentation, "segmentation.json")
        save_json(stance_results, "stance.json")
        save_csv_stance(stance_results, "stance.csv")
        st.success("Saved segmentation.json, stance.json, stance.csv in working directory.")
