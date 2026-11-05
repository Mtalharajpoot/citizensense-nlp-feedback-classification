"""Streamlit demo for CitizenSense. Run locally: streamlit run app/streamlit_app.py"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import analyze_feedback  # noqa: E402

st.set_page_config(page_title="CitizenSense", page_icon="📝")
st.title("CitizenSense: Public Feedback Classifier")
st.caption("Learning prototype trained on 510 synthetic, hand-reviewed samples created with AI assistance. "
           "Scores are relative, not certainty. Predictions are suggestions, not decisions.")

examples = [
    "There is no water supply in our area for three days.",
    "The hospital staff was helpful but the waiting time was too long.",
    "The new garbage truck comes on time.",
]
choice = st.selectbox("Try an example (or type your own below)", [""] + examples)
text = st.text_area("Citizen feedback", value=choice, height=120)

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter some feedback.")
    else:
        with st.spinner("Analyzing..."):
            r = analyze_feedback(text)
        c1, c2, c3 = st.columns(3)
        c1.metric("Category", r["category"], f"score {r['category_score']}", delta_color="off")
        c2.metric("Sentiment", r["sentiment"], f"score {r['sentiment_score']}", delta_color="off")
        c3.metric("Priority (indicative)", r["priority"], f"score {r['priority_score']}", delta_color="off")
        st.write("**Keywords:**", ", ".join(r["keywords"]) or "n/a")
        st.write("**Suggested department:**", r["suggested_department"])
        st.write("**Summary:**", r["summary"])
        st.write("**Suggested action:**", r["suggested_action"])
        if r["needs_manual_review"]:
            st.error("Needs manual review: " + r["review_reason"])
        else:
            st.success("No review flag")
