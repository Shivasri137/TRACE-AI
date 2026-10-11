"""Streamlit interface for TRACE AI."""
import json
import pandas as pd
import streamlit as st
from core import extract_pdf_text, split_into_chunks, verify_document

st.set_page_config(page_title="TRACE AI | Evidence Verification", page_icon="🔎", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
.trace-kicker {letter-spacing: .15em; text-transform: uppercase; font-size: .75rem; color: #7890FF; font-weight: 700;}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="trace-kicker">EVIDENCE INTELLIGENCE</div>', unsafe_allow_html=True)
st.title("TRACE AI")
st.subheader("Don't just generate answers. Verify them.")
st.write("Check claims against source documents, inspect retrieved evidence, and export an auditable report.")
st.info("TRACE AI checks claims against evidence you provide. It is not an independent internet fact-checker.")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Evidence passages per claim", 1, 5, 3)
    st.caption("TF-IDF relevance is not a truth or confidence score.")
    st.caption("Do not upload confidential material.")

mode = st.radio("Evidence source", ["Paste text", "Upload PDF"], horizontal=True)
source_text = ""
if mode == "Paste text":
    source_text = st.text_area("Source document", height=260, placeholder="Paste a report, paper excerpt, or trusted source...")
else:
    uploaded = st.file_uploader("Upload a text-based PDF (up to 50 pages)", type=["pdf"])
    if uploaded:
        try:
            source_text = extract_pdf_text(uploaded.getvalue())
            st.success("PDF text extracted.")
        except ValueError as exc:
            st.error(str(exc))

if source_text.strip():
    left, right = st.columns(2)
    left.metric("Source characters", f"{len(source_text):,}")
    right.metric("Evidence passages", len(split_into_chunks(source_text)))

st.divider()
claims_text = st.text_area("Claims to verify", height=130,
    placeholder="The study enrolled 120 participants. The trial lasted eight weeks.",
    help="Use up to 10 sentence-like claims per run.")
if st.button("Verify claims", type="primary", use_container_width=True):
    if not source_text.strip():
        st.error("Add source evidence before verifying.")
    elif not claims_text.strip():
        st.error("Enter at least one claim.")
    else:
        try:
            with st.spinner("Retrieving evidence and analyzing claims..."):
                st.session_state["trace_results"] = verify_document(claims_text, source_text, top_k)
        except Exception as exc:
            st.error(str(exc))

results = st.session_state.get("trace_results", [])
if results:
    st.divider()
    st.header("Verification report")
    labels = ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE")
    counts = {label: sum(row["verdict"] == label for row in results) for label in labels}
    cols = st.columns(4)
    cols[0].metric("Claims checked", len(results))
    cols[1].metric("Supported", counts[labels[0]])
    cols[2].metric("Contradicted", counts[labels[1]])
    cols[3].metric("Insufficient evidence", counts[labels[2]])
    st.dataframe(pd.DataFrame([{"Claim": row["claim"], "Verdict": row["verdict"],
        "Validated quotations": len(row["citations"])} for row in results]),
        use_container_width=True, hide_index=True)
    export_rows = []
    for i, row in enumerate(results, start=1):
        with st.expander(f"{i}. {row['verdict']} — {row['claim']}", expanded=(i == 1)):
            st.markdown("**Explanation**")
            st.write(row["explanation"] or "No explanation returned.")
            st.markdown("**Validated quotations**")
            if row["citations"]:
                for citation in row["citations"]:
                    st.markdown(f"> {citation['quote']}")
                    st.caption(f"Source passage ID: {citation['source_id']}")
            else:
                st.warning("No quotation passed exact-match validation. Inspect the evidence manually.")
            st.markdown("**Retrieved passages**")
            for passage in row["evidence"]:
                st.caption(f"Passage {passage['source_id']} · TF-IDF relevance {passage['relevance']:.4f}")
                st.write(passage["text"])
            export_rows.append({"claim": row["claim"], "verdict": row["verdict"],
                "explanation": row["explanation"], "citations": row["citations"],
                "retrieved_evidence": row["evidence"]})
    st.download_button("Download report (JSON)", data=json.dumps(export_rows, ensure_ascii=False, indent=2),
        file_name="trace_ai_report.json", mime="application/json", use_container_width=True)
    st.caption("A real quotation can still be irrelevant or misleading in context. Review source passages before relying on a verdict.")
