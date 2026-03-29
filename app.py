"""
Streamlit Web UI for Florida County RAG System

Run with: streamlit run app.py
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import TOPICS, CHUNKS_PATH
from rag.retriever import get_retriever
from rag.answer_engine import RAGAnswerEngine
from utils.county_normalizer import get_canonical_counties
import json

# --- Page styling (Florida / navigator theme) ---
PAGE_CSS = """
<style>
  /* Hide default Streamlit header chrome for a cleaner look */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  header[data-testid="stHeader"] {background: transparent;}

  .block-container {
    padding-top: 1.25rem !important;
    max-width: min(1280px, 100%) !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
  }

  .fpn-hero {
    background:
      radial-gradient(ellipse 95% 75% at 90% -8%, rgba(251, 191, 36, 0.33) 0%, transparent 52%),
      radial-gradient(ellipse 60% 50% at 5% 105%, rgba(56, 189, 248, 0.18) 0%, transparent 55%),
      linear-gradient(152deg, #062a3f 0%, #0c4a6e 34%, #0e7490 62%, #115e59 100%);
    border-radius: 20px;
    padding: 2.75rem 2.25rem 3.25rem 2.25rem;
    margin-bottom: 2rem;
    min-height: clamp(260px, 32vh, 380px);
    box-shadow:
      0 16px 48px rgba(6, 42, 63, 0.42),
      inset 0 1px 0 rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.09);
    color: #f0fdfa;
    position: relative;
    overflow: hidden;
    box-sizing: border-box;
  }
  /* Keep copy in the left half of the card (indent → center); full width on phones. */
  .fpn-hero-inner {
    position: relative;
    z-index: 2;
    width: 100%;
    max-width: min(50%, 42rem);
    margin-left: 0;
    margin-right: auto;
    box-sizing: border-box;
  }
  @media (max-width: 768px) {
    .fpn-hero-inner {
      max-width: 100%;
    }
  }
  .fpn-hero-stats {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.65rem;
    margin-top: 1.65rem;
  }
  .fpn-pill {
    display: inline-flex;
    align-items: baseline;
    gap: 0.35rem;
    padding: 0.5rem 1.05rem;
    border-radius: 999px;
    font-size: 0.88rem;
    font-weight: 500;
    background: rgba(255, 255, 255, 0.11);
    border: 1px solid rgba(255, 255, 255, 0.16);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    color: #ecfeff;
    letter-spacing: 0.02em;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.12);
  }
  .fpn-pill strong {
    font-size: 1.02rem;
    font-weight: 700;
    color: #ffffff;
  }
  .fpn-hero h1 {
    font-size: 2.05rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    margin: 0 0 0.65rem 0 !important;
    line-height: 1.2 !important;
    border: none !important;
    color: #fff !important;
  }
  .fpn-hero p {
    margin: 0 !important;
    opacity: 0.92;
    font-size: 1.12rem;
    line-height: 1.55;
    max-width: 100%;
  }
  .fpn-hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(8px);
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.85rem;
    color: #cffafe;
  }
  /* Bottom wave accent */
  .fpn-hero::after {
    content: "";
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 56px;
    opacity: 0.35;
    pointer-events: none;
    background-repeat: no-repeat;
    background-size: 100% 100%;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1200 48' preserveAspectRatio='none'%3E%3Cpath fill='white' fill-opacity='0.12' d='M0,32 C200,8 400,40 600,24 C800,8 1000,32 1200,16 L1200,48 L0,48 Z'/%3E%3C/svg%3E");
  }

  .fpn-sidebar-title {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.5rem;
  }

  /* Tabs: larger icons + text, use horizontal space */
  div[data-testid="stTabs"] {
    width: 100%;
  }
  div[data-testid="stTabs"] [data-baseweb="tab-list"],
  div[data-testid="stTabs"] [role="tablist"] {
    width: 100% !important;
    gap: 0.75rem !important;
    min-height: auto !important;
    display: flex !important;
  }
  div[data-testid="stTabs"] button[data-baseweb="tab"],
  div[data-testid="stTabs"] button[role="tab"] {
    flex: 1 1 0 !important;
    min-width: 0 !important;
    min-height: 3.5rem !important;
    font-size: 1.35rem !important;
    font-weight: 650 !important;
    padding: 0.95rem 1.15rem !important;
    line-height: 1.35 !important;
    border-radius: 12px !important;
  }
  div[data-testid="stTabs"] button p {
    font-size: 1.35rem !important;
    line-height: 1.35 !important;
    margin: 0 !important;
  }

  .stButton button[kind="primary"] {
    background: linear-gradient(135deg, #0e7490, #0891b2) !important;
    border: none !important;
    font-weight: 600 !important;
  }
</style>
"""

# Official Florida county count (used in hero stat pill)
FLORIDA_COUNTY_COUNT = 67


@st.cache_resource
def load_retriever():
    """Load the vector retriever (cached)."""
    return get_retriever()


@st.cache_resource
def load_answer_engine():
    """Load the RAG answer engine (cached)."""
    return RAGAnswerEngine()


@st.cache_data
def get_counties_from_db():
    """Get list of counties in the database."""
    try:
        counties = set()
        with CHUNKS_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                counties.add(r["county"])
        return sorted(counties)
    except Exception:
        return get_canonical_counties()


def main():
    st.set_page_config(
        page_title="Florida Policy Navigator",
        page_icon="🧭",
        layout="wide",
    )

    st.markdown(PAGE_CSS, unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="fpn-hero">
          <div class="fpn-hero-inner">
            <span class="fpn-hero-badge">County comprehensive plans</span>
            <h1>🐊 🧭 Florida Policy Navigator</h1>
            <p>Explore Florida county plans with semantic search and cited answers.</p>
            <div class="fpn-hero-stats" aria-label="Coverage">
              <span class="fpn-pill"><strong>{FLORIDA_COUNTY_COUNT}</strong> Counties</span>
              <span class="fpn-pill"><strong>270+</strong> Plans</span>
              <span class="fpn-pill"><strong>Cited</strong> Answers</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown('<p class="fpn-sidebar-title">Search scope</p>', unsafe_allow_html=True)
        counties = ["All Counties"] + get_counties_from_db()
        selected_county = st.selectbox(
            "County",
            counties,
            index=0,
            label_visibility="collapsed",
        )
        top_k = st.slider("Number of results", 3, 15, 8)

    tab1, tab2, tab3 = st.tabs(["💬 Ask", "🔍 Search", "📊 Topic analysis"])

    with tab1:
        st.markdown("### 👉 Ask About Florida Policies")
        st.caption("Questions are answered from retrieved plan excerpts with page citations.")

        question = st.text_area(
            "Your question",
            placeholder="e.g., Does Collier County require wildlife surveys before development?",
            height=110,
            label_visibility="collapsed",
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.button("Get answer", type="primary", use_container_width=True)

        if ask_button and question:
            county_filter = None if selected_county == "All Counties" else selected_county

            with st.spinner("Searching plans and drafting an answer…"):
                try:
                    engine = load_answer_engine()
                    result = engine.answer(
                        question,
                        county=county_filter,
                        top_k=top_k,
                    )

                    st.subheader("Answer")

                    confidence_color = {
                        "high": "green",
                        "medium": "orange",
                        "low": "red",
                    }.get(result.confidence, "gray")

                    st.markdown(f"**Confidence:** :{confidence_color}[{result.confidence.upper()}]")
                    st.markdown(result.answer)

                    st.subheader(f"Sources ({len(result.sources)} excerpts)")

                    for i, source in enumerate(result.sources, 1):
                        with st.expander(
                            f"[{i}] {source.citation} (distance: {source.distance:.3f})"
                        ):
                            st.markdown(
                                source.text[:1500]
                                + ("..." if len(source.text) > 1500 else "")
                            )

                except Exception as e:
                    st.error(f"Error: {e}")

        st.divider()
        st.markdown("**Example questions**")
        examples = [
            "What wildlife corridor policies exist in Palm Beach County?",
            "Which counties require land acquisition for conservation?",
            "Do any counties have wildlife crossing requirements?",
            "What open space preservation policies does Alachua County have?",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex[:20]}"):
                st.session_state["example_q"] = ex
                st.rerun()

    with tab2:
        st.header("Semantic search")
        st.markdown("Find relevant policy excerpts without generating a full answer.")

        search_query = st.text_input(
            "Search query",
            placeholder="e.g., wildlife corridor habitat connectivity",
        )

        if st.button("Search", key="search_btn") and search_query:
            county_filter = None if selected_county == "All Counties" else selected_county

            with st.spinner("Searching…"):
                try:
                    retriever = load_retriever()
                    chunks = retriever.retrieve(
                        search_query,
                        top_k=top_k,
                        county_filter=county_filter,
                    )

                    st.subheader(f"Found {len(chunks)} relevant excerpts")

                    for i, chunk in enumerate(chunks, 1):
                        with st.expander(
                            f"**{i}. {chunk.county}** — {chunk.pdf_file} "
                            f"(pp. {chunk.page_start}-{chunk.page_end}) · distance {chunk.distance:.3f}"
                        ):
                            st.markdown(chunk.text)

                except Exception as e:
                    st.error(f"Error: {e}")

    with tab3:
        st.header("Topic analysis by county")
        st.markdown("Quick scan of retrieval strength per environmental topic for one county.")

        analysis_county = st.selectbox(
            "County",
            get_counties_from_db(),
            key="analysis_county",
        )

        if st.button("Analyze topics", key="analyze_btn"):
            with st.spinner(f"Analyzing {analysis_county}…"):
                try:
                    retriever = load_retriever()

                    results = {}
                    for topic_key, topic_info in TOPICS.items():
                        chunks = retriever.retrieve(
                            topic_info["query"],
                            top_k=3,
                            county_filter=analysis_county,
                        )

                        has_evidence = len(chunks) > 0 and chunks[0].distance < 0.6
                        results[topic_key] = {
                            "has_evidence": has_evidence,
                            "chunks": chunks,
                            "best_distance": chunks[0].distance if chunks else 1.0,
                        }

                    st.subheader(f"Results — {analysis_county}")

                    for topic_key, topic_info in TOPICS.items():
                        result = results[topic_key]

                        if result["has_evidence"]:
                            st.markdown(f"✅ **{topic_info['display_name']}** — evidence likely")
                        else:
                            st.markdown(f"❌ **{topic_info['display_name']}** — no clear match")

                        if result["chunks"]:
                            with st.expander(f"Top excerpts — {topic_info['display_name']}"):
                                for chunk in result["chunks"][:2]:
                                    st.markdown(f"**{chunk.citation}** (distance: {chunk.distance:.3f})")
                                    st.markdown(f"> {chunk.text[:500]}...")
                                    st.divider()

                except Exception as e:
                    st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
