"""
GFR RAG Stack Board -- ELITE PROFESSIONAL edition.

Run with:
    streamlit run app.py
"""

import os
import time
import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

import config
from rag_pipeline import answer_query

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="GFR RAG Stack Board",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Custom CSS -- elite dark theme, glassmorphism, animations, grid bg
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background:
            linear-gradient(rgba(11,10,18,0.94), rgba(11,10,18,0.97)),
            repeating-linear-gradient(0deg, rgba(124,92,252,0.04) 0px, transparent 1px, transparent 40px, rgba(124,92,252,0.04) 41px),
            repeating-linear-gradient(90deg, rgba(124,92,252,0.04) 0px, transparent 1px, transparent 40px, rgba(124,92,252,0.04) 41px),
            radial-gradient(circle at 15% 10%, rgba(124,92,252,0.14) 0%, transparent 40%),
            radial-gradient(circle at 85% 90%, rgba(91,63,224,0.12) 0%, transparent 40%),
            #0B0A12;
        background-attachment: fixed;
    }

    h1, h2, h3 { letter-spacing: 0.3px; font-weight: 700; }

    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: #100E1A; }
    ::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #7C5CFC, #4B2FD8); border-radius: 10px; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #14111F 0%, #0E0C16 100%);
        border-right: 1px solid #2A2740;
    }

    .glass-card {
        background: linear-gradient(145deg, rgba(24,21,38,0.85), rgba(20,18,31,0.85));
        backdrop-filter: blur(12px);
        border: 1px solid rgba(124,92,252,0.18);
        border-radius: 18px;
        padding: 18px 22px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    }

    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    .hero {
        padding: 22px 26px;
        border-radius: 22px;
        background: linear-gradient(145deg, rgba(28,24,46,0.6), rgba(18,16,28,0.6));
        border: 1px solid rgba(124,92,252,0.2);
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: "";
        position: absolute; inset: 0;
        background: linear-gradient(120deg, transparent, rgba(124,92,252,0.08), transparent);
        background-size: 200% 200%;
        animation: gradientShift 6s ease infinite;
    }
    .hero h1 {
        font-size: 2.15rem;
        background: linear-gradient(135deg, #ECEAF6, #B9A6FF, #7C5CFC);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
        animation: gradientShift 5s ease infinite;
        position: relative; z-index: 1;
    }
    .hero p { color: #9A94B8; font-size: 0.95rem; position: relative; z-index: 1; margin: 0; }

    .stat-strip {
        display: flex; gap: 10px; margin-top: 14px; position: relative; z-index: 1; flex-wrap: wrap;
    }
    .stat-pill {
        background: rgba(124,92,252,0.1);
        border: 1px solid rgba(124,92,252,0.25);
        border-radius: 12px;
        padding: 8px 16px;
        font-size: 0.78rem;
        color: #C6B8FF;
        font-family: 'JetBrains Mono', monospace;
    }
    .stat-pill b { color: #ECEAF6; }

    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .chat-row {
        display: flex; align-items: flex-start; gap: 12px;
        margin-bottom: 22px; animation: fadeSlideIn 0.35s ease-out;
    }
    .chat-row.user { flex-direction: row-reverse; }
    .avatar {
        width: 38px; height: 38px; min-width: 38px; border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.1rem; box-shadow: 0 4px 14px rgba(0,0,0,0.35);
    }
    .avatar.bot { background: linear-gradient(135deg, #7C5CFC, #4B2FD8); }
    .avatar.user { background: linear-gradient(135deg, #2E2B45, #1C1A2B); border: 1px solid #3A3660; }
    .bubble {
        max-width: 72%; padding: 14px 18px; border-radius: 18px;
        font-size: 0.96rem; line-height: 1.55; box-shadow: 0 4px 18px rgba(0,0,0,0.25);
    }
    .bubble.bot {
        background: linear-gradient(145deg, #1B1830, #151223);
        border: 1px solid rgba(124,92,252,0.22);
        color: #ECEAF6; border-top-left-radius: 4px;
    }
    .bubble.user {
        background: linear-gradient(135deg, #7C5CFC, #5B3FE0);
        color: white; border-top-right-radius: 4px;
    }

    .source-chip {
        display: inline-block; background: rgba(124,92,252,0.14);
        border: 1px solid rgba(124,92,252,0.3); color: #C6B8FF;
        border-radius: 999px; padding: 4px 12px; font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace; margin: 3px 4px 3px 0;
    }

    @keyframes shimmer {
        0% { background-position: -400px 0; }
        100% { background-position: 400px 0; }
    }
    .shimmer-line {
        height: 12px; border-radius: 6px; margin-bottom: 8px;
        background: linear-gradient(90deg, #1B1830 0%, #2A2450 50%, #1B1830 100%);
        background-size: 400px 100%;
        animation: shimmer 1.4s infinite linear;
    }

    .quick-chip-note { color: #6E688C; font-size: 0.78rem; margin: 4px 0 10px 0; }

    .gfr-kpi {
        background: linear-gradient(145deg, #1D1A30, #14121F);
        border: 1px solid #362F5C; border-radius: 18px; padding: 22px;
        text-align: center; transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .gfr-kpi:hover { transform: translateY(-3px); border-color: #7C5CFC; }
    .gfr-kpi h2 {
        font-size: 2.3rem; margin: 6px 0 0 0;
        background: linear-gradient(135deg, #C6B8FF, #7C5CFC);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800;
    }
    .gfr-kpi p {
        margin: 0; color: #9A94B8; font-size: 0.78rem;
        text-transform: uppercase; letter-spacing: 1.2px; font-weight: 600;
    }

    .gfr-badge {
        display: inline-block; background: rgba(124,92,252,0.15);
        border: 1px solid rgba(124,92,252,0.3); color: #C6B8FF;
        border-radius: 999px; padding: 4px 14px; font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace; margin-right: 6px;
    }

    .tech-badge {
        display: inline-block; background: rgba(79,224,160,0.1);
        border: 1px solid rgba(79,224,160,0.3); color: #7FE8B8;
        border-radius: 8px; padding: 5px 12px; font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace; margin: 3px 4px 3px 0;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #7C5CFC, #5B3FE0);
        color: white; border: none; border-radius: 12px;
        padding: 0.55rem 1.5rem; font-weight: 600; transition: all 0.2s ease;
        box-shadow: 0 4px 14px rgba(124,92,252,0.25);
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #8E70FF, #6C4EF2); color: white;
        transform: translateY(-1px); box-shadow: 0 6px 20px rgba(124,92,252,0.4);
    }

    div[data-testid="stTextInput"] input {
        background: #14111F !important; border: 1px solid #2E2A4A !important;
        border-radius: 14px !important; color: #ECEAF6 !important; padding: 12px 16px !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #7C5CFC !important; box-shadow: 0 0 0 3px rgba(124,92,252,0.15) !important;
    }

    .pipeline-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin: 16px 0; }
    .pipeline-step {
        background: linear-gradient(145deg, #1D1A30, #14121F);
        border: 1px solid #362F5C; border-radius: 14px;
        padding: 12px 16px; font-size: 0.82rem; color: #ECEAF6; text-align: center; min-width: 120px;
    }
    .pipeline-arrow { color: #7C5CFC; font-size: 1.2rem; }

    .footer-note {
        text-align: center; color: #524C70; font-size: 0.75rem;
        margin-top: 40px; padding-top: 20px; border-top: 1px solid #201C33;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Cached resources
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_vectorstore():
    """Loads the Chroma vector store once and reuses it across reruns."""
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings

    embeddings = OllamaEmbeddings(model=config.EMBED_MODEL, base_url="http://localhost:11434")
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=config.VECTORSTORE_DIR,
    )


def copy_button(text: str, key: str):
    """Renders a small 'copy to clipboard' button using a tiny embedded script."""
    safe_text = json.dumps(text)
    components.html(
        f"""
        <button id="copy-{key}" style="
            background: rgba(124,92,252,0.12);
            border: 1px solid rgba(124,92,252,0.3);
            color: #C6B8FF; border-radius: 8px; padding: 4px 10px;
            font-size: 0.72rem; cursor: pointer; font-family: 'JetBrains Mono', monospace;
        ">📋 Copy</button>
        <script>
        document.getElementById("copy-{key}").addEventListener("click", function() {{
            navigator.clipboard.writeText({safe_text});
            this.innerText = "✓ Copied";
            setTimeout(() => this.innerText = "📋 Copy", 1500);
        }});
        </script>
        """,
        height=36,
    )


# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "question_count" not in st.session_state:
    st.session_state.question_count = 0
if "question_box" not in st.session_state:
    st.session_state.question_box = ""
if "clear_question_box" not in st.session_state:
    st.session_state.clear_question_box = False

# Safe place to clear the input: BEFORE the text_input widget is instantiated
# this run. Setting st.session_state.question_box directly after the widget
# is created raises StreamlitAPIException, so we use a flag instead.
if st.session_state.clear_question_box:
    st.session_state.question_box = ""
    st.session_state.clear_question_box = False

SUGGESTED_QUESTIONS = [
    "What does Rule 26 say about Controlling Officers?",
    "What is 'Competent Authority' under the GFR?",
    "What does Rule 130 say about Original Works?",
    "What is the Global Tender Enquiry threshold?",
]

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📘 GFR Stack Board")
    st.caption("Elite local RAG system for the General Financial Rules document.")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["💬 Ask GFR", "📊 Evaluation Dashboard", "📚 Rule Explorer", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Models (local, via Ollama)**")
    st.markdown(
        f"""
        <span class="gfr-badge">LLM: {config.LLM_MODEL}</span><br><br>
        <span class="gfr-badge">Embeddings: {config.EMBED_MODEL}</span>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    pdf_exists = os.path.exists(config.PDF_PATH)
    store_exists = os.path.exists(config.VECTORSTORE_DIR) and len(os.listdir(config.VECTORSTORE_DIR)) > 0
    st.markdown("**Status**")
    st.write(("✅" if pdf_exists else "❌") + " PDF loaded")
    st.write(("✅" if store_exists else "❌") + " Vector store built")

    if st.session_state.history:
        st.markdown("---")
        st.markdown("**Session**")
        st.write(f"💬 {st.session_state.question_count} question(s) asked")
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.history = []
            st.rerun()

# ----------------------------------------------------------------------
# PAGE 1: Ask GFR
# ----------------------------------------------------------------------
if page == "💬 Ask GFR":
    st.markdown(
        f"""
        <div class="hero">
            <h1>Ask the GFR document</h1>
            <p>Answers are generated only from retrieved chunks of your GFR PDF — fully local, no paid API.</p>
            <div class="stat-strip">
                <div class="stat-pill">💬 <b>{st.session_state.question_count}</b> asked this session</div>
                <div class="stat-pill">🤖 <b>{config.LLM_MODEL}</b></div>
                <div class="stat-pill">🧩 <b>{config.EMBED_MODEL}</b></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="quick-chip-note">Quick questions — click to try one:</p>', unsafe_allow_html=True)
    chip_cols = st.columns(len(SUGGESTED_QUESTIONS))
    for c, q in zip(chip_cols, SUGGESTED_QUESTIONS):
        with c:
            if st.button(q, key=f"chip_{q}", use_container_width=True):
                st.session_state.question_box = q
                st.rerun()

    col1, col2 = st.columns([5, 1])
    with col1:
        question = st.text_input(
            "Question",
            key="question_box",
            placeholder="e.g. What does Rule 26 say about Controlling Officers?",
            label_visibility="collapsed",
        )
    with col2:
        ask = st.button("Ask →", use_container_width=True)

    if ask and question.strip():
        if not store_exists:
            st.error("Vector store not found. Run `python ingest.py` first.")
        else:
            placeholder = st.empty()
            with placeholder.container():
                st.markdown(
                    """
                    <div class="chat-row bot">
                        <div class="avatar bot">🤖</div>
                        <div class="bubble bot" style="width: 60%;">
                            <div class="shimmer-line" style="width: 90%;"></div>
                            <div class="shimmer-line" style="width: 75%;"></div>
                            <div class="shimmer-line" style="width: 85%;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            try:
                start = time.time()
                result = answer_query(question.strip())
                result["elapsed"] = round(time.time() - start, 1)
                result["question"] = question.strip()
                st.session_state.history.insert(0, result)
                st.session_state.question_count += 1
                st.session_state.clear_question_box = True
                placeholder.empty()
                st.rerun()
            except Exception as e:
                placeholder.empty()
                st.error(f"Something went wrong: {e}")

    st.markdown("<br>", unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown(
            """
            <div class="glass-card" style="text-align:center; padding:40px;">
                <div style="font-size:2rem; margin-bottom:8px;">💡</div>
                <p style="color:#9A94B8; margin:0;">Ask a question above to get started.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for idx, item in enumerate(st.session_state.history):
        st.markdown(
            f"""
            <div class="chat-row user">
                <div class="avatar user">🧑</div>
                <div class="bubble user">{item['question']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        elapsed = item.get("elapsed")
        elapsed_html = f'<div style="font-size:0.72rem; color:#7A7398; margin-top:8px;">⚡ {elapsed}s</div>' if elapsed else ""
        st.markdown(
            f"""
            <div class="chat-row bot">
                <div class="avatar bot">🤖</div>
                <div class="bubble bot">
                    {item['answer']}
                    {elapsed_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        b_col1, b_col2 = st.columns([1, 9])
        with b_col1:
            copy_button(item["answer"], key=f"ans_{idx}")

        with st.expander(f"📎 Retrieved context ({len(item['contexts'])} chunks)"):
            for i, ctx in enumerate(item["contexts"], 1):
                st.markdown(f"**Chunk {i}**")
                st.code(ctx, language=None)

        st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# PAGE 2: Evaluation Dashboard
# ----------------------------------------------------------------------
elif page == "📊 Evaluation Dashboard":
    st.markdown(
        """
        <div class="hero">
            <h1>Evaluation Dashboard</h1>
            <p>Results from <code>python evaluate.py</code> — RAGAS scores using your local Ollama model as judge.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    results_path = os.path.join(config.BASE_DIR, "evaluation_results.csv")

    if not os.path.exists(results_path):
        st.warning("No evaluation results found yet. Run `python evaluate.py` first, then reload this page.")
    else:
        df = pd.read_csv(results_path)
        numeric_cols = [c for c in df.select_dtypes(include="number").columns]

        if not numeric_cols:
            st.warning("No numeric metric columns found in evaluation_results.csv.")

        kpi_cols = st.columns(len(numeric_cols)) if numeric_cols else []
        for col, metric in zip(kpi_cols, numeric_cols):
            avg = df[metric].mean()
            avg_display = "N/A" if pd.isna(avg) else f"{avg:.2f}"
            with col:
                st.markdown(
                    f"""
                    <div class="gfr-kpi">
                        <p>{metric.replace('_', ' ')}</p>
                        <h2>{avg_display}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        chart_col1, chart_col2 = st.columns([1, 1])

        with chart_col1:
            if numeric_cols:
                avg_values = [df[c].mean() if not pd.isna(df[c].mean()) else 0 for c in numeric_cols]
                labels = [c.replace("_", " ").title() for c in numeric_cols]
                fig_radar = go.Figure()
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=avg_values + [avg_values[0]],
                        theta=labels + [labels[0]],
                        fill="toself",
                        fillcolor="rgba(124,92,252,0.25)",
                        line=dict(color="#7C5CFC", width=2),
                        marker=dict(color="#C6B8FF", size=6),
                    )
                )
                fig_radar.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(124,92,252,0.15)", tickfont=dict(size=9, color="#7A7398")),
                        angularaxis=dict(gridcolor="rgba(124,92,252,0.15)", tickfont=dict(size=11, color="#C6B8FF")),
                        bgcolor="rgba(0,0,0,0)",
                    ),
                    showlegend=False,
                    height=380,
                    margin=dict(t=30, b=20, l=40, r=40),
                    title=dict(text="Metric Overview", font=dict(size=14, color="#9A94B8")),
                )
                st.plotly_chart(fig_radar, use_container_width=True)

        with chart_col2:
            if numeric_cols:
                avg_values = [df[c].mean() if not pd.isna(df[c].mean()) else 0 for c in numeric_cols]
                fig_bar = go.Figure(
                    data=[
                        go.Bar(
                            x=numeric_cols,
                            y=avg_values,
                            marker=dict(
                                color=avg_values,
                                colorscale=[[0, "#E0526B"], [0.5, "#E0B84B"], [1, "#7C5CFC"]],
                                cmin=0, cmax=1,
                            ),
                            text=[f"{v:.2f}" if pd.notna(v) else "N/A" for v in avg_values],
                            textposition="outside",
                            textfont=dict(color="#ECEAF6"),
                        )
                    ]
                )
                fig_bar.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(range=[0, 1], title="Score", gridcolor="rgba(124,92,252,0.08)"),
                    xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                    margin=dict(t=40, b=20),
                    height=380,
                    title=dict(text="Score Breakdown", font=dict(size=14, color="#9A94B8")),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("### Per-question breakdown")

        def highlight_scores(val):
            if isinstance(val, (int, float)) and 0 <= val <= 1:
                if val >= 0.8:
                    return "background-color: rgba(79,224,160,0.15); color: #7FE8B8;"
                elif val >= 0.5:
                    return "background-color: rgba(224,184,75,0.15); color: #F0D080;"
                else:
                    return "background-color: rgba(224,82,107,0.15); color: #F08A9A;"
            return ""

        styler = df.style
        # pandas >= 2.1 renamed Styler.applymap -> Styler.map (applymap is
        # deprecated/removed). Support both so this doesn't break across envs.
        if hasattr(styler, "map"):
            styled_df = styler.map(highlight_scores, subset=numeric_cols)
        else:
            styled_df = styler.applymap(highlight_scores, subset=numeric_cols)
        st.dataframe(styled_df, use_container_width=True)

        st.download_button(
            "⬇️ Download full results CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="evaluation_results.csv",
            mime="text/csv",
        )

# ----------------------------------------------------------------------
# PAGE 3: Rule Explorer (semantic search preview, no LLM generation)
# ----------------------------------------------------------------------
elif page == "📚 Rule Explorer":
    st.markdown(
        """
        <div class="hero">
            <h1>Rule Explorer</h1>
            <p>Search the vector store directly — see exactly which chunks retrieval would surface, without generating an LLM answer. Useful for debugging or just browsing GFR by topic.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not store_exists:
        st.error("Vector store not found. Run `python ingest.py` first.")
    else:
        search_col1, search_col2 = st.columns([5, 1])
        with search_col1:
            search_term = st.text_input(
                "Search",
                placeholder="e.g. Controlling Officer, Global Tender Enquiry, Bid Security...",
                label_visibility="collapsed",
            )
        with search_col2:
            top_k = st.selectbox("Results", [3, 5, 8, 10], index=1, label_visibility="collapsed")

        if search_term.strip():
            vectordb = get_vectorstore()
            results = vectordb.similarity_search(search_term.strip(), k=top_k)

            st.markdown(f"<p style='color:#9A94B8; font-size:0.85rem;'>Found {len(results)} chunk(s)</p>", unsafe_allow_html=True)

            for i, r in enumerate(results, 1):
                rule_no = r.metadata.get("rule_number", "N/A")
                page_no = r.metadata.get("page", "N/A")
                defined_term = r.metadata.get("defined_term")

                chips = f'<span class="source-chip">📖 {rule_no}</span><span class="source-chip">📄 Page {page_no}</span>'
                if defined_term:
                    chips += f'<span class="source-chip">🏷️ {defined_term}</span>'

                st.markdown(
                    f"""
                    <div class="glass-card" style="margin-bottom: 14px;">
                        <div style="margin-bottom: 10px;">{chips}</div>
                        <div style="font-size: 0.88rem; color: #D8D4EC; line-height: 1.6; white-space: pre-wrap;">{r.page_content}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                """
                <div class="glass-card" style="text-align:center; padding:40px;">
                    <div style="font-size:2rem; margin-bottom:8px;">🔍</div>
                    <p style="color:#9A94B8; margin:0;">Type a term above to explore the vector store.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ----------------------------------------------------------------------
# PAGE 4: About
# ----------------------------------------------------------------------
else:
    st.markdown(
        """
        <div class="hero">
            <h1>About this project</h1>
            <p>A fully local Retrieval-Augmented Generation system, evaluated end-to-end with RAGAS.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="glass-card">
        This is a fully local Retrieval-Augmented Generation (RAG) system built on the
        <b>General Financial Rules (GFR)</b> document, evaluated with <b>RAGAS</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Pipeline")
    st.markdown(
        """
        <div class="pipeline-row">
            <div class="pipeline-step">📄 PDF</div>
            <div class="pipeline-arrow">→</div>
            <div class="pipeline-step">✂️ Rule-boundary chunking</div>
            <div class="pipeline-arrow">→</div>
            <div class="pipeline-step">🧩 Ollama embeddings</div>
            <div class="pipeline-arrow">→</div>
            <div class="pipeline-step">🗄️ Chroma vector store</div>
            <div class="pipeline-arrow">→</div>
            <div class="pipeline-step">🤖 Ollama LLM</div>
            <div class="pipeline-arrow">→</div>
            <div class="pipeline-step">💬 Answer</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Chunking strategy")
    st.markdown(
        """
        <div class="glass-card">
        Instead of blind fixed-size character splitting, chunks are aligned to GFR's actual
        <b>rule and definition boundaries</b>. Each rule becomes its own chunk, and Rule 2
        (Definitions) is further split so that each individual definition — "Competent Authority",
        "Controlling Officer", etc. — gets its own precise, focused chunk. This significantly
        improved retrieval precision over naive fixed-size chunking.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Evaluation metrics")
    st.markdown(
        """
        <div class="glass-card">
        <b>Faithfulness</b> — are the answer's claims grounded in retrieved context (no hallucination)?<br>
        <b>Answer relevancy</b> — does the answer actually address the question asked?<br>
        <b>Context precision</b> — are the retrieved chunks actually relevant to the question?<br>
        <b>Answer similarity</b> — how close is the answer to the reference (ground truth)?<br><br>
        All scored using a local Ollama model as judge — no paid API involved.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Tech stack")
    st.markdown(
        """
        <div>
            <span class="tech-badge">Python</span>
            <span class="tech-badge">Streamlit</span>
            <span class="tech-badge">LangChain</span>
            <span class="tech-badge">Chroma</span>
            <span class="tech-badge">Ollama</span>
            <span class="tech-badge">RAGAS</span>
            <span class="tech-badge">Plotly</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="footer-note">
            Built entirely with local, free tools — no paid API keys anywhere in this pipeline.
        </div>
        """,
        unsafe_allow_html=True,
    )
