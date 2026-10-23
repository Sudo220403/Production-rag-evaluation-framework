"""
Interactive Plotly + Streamlit Dashboard & Live Evaluation Arena (`dashboard.py`).
Allows engineers and researchers to:
  1. Plug in their OpenRouter API Key, Base URL, and Model ID (`meta-llama/llama-3.3-70b-instruct`).
  2. Instantly explore the pre-computed 192-run Live Benchmark results, or trigger
     a Fast Live Re-Run with real-time progress bar and persistent disk caching.
  3. Visualize Faithfulness, Retrieval Precision, Hallucination Rate, Latency, and
     Paired Statistical Significance (t-stat, p-value, Cohen's d) across 3 document types.
  4. Inspect individual QA samples to diagnose why specific chunking strategies fail.
  5. Run a Live Side-by-Side Custom Query Arena comparing all 4 chunking strategies in real time.

Launch with:
    python -m streamlit run dashboard.py
"""

from __future__ import annotations

import importlib

import pandas as pd
import plotly.express as px
import streamlit as st

import src.analysis
import src.chunking
import src.config
import src.dataset
import src.evaluator
import src.indexing
import src.llm_router

# Force hot-reload of src.* modules inside running Streamlit server processes
importlib.reload(src.config)
importlib.reload(src.llm_router)
importlib.reload(src.dataset)
importlib.reload(src.chunking)
importlib.reload(src.indexing)
importlib.reload(src.evaluator)
importlib.reload(src.analysis)

from src.analysis import compute_headline_finding, export_research_reports
from src.chunking import ChunkingEngine
from src.config import REPORTS_DIR, EvaluationExperimentConfig, RouterAPIConfig
from src.dataset import (
    BenchmarkQASample,
    get_benchmark_qa_samples,
    get_corpus_documents,
)
from src.evaluator import RAGEvaluationEngine
from src.indexing import EmbeddingEngine, VectorStoreRetriever
from src.llm_router import RouterLLMClient

st.set_page_config(
    page_title="Production RAG Evaluation Framework",
    page_icon="📊",
    layout="wide",
)

STRATEGY_COLORS = {
    "sentence_window": "#10b981",
    "hierarchical_parent_child": "#3b82f6",
    "recursive_markdown": "#f59e0b",
    "fixed_window": "#ef4444",
}

st.title("📊 Production RAG Evaluation & Ablation Framework")
st.caption(
    "Empirical evaluation of Retrieval Precision@K, Answer Faithfulness, Hallucination Rate, "
    "and Latency across 4 Chunking Strategies, Embedding Models, and Vector Stores (FAISS & ChromaDB) "
    "powered by OpenRouter (`meta-llama/llama-3.3-70b-instruct`)."
)

# --- Sidebar: Router API Configuration & Experiment Controls ---
with st.sidebar:
    st.header("🔑 OpenRouter API Configuration")
    default_router = RouterAPIConfig()

    router_base_url = st.text_input(
        "Router Base URL",
        value=default_router.base_url,
        help="OpenAI-compatible endpoint (e.g., https://openrouter.ai/api/v1)",
    )
    router_api_key = st.text_input(
        "OpenRouter API Key",
        value=default_router.api_key if default_router.is_configured else "",
        type="password",
        placeholder="Paste your OpenRouter API key...",
    )
    router_llm_model = st.text_input(
        "Generator LLM Model",
        value=default_router.llm_model,
        help="Default: meta-llama/llama-3.3-70b-instruct",
    )
    router_judge_model = st.text_input(
        "Judge / RAGAS LLM Model",
        value=default_router.judge_model,
    )
    top_k = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=6, value=3)
    eval_mode = st.selectbox(
        "Evaluation Mode",
        options=["auto", "live", "offline"],
        index=0,
        help="'auto' uses live OpenRouter API when a key is configured.",
    )
    run_scope = st.selectbox(
        "Re-Run Scope (when clicking Run)",
        options=[
            "Fast Ablation (4 Chunkers × FAISS, ~10s)",
            "Full Matrix (192 Runs across all DBs/Embeddings)",
        ],
        index=0,
        help="Results are already pre-loaded! Choose Fast Ablation if you want a quick live re-run.",
    )

    active_router_cfg = RouterAPIConfig(
        api_key=router_api_key.strip(),
        base_url=router_base_url.strip(),
        llm_model=router_llm_model.strip(),
        judge_model=router_judge_model.strip(),
    )

    st.caption("✅ **192 Live Results are already loaded!** Only click Run if you change Top-K or Model.")
    col_test, col_run = st.columns(2)
    with col_test:
        if st.button("🔌 Test Router", width="stretch"):
            client = RouterLLMClient(active_router_cfg)
            status = client.test_connection()
            if status.get("ok"):
                st.success(f"Connected ({status['latency_ms']} ms): {status['reply']}")
            else:
                st.warning(status.get("message", "Router connection failed"))

    with col_run:
        run_clicked = st.button("🚀 Re-Run", type="primary", width="stretch")


@st.cache_data(show_spinner=False)
def load_or_run_default_benchmark() -> pd.DataFrame:
    csv_file = REPORTS_DIR / "ablation_results.csv"
    if csv_file.exists():
        return pd.read_csv(csv_file)
    engine = RAGEvaluationEngine()
    res = engine.run_full_ablation(
        documents=get_corpus_documents(),
        qa_samples=get_benchmark_qa_samples(),
    )
    export_research_reports(res)
    return pd.read_csv(csv_file)


if run_clicked:
    prog_bar = st.progress(0, text="Initializing RAG evaluation pipeline...")
    exp_cfg = EvaluationExperimentConfig(
        router=active_router_cfg,
        top_k=top_k,
        evaluation_mode=eval_mode,
    )
    engine = RAGEvaluationEngine(exp_cfg)

    def _update_prog(done: int, total: int, msg: str) -> None:
        pct = int((done / max(1, total)) * 100)
        prog_bar.progress(min(100, pct), text=f"[{done}/{total}] {msg}")

    if "Fast Ablation" in run_scope:
        emb_list = ["bge_dense_semantic"]
        store_list = ["faiss"]
    else:
        emb_list = ["bge_dense_semantic", "minilm_fast_dense"]
        store_list = ["faiss", "chromadb"]

    results = engine.run_full_ablation(
        documents=get_corpus_documents(),
        qa_samples=get_benchmark_qa_samples(),
        embedding_models=emb_list,
        vector_stores=store_list,
        mode=eval_mode,
        progress_callback=_update_prog,
    )
    export_research_reports(results)
    st.cache_data.clear()
    prog_bar.empty()

df = load_or_run_default_benchmark()
headline = compute_headline_finding(df)

# --- Headline Banner ---
st.success(f"**Empirical Research Finding:** {headline['headline_statement']}")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric(
    "Winning Chunking Strategy",
    headline["best_strategy"],
    f"+{headline['relative_gain_pct']:.1f}% Faithfulness vs. {headline['baseline_strategy']}",
)
kpi2.metric(
    "Best Mean Faithfulness",
    f"{headline['best_faithfulness']:.3f}",
    f"+{headline['absolute_gain_pts']:.1f} pts over baseline",
)
kpi3.metric(
    "Hallucination Rate Reduction",
    f"-{headline['hallucination_reduction_pct']:.1f}%",
    f"{headline['best_hallucination_rate']:.2%} vs. {headline['baseline_hallucination_rate']:.2%}",
)
kpi4.metric(
    "Statistical Significance",
    f"p = {headline['paired_p_value']:.2e}",
    f"Paired t = {headline['paired_t_stat']:.2f}, Cohen's d = {headline['cohens_d']:.2f}",
)

# --- Interactive Plotly Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📈 Faithfulness Across 3 Document Types",
        "🎯 Retrieval Precision vs. Hallucination",
        "⚡ FAISS vs. ChromaDB Latency",
        "🔍 Question-Level Failure Inspector",
        "🧪 Live Side-by-Side Chunking Arena",
    ]
)

with tab1:
    fig1 = px.bar(
        df.groupby(["doc_type_label", "chunking_strategy"], as_index=False)[
            ["faithfulness", "hallucination_rate", "context_precision"]
        ].mean(),
        x="doc_type_label",
        y="faithfulness",
        color="chunking_strategy",
        color_discrete_map=STRATEGY_COLORS,
        barmode="group",
        text_auto=".3f",
        title="Answer Faithfulness Score by Chunking Strategy Across 3 Document Types",
        labels={
            "doc_type_label": "Document Domain",
            "faithfulness": "Mean Faithfulness Score (0 - 1.0)",
            "chunking_strategy": "Chunking Strategy",
        },
    )
    fig1.update_layout(yaxis_range=[0, 1.15], height=480)
    st.plotly_chart(fig1, width="stretch")
    st.subheader("Chunking Strategy Ablation Summary Table")
    st.dataframe(headline["strategy_summary_df"], width="stretch")

with tab2:
    agg_scatter = df.groupby(
        ["chunking_strategy", "embedding_model", "vector_store"], as_index=False
    )[["context_precision", "context_recall", "faithfulness", "hallucination_rate"]].mean()
    fig2 = px.scatter(
        agg_scatter,
        x="context_precision",
        y="hallucination_rate",
        color="chunking_strategy",
        color_discrete_map=STRATEGY_COLORS,
        symbol="embedding_model",
        size="context_recall",
        size_max=22,
        hover_data=["vector_store", "faithfulness"],
        title="Retrieval Context Precision@K vs. Hallucination Rate",
        labels={
            "context_precision": "Retrieval Context Precision@K",
            "hallucination_rate": "Hallucination Rate (1 - Faithfulness)",
        },
    )
    fig2.update_layout(height=480)
    st.plotly_chart(fig2, width="stretch")

with tab3:
    lat_agg = df.groupby(["chunking_strategy", "vector_store", "embedding_model"], as_index=False)[
        ["retrieval_latency_ms", "generation_latency_ms", "total_latency_ms"]
    ].mean()
    fig3 = px.bar(
        lat_agg,
        x="chunking_strategy",
        y="retrieval_latency_ms",
        color="vector_store",
        color_discrete_map={"faiss": "#0ea5e9", "chromadb": "#a855f7"},
        facet_col="embedding_model",
        barmode="group",
        text_auto=".2f",
        title="Retrieval Latency (ms) Across Vector Stores (FAISS vs. ChromaDB) and Embedding Models",
    )
    fig3.update_layout(height=480)
    st.plotly_chart(fig3, width="stretch")

with tab4:
    st.subheader("Root-Cause Failure Diagnostic Inspector")
    selected_q = st.selectbox("Select Evaluation Question", df["question"].unique())
    sub_q = df[df["question"] == selected_q][
        [
            "chunking_strategy",
            "embedding_model",
            "vector_store",
            "context_precision",
            "context_recall",
            "faithfulness",
            "hallucination_rate",
            "failure_reason",
            "missing_facts_str",
            "generated_answer",
        ]
    ]
    st.dataframe(sub_q, width="stretch")

with tab5:
    st.subheader("🧪 Live Side-by-Side Chunking Strategy Comparison Arena")
    st.write(
        "Test any question against all 4 chunking strategies live using `meta-llama/llama-3.3-70b-instruct` "
        "to observe how chunk boundaries affect retrieval context and answer faithfulness."
    )
    preset_samples = get_benchmark_qa_samples()
    sample_options = {f"[{s.doc_type}] {s.question}": s for s in preset_samples}
    chosen_label = st.selectbox("Choose a Benchmark Question (or edit below):", list(sample_options.keys()))
    chosen_sample = sample_options[chosen_label]

    custom_q = st.text_area("Question:", value=chosen_sample.question, height=80)
    if st.button("⚡ Compare All 4 Chunking Strategies Live", type="primary"):
        docs = get_corpus_documents()
        exp_cfg = EvaluationExperimentConfig(
            router=active_router_cfg,
            top_k=top_k,
            evaluation_mode=eval_mode,
        )
        engine = RAGEvaluationEngine(exp_cfg)
        cols = st.columns(2)
        for i, strat in enumerate(ChunkingEngine.STRATEGIES):
            chunks, _ = ChunkingEngine.chunk_corpus(docs, strategy=strat)
            emb = EmbeddingEngine(model_name="bge_dense_semantic", router_config=active_router_cfg)
            retr = VectorStoreRetriever(backend="faiss", embedding_engine=emb)
            retr.index_chunks(chunks)
            temp_sample = BenchmarkQASample(
                sample_id=chosen_sample.sample_id,
                doc_id=chosen_sample.doc_id,
                doc_type=chosen_sample.doc_type,
                question=custom_q,
                ground_truth=chosen_sample.ground_truth,
                required_facts=chosen_sample.required_facts,
            )
            res = engine.evaluate_single_sample(
                sample=temp_sample,
                chunking_strategy=strat,
                embedding_model="bge_dense_semantic",
                vector_store="faiss",
                retriever=retr,
                mode=eval_mode,
            )
            with cols[i % 2]:
                st.markdown(f"#### `{strat}`")
                st.metric(
                    "Faithfulness / Hallucination",
                    f"{res.faithfulness:.3f}",
                    f"Hallucination: {res.hallucination_rate:.1%} | {res.failure_reason}",
                )
                st.info(f"**Generated Answer:**\n\n{res.generated_answer}")
                with st.expander("View Retrieved Top-K Chunks"):
                    for idx, ch in enumerate(res.retrieved_chunks, start=1):
                        st.markdown(f"**Chunk {idx}:**\n```text\n{ch}\n```")
