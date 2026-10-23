"""
CLI Entrypoint for the Production RAG Evaluation Framework (`run_benchmark.py`).

Usage examples:
  1. Run full ablation immediately (auto-detects Router API key or runs calibrated hybrid benchmark):
     python run_benchmark.py

  2. Run with your Router API key & custom gateway URL:
     python run_benchmark.py --router-api-key "sk-..." --router-base-url "https://openrouter.ai/api/v1" --router-model "openai/gpt-4o-mini" --mode live

  3. Test Router API connectivity only:
     python run_benchmark.py --test-router --router-api-key "sk-..."
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from src.analysis import export_research_reports
from src.chunking import ChunkingEngine
from src.config import EvaluationExperimentConfig, RouterAPIConfig
from src.dataset import (
    get_benchmark_qa_samples,
    get_corpus_documents,
    persist_dataset_artifacts,
)
from src.evaluator import RAGEvaluationEngine
from src.llm_router import RouterLLMClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Production RAG Evaluation Framework — Multi-Dimensional Ablation Runner"
    )
    parser.add_argument(
        "--router-api-key",
        type=str,
        default=None,
        help="Router API Key (overrides ROUTER_API_KEY in .env)",
    )
    parser.add_argument(
        "--router-base-url",
        type=str,
        default=None,
        help="Router API Base URL (e.g., https://openrouter.ai/api/v1)",
    )
    parser.add_argument(
        "--router-model",
        type=str,
        default=None,
        help="Router LLM model identifier (e.g., openai/gpt-4o-mini, anthropic/claude-3.5-sonnet)",
    )
    parser.add_argument(
        "--router-judge-model",
        type=str,
        default=None,
        help="Router Judge LLM model identifier for RAGAS / Faithfulness scoring",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "live", "offline"],
        default="auto",
        help="Evaluation mode: 'auto' (uses Router API if key is set), 'live', or 'offline'",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks retrieved per query (default: 3)",
    )
    parser.add_argument(
        "--embedding-models",
        nargs="+",
        default=["bge_dense_semantic", "minilm_fast_dense"],
        help="Embedding models to evaluate",
    )
    parser.add_argument(
        "--vector-stores",
        nargs="+",
        default=["faiss", "chromadb"],
        help="Vector databases to benchmark (faiss, chromadb)",
    )
    parser.add_argument(
        "--test-router",
        action="store_true",
        help="Test connectivity to the configured Router API and exit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    router_cfg = RouterAPIConfig()
    if args.router_api_key:
        router_cfg.api_key = args.router_api_key
    if args.router_base_url:
        router_cfg.base_url = args.router_base_url
    if args.router_model:
        router_cfg.llm_model = args.router_model
    if args.router_judge_model:
        router_cfg.judge_model = args.router_judge_model

    if args.test_router:
        client = RouterLLMClient(router_cfg)
        status = client.test_connection()
        print("\n=== Router API Connection Test ===")
        for k, v in status.items():
            print(f"  {k}: {v}")
        return 0 if status.get("ok") else 1

    exp_cfg = EvaluationExperimentConfig(
        router=router_cfg,
        top_k=args.top_k,
        evaluation_mode=args.mode,
    )

    print("=" * 88)
    print(" PRODUCTION RAG EVALUATION FRAMEWORK - MULTI-DOMAIN ABLATION STUDY")
    print("=" * 88)
    print(f"  Router Base URL : {router_cfg.base_url}")
    print(f"  Router API Key  : {router_cfg.masked_key()}")
    print(f"  Router LLM Model: {router_cfg.llm_model}")
    print(f"  Resolved Mode   : {exp_cfg.resolve_mode().upper()}")
    print(f"  Top-K Retrieval : {exp_cfg.top_k}")
    print("-" * 88)

    # 1. Persist benchmark dataset artifacts
    ds_info = persist_dataset_artifacts()
    documents = get_corpus_documents()
    qa_samples = get_benchmark_qa_samples()
    print(
        f"Loaded {len(documents)} domain documents ({', '.join(d.doc_type for d in documents)}) "
        f"and {len(qa_samples)} ground-truth multi-clause QA samples."
    )

    # 2. Run ablation matrix
    engine = RAGEvaluationEngine(exp_cfg)
    results = engine.run_full_ablation(
        documents=documents,
        qa_samples=qa_samples,
        chunking_strategies=list(ChunkingEngine.STRATEGIES),
        embedding_models=args.embedding_models,
        vector_stores=args.vector_stores,
        mode=args.mode,
    )

    # 3. Analyze & export reports
    report_bundle = export_research_reports(results)
    headline = report_bundle["headline"]
    summary_df = headline["strategy_summary_df"]

    print("\n=== CHUNKING STRATEGY ABLATION SUMMARY (ACROSS ALL 3 DOCUMENT TYPES) ===")
    print(summary_df.to_string(index=False))

    print("\n=== HEADLINE RESEARCH FINDING ===")
    print(f"  >> {headline['headline_statement']}")

    print("\n=== PER-DOCUMENT-TYPE FAITHFULNESS COMPARISON ===")
    for doc_type, metrics in headline["per_doc_type_gains"].items():
        print(
            f"  - {doc_type:<22}: {headline['best_strategy']} = {metrics['best_faithfulness']:.3f} "
            f"vs. {headline['baseline_strategy']} = {metrics['baseline_faithfulness']:.3f} "
            f"(+{metrics['relative_improvement_pct']:.1f}% relative / +{metrics['absolute_improvement_pts']:.1f} pts)"
        )

    print("\n=== GENERATED ARTIFACTS & DASHBOARD ===")
    print(f"  - Corpus & Ground-Truth QA : {ds_info['qa_dataset_path']}")
    print(f"  - Full Ablation CSV        : {report_bundle['csv_path']}")
    print(f"  - Research Report (MD)     : {report_bundle['markdown_report_path']}")
    print(f"  - Interactive Plotly HTML  : {report_bundle['html_dashboard_path']}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
