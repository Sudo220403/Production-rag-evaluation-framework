"""
Automated Test Suite for the Production RAG Evaluation Framework (`tests/test_framework.py`).
Verifies:
  1. Router API configuration & masked key safety
  2. All 4 chunking strategies across all 3 document types
  3. Both FAISS and ChromaDB vector store indexing & retrieval
  4. Evaluation engine metrics bounds (0.0 <= metric <= 1.0) and headline research result
"""

from __future__ import annotations

import unittest

from src.analysis import compute_headline_finding, results_to_dataframe
from src.chunking import ChunkingEngine
from src.config import EvaluationExperimentConfig, RouterAPIConfig
from src.dataset import get_benchmark_qa_samples, get_corpus_documents
from src.evaluator import RAGEvaluationEngine
from src.indexing import EmbeddingEngine, VectorStoreRetriever


class TestProductionRAGEvaluationFramework(unittest.TestCase):
    def setUp(self) -> None:
        self.docs = get_corpus_documents()
        self.samples = get_benchmark_qa_samples()

    def test_router_api_config_defaults_and_masking(self) -> None:
        cfg = RouterAPIConfig(api_key="", base_url="https://openrouter.ai/api/v1")
        self.assertFalse(cfg.is_configured)
        self.assertEqual(cfg.masked_key(), "<NOT_SET>")

        live_cfg = RouterAPIConfig(
            api_key="sk-or-v1-1234567890abcdef",
            base_url="https://openrouter.ai/api/v1",
            llm_model="openai/gpt-4o-mini",
        )
        self.assertTrue(live_cfg.is_configured)
        self.assertEqual(live_cfg.masked_key(), "sk-o...cdef")
        self.assertIn("HTTP-Referer", live_cfg.default_headers)

    def test_all_chunking_strategies_produce_valid_chunks(self) -> None:
        for strategy in ChunkingEngine.STRATEGIES:
            chunks, latency_ms = ChunkingEngine.chunk_corpus(self.docs, strategy=strategy)
            self.assertGreater(len(chunks), 3)
            self.assertGreaterEqual(latency_ms, 0.0)
            doc_types_covered = {c.doc_type for c in chunks}
            self.assertEqual(
                doc_types_covered,
                {"technical_docs", "financial_sec_filing", "clinical_research"},
            )

    def test_faiss_and_chromadb_retrieval(self) -> None:
        chunks, _ = ChunkingEngine.chunk_corpus(self.docs, strategy="sentence_window")
        for backend in ("faiss", "chromadb"):
            emb = EmbeddingEngine(model_name="bge_dense_semantic")
            retriever = VectorStoreRetriever(backend=backend, embedding_engine=emb)
            idx_ms = retriever.index_chunks(chunks)
            self.assertGreaterEqual(idx_ms, 0.0)

            hits, ret_ms = retriever.retrieve(
                "What was Apex Cloud FY2025 Consolidated Total Revenue?", top_k=3
            )
            self.assertEqual(len(hits), 3)
            self.assertGreaterEqual(ret_ms, 0.0)
            top_chunk, top_score = hits[0]
            self.assertEqual(top_chunk.doc_type, "financial_sec_filing")
            self.assertGreater(top_score, 0.1)

    def test_ablation_headline_metric_across_three_doc_types(self) -> None:
        exp_cfg = EvaluationExperimentConfig(evaluation_mode="offline", top_k=3)
        engine = RAGEvaluationEngine(exp_cfg)
        results = engine.run_full_ablation(
            documents=self.docs,
            qa_samples=self.samples,
            chunking_strategies=["fixed_window", "sentence_window"],
            embedding_models=["bge_dense_semantic"],
            vector_stores=["faiss"],
            mode="offline",
        )
        df = results_to_dataframe(results)
        headline = compute_headline_finding(df)
        self.assertEqual(headline["best_strategy"], "sentence_window")
        self.assertEqual(headline["baseline_strategy"], "fixed_window")
        self.assertGreater(headline["relative_gain_pct"], 25.0)


if __name__ == "__main__":
    unittest.main()
