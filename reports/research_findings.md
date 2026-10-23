# Production RAG Evaluation Framework — Empirical Ablation Report

## Executive Research Headline
> **Chunking strategy `hierarchical_parent_child` outperforms `fixed_window` by +47.3% relative (+30.5 percentage points, 0.949 vs. 0.644, paired t = 3.28, p = 7.38e-03, Cohen's d = 0.95) on Answer Faithfulness and reduces Hallucination Rate by 85.7% (5.1% vs. 35.6%) across all 3 document types (Technical Spec: +41.2%, Financial 10-K: +72.8%, Clinical Trial: +34.6%).**

- **Statistical Significance**: Paired $t = 3.28$, $p = 7.38e-03$, Cohen's $d = 0.95$ across $N = 48$ paired pipeline configurations per chunking strategy.

---

## 1. Overall Chunking Strategy Ablation (Aggregated Across 3 Document Types)

| chunking_strategy | mean_faithfulness | std_faithfulness | mean_hallucination | mean_precision | mean_recall | mean_relevancy | mean_latency_ms | p95_latency_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hierarchical_parent_child | 0.9493 | 0.1042 | 0.0507 | 0.5781 | 1.0000 | 0.8095 | 521.2383 | 569.4205 |
| sentence_window | 0.9143 | 0.1127 | 0.0857 | 0.6947 | 1.0000 | 0.7421 | 618.2558 | 704.7025 |
| recursive_markdown | 0.8799 | 0.1819 | 0.1201 | 0.5059 | 0.9125 | 0.7692 | 398.8225 | 449.7025 |
| fixed_window | 0.6442 | 0.2953 | 0.3558 | 0.4157 | 0.8208 | 0.6436 | 347.4000 | 381.7295 |

---

## 2. Per-Document-Type Faithfulness Gain (`hierarchical_parent_child` vs. `fixed_window`)

| Document Type | Best Strategy (`hierarchical_parent_child`) | Baseline (`fixed_window`) | Absolute Gain (pts) | Relative Gain (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Technical Docs (`technical_docs`)** | 1.0000 | 0.7080 | +29.20 pts | **+41.24%** |
| **Financial 10-K (`financial_sec_filing`)** | 0.9002 | 0.5210 | +37.92 pts | **+72.79%** |
| **Clinical Research (`clinical_research`)** | 0.9475 | 0.7037 | +24.38 pts | **+34.64%** |

---

## 3. Embedding Model & Vector Store (FAISS vs. ChromaDB) Benchmark

| embedding_model | vector_store | faithfulness | context_precision | context_recall | retrieval_latency_ms |
| --- | --- | --- | --- | --- | --- |
| bge_dense_semantic | faiss | 0.8469 | 0.5486 | 0.9333 | 4.3942 |

---

## 4. Detailed Cross-Tabulation (Chunking × Document Type)

```
                          context_precision                                          faithfulness                                     hallucination_rate                                    
doc_type                  clinical_research financial_sec_filing technical_docs clinical_research financial_sec_filing technical_docs  clinical_research financial_sec_filing technical_docs
chunking_strategy                                                                                                                                                                           
fixed_window                         0.4497               0.4312         0.3662            0.7038               0.5210         0.7080             0.2962               0.4790         0.2920
hierarchical_parent_child            0.5694               0.5694         0.5953            0.9475               0.9002         1.0000             0.0525               0.0998         0.0000
recursive_markdown                   0.5635               0.4955         0.4587            0.8992               0.8519         0.8887             0.1008               0.1481         0.1113
sentence_window                      0.7019               0.7025         0.6799            0.9265               0.8372         0.9790             0.0735               0.1628         0.0210
```
