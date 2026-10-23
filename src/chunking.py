"""
Chunking Strategies Module (`src/chunking.py`).
Implements four distinct document chunking architectures for RAG ablation studies:
  1. `fixed_window`: Fixed-size sliding character window (high boundary fragmentation)
  2. `recursive_markdown`: LangChain-compatible recursive structural splitter
  3. `sentence_window`: Sentence-boundary sliding window with +/-1 context expansion
  4. `hierarchical_parent_child`: Fine-grained child indexing mapped to full section/parent context
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from src.dataset import DocumentCorpusItem


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    doc_type: str
    strategy: str
    search_text: str      # Text embedded for vector similarity search
    retrieval_text: str   # Context payload passed to the LLM upon retrieval
    char_length: int
    metadata: Dict[str, str] = field(default_factory=dict)


class ChunkingEngine:
    """Executes and benchmarks different chunking strategies across corpus documents."""

    STRATEGIES = (
        "fixed_window",
        "recursive_markdown",
        "sentence_window",
        "hierarchical_parent_child",
    )

    STRATEGY_LABELS = {
        "fixed_window": "Fixed-Window (260c / 25c overlap)",
        "recursive_markdown": "Recursive Markdown-Aware (480c / 60c overlap)",
        "sentence_window": "Sentence-Window (3-Sent Sliding Context)",
        "hierarchical_parent_child": "Hierarchical Parent-Child (Clause -> Section)",
    }

    @classmethod
    def chunk_corpus(
        cls,
        documents: List[DocumentCorpusItem],
        strategy: str,
    ) -> Tuple[List[DocumentChunk], float]:
        """
        Chunks all documents using the specified strategy.
        Returns `(chunks, chunking_latency_ms)`.
        """
        if strategy not in cls.STRATEGIES:
            raise ValueError(f"Unknown chunking strategy '{strategy}'. Valid: {cls.STRATEGIES}")

        t0 = time.perf_counter()
        all_chunks: List[DocumentChunk] = []

        for doc in documents:
            if strategy == "fixed_window":
                doc_chunks = cls._chunk_fixed_window(doc, chunk_size=260, overlap=25)
            elif strategy == "recursive_markdown":
                doc_chunks = cls._chunk_recursive_markdown(doc, chunk_size=480, overlap=60)
            elif strategy == "sentence_window":
                doc_chunks = cls._chunk_sentence_window(doc, window_sentences=3, stride=2)
            elif strategy == "hierarchical_parent_child":
                doc_chunks = cls._chunk_hierarchical_parent_child(doc)
            else:
                doc_chunks = []
            all_chunks.extend(doc_chunks)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return all_chunks, round(latency_ms, 3)

    @staticmethod
    def _chunk_fixed_window(
        doc: DocumentCorpusItem,
        chunk_size: int = 260,
        overlap: int = 25,
    ) -> List[DocumentChunk]:
        """
        Naive fixed-character sliding window. Often cuts sentences, numbers, and
        conditional qualifiers mid-clause.
        """
        text = doc.content
        chunks: List[DocumentChunk] = []
        step = max(1, chunk_size - overlap)
        idx = 0
        chunk_num = 0
        while idx < len(text):
            piece = text[idx : idx + chunk_size].strip()
            if piece:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc.doc_id}_fixed_{chunk_num}",
                        doc_id=doc.doc_id,
                        doc_type=doc.doc_type,
                        strategy="fixed_window",
                        search_text=piece,
                        retrieval_text=piece,
                        char_length=len(piece),
                        metadata={"title": doc.title, "chunk_index": str(chunk_num)},
                    )
                )
                chunk_num += 1
            idx += step
        return chunks

    @staticmethod
    def _chunk_recursive_markdown(
        doc: DocumentCorpusItem,
        chunk_size: int = 480,
        overlap: int = 60,
    ) -> List[DocumentChunk]:
        """
        Uses LangChain's RecursiveCharacterTextSplitter (or structural fallback)
        with markdown-aware separators.
        """
        pieces: List[str] = []
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                separators=["\n## ", "\n\n", "\n", ". ", " "],
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                keep_separator=True,
            )
            pieces = [p.strip() for p in splitter.split_text(doc.content) if p.strip()]
        except ImportError:
            # Pure-Python recursive structural splitter
            raw_paras = re.split(r"\n(?=#+|\n)", doc.content)
            current = ""
            for para in raw_paras:
                para = para.strip()
                if not para:
                    continue
                if len(current) + len(para) + 2 <= chunk_size:
                    current = f"{current}\n\n{para}".strip()
                else:
                    if current:
                        pieces.append(current)
                    if len(para) <= chunk_size:
                        current = para
                    else:
                        # Split long paragraph by sentence
                        sents = re.split(r"(?<=[.!?])\s+", para)
                        temp = ""
                        for s in sents:
                            if len(temp) + len(s) + 1 <= chunk_size:
                                temp = f"{temp} {s}".strip()
                            else:
                                if temp:
                                    pieces.append(temp)
                                temp = s
                        current = temp
            if current:
                pieces.append(current)

        return [
            DocumentChunk(
                chunk_id=f"{doc.doc_id}_rec_{i}",
                doc_id=doc.doc_id,
                doc_type=doc.doc_type,
                strategy="recursive_markdown",
                search_text=piece,
                retrieval_text=piece,
                char_length=len(piece),
                metadata={"title": doc.title, "chunk_index": str(i)},
            )
            for i, piece in enumerate(pieces)
        ]

    @staticmethod
    def _chunk_sentence_window(
        doc: DocumentCorpusItem,
        window_sentences: int = 3,
        stride: int = 2,
    ) -> List[DocumentChunk]:
        """
        Splits document into complete sentences (preserving section titles) and constructs
        overlapping multi-sentence windows so every claim retains local sentence context.
        """
        lines = doc.content.splitlines()
        sentences: List[str] = []
        current_header = doc.title

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                current_header = stripped.lstrip("#").strip()
                continue
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", stripped) if s.strip()]
            for s in sents:
                sentences.append(f"[{current_header}] {s}")

        chunks: List[DocumentChunk] = []
        idx = 0
        chunk_num = 0
        while idx < len(sentences):
            window = sentences[idx : idx + window_sentences]
            if not window:
                break
            expanded = sentences[max(0, idx - 1) : min(len(sentences), idx + window_sentences + 1)]
            search_txt = " ".join(window)
            retrieval_txt = " ".join(expanded)
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc.doc_id}_sentwin_{chunk_num}",
                    doc_id=doc.doc_id,
                    doc_type=doc.doc_type,
                    strategy="sentence_window",
                    search_text=search_txt,
                    retrieval_text=retrieval_txt,
                    char_length=len(retrieval_txt),
                    metadata={"title": doc.title, "chunk_index": str(chunk_num)},
                )
            )
            chunk_num += 1
            idx += max(1, stride)
        return chunks

    @staticmethod
    def _chunk_hierarchical_parent_child(
        doc: DocumentCorpusItem,
    ) -> List[DocumentChunk]:
        """
        Splits document into semantic parent sections (`## ...`), then indexes fine-grained
        child paragraphs/clauses (`search_text`) that resolve back to the full parent
        section (`retrieval_text`). Eliminates orphaned footnotes and split conditional rules.
        """
        raw_sections = re.split(r"(?=\n## )", "\n" + doc.content)
        chunks: List[DocumentChunk] = []
        chunk_num = 0

        for sec in raw_sections:
            sec_clean = sec.strip()
            if not sec_clean:
                continue
            lines = sec_clean.splitlines()
            header = lines[0].lstrip("#").strip() if lines else doc.title
            parent_context = f"[{doc.title} — {header}]\n{sec_clean}"

            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", sec_clean) if p.strip()]
            for para in paragraphs:
                if para.startswith("#") and len(paragraphs) > 1:
                    continue
                child_search = f"{header}: {para}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc.doc_id}_hier_{chunk_num}",
                        doc_id=doc.doc_id,
                        doc_type=doc.doc_type,
                        strategy="hierarchical_parent_child",
                        search_text=child_search,
                        retrieval_text=parent_context,
                        char_length=len(parent_context),
                        metadata={
                            "title": doc.title,
                            "section": header,
                            "chunk_index": str(chunk_num),
                        },
                    )
                )
                chunk_num += 1
        return chunks
