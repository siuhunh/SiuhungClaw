"""
RAG 检索优化要点（业界常见做法，在代码中落地）：

1. **统一分块**：向量索引与 BM25 使用同一批 `nodes`，避免「向量按整篇、关键词按段」的错位。
2. **分块尺寸与重叠**：较大 chunk + overlap 提升语义完整度；overlap 降低边界截断损失。
3. **提高召回再截断**：vector / BM25 略提高 top_k，融合后再按条数与字数截断，减少漏检。
4. **纯 BM25 路径**：长文档按段落滑窗切块并带 `路径#partN` 伪文档 id，避免整篇仅一个稀疏向量词袋。
5. **索引失效**：若增删 `storage/knowledge/` 后检索异常，删除 `storage/index/knowledge/` 后重启以全量重建向量索引。

可选后续（未内置）：Cross-Encoder 重排、HyDE、多查询 LLM 扩展、Markdown 结构感知解析、查询改写。
"""

import logging
from typing import Any, Callable

from langchain_core.tools import BaseTool

from backend.app.core.config import INDEX_DIR, KNOWLEDGE_DIR

logger = logging.getLogger(__name__)

# --- 可调参数（与业界默认取向一致：偏召回、再靠融合与上下文长度截断） ---
RAG_CHUNK_SIZE = 1024
RAG_CHUNK_OVERLAP = 160
RAG_VECTOR_TOP_K = 8
RAG_BM25_TOP_K = 8
RAG_FUSION_OUTPUT_NODES = 12
RAG_SNIPPET_CHARS = 2000
RAG_BM25_PARAGRAPH_MAX_CHARS = 1600


def _paragraph_chunks(text: str, max_chars: int) -> list[str]:
    """按空行分段再合并，避免 BM25 把整本书当一条稀疏文档。"""
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not parts:
        return [text[:max_chars]] if text.strip() else []
    chunks: list[str] = []
    buf = ""
    for p in parts:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = f"{buf}\n\n{p}" if buf else p
        else:
            if buf:
                chunks.append(buf)
            if len(p) <= max_chars:
                buf = p
            else:
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i : i + max_chars])
                buf = ""
        while len(buf) > max_chars:
            chunks.append(buf[:max_chars])
            buf = buf[max_chars:].lstrip()
    if buf:
        chunks.append(buf)
    return chunks


def _collect_text_docs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if not KNOWLEDGE_DIR.exists():
        return pairs
    for ext in ("*.md", "*.txt", "*.MD", "*.TXT"):
        for path in KNOWLEDGE_DIR.rglob(ext):
            if path.is_file():
                try:
                    rel = str(path.relative_to(KNOWLEDGE_DIR))
                    body = path.read_text(encoding="utf-8", errors="ignore")
                    if len(body) <= RAG_BM25_PARAGRAPH_MAX_CHARS:
                        pairs.append((rel, body))
                    else:
                        for i, ch in enumerate(_paragraph_chunks(body, RAG_BM25_PARAGRAPH_MAX_CHARS)):
                            pairs.append((f"{rel}#part{i}", ch))
                except OSError:
                    continue
    return pairs


def _bm25_search_factory() -> Callable[[str], str]:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:

        def _no_bm25(query: str) -> str:
            return "rank-bm25 not installed; cannot run keyword search."

        return _no_bm25

    pairs = _collect_text_docs()
    if not pairs:
        return lambda q: "Knowledge base is empty. Add .md/.txt under storage/knowledge/."

    tokenized_corpus = [doc.lower().split() for _, doc in pairs]
    bm25 = BM25Okapi(tokenized_corpus)

    def search(query: str) -> str:
        q = query.lower().split()
        if not q:
            return "Empty query."
        scores = bm25.get_scores(q)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
        chunks: list[str] = []
        for i in ranked:
            if scores[i] <= 0:
                continue
            rel, body = pairs[i]
            chunks.append(f"### {rel} (score={scores[i]:.3f})\n{body[:RAG_SNIPPET_CHARS]}")
        return "\n\n".join(chunks) if chunks else "No BM25 hits."

    return search


def _try_build_llama_hybrid(settings: Any) -> Callable[[str], str] | None:
    if not (settings.llm.api_key or "").strip():
        logger.info("No LLM API key; skip LlamaIndex vector index (BM25 fallback still available).")
        return None

    try:
        from llama_index.core import (
            SimpleDirectoryReader,
            VectorStoreIndex,
            StorageContext,
            Settings as LISettings,
            load_index_from_storage,
        )
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core.retrievers import QueryFusionRetriever
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.retrievers.bm25 import BM25Retriever
    except ImportError as e:
        logger.warning("LlamaIndex not available: %s", e)
        return None

    if not KNOWLEDGE_DIR.exists() or not any(KNOWLEDGE_DIR.iterdir()):
        return lambda q: "Knowledge base is empty. Add PDF/MD/TXT under storage/knowledge/."

    persist = str(INDEX_DIR.resolve())
    embed_model_name = (settings.embedding.model or "text-embedding-3-small").strip()
    api_base = (settings.embedding.base_url or settings.llm.base_url or "").strip() or None

    try:
        LISettings.embed_model = OpenAIEmbedding(
            model=embed_model_name,
            api_key=settings.llm.api_key.strip(),
            api_base=api_base,
        )
    except Exception as e:
        logger.warning("LlamaIndex embedding init failed: %s", e)
        return None

    try:
        reader = SimpleDirectoryReader(input_dir=str(KNOWLEDGE_DIR), recursive=True)
        documents = reader.load_data()
        if not documents:
            return lambda q: "No loadable documents in storage/knowledge/."

        splitter = SentenceSplitter(
            chunk_size=RAG_CHUNK_SIZE,
            chunk_overlap=RAG_CHUNK_OVERLAP,
        )
        nodes = splitter.get_nodes_from_documents(documents)
        if not nodes:
            return lambda q: "No chunks after splitting; check storage/knowledge/."

        index: VectorStoreIndex
        try:
            if INDEX_DIR.exists() and any(INDEX_DIR.iterdir()):
                storage_context = StorageContext.from_defaults(persist_dir=persist)
                index = load_index_from_storage(storage_context)
                logger.info(
                    "RAG: loaded vector index from %s (if knowledge files changed, delete this dir to rebuild).",
                    persist,
                )
            else:
                raise FileNotFoundError("no index")
        except Exception:
            sc = StorageContext.from_defaults()
            # 与 BM25 共用同一 nodes，保证向量通道与关键词通道块对齐
            index = VectorStoreIndex(nodes, storage_context=sc, show_progress=False)
            index.storage_context.persist(persist_dir=persist)
            logger.info(
                "RAG: built new vector index (%s chunks) under %s",
                len(nodes),
                persist,
            )

        vector_retriever = index.as_retriever(similarity_top_k=RAG_VECTOR_TOP_K)
        bm25_retriever = None
        try:
            bm25_retriever = BM25Retriever.from_defaults(
                nodes=nodes,
                similarity_top_k=RAG_BM25_TOP_K,
            )
        except Exception as e:
            logger.warning("BM25Retriever init failed: %s", e)

        qe = index.as_query_engine(similarity_top_k=max(RAG_VECTOR_TOP_K, 6))

        def search_qe(query: str) -> str:
            return str(qe.query(query))

        if bm25_retriever is None:
            return search_qe

        try:
            fusion = QueryFusionRetriever(
                [vector_retriever, bm25_retriever],
                num_queries=1,
                mode="reciprocal_rerank",
            )

            def search_fusion(query: str) -> str:
                nodes_out = fusion.retrieve(query)
                parts = []
                for n in nodes_out[:RAG_FUSION_OUTPUT_NODES]:
                    parts.append(n.get_content()[:RAG_SNIPPET_CHARS])
                return "\n\n---\n\n".join(parts) if parts else "No retrieval results."

            return search_fusion
        except Exception as e:
            logger.warning("QueryFusionRetriever unavailable (%s); using vector query engine.", e)
            return search_qe
    except Exception as e:
        logger.warning("LlamaIndex pipeline failed: %s", e)
        return None


def build_knowledge_search_tool(settings: Any) -> BaseTool:
    """§3.9.5 — hybrid when LlamaIndex + embeddings work; else BM25 on text files."""
    hybrid = _try_build_llama_hybrid(settings)
    bm25_only = _bm25_search_factory()
    runner = hybrid or bm25_only

    class SearchKnowledgeBaseTool(BaseTool):
        name: str = "search_knowledge_base"
        description: str = (
            "Search the local knowledge base under storage/knowledge (not chat history). "
            "Uses vector+BM25 fusion when the LlamaIndex index is available; otherwise keyword search."
        )

        def _run(self, query: str, **kwargs: Any) -> str:
            return runner(query.strip() or "")

        async def _arun(self, query: str, **kwargs: Any) -> str:
            return self._run(query, **kwargs)

    return SearchKnowledgeBaseTool()
