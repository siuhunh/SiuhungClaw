import logging
from datetime import datetime, timezone
from typing import Any

from backend.app.core.config import AppSettings
from backend.app.core.model_factory import build_embedding_model

logger = logging.getLogger("ace_claw")

_store: "MilvusVectorMemoryStore | None" = None


class MilvusVectorMemoryStore:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._cfg = settings.vectordb
        self._embedding = build_embedding_model(settings)
        self._connected = False
        self._collection = None
        self._pymilvus = None
        self._has_memory_kind = False

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled and self._cfg.provider.lower() == "milvus"

    @property
    def supports_memory_kinds(self) -> bool:
        return self._has_memory_kind

    def initialize(self) -> None:
        if not self.enabled:
            logger.info("vector memory disabled by config")
            return
        try:
            from pymilvus import (
                Collection,
                CollectionSchema,
                DataType,
                FieldSchema,
                connections,
                utility,
            )
        except Exception as e:
            logger.warning("pymilvus not available, vector memory disabled: %s", e)
            return

        try:
            self._pymilvus = {
                "Collection": Collection,
                "CollectionSchema": CollectionSchema,
                "DataType": DataType,
                "FieldSchema": FieldSchema,
                "connections": connections,
                "utility": utility,
            }
            uri = (self._cfg.uri or "").strip()
            kwargs: dict[str, Any] = {"alias": "aceclaw_milvus"}
            if uri:
                kwargs["uri"] = uri
            else:
                kwargs["host"] = self._cfg.host
                kwargs["port"] = self._cfg.port
            if self._cfg.token:
                kwargs["token"] = self._cfg.token
            if self._cfg.db_name:
                kwargs["db_name"] = self._cfg.db_name
            connections.connect(**kwargs)

            collection_name = self._cfg.collection_name
            utility_mod = self._pymilvus["utility"]
            if utility_mod.has_collection(collection_name, using="aceclaw_milvus"):
                collection = Collection(collection_name, using="aceclaw_milvus")
                field_names = {f.name for f in collection.schema.fields}
                self._has_memory_kind = "memory_kind" in field_names
                if not self._has_memory_kind:
                    logger.warning(
                        "milvus collection %s has no memory_kind field; using legacy inserts/search. "
                        "Drop the collection or use a new MILVUS_COLLECTION_NAME for short/long split.",
                        collection_name,
                    )
            else:
                collection = self._create_collection(collection_name)
                self._has_memory_kind = True
            self._collection = collection
            self._collection.load()
            self._connected = True
            logger.info("milvus vector memory initialized collection=%s", collection_name)
        except Exception as e:
            logger.warning("milvus init failed, vector memory disabled: %s", e)
            self._connected = False
            self._collection = None

    def _create_collection(self, collection_name: str):
        FieldSchema = self._pymilvus["FieldSchema"]
        DataType = self._pymilvus["DataType"]
        CollectionSchema = self._pymilvus["CollectionSchema"]
        Collection = self._pymilvus["Collection"]

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="ts", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="memory_kind", dtype=DataType.VARCHAR, max_length=16),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self._cfg.embedding_dim),
        ]
        schema = CollectionSchema(fields=fields, description="AceClaw conversation memories (short/long)")
        collection = Collection(
            name=collection_name,
            schema=schema,
            using="aceclaw_milvus",
        )
        try:
            collection.create_index(
                field_name="embedding",
                index_params={
                    "index_type": "AUTOINDEX",
                    "metric_type": self._cfg.metric_type,
                    "params": {},
                },
            )
        except Exception:
            collection.create_index(
                field_name="embedding",
                index_params={
                    "index_type": "HNSW",
                    "metric_type": self._cfg.metric_type,
                    "params": {"M": 16, "efConstruction": 200},
                },
            )
        return collection

    def _embed_query(self, text: str) -> list[float]:
        if hasattr(self._embedding, "embed_query"):
            return list(self._embedding.embed_query(text))
        if hasattr(self._embedding, "embed_documents"):
            return list(self._embedding.embed_documents([text])[0])
        raise RuntimeError("embedding model does not support query embeddings")

    def remember_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        text = f"user: {user_message}\nassistant: {assistant_message}"
        self.remember_text(session_id, text, ts=ts, memory_kind="short")

    def remember_long_fact(self, session_id: str, fact: str) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        self.remember_text(session_id, fact, ts=ts, memory_kind="long")

    def remember_text(
        self,
        session_id: str,
        text: str,
        *,
        ts: str | None = None,
        memory_kind: str = "short",
    ) -> None:
        if not self._connected or self._collection is None:
            return
        stamp = ts or datetime.now(timezone.utc).isoformat()
        if len(text) > 8000:
            text = text[:8000]
        kind = (memory_kind or "short")[:16]
        try:
            vec = self._embed_query(text)
            if self._has_memory_kind:
                self._collection.insert(
                    [
                        [session_id],
                        [stamp],
                        [kind],
                        [text],
                        [vec],
                    ]
                )
            else:
                self._collection.insert([[session_id], [stamp], [text], [vec]])
        except Exception as e:
            logger.warning("milvus remember_text failed: %s", e)

    def search(
        self,
        session_id: str,
        query: str,
        *,
        memory_kinds: list[str] | None = None,
        limit: int | None = None,
    ) -> list[str]:
        if not self._connected or self._collection is None:
            return []
        if not query.strip():
            return []
        lim = limit if limit is not None else max(1, self._cfg.top_k)
        try:
            vec = self._embed_query(query)
            if self._has_memory_kind and memory_kinds:
                kinds = [k[:16] for k in memory_kinds if k]
                kind_expr = " or ".join(f'memory_kind == "{k}"' for k in kinds)
                expr = f'session_id == "{session_id}" and ({kind_expr})'
            else:
                expr = f'session_id == "{session_id}"'
            out_fields = ["text", "ts", "memory_kind"] if self._has_memory_kind else ["text", "ts"]
            results = self._collection.search(
                data=[vec],
                anns_field="embedding",
                param={"metric_type": self._cfg.metric_type, "params": {"nprobe": 10}},
                limit=max(1, lim),
                expr=expr,
                output_fields=out_fields,
            )
            out: list[str] = []
            if results:
                for hit in results[0]:
                    ent = hit.entity
                    text = ent.get("text") if ent else ""
                    ts = ent.get("ts") if ent else ""
                    mk = ent.get("memory_kind", "") if ent else ""
                    if text:
                        if mk:
                            out.append(f"[{mk}][{ts}] {text}")
                        else:
                            out.append(f"[{ts}] {text}")
            return out
        except Exception as e:
            logger.warning("milvus search failed: %s", e)
            return []


def init_vector_memory(settings: AppSettings) -> None:
    global _store
    _store = MilvusVectorMemoryStore(settings)
    _store.initialize()


def get_vector_memory() -> MilvusVectorMemoryStore | None:
    return _store
