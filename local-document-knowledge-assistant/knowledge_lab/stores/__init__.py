"""Vector adapters accept explicit embeddings; none invoke embedding models."""

import json
import re

import httpx
import numpy as np

from ..config import LabError, Settings
from ..repository import Repository


class SQLiteStore:
    def __init__(self, namespace, settings, repository, index_type):
        self.namespace, self.repository = namespace, repository

    def build(self, ids, vectors):
        # Canonical vectors are committed to the chunks table by the service.
        pass

    def search(self, vector, limit):
        rows = self.repository.chunks(self.namespace.removeprefix("ldk_"))
        if not rows:
            return []
        scores = np.asarray([r["vector"] for r in rows], dtype=np.float32) @ vector
        order = np.argsort(-scores, kind="stable")[:limit]
        return [(rows[i]["chunk"]["id"], float(scores[i])) for i in order]

    def drop(self):
        pass


class FaissStore:
    def __init__(self, namespace, settings, repository, index_type):
        try:
            import faiss
        except ImportError as exc:
            raise LabError("Install the 'faiss' extra to use FAISS.", 503) from exc
        self.faiss, self.index_type = faiss, index_type
        directory = settings.data_dir / "faiss"
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{namespace}.index"
        self.ids_path = directory / f"{namespace}.json"

    def build(self, ids, vectors):
        dimensions = vectors.shape[1]
        if self.index_type == "hnsw":
            index = self.faiss.IndexHNSWFlat(dimensions, 32, self.faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 100
            index.hnsw.efSearch = 64
        else:
            index = self.faiss.IndexFlatIP(dimensions)
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self.faiss.write_index(index, str(self.path))
        self.ids_path.write_text(json.dumps(ids))

    def search(self, vector, limit):
        index = self.faiss.read_index(str(self.path))
        ids = json.loads(self.ids_path.read_text())
        if self.index_type == "hnsw":
            index.hnsw.efSearch = max(64, limit)
        scores, indices = index.search(np.asarray([vector], dtype=np.float32), min(limit, len(ids)))
        return [
            (ids[int(i)], float(s)) for i, s in zip(indices[0], scores[0], strict=True) if i >= 0
        ]

    def drop(self):
        self.path.unlink(missing_ok=True)
        self.ids_path.unlink(missing_ok=True)


class ChromaStore:
    def __init__(self, namespace, settings, repository, index_type):
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as exc:
            raise LabError("Install the 'chroma' extra to use Chroma.", 503) from exc
        self.namespace = namespace
        self.client = chromadb.PersistentClient(
            path=str(settings.data_dir / "chroma"),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    def build(self, ids, vectors):
        collection = self.client.create_collection(
            self.namespace, embedding_function=None, configuration={"hnsw": {"space": "cosine"}}
        )
        for offset in range(0, len(ids), 1000):
            collection.add(
                ids=ids[offset : offset + 1000], embeddings=vectors[offset : offset + 1000].tolist()
            )

    def search(self, vector, limit):
        collection = self.client.get_collection(self.namespace, embedding_function=None)
        result = collection.query(
            query_embeddings=[vector.tolist()],
            n_results=min(limit, collection.count()),
            include=["distances"],
        )
        return [
            (identifier, 1 - float(distance))
            for identifier, distance in zip(result["ids"][0], result["distances"][0], strict=True)
        ]

    def drop(self):
        existing = {c if isinstance(c, str) else c.name for c in self.client.list_collections()}
        if self.namespace in existing:
            self.client.delete_collection(self.namespace)


class QdrantStore:
    """REST API keeps this adapter small and never invokes Qdrant's inference APIs."""

    def __init__(self, namespace, settings, repository, index_type):
        self.namespace, self.settings = namespace, settings

    def request(self, method, path="", body=None, missing_ok=False):
        with httpx.Client(timeout=self.settings.request_timeout, trust_env=False) as client:
            response = client.request(
                method, f"{self.settings.qdrant_url}/collections/{self.namespace}{path}", json=body
            )
            if missing_ok and response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()["result"]

    def build(self, ids, vectors):
        self.request("PUT", body={"vectors": {"size": vectors.shape[1], "distance": "Cosine"}})
        for offset in range(0, len(ids), 256):
            self.request(
                "PUT",
                "/points?wait=true",
                {
                    "points": [
                        {"id": key, "vector": vector.tolist()}
                        for key, vector in zip(
                            ids[offset : offset + 256], vectors[offset : offset + 256], strict=True
                        )
                    ]
                },
            )

    def search(self, vector, limit):
        result = self.request(
            "POST",
            "/points/query",
            {
                "query": vector.tolist(),
                "limit": limit,
                "with_payload": False,
                "params": {"hnsw_ef": max(64, limit)},
            },
        )
        return [(str(p["id"]), float(p["score"])) for p in result["points"]]

    def drop(self):
        self.request("DELETE", missing_ok=True)


class PgvectorStore:
    def __init__(self, namespace, settings, repository, index_type):
        try:
            import psycopg
        except ImportError as exc:
            raise LabError(
                "Install the 'pgvector' extra to use PostgreSQL + pgvector.", 503
            ) from exc
        self.driver, self.dsn = psycopg, settings.postgres_dsn
        self.namespace, self.index_type = namespace, index_type

    @staticmethod
    def literal(vector):
        return "[" + ",".join(str(float(x)) for x in vector) + "]"

    def build(self, ids, vectors):
        dimensions = vectors.shape[1]
        if self.index_type == "hnsw" and dimensions > 2000:
            raise LabError("pgvector HNSW vector indexes support at most 2000 dimensions here.")
        with self.driver.connect(self.dsn, connect_timeout=5) as db, db.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(
                f"CREATE TABLE {self.namespace} "
                f"(id TEXT PRIMARY KEY, embedding vector({dimensions}))"
            )
            cursor.executemany(
                f"INSERT INTO {self.namespace} VALUES (%s, %s::vector)",
                [(key, self.literal(v)) for key, v in zip(ids, vectors, strict=True)],
            )
            if self.index_type == "hnsw":
                cursor.execute(
                    f"CREATE INDEX ON {self.namespace} "
                    "USING hnsw (embedding vector_cosine_ops) "
                    "WITH (m=32, ef_construction=100)"
                )

    def search(self, vector, limit):
        query = self.literal(vector)
        with self.driver.connect(self.dsn, connect_timeout=5) as db, db.cursor() as cursor:
            if self.index_type == "hnsw":
                cursor.execute(f"SET LOCAL hnsw.ef_search = {max(64, limit)}")
            cursor.execute(
                f"SELECT id, 1 - (embedding <=> %s::vector) FROM {self.namespace} "
                "ORDER BY embedding <=> %s::vector LIMIT %s",
                (query, query, limit),
            )
            return [(row[0], float(row[1])) for row in cursor.fetchall()]

    def drop(self):
        with self.driver.connect(self.dsn, connect_timeout=5) as db:
            db.execute(f"DROP TABLE IF EXISTS {self.namespace}")


ADAPTERS = {
    "sqlite": SQLiteStore,
    "faiss": FaissStore,
    "chroma": ChromaStore,
    "qdrant": QdrantStore,
    "pgvector": PgvectorStore,
}


def make_store(index: dict, settings: Settings, repository: Repository):
    namespace = "ldk_" + index["id"]
    # These identifiers are server-generated. Still validate before interpolating SQL/paths.
    if not re.fullmatch(r"ldk_[a-f0-9]{32}", namespace):
        raise LabError("Invalid internal index identifier.")
    config = index["config"]
    return ADAPTERS[config["store"]](namespace, settings, repository, config["vector_index"])
