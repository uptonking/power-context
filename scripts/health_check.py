#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from typing import Dict, Any

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding

# Ensure /work (repo root) is on sys.path when run from /work/scripts
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from scripts.utils import sanitize_vector_name


def assert_true(cond: bool, msg: str):
    if not cond:
        print(f"[FAIL] {msg}")
        sys.exit(1)
    else:
        print(f"[OK] {msg}")


def main():
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY")
    collection = os.environ.get("COLLECTION_NAME", "my-collection")
    model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

    print(f"Health check -> {qdrant_url} collection={collection} model={model_name}")

    # Init embedding to derive dimension and test embedding
    model = TextEmbedding(model_name=model_name)
    dim = len(next(model.embed(["health dim probe"])))
    vec_name_expect = sanitize_vector_name(model_name)

    client = QdrantClient(url=qdrant_url, api_key=api_key or None)

    # Get all collections and check each one
    try:
        collections_response = client.get_collections()
        collections = [c.name for c in collections_response.collections]
        print(f"Found collections: {collections}")
    except Exception as e:
        print(f"Error getting collections: {e}")
        sys.exit(1)

    if not collections:
        print("No collections found - nothing to health check")
        return

    # Check each collection
    for collection_name in collections:
        print(f"Checking collection: {collection_name}")

        # 1) Collection exists and has expected named vector/dimension
        info = client.get_collection(collection_name)
        cfg = info.config.params.vectors
        if isinstance(cfg, dict):
            present_names = list(cfg.keys())
            assert_true(len(present_names) >= 1, "Collection has at least one named vector")
            assert_true(
                vec_name_expect in present_names,
                f"Expected vector name present: {vec_name_expect} in {present_names}",
            )
            got_dim = cfg[vec_name_expect].size
        else:
            present_names = ["<unnamed>"]
            got_dim = cfg.size
        assert_true(
            got_dim == dim, f"Vector dimension matches embedding ({got_dim} == {dim})"
        )

        # 2) HNSW tuned params (best effort; allow >= thresholds)
        hcfg = info.config.hnsw_config
        try:
            m = getattr(hcfg, "m", None)
            efc = getattr(hcfg, "ef_construct", None)
            assert_true(m is None or m >= 16, f"HNSW m>=16 (got {m})")
            assert_true(efc is None or efc >= 256, f"HNSW ef_construct>=256 (got {efc})")
        except Exception:
            print("[WARN] Could not read HNSW config; continuing")

        # 3) Test queries on this collection
        probe_text = "split code into overlapping line chunks"
        probe_vec = next(model.embed([probe_text])).tolist()

        # Unfiltered query
        qp = client.query_points(
            collection_name=collection_name,
            query=probe_vec,
            using=vec_name_expect,
            limit=3,
            with_payload=True,
            search_params=models.SearchParams(hnsw_ef=128),
        )
        res_points = getattr(qp, "points", qp)
        assert_true(isinstance(res_points, list), "query_points returns a list of points")

        # Filtered by language + kind (should not error; may return 0 results if dataset sparse)
        flt = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.language", match=models.MatchValue(value="python")
                ),
                models.FieldCondition(
                    key="metadata.kind", match=models.MatchValue(value="function")
                ),
            ]
        )
        qp2 = client.query_points(
            collection_name=collection_name,
            query=probe_vec,
            using=vec_name_expect,
            query_filter=flt,
            limit=3,
            with_payload=True,
        )
        res2 = getattr(qp2, "points", qp2) or []
        # If results exist, ensure payload has kind/symbol keys
        if res2:
            md: Dict[str, Any] = (res2[0].payload or {}).get("metadata") or {}
            assert_true(
                "kind" in md and "symbol" in md,
                "payload includes metadata.kind and metadata.symbol",
            )
        else:
            print("[OK] Filtered query ran (no results is acceptable depending on data)")

        print(f"[OK] Collection {collection_name} health check passed")

    print(f"[OK] All {len(collections)} collections passed health check")


if __name__ == "__main__":
    main()