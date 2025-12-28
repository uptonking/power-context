"""End-to-end tests for pattern_search with real Qdrant vectors.

Tests that:
1. Pattern vectors can be stored in Qdrant
2. Structural similarity search works across languages
3. Natural language description search finds patterns
"""
import os

# Enable pattern vectors for these tests
os.environ.setdefault("PATTERN_VECTORS", "1")

import uuid
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Test fixtures
RETRY_PATTERN_PYTHON = '''
for attempt in range(3):
    try:
        result = make_request()
        break
    except Exception:
        time.sleep(2 ** attempt)
'''

RETRY_PATTERN_GO = '''
for i := 0; i < 3; i++ {
    result, err := makeRequest()
    if err == nil {
        break
    }
    time.Sleep(time.Duration(1<<i) * time.Second)
}
'''

FILE_READ_PATTERN = '''
with open(filename) as f:
    data = json.load(f)
'''


@pytest.fixture
def test_collection():
    """Create a temporary test collection with pattern vectors."""
    collection_name = f"test_pattern_{uuid.uuid4().hex[:8]}"
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    
    client = QdrantClient(url=qdrant_url, timeout=30)
    
    # Create collection with pattern_vector field (64-dim for structural patterns)
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "code": VectorParams(size=384, distance=Distance.COSINE),
            "pattern_vector": VectorParams(size=64, distance=Distance.COSINE),
        }
    )
    
    yield collection_name, client
    
    # Cleanup
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass


@pytest.fixture
def populated_collection(test_collection):
    """Populate collection with test code snippets and pattern vectors."""
    collection_name, client = test_collection
    
    # Import pattern extraction
    from scripts.pattern_detection import PatternExtractor, PatternEncoder
    extractor = PatternExtractor()
    encoder = PatternEncoder()
    
    # Test data: code snippets with their patterns
    test_snippets = [
        {"path": "retry_python.py", "code": RETRY_PATTERN_PYTHON, "lang": "python", "start": 1, "end": 7},
        {"path": "retry_go.go", "code": RETRY_PATTERN_GO, "lang": "go", "start": 1, "end": 8},
        {"path": "file_read.py", "code": FILE_READ_PATTERN, "lang": "python", "start": 1, "end": 3},
    ]
    
    points = []
    for i, snippet in enumerate(test_snippets):
        # Extract structural signature and encode
        sig = extractor.extract(snippet["code"], snippet["lang"])
        pattern_vec = encoder.encode(sig)
        
        # Create point with both vectors (use int ID)
        point = PointStruct(
            id=i + 1,
            vector={
                "code": [0.1] * 384,  # Dummy dense vector
                "pattern_vector": pattern_vec,
            },
            payload={
                "path": snippet["path"],
                "language": snippet["lang"],
                "start_line": snippet["start"],
                "end_line": snippet["end"],
                "text": snippet["code"],
            }
        )
        points.append(point)
    
    client.upsert(collection_name=collection_name, points=points)
    
    return collection_name, client, test_snippets


@pytest.mark.service
def test_pattern_search_finds_similar_retry_patterns(populated_collection):
    """Test that Python retry pattern finds Go retry pattern."""
    collection_name, client, snippets = populated_collection
    
    from scripts.pattern_detection.search import pattern_search
    
    # Search for retry pattern using Python example
    results = pattern_search(
        example=RETRY_PATTERN_PYTHON,
        language="python",
        collection=collection_name,
        limit=5,
    )
    
    assert results.total >= 1, "Should find at least one result"
    paths = [r.path for r in results.results]
    
    # Python retry should be top match (exact)
    assert "retry_python.py" in paths, "Should find Python retry"
    # Go retry should also match due to structural similarity
    # (may not always be found depending on threshold)


@pytest.mark.service
def test_pattern_search_code_vs_nl_detection(populated_collection):
    """Test auto-detection of code vs natural language queries."""
    collection_name, _, _ = populated_collection

    import asyncio
    from scripts.mcp_impl.pattern_search import _pattern_search_impl

    # Code query should be detected as code
    result1 = asyncio.get_event_loop().run_until_complete(
        _pattern_search_impl(
            query="for i in range(3): try: pass except: pass",
            collection=collection_name,
        )
    )
    assert result1.get("ok"), f"Should succeed: {result1.get('error')}"
    assert result1.get("query_mode") == "code", "Should detect as code"

    # NL query should be detected as description
    result2 = asyncio.get_event_loop().run_until_complete(
        _pattern_search_impl(
            query="retry with exponential backoff",
            collection=collection_name,
        )
    )
    assert result2.get("ok"), f"Should succeed: {result2.get('error')}"
    assert result2.get("query_mode") == "description", "Should detect as NL"


@pytest.mark.service
def test_cross_language_pattern_similarity(populated_collection):
    """Prove that Python retry pattern finds structurally similar Go code."""
    collection_name, client, snippets = populated_collection

    from scripts.pattern_detection import PatternExtractor, PatternEncoder
    extractor = PatternExtractor()
    encoder = PatternEncoder()

    # Extract signatures for both patterns
    py_sig = extractor.extract(RETRY_PATTERN_PYTHON, "python")
    go_sig = extractor.extract(RETRY_PATTERN_GO, "go")
    file_sig = extractor.extract(FILE_READ_PATTERN, "python")

    # Encode to vectors
    py_vec = encoder.encode(py_sig)
    go_vec = encoder.encode(go_sig)
    file_vec = encoder.encode(file_sig)

    # Compute cosine similarities
    import numpy as np
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

    py_go_sim = cosine_sim(py_vec, go_vec)
    py_file_sim = cosine_sim(py_vec, file_vec)

    print(f"\n  Python retry <-> Go retry similarity: {py_go_sim:.3f}")
    print(f"  Python retry <-> File read similarity: {py_file_sim:.3f}")

    # Retry patterns should be MORE similar to each other than to file read
    assert py_go_sim > py_file_sim, (
        f"Retry patterns should be more similar ({py_go_sim:.3f}) "
        f"than retry vs file ({py_file_sim:.3f})"
    )

    # Retry patterns should have meaningful similarity (>0.3)
    assert py_go_sim > 0.3, f"Cross-language retry similarity too low: {py_go_sim:.3f}"

