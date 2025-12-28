"""Unit tests for pattern_search functionality.

Tests that:
1. Pattern extraction produces consistent signatures
2. Cross-language patterns have high similarity
3. Code vs NL detection works correctly
"""
import pytest

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


def test_cross_language_pattern_similarity():
    """Prove that Python retry pattern is structurally similar to Go retry."""
    from scripts.pattern_detection import PatternExtractor, PatternEncoder
    import numpy as np

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


def test_pattern_vector_dimensions():
    """Test that pattern vectors have correct dimensions."""
    from scripts.pattern_detection import PatternExtractor, PatternEncoder

    extractor = PatternExtractor()
    encoder = PatternEncoder()

    sig = extractor.extract(RETRY_PATTERN_PYTHON, "python")
    vec = encoder.encode(sig)

    assert len(vec) == 64, f"Expected 64-dim vector, got {len(vec)}"
    assert all(isinstance(v, float) for v in vec), "All values should be floats"


def test_code_vs_nl_detection():
    """Test auto-detection of code vs natural language queries."""
    from scripts.mcp_impl.pattern_search import _looks_like_code

    # Code examples should be detected as code
    assert _looks_like_code("for i in range(3): try: pass except: pass")
    assert _looks_like_code("if err != nil { return err }")
    assert _looks_like_code("def foo(): return 42")

    # NL descriptions should not be detected as code
    assert not _looks_like_code("retry with exponential backoff")
    assert not _looks_like_code("find error handling patterns")
    assert not _looks_like_code("resource cleanup code")

