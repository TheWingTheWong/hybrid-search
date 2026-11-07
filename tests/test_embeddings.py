import numpy as np
import pytest

from hybrid_search.embeddings import HashingEmbedder, create_backend, load_backend

TEXTS = [
    "Raft elects a leader with randomized election timeouts",
    "Bloom filters answer membership queries with false positives",
    "Consistent hashing places nodes and keys on a ring",
    "Exponential backoff with jitter spreads out retries",
]


@pytest.fixture(scope="module")
def emb() -> HashingEmbedder:
    return HashingEmbedder().fit(TEXTS)


def test_vectors_are_unit_norm_and_shaped(emb: HashingEmbedder) -> None:
    v = emb.embed(TEXTS)
    assert v.shape == (4, 512) and v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


def test_empty_text_embeds_to_zero(emb: HashingEmbedder) -> None:
    assert not np.any(emb.embed([""]))


def test_deterministic_across_instances(emb: HashingEmbedder) -> None:
    other = HashingEmbedder().fit(TEXTS)
    assert np.array_equal(emb.embed(TEXTS), other.embed(TEXTS))


def test_typos_stay_close(emb: HashingEmbedder) -> None:
    docs = emb.embed(TEXTS)
    q = emb.embed_query("bloom filtr membrship")
    sims = docs @ q
    assert int(np.argmax(sims)) == 1
    assert sims[1] > 2 * np.delete(sims, 1).max()


def test_retrieval_picks_topical_doc(emb: HashingEmbedder) -> None:
    docs = emb.embed(TEXTS)
    for i, q in enumerate(["leader election", "bloom filter", "hash ring", "retry backoff"]):
        assert int(np.argmax(docs @ emb.embed_query(q))) == i


def test_fit_learns_idf(emb: HashingEmbedder) -> None:
    assert not np.allclose(emb.idf, 1.0)
    assert emb.idf.min() >= 1.0


def test_state_round_trip(emb: HashingEmbedder) -> None:
    clone = load_backend(emb.name, emb.config(), emb.arrays())
    assert np.array_equal(clone.embed(TEXTS), emb.embed(TEXTS))


def test_projection_is_seeded() -> None:
    a, b = HashingEmbedder(seed=1), HashingEmbedder(seed=2)
    assert not np.array_equal(a.embed(["raft"]), b.embed(["raft"]))


def test_create_backend_specs() -> None:
    assert isinstance(create_backend("hashing"), HashingEmbedder)
    assert isinstance(create_backend("hash", dim=64), HashingEmbedder)
    with pytest.raises(ValueError, match="unknown"):
        create_backend("word2vec")
    with pytest.raises(ValueError, match="no model"):
        create_backend("hashing:foo")


def test_sentence_transformers_missing_extra_is_explained() -> None:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="--extra st"):
            create_backend("st")
    else:  # pragma: no cover - only when the optional extra is installed
        pytest.skip("sentence-transformers is installed")
