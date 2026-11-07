import pytest

from hybrid_search.fusion import normalize_scores, reciprocal_rank_fusion, weighted_fusion


def test_rrf_formula() -> None:
    fused = dict(reciprocal_rank_fusion({"a": ["x", "y"], "b": ["y", "z"]}, k=60))
    assert fused["y"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["x"] == pytest.approx(1 / 61)
    assert fused["z"] == pytest.approx(1 / 62)


def test_rrf_rewards_agreement() -> None:
    fused = reciprocal_rank_fusion({"a": ["x", "y", "z"], "b": ["z", "y", "x"]})
    assert fused[0][0] in {"x", "z"} or fused[0][0] == "y"
    # y is 2nd in both lists; x and z are 1st and 3rd. 2/(62) vs 1/61 + 1/63:
    assert dict(fused)["y"] < dict(fused)["x"]


def test_rrf_weights_and_validation() -> None:
    fused = reciprocal_rank_fusion({"a": ["x"], "b": ["y"]}, weights={"a": 2.0, "b": 1.0})
    assert [d for d, _ in fused] == ["x", "y"]
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"a": ["x"]}, k=-1)


def test_normalize_scores() -> None:
    assert normalize_scores({"a": 2.0, "b": 4.0}) == {"a": 0.0, "b": 1.0}
    assert normalize_scores({"a": 3.0}) == {"a": 1.0}
    z = normalize_scores({"a": 1.0, "b": 3.0}, "zscore")
    assert z["a"] == pytest.approx(-1.0) and z["b"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        normalize_scores({"a": 1.0}, "softmax")


def test_weighted_fusion_blends_and_fills_missing() -> None:
    fused = dict(
        weighted_fusion(
            {"bm25": [("x", 10.0), ("y", 5.0), ("w", 0.0)], "dense": [("y", 0.9), ("z", 0.1)]},
            weights={"bm25": 0.5, "dense": 0.5},
        )
    )
    assert fused["x"] == pytest.approx(0.5)  # 0.5 * 1.0 + 0.5 * 0 (missing -> min)
    assert fused["y"] == pytest.approx(0.5 * 0.5 + 0.5 * 1.0)
    assert fused["z"] == pytest.approx(0.0 + 0.0)


def test_weighted_fusion_alpha_extremes() -> None:
    results = {"bm25": [("x", 3.0), ("y", 1.0)], "dense": [("y", 0.9), ("x", 0.2)]}
    assert weighted_fusion(results, {"bm25": 1.0, "dense": 0.0})[0][0] == "x"
    assert weighted_fusion(results, {"bm25": 0.0, "dense": 1.0})[0][0] == "y"
