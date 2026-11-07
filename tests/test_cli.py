import json
from pathlib import Path

import pytest

from hybrid_search.cli import main

from .conftest import CORPUS, QUERIES


@pytest.fixture()
def index_dir(tiny_corpus: Path, tmp_path: Path) -> Path:
    out = tmp_path / "idx"
    assert main(["index", str(tiny_corpus), "--index-dir", str(out)]) == 0
    return out


def test_index_then_search_json(index_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    capsys.readouterr()
    assert main(["search", "raft leader", "-i", str(index_dir), "-k", "2", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["results"][0]["doc_id"] == "notes/raft"
    assert payload["results"][0]["snippet"]["highlights"]


def test_pretty_search_output(index_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    capsys.readouterr()
    main(["search", "bloom filter", "-i", str(index_dir), "--explain", "-f", "tags=probabilistic"])
    out = capsys.readouterr().out
    assert "Bloom Filters" in out and "bm25 #1" in out and "filters: tags=probabilistic" in out
    assert "\x1b[" not in out  # no ANSI codes when not a TTY


def test_no_results_message(index_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["search", "raft", "-i", str(index_dir), "-f", "year>3000"])
    assert "no results" in capsys.readouterr().out


def test_stats(index_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["stats", "-i", str(index_dir)]) == 0
    out = capsys.readouterr().out
    assert "chunks" in out and "hnsw" in out


def test_errors_exit_2(index_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["search", "x", "-i", str(index_dir), "-f", "nonsense"]) == 2
    assert main(["search", "x", "-i", str(tmp_path / "missing")]) == 2
    assert "hs: error" in capsys.readouterr().err


def test_eval_markdown(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_json = tmp_path / "report.json"
    rc = main(["eval", "--corpus", str(CORPUS), "--queries", str(QUERIES), "--markdown",
               "--ann-recall", "--json", str(out_json)])  # fmt: skip
    assert rc == 0
    out = capsys.readouterr().out
    assert "| Hybrid, RRF |" in out and "recall@10" in out
    report = json.loads(out_json.read_text())
    assert len(report["runs"]) == 6 and report["ann_corpus"]
