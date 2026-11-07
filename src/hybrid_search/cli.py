"""Command line interface: ``hs index | search | stats | serve | eval``."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .engine import FUSIONS, MODES, EngineConfig, SearchEngine, SearchResult
from .filters import FilterError
from .term import Style, terminal_width, wrap_highlighted

DEFAULT_INDEX_DIR = os.environ.get("HS_INDEX", ".hs")


def _add_index_dir(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--index-dir", "-i", default=DEFAULT_INDEX_DIR,
        help="index directory (default: $HS_INDEX or .hs)",
    )  # fmt: skip


def _add_build_options(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("index build options")
    g.add_argument("--backend", default="hashing",
                   help="embedding backend: hashing (default) or st[:model-name]")
    g.add_argument("--ann", choices=("hnsw", "exact"), default="hnsw",
                   help="dense index type (default: hnsw)")
    g.add_argument("--scorer", choices=("bm25f", "bm25"), default="bm25f")
    g.add_argument("--max-tokens", type=int, default=160, help="chunk size in tokens")
    g.add_argument("--overlap", type=int, default=32, help="token overlap between windows")
    g.add_argument("--hnsw-m", type=int, default=16)
    g.add_argument("--ef-construction", type=int, default=100)
    g.add_argument("--ef-search", type=int, default=64)  # fmt: skip


def config_from_args(args: argparse.Namespace) -> EngineConfig:
    return EngineConfig(
        backend=args.backend,
        ann=args.ann,
        scorer=args.scorer,
        max_tokens=args.max_tokens,
        overlap=args.overlap,
        hnsw_m=args.hnsw_m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )


def add_search_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("-k", type=int, default=5, help="number of results (default: 5)")
    p.add_argument("--mode", choices=MODES, default="hybrid")
    p.add_argument("--fusion", choices=FUSIONS, default="rrf")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="dense weight in [0,1]; BM25 gets 1-alpha (default: 0.5)")
    p.add_argument("--rrf-k", type=int, default=60, help="RRF rank constant (default: 60)")
    p.add_argument("--filter", "-f", action="append", default=[], metavar="EXPR",
                   help="metadata filter, repeatable: tags=raft, year>=2022, source=*.md")
    p.add_argument("--docs", action="store_true", help="return at most one chunk per document")  # fmt: skip


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hs",
        description="Local hybrid search: BM25F + dense vectors + rank fusion.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("index", help="index Markdown, text and JSON Lines files")
    p.add_argument("paths", nargs="+", help="files or directories to index")
    _add_index_dir(p)
    _add_build_options(p)
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", help="query an index")
    p.add_argument("query")
    _add_index_dir(p)
    add_search_options(p)
    p.add_argument("--json", action="store_true", help="print results as JSON")
    p.add_argument("--explain", action="store_true", help="show per-retriever ranks and scores")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("stats", help="show index statistics")
    _add_index_dir(p)
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("serve", help="serve the JSON HTTP API")
    _add_index_dir(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("eval", help="evaluate retrieval quality on a labeled query set")
    p.add_argument("--corpus", nargs="+", default=["data/corpus"], help="paths to index")
    p.add_argument("--queries", default="data/queries.jsonl", help="labeled queries (JSONL)")
    p.add_argument("-k", type=int, default=10, help="cutoff for MRR and nDCG (default: 10)")
    p.add_argument("--ann-recall", action="store_true", help="also measure HNSW recall vs exact")
    p.add_argument("--synthetic", type=int, default=0, metavar="N",
                   help="also benchmark HNSW on N synthetic vectors (e.g. 5000)")
    p.add_argument("--markdown", action="store_true", help="print GitHub-flavored tables")
    p.add_argument("--json", metavar="PATH", help="write the full report (incl. per query) here")
    p.add_argument("--failures", action="store_true",
                   help="list queries where the hybrid run misses at rank 1")
    _add_build_options(p)
    p.set_defaults(func=cmd_eval)
    return parser


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_index(args: argparse.Namespace) -> int:
    st = Style()
    t0 = time.perf_counter()
    engine = SearchEngine.from_paths(args.paths, config_from_args(args))
    out = engine.save(args.index_dir)
    s = engine.stats
    print(
        f"{st('indexed', 'green', 'bold')} {s['documents']} documents -> {s['chunks']} chunks, "
        f"{s['terms']} terms, {s['dim']}-d {engine.embedder.name} vectors "
        f"({engine.vectors.kind}) in {time.perf_counter() - t0:.2f}s"
    )
    print(st(f"wrote {out}/", "dim"))
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    engine = SearchEngine.load(args.index_dir)
    t0 = time.perf_counter()
    results = engine.search(
        args.query, args.k, mode=args.mode, fusion=args.fusion, alpha=args.alpha,
        rrf_k=args.rrf_k, filters=args.filter, unique_docs=args.docs,
    )  # fmt: skip
    elapsed_ms = (time.perf_counter() - t0) * 1000
    if args.json:
        payload = {
            "query": args.query,
            "mode": args.mode,
            "fusion": args.fusion if args.mode == "hybrid" else None,
            "took_ms": round(elapsed_ms, 2),
            "results": [r.to_dict() for r in results],
        }
        json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
        print()
        return 0
    print_results(args.query, results, engine, elapsed_ms, args)
    return 0


def print_results(
    query: str,
    results: Sequence[SearchResult],
    engine: SearchEngine,
    elapsed_ms: float,
    args: argparse.Namespace,
) -> None:
    st = Style()
    width = terminal_width()
    if not results:
        print(st(f'no results for "{query}"', "yellow"))
        return
    for r in results:
        c = r.chunk
        crumb = " > ".join(x for x in (c.title, c.heading) if x)
        score = f"{r.score:.4f}"
        rank = st(f"{r.rank:>2}", "bold", "cyan")
        pad = max(1, width - 4 - len(crumb) - len(score))
        print(f"{rank}  {st(crumb, 'bold')}{' ' * pad}{st(score, 'dim')}")
        meta = [c.source]
        tags = c.metadata.get("tags")
        if isinstance(tags, list) and tags:
            meta.append(" ".join(f"#{t}" for t in tags))
        print("    " + st("  ".join(meta), "gray"))
        if args.explain:
            print("    " + _explain(r, st))
        if r.snippet:
            for line in wrap_highlighted(r.snippet.text, r.snippet.spans, width, st, indent="    "):
                print(line)
        print()
    how = args.mode if args.mode != "hybrid" else f"hybrid/{args.fusion}"
    filt = f", filters: {' AND '.join(args.filter)}" if args.filter else ""
    print(
        st(
            f'{len(results)} results for "{query}" in {elapsed_ms:.1f} ms '
            f"({how}, {len(engine.chunks)} chunks{filt})",
            "dim",
        )
    )


def _explain(r: SearchResult, st: Style) -> str:
    parts = []
    if r.bm25_rank is not None:
        parts.append(f"bm25 #{r.bm25_rank} ({r.bm25_score:.2f})")
    else:
        parts.append(st("bm25 -", "dim"))
    if r.dense_rank is not None:
        parts.append(f"dense #{r.dense_rank} ({r.dense_score:.3f})")
    else:
        parts.append(st("dense -", "dim"))
    return st("   ".join(parts), "magenta")


def cmd_stats(args: argparse.Namespace) -> int:
    path = Path(args.index_dir)
    engine = SearchEngine.load(path)
    manifest = json.loads((path / "manifest.json").read_text())
    st = Style()
    rows = [
        ("index", str(path.resolve())),
        ("created", manifest.get("created", "?")),
        ("documents", engine.stats.get("documents", len({c.doc_id for c in engine.chunks}))),
        ("chunks", len(engine.chunks)),
        ("terms", engine.lexical.n_terms),
        ("scorer", engine.config.scorer),
        ("embedder", f"{engine.embedder.name} ({engine.embedder.dim}-d)"),
        ("vector index", engine.vectors.kind),
        ("size on disk", _human_size(sum(f.stat().st_size for f in path.iterdir()))),
    ]
    for k, v in rows:
        print(f"{st(f'{k:>13}', 'cyan')}  {v}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import make_server

    engine = SearchEngine.load(args.index_dir)
    server = make_server(engine, args.host, args.port)
    host, port = server.server_address[:2]
    print(f"serving {len(engine.chunks)} chunks on http://{host}:{port}  (Ctrl-C to stop)")
    print(f'  curl "http://{host}:{port}/search?q=raft+leader+election&k=3"')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .evaluation import ann_recall, evaluate, load_queries, synthetic_ann_benchmark

    st = Style()
    queries = load_queries(args.queries)
    t0 = time.perf_counter()
    engine = SearchEngine.from_paths(args.corpus, config_from_args(args))
    s = engine.stats
    print(st(
        f"corpus: {s['documents']} documents, {s['chunks']} chunks, {s['terms']} terms; "
        f"{len(queries)} labeled queries; embedder {engine.embedder.name} ({s['dim']}-d); "
        f"built in {time.perf_counter() - t0:.2f}s", "dim",
    ))  # fmt: skip
    print()
    report = evaluate(engine, queries, k=args.k)
    if args.markdown:
        print(report.to_markdown())
        print()
        print(report.by_type_markdown())
    else:
        cols = report.metric_names + ["p50 ms"]
        rows = [
            [r.spec.name] + [f"{r.metrics[m]:.3f}" for m in report.metric_names]
            + [f"{r.latency_ms['p50']:.2f}"]
            for r in report.runs
        ]  # fmt: skip
        _print_table(["run", *cols], rows, st)
        print()
        types = list(report.runs[0].by_type)
        metric = f"nDCG@{args.k}"
        _print_table(
            [f"{metric} by type", *(f"{t} ({report.meta['type_counts'][t]})" for t in types)],
            [[r.spec.name] + [f"{r.by_type[t][metric]:.3f}" for t in types] for r in report.runs],
            st,
        )

    if args.failures:
        hybrid = next(r for r in report.runs if r.spec.mode == "hybrid")
        misses = [q for q in hybrid.per_query if q[f"MRR@{args.k}"] < 1.0]
        print()
        noun = "query" if len(misses) == 1 else "queries"
        print(st(f"{hybrid.spec.name}: {len(misses)} {noun} without a relevant doc at rank 1", "bold"))
        for q in misses:
            print(f"  {q['id']}  {q['query']!r:60}  top3={q['top3']}")

    extra: dict[str, object] = {}
    if args.ann_recall:
        rows = ann_recall(engine, [q.query for q in queries], k=10)
        extra["ann_corpus"] = rows
        print()
        print(st(f"HNSW vs exact on corpus vectors ({len(engine.chunks)} chunks, "
                 f"{len(queries)} queries)", "bold"))  # fmt: skip
        _print_ann(rows, st, args.markdown)
    if args.synthetic:
        bench = synthetic_ann_benchmark(n=args.synthetic)
        extra["ann_synthetic"] = bench
        print()
        print(st(
            f"HNSW vs exact on {bench['n']} synthetic {bench['dim']}-d vectors "
            f"(M={bench['M']}, ef_construction={bench['ef_construction']}, "
            f"build {bench['build_s']:.1f}s)", "bold",
        ))  # fmt: skip
        _print_ann(bench["rows"], st, args.markdown)

    if args.json:
        payload = {**report.to_dict(), **extra}
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n")
        print(st(f"\nwrote {args.json}", "dim"))
    return 0


def _print_ann(rows: list[dict], st: Style, markdown: bool) -> None:
    header = ["ef_search", "recall@10", "HNSW p50 ms", "exact p50 ms"]
    body = [
        [str(r["ef"]), f"{r['recall@10']:.3f}", f"{r['hnsw_ms']:.3f}", f"{r['exact_ms']:.3f}"]
        for r in rows
    ]
    if markdown:
        print("| " + " | ".join(header) + " |")
        print("|" + "---:|" * len(header))
        for b in body:
            print("| " + " | ".join(b) + " |")
    else:
        _print_table(header, body, st)


def _print_table(header: list[str], rows: list[list[str]], st: Style) -> None:
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows, strict=True)]

    def fmt(cells: list[str]) -> str:
        first = f"{cells[0]:<{widths[0]}}"
        rest = [f"{c:>{w}}" for c, w in zip(cells[1:], widths[1:], strict=True)]
        return "  ".join([first, *rest])

    print(st(fmt(header), "bold"))
    print(st("  ".join("-" * w for w in widths), "dim"))
    for r in rows:
        print(fmt(r))


def _human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, FilterError, ValueError, ImportError) as e:
        print(f"hs: error: {e}", file=sys.stderr)
        return 2
    except BrokenPipeError:  # e.g. `hs search ... | head`
        return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
