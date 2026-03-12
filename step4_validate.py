"""
step4_validate.py
------------------
Full validation suite for the Bible Graph Explorer pipeline.

Runs all checks from CLAUDE.md "Validation Queries" section plus extended
quality checks.  Prints a pass/fail table and exits with code 0 (all pass)
or 1 (any failure).

Usage:
    python step4_validate.py
    python step4_validate.py --verbose    # show sample data for each check
    python step4_validate.py --fix-hints  # print Cypher hints for any failures
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constants  (mirror CLAUDE.md)
# ---------------------------------------------------------------------------

EXPECTED_VERSES          = 31_102
EXPECTED_BOOKS           = 66
EXPECTED_OT_BOOKS        = 39
EXPECTED_NT_BOOKS        = 27
MIN_VOTES_REFERENCES     = 10
MAX_RANK_REFERENCES      = 20
MAX_RANK_IS_SIMILAR      = 5
MIN_CHAR_COUNT           = 25
EMBEDDING_DIMENSIONS     = 1536
INDEX_NAME               = "verse_embedding_idx"

# ---------------------------------------------------------------------------
# Check registry
# ---------------------------------------------------------------------------

# Each check: (id, description, query, assertion_fn, fix_hint)
# assertion_fn receives the raw result dict and returns (passed: bool, detail: str)

def _exact(key, expected):
    def fn(row):
        val = row.get(key, 0)
        return val == expected, f"{val} (expected {expected})"
    return fn

def _gte(key, minimum):
    def fn(row):
        val = row.get(key, 0)
        return val >= minimum, f"{val} (min {minimum})"
    return fn

def _lte(key, maximum):
    def fn(row):
        val = row.get(key)
        if val is None:
            return True, "no data (ok)"
        return val <= maximum, f"{val} (max {maximum})"
    return fn

def _is_zero(key):
    def fn(row):
        val = row.get(key, 0)
        return val == 0, f"{val} (expected 0)"
    return fn

def _is_nonzero(key):
    def fn(row):
        val = row.get(key, 0)
        return val > 0, f"{val} (expected > 0)"
    return fn


CHECKS: list[dict] = [
    # ── Node counts ──────────────────────────────────────────────────────
    {
        "id":    "node.verse_count",
        "desc":  "Total Verse nodes",
        "query": "MATCH (v:Verse) RETURN count(v) AS n",
        "assert": _exact("n", EXPECTED_VERSES),
        "hint":  (
            "Expected 31,102 verses.  Check CSV parse output for skipped rows.\n"
            "Run: python step1_ingest_web.py --csv <file> --dry-run"
        ),
    },
    {
        "id":    "node.book_count",
        "desc":  "Total Book nodes",
        "query": "MATCH (b:Book) RETURN count(b) AS n",
        "assert": _exact("n", EXPECTED_BOOKS),
        "hint":  "Expected 66 books.  Re-run step1_ingest_web.py.",
    },
    {
        "id":    "node.ot_books",
        "desc":  "OT Book nodes",
        "query": "MATCH (b:Book {testament:'OT'}) RETURN count(b) AS n",
        "assert": _exact("n", EXPECTED_OT_BOOKS),
        "hint":  "Expected 39 OT books.",
    },
    {
        "id":    "node.nt_books",
        "desc":  "NT Book nodes",
        "query": "MATCH (b:Book {testament:'NT'}) RETURN count(b) AS n",
        "assert": _exact("n", EXPECTED_NT_BOOKS),
        "hint":  "Expected 27 NT books.",
    },
    # ── CONTAINS edges ───────────────────────────────────────────────────
    {
        "id":    "edge.contains_total",
        "desc":  "CONTAINS edges (structural)",
        "query": "MATCH ()-[r:CONTAINS]->() RETURN count(r) AS n",
        "assert": _is_nonzero("n"),
        "hint":  "No CONTAINS edges found.  Re-run step1_ingest_web.py.",
    },
    # ── REFERENCES edges ─────────────────────────────────────────────────
    {
        "id":    "edge.references_total",
        "desc":  "REFERENCES edges total",
        "query": "MATCH ()-[r:REFERENCES]->() RETURN count(r) AS n",
        "assert": _is_nonzero("n"),
        "hint":  (
            "Zero REFERENCES edges.  Likely an OSIS ID mismatch between WEB and OpenBible.\n"
            "Run: MATCH (v:Verse) RETURN v.id LIMIT 5  and compare against the raw CSV."
        ),
    },
    {
        "id":    "edge.references_min_votes",
        "desc":  f"REFERENCES min(votes) >= {MIN_VOTES_REFERENCES}",
        "query": "MATCH ()-[r:REFERENCES]->() RETURN min(r.votes) AS min_votes",
        "assert": _gte("min_votes", MIN_VOTES_REFERENCES),
        "hint":  f"Some REFERENCES edges have votes < {MIN_VOTES_REFERENCES}.  Re-run step2_references.py.",
    },
    {
        "id":    "edge.references_max_degree",
        "desc":  f"REFERENCES max outgoing degree <= {MAX_RANK_REFERENCES}",
        "query": """
            MATCH (v:Verse)-[:REFERENCES]->()
            WITH v, count(*) AS degree ORDER BY degree DESC LIMIT 1
            RETURN degree
        """,
        "assert": _lte("degree", MAX_RANK_REFERENCES),
        "hint":  f"Some verses have > {MAX_RANK_REFERENCES} outgoing REFERENCES.  Check step2_references.py filtering.",
    },
    # ── Embeddings ───────────────────────────────────────────────────────
    {
        "id":    "embed.count",
        "desc":  "Verses with embeddings",
        "query": "MATCH (v:Verse) WHERE v.embedding IS NOT NULL RETURN count(v) AS n",
        "assert": _is_nonzero("n"),
        "hint":  "No embeddings found.  Run step3a_embeddings.py.",
    },
    {
        "id":    "embed.dimensions",
        "desc":  f"Embedding dimensions == {EMBEDDING_DIMENSIONS}",
        "query": (
            "MATCH (v:Verse) WHERE v.embedding IS NOT NULL "
            "RETURN size(v.embedding) AS dims LIMIT 1"
        ),
        "assert": _exact("dims", EMBEDDING_DIMENSIONS),
        "hint":  (
            f"Expected {EMBEDDING_DIMENSIONS}-dimensional embeddings (text-embedding-3-small).\n"
            "Check EMBEDDING_MODEL and EMBEDDING_DIMENSIONS in step3a_embeddings.py."
        ),
    },
    {
        "id":    "embed.no_short_verses",
        "desc":  f"Short verses (char_count < {MIN_CHAR_COUNT}) have no embeddings",
        "query": (
            f"MATCH (v:Verse) WHERE v.char_count < {MIN_CHAR_COUNT} "
            "AND v.embedding IS NOT NULL RETURN count(v) AS n"
        ),
        "assert": _is_zero("n"),
        "hint":  "Short verses should not have embeddings.  Check min_chars filter in step3a.",
    },
    # ── IS_SIMILAR edges ─────────────────────────────────────────────────
    {
        "id":    "edge.is_similar_total",
        "desc":  "IS_SIMILAR edges total",
        "query": "MATCH ()-[r:IS_SIMILAR]->() RETURN count(r) AS n",
        "assert": _is_nonzero("n"),
        "hint":  "Zero IS_SIMILAR edges.  Run step3c_knn.py.",
    },
    {
        "id":    "edge.is_similar_max_degree",
        "desc":  f"IS_SIMILAR max outgoing degree <= {MAX_RANK_IS_SIMILAR}",
        "query": """
            MATCH (v:Verse)-[:IS_SIMILAR]->()
            WITH v, count(*) AS degree ORDER BY degree DESC LIMIT 1
            RETURN degree
        """,
        "assert": _lte("degree", MAX_RANK_IS_SIMILAR),
        "hint":  f"Some verses have > {MAX_RANK_IS_SIMILAR} outgoing IS_SIMILAR edges.  Check step3c.",
    },
    {
        "id":    "edge.is_similar_score_range",
        "desc":  "IS_SIMILAR scores are in [0, 1]",
        "query": (
            "MATCH ()-[r:IS_SIMILAR]->() "
            "RETURN min(r.score) AS min_s, max(r.score) AS max_s"
        ),
        "assert": lambda row: (
            (row.get("min_s", 0) >= 0 and row.get("max_s", 0) <= 1.0),
            f"min={row.get('min_s'):.4f}  max={row.get('max_s'):.4f}",
        ),
        "hint":  "IS_SIMILAR scores should be cosine similarity in [0, 1].  Check score computation.",
    },
    # ── Isolation check ──────────────────────────────────────────────────
    {
        "id":    "graph.no_isolated_verses",
        "desc":  "Isolated verses (no REFERENCES, no IS_SIMILAR) == 0",
        "query": """
            MATCH (v:Verse)
            WHERE NOT (v)-[:REFERENCES]-() AND NOT (v)-[:IS_SIMILAR]-()
            RETURN count(v) AS n
        """,
        "assert": _is_zero("n"),
        "hint":  (
            "Some verses are completely isolated.  This may be expected for very short\n"
            "verses excluded from IS_SIMILAR and rare verses with no cross-references.\n"
            "Check: MATCH (v:Verse) WHERE NOT (v)-[:REFERENCES]-() AND NOT (v)-[:IS_SIMILAR]-()\n"
            "       RETURN v.reference, v.char_count ORDER BY v.char_count LIMIT 20"
        ),
    },
    # ── No self-loops ────────────────────────────────────────────────────
    {
        "id":    "graph.no_self_references",
        "desc":  "Self-referencing REFERENCES edges == 0",
        "query": "MATCH (v:Verse)-[:REFERENCES]->(v) RETURN count(v) AS n",
        "assert": _is_zero("n"),
        "hint":  "Self-reference edges found.  Check Layer 2 filter in step2_references.py.",
    },
    {
        "id":    "graph.no_self_similar",
        "desc":  "Self-referencing IS_SIMILAR edges == 0",
        "query": "MATCH (v:Verse)-[:IS_SIMILAR]->(v) RETURN count(v) AS n",
        "assert": _is_zero("n"),
        "hint":  "Self-similarity edges found.  Check WHERE similar.id <> $verse_id in step3c.",
    },
]

# ---------------------------------------------------------------------------
# Spot-check verses (CLAUDE.md)
# ---------------------------------------------------------------------------

SPOT_CHECK_VERSES = [
    {
        "id":       "ROM.8.28",
        "label":    "Romans 8:28",
        "why":      "High-traffic NT verse — should have rich neighbors in both edge types",
        "min_refs": 3,
        "min_sim":  3,
    },
    {
        "id":       "JHN.3.16",
        "label":    "John 3:16",
        "why":      "Most cross-referenced verse — good stress test for hairball prevention",
        "min_refs": 5,
        "min_sim":  3,
    },
    {
        "id":       "PSA.23.1",
        "label":    "Psalm 23:1",
        "why":      "OT anchor — validates OT→NT semantic edge discovery",
        "min_refs": 1,
        "min_sim":  3,
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_checks(session, verbose: bool = False, fix_hints: bool = False) -> tuple[int, int]:
    """
    Run all registered checks.
    Returns (passed_count, total_count).
    """
    passed = 0
    total  = len(CHECKS)

    header = f"{'ID':<35}  {'Description':<45}  {'Result':<12}  {'Detail'}"
    separator = "─" * len(header)
    log.info("\n%s\n%s\n%s", separator, header, separator)

    for check in CHECKS:
        try:
            result = session.run(check["query"]).single()
            row = dict(result) if result else {}
            ok, detail = check["assert"](row)
        except Exception as exc:
            ok     = False
            detail = f"ERROR: {exc}"

        status = "✓ PASS" if ok else "✗ FAIL"
        log.info(
            "  %-35s  %-45s  %-12s  %s",
            check["id"], check["desc"], status, detail,
        )

        if not ok and fix_hints:
            log.info("    FIX HINT: %s", check["hint"])

        if ok:
            passed += 1

    log.info("%s\n  Result: %d / %d checks passed.\n%s", separator, passed, total, separator)
    return passed, total


def run_spot_checks(session, verbose: bool = False) -> int:
    """
    Manual spot-check of key verses.
    Returns number of verses that passed their minimum thresholds.
    """
    log.info("─── Spot-Check Verses ────────────────────────────────")
    ok_count = 0

    for sv in SPOT_CHECK_VERSES:
        vid   = sv["id"]
        label = sv["label"]

        # Node exists?
        verse = session.run(
            "MATCH (v:Verse {id: $id}) RETURN v.reference AS ref, v.char_count AS cc",
            id=vid,
        ).single()
        if not verse:
            log.error("  ✗  %s (%s) — NOT FOUND", label, vid)
            continue
        log.info("  ✓  %s  (%d chars)", verse["ref"], verse["cc"])

        # REFERENCES neighbors (outgoing)
        refs = session.run(
            """
            MATCH (v:Verse {id: $id})-[r:REFERENCES]->(ref:Verse)
            RETURN ref.reference AS ref, r.votes AS votes, r.rank AS rank
            ORDER BY r.rank LIMIT 5
            """,
            id=vid,
        ).data()
        ref_ok = len(refs) >= sv["min_refs"]
        log.info(
            "  %s  REFERENCES outgoing: %d  (min: %d)",
            "✓" if ref_ok else "⚠", len(refs), sv["min_refs"],
        )
        if verbose:
            for r in refs:
                log.info("       rank %-2d  %-30s  votes=%d", r["rank"], r["ref"], r["votes"])

        # IS_SIMILAR neighbors (undirected — CLAUDE.md correct pattern)
        sims = session.run(
            """
            MATCH (v:Verse {id: $id})-[r:IS_SIMILAR]-(sim:Verse)
            RETURN sim.reference AS ref, r.score AS score, r.rank AS rank
            ORDER BY r.score DESC LIMIT 5
            """,
            id=vid,
        ).data()
        sim_ok = len(sims) >= sv["min_sim"]
        log.info(
            "  %s  IS_SIMILAR (undirected): %d  (min: %d)",
            "✓" if sim_ok else "⚠", len(sims), sv["min_sim"],
        )
        if verbose:
            for s in sims:
                log.info(
                    "       rank %-2s  %-30s  score=%.4f",
                    s["rank"] if s["rank"] else "?",
                    s["ref"], s["score"],
                )

        if ref_ok or sim_ok:
            ok_count += 1

    log.info("──────────────────────────────────────────────────────")
    return ok_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 4: Full pipeline validation suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--env",        default=None, type=str)
    parser.add_argument("--verbose",    action="store_true",
                        help="Show sample neighbor data for spot-checks")
    parser.add_argument("--fix-hints",  action="store_true",
                        help="Print Cypher fix hints for failed checks")
    args = parser.parse_args()

    _load_dotenv(args.env)
    from utils.neo4j_conn import get_driver

    driver = get_driver(args.env)
    try:
        with driver.session() as session:
            start = time.time()

            passed, total = run_checks(
                session,
                verbose   = args.verbose,
                fix_hints = args.fix_hints,
            )

            spot_ok = run_spot_checks(session, verbose=args.verbose)

            elapsed = time.time() - start
            log.info("Total validation time: %.1fs", elapsed)

        all_passed = (passed == total)
        if all_passed:
            log.info(
                "✓  All %d checks passed.  Graph is ready for Bloom / Streamlit.",
                total,
            )
            log.info("Next: configure Neo4j Bloom perspective (Step 5).")
            return 0
        else:
            failed = total - passed
            log.error(
                "✗  %d / %d check(s) failed.  "
                "Re-run with --fix-hints for remediation guidance.",
                failed, total,
            )
            return 1
    finally:
        driver.close()


def _load_dotenv(env_path=None):
    try:
        from dotenv import load_dotenv
        import os
        p = env_path or ".env"
        if os.path.exists(p):
            load_dotenv(p, override=False)
    except ImportError:
        pass


if __name__ == "__main__":
    sys.exit(main())
