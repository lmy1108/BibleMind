"""
step3c_knn.py
--------------
Run offline KNN using the HNSW vector index and write IS_SIMILAR edges.

For each qualifying verse:
    - Query the Top 6 nearest neighbors (+1 to account for self in results)
    - Exclude the verse itself
    - Keep Top 5 (MAX_RANK_IS_SIMILAR)
    - MERGE directed IS_SIMILAR edges with score and rank properties

Directionality note (PRD v2.1, Section V Step 3c):
    Edges are WRITTEN directed: (v)-[:IS_SIMILAR]->(similar)
    Edges are QUERIED undirected: (v)-[:IS_SIMILAR]-(similar)
    This captures asymmetric KNN results while supporting bidirectional exploration.
    See CLAUDE.md Cypher Patterns section.

MERGE semantics:
    Uses MERGE ON CREATE — does not update score/rank if edge already exists.
    Re-runs are safe.

Usage:
    python step3c_knn.py
    python step3c_knn.py --dry-run       # run KNN on first 10 verses, no writes
    python step3c_knn.py --batch-size 100
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

MAX_RANK_IS_SIMILAR  = 5    # Max outgoing IS_SIMILAR edges per verse
MIN_CHAR_COUNT       = 25   # Only verses that had embeddings generated
INDEX_NAME           = "verse_embedding_idx"
KNN_FETCH_K          = MAX_RANK_IS_SIMILAR + 1   # +1 because self appears in results
BATCH_SIZE_DEFAULT   = 200  # verses per Neo4j write transaction

# ---------------------------------------------------------------------------
# Cypher
# ---------------------------------------------------------------------------

# Fetch all qualifying verse IDs (those with embeddings)
FETCH_VERSE_IDS_Q = """
MATCH (v:Verse)
WHERE v.embedding IS NOT NULL
RETURN v.id AS id
ORDER BY v.id
"""

# Per-verse KNN: query index, exclude self, keep Top MAX_RANK
# Note: `rank()` window function not available in MERGE context;
# we compute rank in Python and pass it in the write query.
KNN_QUERY_Q = f"""
MATCH (v:Verse {{id: $verse_id}})
CALL db.index.vector.queryNodes('{INDEX_NAME}', {KNN_FETCH_K}, v.embedding)
YIELD node AS similar, score
WHERE similar.id <> $verse_id
RETURN similar.id AS similar_id, round(score, 4) AS score
ORDER BY score DESC
LIMIT {MAX_RANK_IS_SIMILAR}
"""

# Write IS_SIMILAR edges (directed; ON CREATE only — CLAUDE.md Hard Rule #1 analogue)
MERGE_IS_SIMILAR_Q = """
UNWIND $rows AS row
MATCH (a:Verse {id: row.verse_id})
MATCH (b:Verse {id: row.similar_id})
MERGE (a)-[r:IS_SIMILAR]->(b)
ON CREATE SET r.score = row.score,
              r.rank  = row.rank
"""

# Progress / validation queries
COUNT_IS_SIMILAR_Q = "MATCH ()-[r:IS_SIMILAR]->() RETURN count(r) AS n"
MAX_DEGREE_Q = """
MATCH (v:Verse)-[:IS_SIMILAR]->()
WITH v, count(*) AS degree
ORDER BY degree DESC LIMIT 1
RETURN v.reference AS verse, degree
"""


# ---------------------------------------------------------------------------
# KNN loop
# ---------------------------------------------------------------------------

def run_knn(
    session,
    verse_ids: list[str],
    write_batch_size: int,
    dry_run: bool = False,
) -> int:
    """
    Iterate over all verse IDs, run KNN for each, buffer results, and
    flush to Neo4j in write batches.

    Returns total number of IS_SIMILAR edges written.
    """
    total_verses  = len(verse_ids)
    edge_buffer:  list[dict] = []
    edges_written = 0
    start         = time.time()

    for i, verse_id in enumerate(verse_ids, start=1):
        # ── KNN query ────────────────────────────────────────────────────
        neighbors = session.run(KNN_QUERY_Q, verse_id=verse_id).data()

        if not neighbors:
            log.debug("No neighbors for %s (embedding missing or index gap)", verse_id)
            continue

        # Assign rank (1 = most similar)
        for rank, nb in enumerate(neighbors, start=1):
            edge_buffer.append({
                "verse_id":   verse_id,
                "similar_id": nb["similar_id"],
                "score":      nb["score"],
                "rank":       rank,
            })

        # ── Flush buffer ─────────────────────────────────────────────────
        if len(edge_buffer) >= write_batch_size:
            if not dry_run:
                session.run(MERGE_IS_SIMILAR_Q, rows=edge_buffer)
            edges_written += len(edge_buffer)
            edge_buffer.clear()

        # ── Progress ─────────────────────────────────────────────────────
        if i % 500 == 0 or i == total_verses:
            elapsed = time.time() - start
            rate = i / elapsed if elapsed > 0 else 0
            eta  = (total_verses - i) / rate if rate > 0 else 0
            log.info(
                "KNN: %d / %d verses  (%.1f%%)  %.0f v/s  ETA %.0fs",
                i, total_verses, i / total_verses * 100, rate, eta,
            )

    # ── Flush remainder ──────────────────────────────────────────────────
    if edge_buffer:
        if not dry_run:
            session.run(MERGE_IS_SIMILAR_Q, rows=edge_buffer)
        edges_written += len(edge_buffer)

    log.info(
        "KNN complete: %d IS_SIMILAR edges %s in %.1fs.",
        edges_written,
        "would be written" if dry_run else "written",
        time.time() - start,
    )
    return edges_written


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def run_validation(session) -> bool:
    log.info("─── Validation ───────────────────────────────────────")

    total = session.run(COUNT_IS_SIMILAR_Q).single()["n"]
    log.info("  Total IS_SIMILAR edges: %d", total)
    if total == 0:
        log.error("  ✗  Zero IS_SIMILAR edges written!")
        return False

    # Max outgoing degree (expected: <= MAX_RANK_IS_SIMILAR)
    result = session.run(MAX_DEGREE_Q).single()
    ok = True
    if result:
        degree = result["degree"]
        degree_ok = degree <= MAX_RANK_IS_SIMILAR
        log.info(
            "  %s  Max outgoing degree: %d on '%s'  (limit: %d)",
            "✓" if degree_ok else "✗",
            degree, result["verse"], MAX_RANK_IS_SIMILAR,
        )
        if not degree_ok:
            ok = False

    # Spot-checks: verify key verses have neighbors in both directions
    spot_checks = [
        ("ROM.8.28", "Romans 8:28"),
        ("JHN.3.16", "John 3:16"),
        ("PSA.23.1", "Psalm 23:1"),
    ]
    for osis_id, label in spot_checks:
        # Undirected query (CLAUDE.md Cypher Patterns — correct pattern)
        result = session.run(
            """
            MATCH (v:Verse {id: $id})-[r:IS_SIMILAR]-(sim:Verse)
            RETURN sim.reference AS ref, r.score AS score, r.rank AS rank
            ORDER BY r.score DESC LIMIT 3
            """,
            id=osis_id,
        ).data()
        if result:
            log.info("  ✓  %s top IS_SIMILAR neighbors (undirected):", label)
            for r in result:
                log.info(
                    "       rank %-2s  %-30s  score=%.4f",
                    r["rank"] if r["rank"] else "?",
                    r["ref"], r["score"],
                )
        else:
            log.warning("  ⚠  %s (%s) has no IS_SIMILAR neighbors", label, osis_id)

    # Verify undirected > directed (proves bidirectional capture works)
    directed = session.run(
        "MATCH (v:Verse {id:'ROM.8.28'})-[:IS_SIMILAR]->(s) RETURN count(s) AS n"
    ).single()["n"]
    undirected = session.run(
        "MATCH (v:Verse {id:'ROM.8.28'})-[:IS_SIMILAR]-(s) RETURN count(s) AS n"
    ).single()["n"]
    log.info(
        "  ℹ  ROM.8.28 directed neighbors: %d  |  undirected: %d  "
        "(undirected should be >= directed)",
        directed, undirected,
    )

    log.info("──────────────────────────────────────────────────────")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 3c: Run offline KNN and write IS_SIMILAR edges.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--env",        default=None, type=str)
    parser.add_argument("--batch-size", default=BATCH_SIZE_DEFAULT, type=int,
                        help=f"Verses per Neo4j write tx (default: {BATCH_SIZE_DEFAULT})")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Run KNN on first 10 verses and print results; no writes")
    args = parser.parse_args()

    _load_dotenv(args.env)
    from utils.neo4j_conn import get_driver

    driver = get_driver(args.env)
    try:
        with driver.session() as session:

            # Pre-flight: confirm index is ONLINE
            index_check = session.run(
                f"SHOW INDEXES YIELD name, state WHERE name = '{INDEX_NAME}'"
            ).data()
            if not index_check:
                log.error(
                    "Vector index '%s' does not exist.  Run step3b_build_index.py first.",
                    INDEX_NAME,
                )
                return 1
            state = index_check[0]["state"]
            if state != "ONLINE":
                log.error(
                    "Vector index '%s' is in state '%s', not ONLINE.  "
                    "Wait for it to finish populating before running KNN.  "
                    "(CLAUDE.md Hard Rule #3)",
                    INDEX_NAME, state,
                )
                return 1
            log.info("Pre-flight: index '%s' is ONLINE.", INDEX_NAME)

            # Fetch all qualifying verse IDs
            verse_ids = [r["id"] for r in session.run(FETCH_VERSE_IDS_Q).data()]
            log.info("Qualifying verses: %d", len(verse_ids))

            if args.dry_run:
                sample = verse_ids[:10]
                log.info("── Dry-run: KNN on first %d verses ──", len(sample))
                for vid in sample:
                    neighbors = session.run(KNN_QUERY_Q, verse_id=vid).data()
                    log.info("  %s →", vid)
                    for rank, nb in enumerate(neighbors, start=1):
                        log.info("    rank %d  %-30s  score=%.4f", rank, nb["similar_id"], nb["score"])
                log.info("Dry run complete.  No edges written.")
                return 0

            # Run KNN + write
            edges = run_knn(
                session      = session,
                verse_ids    = verse_ids,
                write_batch_size = args.batch_size,
            )

            if edges == 0:
                log.error("No IS_SIMILAR edges were written.  Check index status and embeddings.")
                return 1

            ok = run_validation(session)

        if ok:
            log.info("Step 3c complete.  %d IS_SIMILAR edges written.", edges)
            log.info("Next: run step4_validate.py for the full validation suite.")
            return 0
        else:
            log.error("Step 3c finished with validation failures.")
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
