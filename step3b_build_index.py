"""
step3b_build_index.py
----------------------
Create the Neo4j HNSW vector index on Verse.embedding and wait for it to
reach ONLINE status before exiting.

CRITICAL — Run this BEFORE step3c_knn.py.
(CLAUDE.md Hard Rule #3: Never run KNN before the vector index is ONLINE.
 A partial index produces silently incorrect nearest-neighbour results.)

Memory prerequisites (must be configured in neo4j.conf BEFORE running):
    dbms.memory.heap.initial_size=2G
    dbms.memory.heap.max_size=4G
    dbms.memory.pagecache.size=1G

    31,102 × 1536 dimensions × 4 bytes ≈ 190 MB of raw vector data.
    Neo4j Desktop's default 1 GB heap will OOM during HNSW construction.
    See CLAUDE.md "Neo4j Memory Configuration" section.

Usage:
    python step3b_build_index.py
    python step3b_build_index.py --timeout 600   # wait up to 10 min for ONLINE
    python step3b_build_index.py --drop-rebuild  # drop existing index and recreate
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

INDEX_NAME           = "verse_embedding_idx"   # NEO4J_VECTOR_INDEX in CLAUDE.md
EMBEDDING_DIMENSIONS = 1536
POLL_INTERVAL_S      = 10      # seconds between status polls
DEFAULT_TIMEOUT_S    = 600     # 10 minutes; HNSW build on 31K verses typically < 2 min

# ---------------------------------------------------------------------------
# Cypher
# ---------------------------------------------------------------------------

CREATE_INDEX_Q = f"""
CREATE VECTOR INDEX {INDEX_NAME} IF NOT EXISTS
FOR (v:Verse) ON (v.embedding)
OPTIONS {{
  indexConfig: {{
    `vector.dimensions`:          {EMBEDDING_DIMENSIONS},
    `vector.similarity_function`: 'cosine'
  }}
}}
"""

# Show index status (Neo4j 5.x)
SHOW_INDEX_Q = f"""
SHOW INDEXES
YIELD name, state, populationPercent, type
WHERE name = '{INDEX_NAME}'
"""

DROP_INDEX_Q = f"DROP INDEX {INDEX_NAME} IF EXISTS"

# Sanity: count verses with embeddings before building
COUNT_EMBEDDED_Q = """
MATCH (v:Verse)
WHERE v.embedding IS NOT NULL
RETURN count(v) AS n
"""

# Quick KNN smoke test — must only run AFTER ONLINE status confirmed
SMOKE_TEST_Q = f"""
MATCH (v:Verse {{id: 'ROM.8.28'}})
CALL db.index.vector.queryNodes('{INDEX_NAME}', 3, v.embedding)
YIELD node AS similar, score
WHERE similar.id <> 'ROM.8.28'
RETURN similar.reference AS ref, round(score, 4) AS score
LIMIT 3
"""


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

def get_index_state(session) -> tuple[str, float] | None:
    """
    Returns (state, populationPercent) or None if the index does not exist.
    state is one of: 'POPULATING', 'ONLINE', 'FAILED', 'NOT_FOUND', etc.
    """
    result = session.run(SHOW_INDEX_Q).data()
    if not result:
        return None
    row = result[0]
    return row["state"], row.get("populationPercent", 0.0)


def create_index(session) -> None:
    session.run(CREATE_INDEX_Q)
    log.info("CREATE VECTOR INDEX issued (IF NOT EXISTS — safe to re-run).")


def drop_index(session) -> None:
    session.run(DROP_INDEX_Q)
    log.info("Existing index dropped.")


def wait_for_online(session, timeout_s: int = DEFAULT_TIMEOUT_S) -> bool:
    """
    Poll the index status every POLL_INTERVAL_S seconds until ONLINE or timeout.

    Returns True if ONLINE, False if timed out or FAILED.
    """
    deadline = time.time() + timeout_s
    log.info("Waiting for index '%s' to reach ONLINE status (timeout: %ds) …", INDEX_NAME, timeout_s)

    while time.time() < deadline:
        state_info = get_index_state(session)

        if state_info is None:
            log.warning("Index not found — waiting …")
            time.sleep(POLL_INTERVAL_S)
            continue

        state, pct = state_info

        if state == "ONLINE":
            log.info("✓  Index is ONLINE (100%% populated).")
            return True

        if state == "FAILED":
            log.error(
                "✗  Index build FAILED.  Common cause: insufficient heap memory.\n"
                "   Fix: increase dbms.memory.heap.max_size to 4G in neo4j.conf,\n"
                "        restart Neo4j, then re-run this script with --drop-rebuild."
            )
            return False

        if state == "POPULATING":
            log.info("  Populating … %.1f%%  (checking again in %ds)", pct, POLL_INTERVAL_S)
        else:
            log.info("  State: %s  %.1f%%", state, pct)

        time.sleep(POLL_INTERVAL_S)

    log.error(
        "Timeout after %ds.  Index state: %s.  "
        "Try increasing --timeout or checking Neo4j memory settings.",
        timeout_s, get_index_state(session),
    )
    return False


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def run_validation(session) -> bool:
    log.info("─── Validation ───────────────────────────────────────")

    # 1. Confirm index is ONLINE
    state_info = get_index_state(session)
    if not state_info or state_info[0] != "ONLINE":
        log.error("  ✗  Index is not ONLINE: %s", state_info)
        return False
    log.info("  ✓  Index state: ONLINE")

    # 2. Count embedded verses
    n = session.run(COUNT_EMBEDDED_Q).single()["n"]
    log.info("  ℹ  Verses with embeddings: %d", n)

    # 3. KNN smoke test on Romans 8:28
    log.info("  Running KNN smoke test on ROM.8.28 …")
    results = session.run(SMOKE_TEST_Q).data()
    if results:
        log.info("  ✓  KNN smoke test passed.  Top %d semantic neighbors of Romans 8:28:", len(results))
        for r in results:
            log.info("       %-30s  score=%.4f", r["ref"], r["score"])
    else:
        log.error(
            "  ✗  KNN smoke test returned no results.  "
            "Index may not be fully populated, or ROM.8.28 has no embedding."
        )
        return False

    log.info("──────────────────────────────────────────────────────")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 3b: Build HNSW vector index on Verse.embedding.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--env",          default=None, type=str)
    parser.add_argument("--timeout",      default=DEFAULT_TIMEOUT_S, type=int,
                        help=f"Max seconds to wait for ONLINE status (default: {DEFAULT_TIMEOUT_S})")
    parser.add_argument("--drop-rebuild", action="store_true",
                        help="Drop the existing index and rebuild from scratch")
    args = parser.parse_args()

    _load_dotenv(args.env)
    from utils.neo4j_conn import get_driver

    driver = get_driver(args.env)
    try:
        with driver.session() as session:

            # Pre-flight: ensure embeddings exist
            n_embedded = session.run(COUNT_EMBEDDED_Q).single()["n"]
            if n_embedded == 0:
                log.error(
                    "No Verse nodes have an embedding property.  "
                    "Run step3a_embeddings.py first."
                )
                return 1
            log.info("Pre-flight: %d verses have embeddings.", n_embedded)

            if args.drop_rebuild:
                log.info("--drop-rebuild: dropping existing index …")
                drop_index(session)
                time.sleep(2)   # brief pause for Neo4j to deregister

            # Check current state
            state_info = get_index_state(session)

            if state_info and state_info[0] == "ONLINE" and not args.drop_rebuild:
                log.info(
                    "Index '%s' already exists and is ONLINE.  "
                    "Use --drop-rebuild to recreate it.",
                    INDEX_NAME,
                )
            else:
                create_index(session)

            # Wait for ONLINE
            online = wait_for_online(session, timeout_s=args.timeout)
            if not online:
                return 1

            # Validate
            ok = run_validation(session)

        if ok:
            log.info("Step 3b complete.  HNSW vector index is ONLINE.")
            log.info("Next: run step3c_knn.py to write IS_SIMILAR edges.")
            return 0
        else:
            log.error("Step 3b finished with validation failures.")
            return 1
    finally:
        driver.close()


def _load_dotenv(env_path: str | None) -> None:
    try:
        from dotenv import load_dotenv
        p = env_path or ".env"
        if os.path.exists(p):
            load_dotenv(p, override=False)
    except ImportError:
        pass


import os   # noqa: E402 — needed by _load_dotenv above

if __name__ == "__main__":
    sys.exit(main())
