"""
step3a_embeddings.py
---------------------
Generate OpenAI embeddings for all qualifying Verse nodes and write them
back to Neo4j as `v.embedding` (float list, 1536 dimensions).

Qualifying verses: char_count >= MIN_CHAR_COUNT (25).  Short verses like
"Jesus wept." (11 chars) produce noisy embeddings and are intentionally excluded.

Cost estimate (text-embedding-3-small, $0.02 / 1M tokens):
    ~31,102 verses × ~30 tokens each ≈ 930K tokens ≈ $0.80 total

Checkpoint resume:
    Progress is saved to CHECKPOINT_FILE after every batch.
    If the job crashes or is interrupted, restart it — it picks up from the
    last completed batch with no work lost and no duplicate API calls.

Usage:
    python step3a_embeddings.py
    python step3a_embeddings.py --batch-size 200 --dry-run
    python step3a_embeddings.py --resume            # force resume from checkpoint
    python step3a_embeddings.py --reset-checkpoint  # discard checkpoint, start fresh

Prerequisites:
    - Step 1 complete (Verse nodes exist with char_count property)
    - ANTHROPIC_API_KEY or OPENAI_API_KEY set in .env
    - pip install openai tenacity neo4j python-dotenv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constants  (mirror CLAUDE.md)
# ---------------------------------------------------------------------------

EMBEDDING_MODEL      = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
MIN_CHAR_COUNT       = 25       # verses below this are excluded
CHECKPOINT_FILE      = "embedding_checkpoint.json"
BATCH_SIZE_DEFAULT   = 100      # API call: number of texts per request
WRITE_BATCH_DEFAULT  = 500      # Neo4j write: number of verses per transaction

# ---------------------------------------------------------------------------
# OpenAI client + retry logic
# ---------------------------------------------------------------------------

def _get_openai_client():
    """
    Returns an OpenAI client.  Supports both OPENAI_API_KEY and
    ANTHROPIC_API_KEY env variable names (the latter is listed in CLAUDE.md).
    """
    import openai

    # Prefer explicit OPENAI_API_KEY; fall back to ANTHROPIC_API_KEY
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Neither OPENAI_API_KEY nor ANTHROPIC_API_KEY is set. "
            "Add one to your .env file."
        )
    return openai.OpenAI(api_key=api_key)


def _embed_batch_with_retry(client, texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts with exponential backoff on rate-limit errors.

    Uses tenacity for retry logic:
        - 0 wait under normal conditions (no unnecessary sleep)
        - 1s → 2s → 4s → ... → 60s on rate-limit or connection errors
        - Max 6 attempts before giving up

    Why not time.sleep(0.5)?  See PRD v2.1 Section V Step 3a.
    """
    try:
        from tenacity import (
            retry, wait_exponential, stop_after_attempt,
            retry_if_exception_type,
        )
        import openai as _openai

        @retry(
            wait=wait_exponential(multiplier=1, min=1, max=60),
            stop=stop_after_attempt(6),
            retry=retry_if_exception_type(
                (_openai.RateLimitError, _openai.APIConnectionError)
            ),
            reraise=True,
        )
        def _call():
            response = client.embeddings.create(
                input=texts,
                model=EMBEDDING_MODEL,
                dimensions=EMBEDDING_DIMENSIONS,
            )
            return [e.embedding for e in response.data]

        return _call()

    except ImportError:
        # tenacity not installed — fall back to simple single attempt
        log.warning(
            "tenacity not installed; retries disabled. "
            "Run: pip install tenacity"
        )
        response = client.embeddings.create(
            input=texts,
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMENSIONS,
        )
        return [e.embedding for e in response.data]


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(path: str = CHECKPOINT_FILE) -> dict[str, list[float]]:
    """Load completed verse_id → embedding mappings from disk."""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        log.info("Checkpoint loaded: %d verses already embedded.", len(data))
        return data
    return {}


def save_checkpoint(
    results: dict[str, list[float]],
    path: str = CHECKPOINT_FILE,
) -> None:
    """Persist current results to the checkpoint file (atomic write)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f)
    os.replace(tmp, path)   # atomic on POSIX; avoids corrupt checkpoint on crash


# ---------------------------------------------------------------------------
# Neo4j helpers
# ---------------------------------------------------------------------------

FETCH_VERSES_Q = """
MATCH (v:Verse)
WHERE v.char_count >= $min_chars
RETURN v.id AS id, v.text AS text
ORDER BY v.id
"""

WRITE_EMBEDDINGS_Q = """
UNWIND $rows AS row
MATCH (v:Verse {id: row.id})
SET v.embedding = row.embedding
"""

COUNT_EMBEDDED_Q = """
MATCH (v:Verse)
WHERE v.embedding IS NOT NULL
RETURN count(v) AS n
"""


def fetch_qualifying_verses(session, min_chars: int = MIN_CHAR_COUNT) -> list[dict]:
    """Return [{id, text}, ...] for all verses meeting the char_count threshold."""
    result = session.run(FETCH_VERSES_Q, min_chars=min_chars)
    verses = [{"id": r["id"], "text": r["text"]} for r in result]
    log.info(
        "Fetched %d qualifying verses (char_count >= %d).",
        len(verses), min_chars,
    )
    return verses


def write_embeddings_batch(
    session,
    id_embedding_pairs: list[tuple[str, list[float]]],
) -> None:
    rows = [{"id": vid, "embedding": emb} for vid, emb in id_embedding_pairs]
    session.run(WRITE_EMBEDDINGS_Q, rows=rows)


# ---------------------------------------------------------------------------
# Core embedding loop
# ---------------------------------------------------------------------------

def run_embedding_job(
    session,
    verses: list[dict],
    checkpoint: dict[str, list[float]],
    api_batch_size: int,
    write_batch_size: int,
    checkpoint_file: str = CHECKPOINT_FILE,
) -> dict[str, list[float]]:
    """
    Generate embeddings for all verses not yet in the checkpoint.

    Strategy:
        1. Build list of (id, text) pairs that are NOT yet in checkpoint
        2. Call API in batches of `api_batch_size`
        3. Save checkpoint after every API batch
        4. Write to Neo4j in batches of `write_batch_size`

    Returns the final checkpoint dict (all completed embeddings).
    """
    results = dict(checkpoint)

    # Identify remaining work
    remaining = [(v["id"], v["text"]) for v in verses if v["id"] not in results]
    total_remaining = len(remaining)
    total_all       = len(verses)
    already_done    = total_all - total_remaining

    if total_remaining == 0:
        log.info("All %d embeddings already in checkpoint.  Skipping API calls.", total_all)
    else:
        log.info(
            "Embeddings to generate: %d  (already done: %d / %d)",
            total_remaining, already_done, total_all,
        )

    # ── API batching ────────────────────────────────────────────────────
    client  = _get_openai_client()
    api_done = 0
    start    = time.time()

    for i in range(0, len(remaining), api_batch_size):
        batch = remaining[i : i + api_batch_size]
        ids   = [b[0] for b in batch]
        texts = [b[1] for b in batch]

        embeddings = _embed_batch_with_retry(client, texts)

        for vid, emb in zip(ids, embeddings):
            results[vid] = emb

        api_done += len(batch)
        save_checkpoint(results, checkpoint_file)

        elapsed = time.time() - start
        total_done_now = already_done + api_done
        pct  = total_done_now / total_all * 100
        rate = api_done / elapsed if elapsed > 0 else 0
        log.info(
            "API: %d / %d new  |  total %d / %d  (%.1f%%)  %.0f verses/sec",
            api_done, total_remaining,
            total_done_now, total_all,
            pct, rate,
        )

    # ── Neo4j write ─────────────────────────────────────────────────────
    log.info("Writing embeddings to Neo4j (batch_size=%d) …", write_batch_size)
    all_pairs = [(vid, emb) for vid, emb in results.items()]
    write_start = time.time()
    written = 0

    for i in range(0, len(all_pairs), write_batch_size):
        batch = all_pairs[i : i + write_batch_size]
        write_embeddings_batch(session, batch)
        written += len(batch)
        log.info(
            "Neo4j write: %d / %d  (%.1f%%)",
            written, len(all_pairs), written / len(all_pairs) * 100,
        )

    log.info(
        "Neo4j write complete — %d embeddings in %.1fs.",
        written, time.time() - write_start,
    )
    return results


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def run_validation(session, expected_count: int, partial: bool = False) -> bool:
    log.info("─── Validation ───────────────────────────────────────")
    n = session.run(COUNT_EMBEDDED_Q).single()["n"]
    ok = partial or n >= expected_count * 0.99   # allow 1% tolerance for edge cases
    log.info(
        "  %s  Verses with embeddings: %d  (expected ~%d%s)",
        "✓" if ok else "✗", n, expected_count, " [partial test run]" if partial else "",
    )

    # Spot-check: verify embedding is 1536-dimensional
    spot = session.run(
        "MATCH (v:Verse {id: 'ROM.8.28'}) "
        "RETURN size(v.embedding) AS dims, v.reference AS ref"
    ).single()
    if spot and spot["dims"] is not None:
        dims = spot["dims"]
        dim_ok = dims == EMBEDDING_DIMENSIONS
        log.info(
            "  %s  ROM.8.28 embedding dimensions: %d  (expected %d)",
            "✓" if dim_ok else "✗", dims, EMBEDDING_DIMENSIONS,
        )
        if not dim_ok:
            ok = False
    elif partial:
        log.info("  ℹ  ROM.8.28 spot-check skipped (partial test run)")
    else:
        log.warning("  ⚠  ROM.8.28 not found or has no embedding")
        ok = False

    # Short verse should NOT have an embedding
    short = session.run(
        "MATCH (v:Verse) WHERE v.char_count < $mc AND v.embedding IS NOT NULL "
        "RETURN count(v) AS n",
        mc=MIN_CHAR_COUNT,
    ).single()["n"]
    if short == 0:
        log.info("  ✓  No short verses (char_count < %d) have embeddings.", MIN_CHAR_COUNT)
    else:
        log.warning(
            "  ⚠  %d short verses have embeddings — they will become noisy similarity hubs.",
            short,
        )

    log.info("──────────────────────────────────────────────────────")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 3a: Generate verse embeddings and store in Neo4j.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--env",              default=None, type=str)
    parser.add_argument("--batch-size",       default=BATCH_SIZE_DEFAULT, type=int,
                        help=f"Texts per API call (default: {BATCH_SIZE_DEFAULT})")
    parser.add_argument("--write-batch-size", default=WRITE_BATCH_DEFAULT, type=int,
                        help=f"Verses per Neo4j tx (default: {WRITE_BATCH_DEFAULT})")
    parser.add_argument("--checkpoint-file",  default=CHECKPOINT_FILE,
                        help=f"Checkpoint file path (default: {CHECKPOINT_FILE})")
    parser.add_argument("--min-chars",        default=MIN_CHAR_COUNT, type=int,
                        help=f"Min verse length to embed (default: {MIN_CHAR_COUNT})")
    parser.add_argument("--dry-run",          action="store_true",
                        help="Fetch verses and show count; no API calls or Neo4j writes")
    parser.add_argument("--resume",           action="store_true",
                        help="Load checkpoint and continue (default behaviour, flag is informational)")
    parser.add_argument("--reset-checkpoint", action="store_true",
                        help="Delete checkpoint file and start from scratch")
    parser.add_argument("--limit",            default=None, type=int,
                        help="Embed only the first N verses (for testing)")
    args = parser.parse_args()

    if args.reset_checkpoint and os.path.exists(args.checkpoint_file):
        os.remove(args.checkpoint_file)
        log.info("Checkpoint file deleted.  Starting from scratch.")

    # ── Load checkpoint ──────────────────────────────────────────────────
    checkpoint = load_checkpoint(args.checkpoint_file)

    if args.dry_run:
        # Just connect, count qualifying verses, and exit
        from utils.neo4j_conn import get_driver
        _load_dotenv(args.env)
        driver = get_driver(args.env)
        try:
            with driver.session() as session:
                verses = fetch_qualifying_verses(session, args.min_chars)
            remaining = len([v for v in verses if v["id"] not in checkpoint])
            log.info(
                "Dry run: %d qualifying verses, %d already in checkpoint, %d to embed.",
                len(verses), len(checkpoint), remaining,
            )
            import openai
            # Rough token + cost estimate
            sample_tokens = sum(len(v["text"].split()) * 1.3 for v in verses[:100]) / 100
            total_tokens  = sample_tokens * len(verses)
            cost_estimate = total_tokens / 1_000_000 * 0.02
            log.info(
                "Cost estimate: ~%.0fK tokens → ~$%.2f  (text-embedding-3-small)",
                total_tokens / 1000, cost_estimate,
            )
        finally:
            driver.close()
        return 0

    # ── Full run ─────────────────────────────────────────────────────────
    _load_dotenv(args.env)
    from utils.neo4j_conn import get_driver

    driver = get_driver(args.env)
    try:
        with driver.session() as session:
            verses = fetch_qualifying_verses(session, args.min_chars)

            if not verses:
                log.error(
                    "No qualifying verses found.  Is Step 1 complete? "
                    "Do Verse nodes have a char_count property?"
                )
                return 1

            if args.limit:
                verses = verses[: args.limit]
                log.info("--limit %d: embedding only first %d verses.", args.limit, len(verses))

            final_checkpoint = run_embedding_job(
                session     = session,
                verses      = verses,
                checkpoint  = checkpoint,
                api_batch_size   = args.batch_size,
                write_batch_size = args.write_batch_size,
                checkpoint_file  = args.checkpoint_file,
            )

            ok = run_validation(session, expected_count=len(verses), partial=bool(args.limit))

        if ok:
            log.info("Step 3a complete.  %d embeddings stored.", len(final_checkpoint))
            log.info("Next: run step3b_build_index.py to create the HNSW vector index.")
            return 0
        else:
            log.error("Step 3a finished with validation failures.")
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


if __name__ == "__main__":
    sys.exit(main())
