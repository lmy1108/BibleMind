"""
step2_references.py
--------------------
Load OpenBible cross-references and write REFERENCES edges to Neo4j.

Data source: https://a.openbible.info/data/cross-references.zip
License: CC0 (public domain)

Three-layer quality filter (from PRD v2.1):
    Layer 1 — Drop low-quality edges: votes < MIN_VOTES (10)
    Layer 2 — Drop self-references
    Layer 3 — Keep only Top MAX_RANK (20) outgoing edges per source verse

MERGE semantics (CLAUDE.md Hard Rule #1):
    Use MERGE, not CREATE.
    Do NOT update votes/rank if the edge already exists — source data is static.
    Re-runs are safe and idempotent.

Usage:
    python step2_references.py --csv path/to/cross_references.csv
    python step2_references.py --zip path/to/cross-references.zip
    python step2_references.py --csv path/to/cross_references.csv --dry-run
    python step2_references.py --csv path/to/cross_references.csv --min-votes 5 --max-rank 30

Expected OpenBible CSV format (tab-separated, no header):
    from_verse  to_verse  votes
    e.g.:  Gen.1.1  Rev.21.1  12

Note: OpenBible uses mixed-case OSIS (e.g. "Gen.1.1", "Rom.8.28").
      This script normalises them to UPPERCASE to match our canonical format.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constants  (mirror CLAUDE.md critical constants)
# ---------------------------------------------------------------------------

MIN_VOTES_DEFAULT  = 10
MAX_RANK_DEFAULT   = 20
BATCH_SIZE_DEFAULT = 1000   # REFERENCES is written-once; larger batches are fine

# ---------------------------------------------------------------------------
# OSIS ID normalisation
# ---------------------------------------------------------------------------

# OpenBible uses mixed-case (Gen.1.1, Rom.8.28, Ps.23.1).
# We store UPPERCASE (GEN.1.1, ROM.8.28, PSA.23.1).
# The mapping below covers every known variant in the OpenBible dataset.

_OB_BOOK_MAP: dict[str, str] = {
    # Pentateuch
    "gen":   "GEN", "exo":  "EXO", "exod": "EXO",
    "lev":   "LEV", "num":  "NUM", "deu":  "DEU", "deut": "DEU",
    # History
    "jos":   "JOS", "josh": "JOS",
    "jdg":   "JDG", "judg": "JDG",
    "rut":   "RUT", "ruth": "RUT",
    "1sa":   "1SA", "1sam": "1SA",
    "2sa":   "2SA", "2sam": "2SA",
    "1ki":   "1KI", "1kgs": "1KI",
    "2ki":   "2KI", "2kgs": "2KI",
    "1ch":   "1CH", "1chr": "1CH",
    "2ch":   "2CH", "2chr": "2CH",
    "ezr":   "EZR", "ezra": "EZR",
    "neh":   "NEH", "est":  "EST", "esth": "EST",
    # Poetry / Wisdom
    "job":   "JOB",
    "ps":    "PSA", "psa":  "PSA", "pss":  "PSA", "psalm": "PSA",
    "pro":   "PRO", "prov": "PRO",
    "ecc":   "ECC", "eccl": "ECC",
    "sng":   "SNG", "song": "SNG", "sol":  "SNG",
    # Prophets
    "isa":   "ISA", "jer":  "JER", "lam":  "LAM",
    "ezk":   "EZK", "ezek": "EZK",
    "dan":   "DAN", "hos":  "HOS",
    "jol":   "JOL", "joel": "JOL",
    "amo":   "AMO", "amos": "AMO",
    "oba":   "OBA", "obad": "OBA",
    "jon":   "JNA", "jona": "JNA", "jonah": "JNA",
    "mic":   "MIC", "nah":  "NAH",
    "hab":   "HAB", "zep":  "ZEP", "zeph": "ZEP",
    "hag":   "HAG", "zec":  "ZEC", "zech": "ZEC",
    "mal":   "MAL",
    # Gospels & Acts
    "mat":   "MAT", "matt": "MAT",
    "mrk":   "MRK", "mar":  "MRK", "mk":   "MRK", "mark": "MRK",
    "luk":   "LUK", "lk":   "LUK",  "luke": "LUK",
    "jhn":   "JHN", "joh":  "JHN", "jn":   "JHN", "john": "JHN",
    "act":   "ACT", "acts": "ACT",
    # Epistles
    "rom":   "ROM",
    "1co":   "1CO", "1cor": "1CO",
    "2co":   "2CO", "2cor": "2CO",
    "gal":   "GAL", "eph":  "EPH",
    "php":   "PHP", "phil": "PHP",
    "col":   "COL",
    "1th":   "1TH", "1thess": "1TH",
    "2th":   "2TH", "2thess": "2TH",
    "1ti":   "1TI", "1tim": "1TI",
    "2ti":   "2TI", "2tim": "2TI",
    "tit":   "TIT", "titus":"TIT", "phm":  "PHM", "phlm": "PHM",
    "heb":   "HEB", "jas":  "JAS",
    "1pe":   "1PE", "1pet": "1PE",
    "2pe":   "2PE", "2pet": "2PE",
    "1jn":   "1JN", "1jo":  "1JN", "1john": "1JN",
    "2jn":   "2JN", "2jo":  "2JN", "2john": "2JN",
    "3jn":   "3JN", "3jo":  "3JN", "3john": "3JN",
    "jud":   "JUD", "jude": "JUD",
    "rev":   "REV",
}


def normalise_osis(raw: str) -> str | None:
    """
    Convert an OpenBible OSIS ID to our canonical uppercase format.

    Examples:
        "Gen.1.1"   →  "GEN.1.1"
        "Rom.8.28"  →  "ROM.8.28"
        "Ps.23.1"   →  "PSA.23.1"
        "1Cor.13.4" →  "1CO.13.4"

    Returns None if the book abbreviation is unrecognised.
    Handles range syntax (e.g. "Gen.1.1-Gen.1.2") by taking the START verse.
    """
    # Strip range suffix: "Gen.1.1-Gen.1.2" → "Gen.1.1"
    raw = raw.split("-")[0].strip()

    parts = raw.split(".")
    if len(parts) != 3:
        return None

    book_raw, chapter_raw, verse_raw = parts
    canonical = _OB_BOOK_MAP.get(book_raw.lower())
    if canonical is None:
        return None

    try:
        chapter = int(chapter_raw)
        verse   = int(verse_raw)
    except ValueError:
        return None

    return f"{canonical}.{chapter}.{verse}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _read_lines_from_zip(zip_path: Path) -> list[str]:
    """Extract the first .csv or .txt file from the zip and return its lines."""
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith((".csv", ".txt", ".tsv"))]
        if not names:
            raise ValueError(f"No CSV/TXT file found inside {zip_path.name}")
        target = names[0]
        log.info("Extracting %s from %s", target, zip_path.name)
        data = zf.read(target)
    return data.decode("utf-8-sig").splitlines()


def load_raw_references(
    csv_path: Path | None = None,
    zip_path: Path | None = None,
) -> list[dict]:
    """
    Parse the OpenBible cross-references file.

    Returns a list of dicts: {from_verse, to_verse, votes}
    - All verse IDs are in our canonical UPPERCASE OSIS format.
    - Rows with unrecognisable IDs are skipped with a warning.
    - No filtering is applied here; filtering happens in apply_quality_filters().
    """
    if zip_path:
        lines = _read_lines_from_zip(zip_path)
        reader = csv.reader(io.StringIO("\n".join(lines)), delimiter="\t")
    elif csv_path:
        fh = csv_path.open(encoding="utf-8-sig", newline="")
        # Auto-detect delimiter
        sample = fh.read(512)
        fh.seek(0)
        delimiter = "\t" if "\t" in sample else ","
        reader = csv.reader(fh, delimiter=delimiter)
    else:
        raise ValueError("Provide --csv or --zip")

    rows = []
    skipped_id = 0
    skipped_votes = 0

    for line_no, row in enumerate(reader, start=1):
        # Skip blank lines and comment lines (OpenBible file has a header comment)
        if not row or row[0].startswith("#"):
            continue
        # Some distributions have a header row
        if row[0].lower() in ("from", "from_verse", "verse"):
            continue

        if len(row) < 3:
            log.debug("Line %d: too few columns (%d) — skipped", line_no, len(row))
            continue

        from_raw, to_raw, votes_raw = row[0], row[1], row[2]

        from_id = normalise_osis(from_raw)
        to_id   = normalise_osis(to_raw)

        if from_id is None or to_id is None:
            skipped_id += 1
            log.debug("Line %d: unrecognised ID '%s' or '%s' — skipped", line_no, from_raw, to_raw)
            continue

        try:
            votes = int(float(votes_raw))
        except ValueError:
            skipped_votes += 1
            continue

        rows.append({"from_verse": from_id, "to_verse": to_id, "votes": votes})

    if csv_path:
        fh.close()  # type: ignore[possibly-undefined]

    log.info(
        "Loaded %d raw reference rows  (skipped: %d bad IDs, %d bad votes)",
        len(rows), skipped_id, skipped_votes,
    )
    return rows


# ---------------------------------------------------------------------------
# Quality filters  (PRD v2.1, Section V Step 2)
# ---------------------------------------------------------------------------

def apply_quality_filters(
    rows: list[dict],
    min_votes: int = MIN_VOTES_DEFAULT,
    max_rank: int  = MAX_RANK_DEFAULT,
) -> list[dict]:
    """
    Apply the three-layer filter defined in the PRD.

    Layer 1: votes >= min_votes
    Layer 2: from_verse != to_verse  (no self-references)
    Layer 3: rank <= max_rank per source verse (ranked by votes DESC)

    Returns a new list with `rank` added to each row.
    """
    start_count = len(rows)

    # Layer 1 — votes threshold
    rows = [r for r in rows if r["votes"] >= min_votes]
    after_l1 = len(rows)
    log.info("Layer 1 (votes >= %d): %d → %d rows", min_votes, start_count, after_l1)

    # Layer 2 — self-reference removal
    rows = [r for r in rows if r["from_verse"] != r["to_verse"]]
    after_l2 = len(rows)
    log.info("Layer 2 (no self-refs):  %d → %d rows", after_l1, after_l2)

    # Layer 3 — rank per source verse, keep top max_rank
    # Group by from_verse, sort by votes DESC, assign rank
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[r["from_verse"]].append(r)

    ranked: list[dict] = []
    for from_verse, refs in grouped.items():
        refs.sort(key=lambda x: x["votes"], reverse=True)
        for i, ref in enumerate(refs[:max_rank], start=1):
            ranked.append({**ref, "rank": i})

    after_l3 = len(ranked)
    log.info("Layer 3 (rank <= %d):    %d → %d rows", max_rank, after_l2, after_l3)
    log.info(
        "Final: %d REFERENCES edges across %d source verses",
        after_l3, len(grouped),
    )
    return ranked


# ---------------------------------------------------------------------------
# Neo4j writes
# ---------------------------------------------------------------------------

# MERGE, never CREATE.  Do not overwrite votes/rank on existing edges.
# (CLAUDE.md Hard Rule #1: source data is static)
MERGE_REFERENCES_Q = """
UNWIND $rows AS row
MATCH (a:Verse {id: row.from_verse})
MATCH (b:Verse {id: row.to_verse})
MERGE (a)-[r:REFERENCES]->(b)
ON CREATE SET r.votes = row.votes,
              r.rank  = row.rank
"""

# Verification
COUNT_REFS_Q = "MATCH ()-[r:REFERENCES]->() RETURN count(r) AS n"
MIN_VOTES_Q  = "MATCH ()-[r:REFERENCES]->() RETURN min(r.votes) AS min_votes"
MAX_RANK_Q   = """
MATCH (v:Verse)-[r:REFERENCES]->()
WITH v, count(r) AS degree ORDER BY degree DESC LIMIT 1
RETURN v.reference AS verse, degree
"""


def write_references(
    session,
    rows: list[dict],
    batch_size: int = BATCH_SIZE_DEFAULT,
) -> int:
    """Write REFERENCES edges in batches. Returns number of rows submitted."""
    total   = len(rows)
    written = 0
    start   = time.time()

    for i in range(0, total, batch_size):
        batch = rows[i : i + batch_size]
        session.run(MERGE_REFERENCES_Q, rows=batch)
        written += len(batch)
        elapsed = time.time() - start
        rate = written / elapsed if elapsed > 0 else 0
        log.info(
            "REFERENCES: %d / %d  (%.1f%%)  %.0f rows/sec",
            written, total, written / total * 100, rate,
        )

    log.info("REFERENCES write complete in %.1fs.", time.time() - start)
    return written


def run_validation(session, min_votes: int, max_rank: int) -> bool:
    passed = True
    log.info("─── Validation ───────────────────────────────────────")

    # Total edge count
    total = session.run(COUNT_REFS_Q).single()["n"]
    log.info("  Total REFERENCES edges: %d", total)
    if total == 0:
        log.error("  ✗ ZERO edges written — likely an OSIS ID mismatch!")
        log.error("    Run: MATCH (v:Verse) RETURN v.id LIMIT 5  in Neo4j Browser")
        log.error("    Compare against raw OpenBible IDs to find the mismatch.")
        return False

    # Min votes (must be >= min_votes threshold)
    mv = session.run(MIN_VOTES_Q).single()["min_votes"]
    ok = mv >= min_votes
    log.info("  %s  min(votes): %d  (threshold: %d)", "✓" if ok else "✗", mv, min_votes)
    if not ok:
        passed = False

    # Max outgoing degree (must be <= max_rank)
    result = session.run(MAX_RANK_Q).single()
    if result:
        degree = result["degree"]
        ok = degree <= max_rank
        log.info(
            "  %s  max outgoing degree: %d on '%s'  (limit: %d)",
            "✓" if ok else "✗", degree, result["verse"], max_rank,
        )
        if not ok:
            passed = False

    # Spot-check: John 3:16 — most cross-referenced verse
    spot = session.run(
        """
        MATCH (v:Verse {id: 'JHN.3.16'})-[r:REFERENCES]->(ref:Verse)
        RETURN ref.reference AS ref, r.votes AS votes, r.rank AS rank
        ORDER BY r.rank LIMIT 5
        """
    ).data()
    if spot:
        log.info("  ✓  John 3:16 top REFERENCES:")
        for s in spot:
            log.info("       rank %d  %-25s  votes=%d", s["rank"], s["ref"], s["votes"])
    else:
        log.warning("  ⚠  John 3:16 has no REFERENCES — check data coverage")

    log.info("──────────────────────────────────────────────────────")
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 2: Import OpenBible cross-references as REFERENCES edges.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="Path to extracted cross_references.csv")
    source.add_argument("--zip", type=Path, help="Path to cross-references.zip (auto-extracts)")

    parser.add_argument("--env",       default=None, type=str, help="Path to .env file")
    parser.add_argument("--min-votes", default=MIN_VOTES_DEFAULT, type=int,
                        help=f"Minimum vote count to include (default: {MIN_VOTES_DEFAULT})")
    parser.add_argument("--max-rank",  default=MAX_RANK_DEFAULT,  type=int,
                        help=f"Max outgoing edges per verse (default: {MAX_RANK_DEFAULT})")
    parser.add_argument("--batch-size", default=BATCH_SIZE_DEFAULT, type=int,
                        help=f"Rows per Neo4j transaction (default: {BATCH_SIZE_DEFAULT})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load and filter data; print summary without writing to Neo4j")
    args = parser.parse_args()

    # Validate paths
    src = args.csv or args.zip
    if not src.exists():
        log.error("File not found: %s", src)
        return 1

    # ── Load ────────────────────────────────────────────────────────────
    log.info("Loading OpenBible cross-references from %s …", src.name)
    raw = load_raw_references(
        csv_path=args.csv if args.csv else None,
        zip_path=args.zip if args.zip else None,
    )

    # ── Filter ──────────────────────────────────────────────────────────
    filtered = apply_quality_filters(raw, min_votes=args.min_votes, max_rank=args.max_rank)

    if not filtered:
        log.error("No edges remain after filtering. Check --min-votes / --max-rank.")
        return 1

    if args.dry_run:
        log.info("── Dry-run sample (first 10 edges) ──")
        for r in filtered[:10]:
            log.info("  %s → %s  votes=%d  rank=%d",
                     r["from_verse"], r["to_verse"], r["votes"], r["rank"])
        log.info("Dry run complete.  No data written to Neo4j.")
        return 0

    # ── Write ───────────────────────────────────────────────────────────
    from utils.neo4j_conn import get_driver

    driver = get_driver(args.env)
    try:
        with driver.session() as session:
            write_references(session, filtered, args.batch_size)
            ok = run_validation(session, args.min_votes, args.max_rank)

        if ok:
            log.info("Step 2 complete.  REFERENCES edges imported and validated.")
            log.info("Next: run step3a_embeddings.py to generate verse embeddings.")
            return 0
        else:
            log.error("Step 2 finished with validation failures.  See output above.")
            return 1
    finally:
        driver.close()


if __name__ == "__main__":
    sys.exit(main())
