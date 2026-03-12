"""
step1_ingest_web.py
--------------------
Parse the World English Bible (WEB) CSV and load it into Neo4j as:
    (:Book)-[:CONTAINS]->(:Chapter)-[:CONTAINS]->(:Verse)

This is Phase 1 / Step 1 of the Bible Graph Explorer pipeline.
No external API calls.  No costs.  Must complete successfully before any
other step runs.

OSIS ID format (must match OpenBible cross-references):
    {BOOK_ABBR_UPPER}.{chapter}.{verse}   e.g.  ROM.8.28  GEN.1.1  PSA.23.1

Usage:
    python step1_ingest_web.py --csv path/to/web.csv
    python step1_ingest_web.py --csv path/to/web.csv --dry-run
    python step1_ingest_web.py --csv path/to/web.csv --env /path/to/.env
    python step1_ingest_web.py --csv path/to/web.csv --batch-size 500

Accepted CSV formats (auto-detected):
    Format A  — 4 columns: book_osis, chapter, verse, text
                e.g.  GEN,1,1,"In the beginning..."
    Format B  — 2 columns: reference, text
                e.g.  "GEN 1:1","In the beginning..."
    Format C  — ebible.org "readaloud" layout: #id column + Verse column
                e.g.  GEN1_1  "In the beginning..."
    Format D  — Numbered books (1-66): book_num, chapter, verse, text
                e.g.  1,1,1,"In the beginning..."

See PLAN.md for the download URL and pre-run checklist.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical OSIS book table  (66 books, order = book_order in Neo4j)
# ---------------------------------------------------------------------------
# Each entry: (osis_id, full_name, testament, book_order)
# This table is the single source of truth for IDs across all pipeline steps.
# Alternative abbreviations (ebible.org variants, SBL codes, etc.) map back
# to these canonical IDs in ALTERNATE_ABBR below.

BOOK_TABLE: list[tuple[str, str, str, int]] = [
    # ── Old Testament ────────────────────────────────────────────────────
    ("GEN",  "Genesis",         "OT",  1),
    ("EXO",  "Exodus",          "OT",  2),
    ("LEV",  "Leviticus",       "OT",  3),
    ("NUM",  "Numbers",         "OT",  4),
    ("DEU",  "Deuteronomy",     "OT",  5),
    ("JOS",  "Joshua",          "OT",  6),
    ("JDG",  "Judges",          "OT",  7),
    ("RUT",  "Ruth",            "OT",  8),
    ("1SA",  "1 Samuel",        "OT",  9),
    ("2SA",  "2 Samuel",        "OT", 10),
    ("1KI",  "1 Kings",         "OT", 11),
    ("2KI",  "2 Kings",         "OT", 12),
    ("1CH",  "1 Chronicles",    "OT", 13),
    ("2CH",  "2 Chronicles",    "OT", 14),
    ("EZR",  "Ezra",            "OT", 15),
    ("NEH",  "Nehemiah",        "OT", 16),
    ("EST",  "Esther",          "OT", 17),
    ("JOB",  "Job",             "OT", 18),
    ("PSA",  "Psalms",          "OT", 19),
    ("PRO",  "Proverbs",        "OT", 20),
    ("ECC",  "Ecclesiastes",    "OT", 21),
    ("SNG",  "Song of Solomon", "OT", 22),
    ("ISA",  "Isaiah",          "OT", 23),
    ("JER",  "Jeremiah",        "OT", 24),
    ("LAM",  "Lamentations",    "OT", 25),
    ("EZK",  "Ezekiel",         "OT", 26),
    ("DAN",  "Daniel",          "OT", 27),
    ("HOS",  "Hosea",           "OT", 28),
    ("JOL",  "Joel",            "OT", 29),
    ("AMO",  "Amos",            "OT", 30),
    ("OBA",  "Obadiah",         "OT", 31),
    ("JNA",  "Jonah",           "OT", 32),
    ("MIC",  "Micah",           "OT", 33),
    ("NAH",  "Nahum",           "OT", 34),
    ("HAB",  "Habakkuk",        "OT", 35),
    ("ZEP",  "Zephaniah",       "OT", 36),
    ("HAG",  "Haggai",          "OT", 37),
    ("ZEC",  "Zechariah",       "OT", 38),
    ("MAL",  "Malachi",         "OT", 39),
    # ── New Testament ────────────────────────────────────────────────────
    ("MAT",  "Matthew",         "NT", 40),
    ("MRK",  "Mark",            "NT", 41),
    ("LUK",  "Luke",            "NT", 42),
    ("JHN",  "John",            "NT", 43),
    ("ACT",  "Acts",            "NT", 44),
    ("ROM",  "Romans",          "NT", 45),
    ("1CO",  "1 Corinthians",   "NT", 46),
    ("2CO",  "2 Corinthians",   "NT", 47),
    ("GAL",  "Galatians",       "NT", 48),
    ("EPH",  "Ephesians",       "NT", 49),
    ("PHP",  "Philippians",     "NT", 50),
    ("COL",  "Colossians",      "NT", 51),
    ("1TH",  "1 Thessalonians", "NT", 52),
    ("2TH",  "2 Thessalonians", "NT", 53),
    ("1TI",  "1 Timothy",       "NT", 54),
    ("2TI",  "2 Timothy",       "NT", 55),
    ("TIT",  "Titus",           "NT", 56),
    ("PHM",  "Philemon",        "NT", 57),
    ("HEB",  "Hebrews",         "NT", 58),
    ("JAS",  "James",           "NT", 59),
    ("1PE",  "1 Peter",         "NT", 60),
    ("2PE",  "2 Peter",         "NT", 61),
    ("1JN",  "1 John",          "NT", 62),
    ("2JN",  "2 John",          "NT", 63),
    ("3JN",  "3 John",          "NT", 64),
    ("JUD",  "Jude",            "NT", 65),
    ("REV",  "Revelation",      "NT", 66),
]

# Build fast lookup dicts from the table
_BY_OSIS: dict[str, tuple] = {row[0]: row for row in BOOK_TABLE}
_BY_ORDER: dict[int, tuple] = {row[3]: row for row in BOOK_TABLE}
_BY_NAME: dict[str, tuple] = {row[1].lower(): row for row in BOOK_TABLE}

# Alternative abbreviations that ebible.org or other sources may use.
# Maps non-canonical abbreviation → canonical OSIS ID.
ALTERNATE_ABBR: dict[str, str] = {
    # ebible.org variants
    "EXO": "EXO", "EXOD": "EXO",
    "JUDG": "JDG",
    "SAM1": "1SA", "1SAM": "1SA",
    "SAM2": "2SA", "2SAM": "2SA",
    "KGS1": "1KI", "1KGS": "1KI", "1KIN": "1KI",
    "KGS2": "2KI", "2KGS": "2KI", "2KIN": "2KI",
    "CHR1": "1CH", "1CHR": "1CH",
    "CHR2": "2CH", "2CHR": "2CH",
    "PS":   "PSA", "PSALM": "PSA", "PSALMS": "PSA",
    "ECCL": "ECC",
    "SONG": "SNG", "SOL":  "SNG", "CANT": "SNG",
    "EZEK": "EZK",
    "JON":  "JNA", "JONAH": "JNA",
    "ZEPH": "ZEP",
    "ZECH": "ZEC",
    # New Testament variants
    "MATT": "MAT",
    "MK":   "MRK", "MAR":  "MRK",
    "LK":   "LUK",
    "JN":   "JHN",
    "ACTS": "ACT",
    "1COR": "1CO", "2COR": "2CO",
    "PHIL": "PHP",
    "1THESS": "1TH", "2THESS": "2TH",
    "1TIM": "1TI", "2TIM": "2TI",
    "TITUS": "TIT",
    "PHLM": "PHM", "PHILEM": "PHM", "PRM": "PHM",
    "1PET": "1PE", "2PET": "2PE",
    "1JO": "1JN", "2JO": "2JN", "3JO": "3JN",
    "JUDE": "JUD",
    # WEB VPL-specific abbreviations
    "EZE": "EZK",
    "JOE": "JOL",
    "JOH": "JHN",
    "PHI": "PHP",
    "JAM": "JAS",
}


def resolve_osis(raw: str) -> str | None:
    """
    Resolve any book abbreviation to a canonical OSIS ID.
    Returns None if unrecognised (caller should abort or skip).

    Examples:
        "ROM"   → "ROM"
        "PSALM" → "PSA"
        "1COR"  → "1CO"
        "Matt"  → "MAT"
    """
    normalised = raw.strip().upper()
    if normalised in _BY_OSIS:
        return normalised
    if normalised in ALTERNATE_ABBR:
        return ALTERNATE_ABBR[normalised]
    return None


def book_num_to_osis(n: int) -> str | None:
    """Map 1-66 book number to canonical OSIS ID."""
    row = _BY_ORDER.get(int(n))
    return row[0] if row else None


# ---------------------------------------------------------------------------
# Verse data class
# ---------------------------------------------------------------------------

class VerseRecord:
    """Lightweight struct for one parsed verse row."""

    __slots__ = ("osis_id", "reference", "text", "book_name", "book_osis",
                 "chapter", "verse", "char_count", "testament", "book_order")

    def __init__(
        self,
        book_osis: str,
        chapter: int,
        verse: int,
        text: str,
    ) -> None:
        row = _BY_OSIS[book_osis]
        self.book_osis   = book_osis
        self.book_name   = row[1]
        self.testament   = row[2]
        self.book_order  = row[3]
        self.chapter     = chapter
        self.verse       = verse
        self.text        = text.strip()
        self.char_count  = len(self.text)
        self.osis_id     = f"{book_osis}.{chapter}.{verse}"
        self.reference   = f"{self.book_name} {chapter}:{verse}"

    def to_dict(self) -> dict:
        return {
            "id":         self.osis_id,
            "reference":  self.reference,
            "text":       self.text,
            "book":       self.book_name,
            "chapter":    self.chapter,
            "verse":      self.verse,
            "char_count": self.char_count,
        }


# ---------------------------------------------------------------------------
# CSV Parsers (auto-detected)
# ---------------------------------------------------------------------------

def _sniff_format(path: Path) -> str:
    """
    Read up to 5 rows and guess the CSV format.
    Returns: "A" | "B" | "C" | "D"
    """
    with path.open(encoding="utf-8-sig", newline="") as fh:
        sample = fh.read(4096)

    lines = [l.strip() for l in sample.splitlines() if l.strip() and not l.startswith("#")]
    if not lines:
        raise ValueError("CSV file appears empty.")

    first = lines[0]

    # Format C: ebible.org readaloud — first column looks like GEN1_1
    if re.match(r"^[A-Z1-9]{3,6}\d+_\d+", first.split(",")[0].strip('"')):
        return "C"

    reader = csv.reader([first])
    cols = next(reader)

    if len(cols) >= 4:
        # Format D if first column is a digit (book number)
        if cols[0].strip().isdigit():
            return "D"
        # Format A if first column looks like an abbreviation
        if re.match(r"^[A-Z0-9]{2,5}$", cols[0].strip().upper()):
            return "A"

    if len(cols) == 2:
        # Format B: "GEN 1:1" style reference in col 0
        if re.search(r"\d+:\d+", cols[0]):
            return "B"

    # Default to A and let parse errors surface naturally
    return "A"


def _parse_format_a(path: Path) -> Iterator[VerseRecord]:
    """book_osis, chapter, verse, text"""
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        for line_no, row in enumerate(reader, start=1):
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 4:
                log.warning("Line %d: expected 4 columns, got %d — skipped", line_no, len(row))
                continue
            book_raw, chap_raw, verse_raw, text = row[0], row[1], row[2], row[3]
            osis = resolve_osis(book_raw)
            if osis is None:
                log.warning("Line %d: unrecognised book abbreviation '%s' — skipped", line_no, book_raw)
                continue
            try:
                yield VerseRecord(osis, int(chap_raw), int(verse_raw), text)
            except (ValueError, KeyError) as exc:
                log.warning("Line %d: parse error (%s) — skipped", line_no, exc)


def _parse_format_b(path: Path) -> Iterator[VerseRecord]:
    """reference ('GEN 1:1' or 'Genesis 1:1'), text"""
    pattern = re.compile(r"^(.+?)\s+(\d+):(\d+)$")
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        for line_no, row in enumerate(reader, start=1):
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 2:
                continue
            ref, text = row[0].strip(), row[1]
            m = pattern.match(ref)
            if not m:
                log.warning("Line %d: cannot parse reference '%s' — skipped", line_no, ref)
                continue
            book_raw, chap_raw, verse_raw = m.group(1), m.group(2), m.group(3)
            osis = resolve_osis(book_raw)
            if osis is None:
                # try matching by full name
                osis_row = _BY_NAME.get(book_raw.lower())
                if osis_row:
                    osis = osis_row[0]
            if osis is None:
                log.warning("Line %d: unrecognised book '%s' — skipped", line_no, book_raw)
                continue
            try:
                yield VerseRecord(osis, int(chap_raw), int(verse_raw), text)
            except (ValueError, KeyError) as exc:
                log.warning("Line %d: parse error (%s) — skipped", line_no, exc)


def _parse_format_c(path: Path) -> Iterator[VerseRecord]:
    """ebible readaloud: id=GEN1_1, text column"""
    id_pattern = re.compile(r"^([A-Z1-9]{2,5})(\d+)_(\d+)$")
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        headers = None
        for line_no, row in enumerate(reader, start=1):
            if not row:
                continue
            if line_no == 1 and not re.match(r"^[A-Z1-9]{3,6}\d+_\d+", row[0].strip('"')):
                headers = [h.strip().lower() for h in row]
                continue
            id_col = row[0].strip().strip('"')
            text_col = row[1] if len(row) > 1 else ""
            m = id_pattern.match(id_col)
            if not m:
                continue
            osis = resolve_osis(m.group(1))
            if osis is None:
                log.warning("Line %d: unrecognised book '%s' — skipped", line_no, m.group(1))
                continue
            try:
                yield VerseRecord(osis, int(m.group(2)), int(m.group(3)), text_col)
            except (ValueError, KeyError) as exc:
                log.warning("Line %d: parse error (%s) — skipped", line_no, exc)


def _parse_format_d(path: Path) -> Iterator[VerseRecord]:
    """book_num (1-66), chapter, verse, text"""
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        for line_no, row in enumerate(reader, start=1):
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 4:
                continue
            osis = book_num_to_osis(row[0])
            if osis is None:
                log.warning("Line %d: invalid book number '%s' — skipped", line_no, row[0])
                continue
            try:
                yield VerseRecord(osis, int(row[1]), int(row[2]), row[3])
            except (ValueError, KeyError) as exc:
                log.warning("Line %d: parse error (%s) — skipped", line_no, exc)


def parse_csv(path: Path) -> list[VerseRecord]:
    """Auto-detect format, parse all verses, return list sorted by book_order/chapter/verse."""
    fmt = _sniff_format(path)
    log.info("Detected CSV format: %s", fmt)
    parsers = {"A": _parse_format_a, "B": _parse_format_b,
               "C": _parse_format_c, "D": _parse_format_d}
    verses = list(parsers[fmt](path))
    verses.sort(key=lambda v: (v.book_order, v.chapter, v.verse))
    log.info("Parsed %d verses from %s", len(verses), path.name)
    return verses


# ---------------------------------------------------------------------------
# Neo4j writes
# ---------------------------------------------------------------------------

# Constraint DDL — MUST run before any MERGE (CLAUDE.md Hard Rule #2)
CONSTRAINT_QUERIES = [
    "CREATE CONSTRAINT verse_id_unique IF NOT EXISTS "
    "FOR (v:Verse) REQUIRE v.id IS UNIQUE",

    "CREATE CONSTRAINT book_name_unique IF NOT EXISTS "
    "FOR (b:Book) REQUIRE b.name IS UNIQUE",

    "CREATE CONSTRAINT chapter_id_unique IF NOT EXISTS "
    "FOR (c:Chapter) REQUIRE c.chapter_id IS UNIQUE",
]

# Merge a book node
MERGE_BOOK_Q = """
UNWIND $books AS b
MERGE (node:Book {name: b.name})
  ON CREATE SET node.testament   = b.testament,
                node.book_order  = b.book_order
"""

# Merge chapter nodes + Book→Chapter CONTAINS edge
MERGE_CHAPTER_Q = """
UNWIND $chapters AS c
MATCH  (book:Book {name: c.book_name})
MERGE  (ch:Chapter {chapter_id: c.chapter_id})
  ON CREATE SET ch.number      = c.number,
                ch.verse_count = c.verse_count,
                ch.book        = c.book_name
MERGE  (book)-[:CONTAINS]->(ch)
"""

# Merge verse nodes + Chapter→Verse CONTAINS edge
MERGE_VERSE_Q = """
UNWIND $verses AS v
MATCH  (ch:Chapter {chapter_id: v.chapter_id})
MERGE  (node:Verse {id: v.id})
  ON CREATE SET node.reference  = v.reference,
                node.text       = v.text,
                node.book       = v.book,
                node.chapter    = v.chapter,
                node.verse      = v.verse,
                node.char_count = v.char_count
MERGE  (ch)-[:CONTAINS]->(node)
"""


def create_constraints(session) -> None:
    for q in CONSTRAINT_QUERIES:
        session.run(q)
    log.info("Constraints created (or already existed).")


def write_books(session, verses: list[VerseRecord]) -> None:
    seen: dict[str, dict] = {}
    for v in verses:
        if v.book_name not in seen:
            seen[v.book_name] = {
                "name":       v.book_name,
                "testament":  v.testament,
                "book_order": v.book_order,
            }
    session.run(MERGE_BOOK_Q, books=list(seen.values()))
    log.info("Merged %d Book nodes.", len(seen))


def write_chapters(session, verses: list[VerseRecord]) -> None:
    # Build {chapter_id: verse_count} by counting verses per chapter
    verse_counts: dict[str, int] = defaultdict(int)
    chapter_meta: dict[str, dict] = {}

    for v in verses:
        cid = f"{v.book_osis}.{v.chapter}"
        verse_counts[cid] += 1
        if cid not in chapter_meta:
            chapter_meta[cid] = {
                "chapter_id": cid,
                "number":     v.chapter,
                "book_name":  v.book_name,
            }

    rows = []
    for cid, meta in chapter_meta.items():
        meta["verse_count"] = verse_counts[cid]
        rows.append(meta)

    session.run(MERGE_CHAPTER_Q, chapters=rows)
    log.info("Merged %d Chapter nodes.", len(rows))


def write_verses(session, verses: list[VerseRecord], batch_size: int = 500) -> None:
    total = len(verses)
    written = 0
    start = time.time()

    for i in range(0, total, batch_size):
        batch = verses[i : i + batch_size]
        rows = []
        for v in batch:
            d = v.to_dict()
            d["chapter_id"] = f"{v.book_osis}.{v.chapter}"
            rows.append(d)
        session.run(MERGE_VERSE_Q, verses=rows)
        written += len(batch)
        elapsed = time.time() - start
        pct = written / total * 100
        rate = written / elapsed if elapsed > 0 else 0
        log.info(
            "Verses: %d / %d  (%.1f%%)  %.0f verses/sec",
            written, total, pct, rate,
        )

    log.info("Merge complete — %d Verse nodes written in %.1fs.", total, time.time() - start)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

EXPECTED_VERSES = 31_102

VALIDATION_QUERIES = [
    ("Total verses",        "MATCH (v:Verse)  RETURN count(v) AS n"),
    ("Total books",         "MATCH (b:Book)   RETURN count(b) AS n"),
    ("Total chapters",      "MATCH (c:Chapter) RETURN count(c) AS n"),
    ("OT verses",           "MATCH (v:Verse) WHERE v.book IN [b IN [(x:Book) WHERE x.testament='OT' | x.name] | b] RETURN count(v) AS n"),
    ("CONTAINS edges total","MATCH ()-[r:CONTAINS]->() RETURN count(r) AS n"),
]

# Simpler queries that avoid complex subqueries
SIMPLE_VALIDATION: list[tuple[str, str, int | None]] = [
    ("Total verses",   "MATCH (v:Verse)   RETURN count(v) AS n",   EXPECTED_VERSES),
    ("Total books",    "MATCH (b:Book)    RETURN count(b) AS n",    66),
    ("Total chapters", "MATCH (c:Chapter) RETURN count(c) AS n",    None),  # ~1189
    ("OT books",       "MATCH (b:Book {testament:'OT'}) RETURN count(b) AS n", 39),
    ("NT books",       "MATCH (b:Book {testament:'NT'}) RETURN count(b) AS n", 27),
]


def run_validation(session) -> bool:
    passed = True
    log.info("─── Validation ───────────────────────────────────────")
    for label, query, expected in SIMPLE_VALIDATION:
        result = session.run(query).single()
        actual = result["n"] if result else 0
        if expected is not None:
            ok = actual == expected
            status = "✓ PASS" if ok else "✗ FAIL"
            log.info("  %s  %s: %d (expected %d)", status, label, actual, expected)
            if not ok:
                passed = False
        else:
            log.info("  ℹ  %s: %d", label, actual)

    # Spot-check: spot-check verses from CLAUDE.md
    spot_checks = ["ROM.8.28", "JHN.3.16", "PSA.23.1"]
    for osis_id in spot_checks:
        result = session.run(
            "MATCH (v:Verse {id: $id}) RETURN v.reference AS ref, v.char_count AS cc",
            id=osis_id,
        ).single()
        if result:
            log.info("  ✓  Spot-check %s → '%s'  (%d chars)", osis_id, result["ref"], result["cc"])
        else:
            log.error("  ✗  Spot-check %s NOT FOUND — check OSIS ID alignment!", osis_id)
            passed = False

    log.info("──────────────────────────────────────────────────────")
    return passed


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 1: Ingest WEB Bible text into Neo4j.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--csv", required=True, type=Path,
        help="Path to the WEB Bible CSV file.",
    )
    parser.add_argument(
        "--env", default=None, type=str,
        help="Path to .env file (optional; defaults to ./.env).",
    )
    parser.add_argument(
        "--batch-size", default=500, type=int,
        help="Verses per Neo4j transaction (default: 500).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse CSV and print summary; do NOT write to Neo4j.",
    )
    args = parser.parse_args()

    csv_path: Path = args.csv
    if not csv_path.exists():
        log.error("CSV file not found: %s", csv_path)
        return 1

    # ── Parse ────────────────────────────────────────────────────────────
    log.info("Parsing %s …", csv_path)
    verses = parse_csv(csv_path)

    if not verses:
        log.error("No verses were parsed.  Check CSV format (see --help).")
        return 1

    # Summary
    books_seen = {v.book_name for v in verses}
    log.info(
        "Parse summary: %d verses | %d books | first=%s | last=%s",
        len(verses), len(books_seen),
        verses[0].osis_id, verses[-1].osis_id,
    )

    if args.dry_run:
        log.info("── Dry-run sample (first 10 verses) ──")
        for v in verses[:10]:
            log.info("  %s  |  %s  |  %d chars  |  '%s…'",
                     v.osis_id, v.reference, v.char_count, v.text[:60])
        log.info("Dry run complete.  No data written to Neo4j.")
        return 0

    # ── Neo4j writes ─────────────────────────────────────────────────────
    from utils.neo4j_conn import get_driver  # deferred import (not needed in dry-run)

    driver = get_driver(args.env)
    try:
        with driver.session() as session:
            log.info("Creating uniqueness constraints …")
            create_constraints(session)

            log.info("Writing Book nodes …")
            write_books(session, verses)

            log.info("Writing Chapter nodes …")
            write_chapters(session, verses)

            log.info("Writing Verse nodes (batch_size=%d) …", args.batch_size)
            write_verses(session, verses, args.batch_size)

            # ── Validation ───────────────────────────────────────────────
            ok = run_validation(session)

        if ok:
            log.info("Step 1 complete.  All validation checks passed.")
            log.info("Next: run step2_references.py to import REFERENCES edges.")
            return 0
        else:
            log.error("Step 1 finished with validation failures.  Review output above.")
            log.error("Common cause: OSIS ID mismatch.  Compare 5 IDs from WEB CSV "
                      "against OpenBible cross-references before running Step 2.")
            return 1
    finally:
        driver.close()


if __name__ == "__main__":
    sys.exit(main())
