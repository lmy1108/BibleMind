# Bible Graph Explorer — Implementation Plan

**Date**: 2026-03-11
**Based on**: PRD v2.1 + CLAUDE.md
**Scope**: Phase 1 — WEB ingestion + REFERENCES edges (Step 1 of 5)

---

## What Step 1 Accomplishes

Reads the World English Bible (WEB) text from a local CSV, creates three node labels
(`Book`, `Chapter`, `Verse`) and the structural `CONTAINS` edges, then verifies
the expected verse count (31,102) before any subsequent step runs.

Step 1 has zero external API calls and costs nothing. It is a prerequisite for every
other step — all later MERGE operations depend on `Verse.id` uniqueness constraints
that are created here.

---

## Data Flow

```
ebible.org WEB CSV download
        │
        ▼
 step1_ingest_web.py
        │
        ├── 1. Create uniqueness constraints (BEFORE any MERGE)
        │       Book.name, Chapter composite, Verse.id
        │
        ├── 2. MERGE Book nodes (66 rows)
        │       {name, testament, book_order}
        │
        ├── 3. MERGE Chapter nodes + Book→Chapter CONTAINS edges
        │       {number, verse_count} per book
        │
        ├── 4. MERGE Verse nodes + Chapter→Verse CONTAINS edges
        │       {id (OSIS), reference, text, book, chapter, verse, char_count}
        │       Batched at 500 rows per transaction
        │
        └── 5. Validation query — assert count(Verse) == 31,102
```

---

## Key Decisions

| Decision | Rationale |
|---|---|
| Constraints first, data second | Without constraints, MERGE triggers a full scan on every row — 31K rows × full scan = catastrophic performance |
| Batch size 500 verses per tx | Balances memory overhead against transaction overhead; Neo4j's sweet spot for bulk MERGE is 200–1000 rows |
| OSIS IDs generated in Python, not Cypher | Keeps ID logic testable and auditable outside Neo4j; CLAUDE.md format: `{BOOK_ABBR_UPPER}.{chapter}.{verse}` |
| `char_count` stored at ingest time | IS_SIMILAR filtering (Step 3) needs `char_count >= 25`; computing it at write time costs nothing and avoids a later UPDATE pass |
| `verse_count` stored on Chapter | Bloom UI can display chapter size; costs one extra pass over already-loaded data |

---

## OSIS Book Abbreviations (canonical set — must match OpenBible)

These are the 66 abbreviations stored in `Verse.id` and matched against in Step 2.

```
OT: GEN EXO LEV NUM DEU JOS JDG RUT 1SA 2SA 1KI 2KI 1CH 2CH EZR NEH EST
    JOB PSA PRO ECC SNG ISA JER LAM EZK DAN HOS JOL AMO OBA JNA MIC NAH
    HAB ZEP HAG ZEC MAL

NT: MAT MRK LUK JHN ACT ROM 1CO 2CO GAL EPH PHP COL 1TH 2TH 1TI 2TI
    TIT PHM HEB JAS 1PE 2PE 1JN 2JN 3JN JUD REV
```

---

## Pre-Run Checklist

Before executing `step1_ingest_web.py`:

- [ ] Neo4j Desktop is running (version ≥ 5.11)
- [ ] `.env` file exists with `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- [ ] WEB CSV file downloaded from https://ebible.org/find/show.php?id=eng-web
      (`eng-web_readaloud.csv` or equivalent — see script `--help` for accepted formats)
- [ ] Python deps installed: `pip install neo4j pandas python-dotenv`

---

## Running Step 1

```bash
# Dry run — parses CSV and prints first 10 rows; no Neo4j writes
python step1_ingest_web.py --csv path/to/web.csv --dry-run

# Full run
python step1_ingest_web.py --csv path/to/web.csv

# If your .env is not in the current directory
python step1_ingest_web.py --csv path/to/web.csv --env /path/to/.env
```

Expected runtime: **30–120 seconds** on a local Neo4j instance (depends on disk speed).

---

## What Comes Next

| Step | File | Depends On |
|---|---|---|
| Step 2 | `step2_references.py` | Step 1 complete + `Verse.id` constraints exist |
| Step 3a | `step3a_embeddings.py` | Step 1 complete + `ANTHROPIC_API_KEY` set |
| Step 3b | `step3b_build_index.py` | Step 3a checkpoint file fully populated |
| Step 3c | `step3c_knn.py` | Step 3b index status = ONLINE |
| Step 4 | `step4_validate.py` | All prior steps complete |

**Do not skip Step 1's validation assertion.** If the verse count is not 31,102 the
OSIS IDs are likely malformed, and Step 2's MERGE operations will silently create zero
edges with no errors thrown.
