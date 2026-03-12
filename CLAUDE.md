# Bible Graph Explorer — Claude Code Project Memory

This file is read automatically by Claude Code on every invocation (interactive and headless).
Keep it accurate. When project constants change, update this file first.

---

## Project Overview

A Bible knowledge graph built on **World English Bible (WEB)** text, stored in Neo4j.
Exposes two relationship types between verses:
- `REFERENCES` — human-curated cross-references from OpenBible
- `IS_SIMILAR` — semantic similarity via OpenAI embeddings + HNSW KNN

Primary UI: Neo4j Bloom (local/team). Fallback UI: Streamlit + streamlit-agraph.

---

## Critical Constants

```python
# Verse ID format — MUST match OpenBible OSIS format
# Example: "ROM.8.28", "GEN.1.1", "PSA.23.1"
VERSE_ID_FORMAT = "{BOOK_ABBR_UPPER}.{chapter}.{verse}"

# Data quality thresholds
MIN_VOTES_REFERENCES = 10       # Drop REFERENCES edges with votes below this
MAX_RANK_REFERENCES  = 20       # Max outgoing REFERENCES edges per verse
MAX_RANK_IS_SIMILAR  = 5        # Max outgoing IS_SIMILAR edges per verse
MIN_CHAR_COUNT       = 25       # Min verse length to participate in IS_SIMILAR

# Embedding config
EMBEDDING_MODEL      = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
CHECKPOINT_FILE      = "embedding_checkpoint.json"

# Neo4j
NEO4J_VECTOR_INDEX   = "verse_embedding_idx"
NEO4J_VERSION_MIN    = "5.11"   # Required for native HNSW vector index
TOTAL_VERSES         = 31102    # Expected verse count after full WEB import
```

---

## Graph Schema

### Nodes
| Label | Key Properties |
|---|---|
| `Book` | `name` (unique), `testament` ("OT"/"NT"), `book_order` (1–66) |
| `Chapter` | `number`, `verse_count` |
| `Verse` | `id` (unique, OSIS), `reference`, `text`, `book`, `chapter`, `verse`, `char_count` |

### Edges
| Type | Direction | Properties |
|---|---|---|
| `CONTAINS` | Book→Chapter→Verse | none |
| `REFERENCES` | Verse→Verse | `votes` (int), `rank` (int, 1=strongest) |
| `IS_SIMILAR` | Verse→Verse | `score` (float, cosine), `rank` (int, 1=strongest) |

---

## Implementation Steps (Current State)

Track progress here as each step completes.

- [ ] **Step 1**: WEB text ingested — Book / Chapter / Verse nodes + CONTAINS edges
- [ ] **Step 2**: REFERENCES edges imported from OpenBible (votes ≥ 10, rank ≤ 20)
- [ ] **Step 3a**: Embeddings generated for all verses (checkpoint file present)
- [ ] **Step 3b**: HNSW vector index built and status = ONLINE
- [ ] **Step 3c**: IS_SIMILAR edges written (rank ≤ 5 per verse)
- [ ] **Step 4**: Validation queries passed (see Section VII of PRD)
- [ ] **Step 5**: Bloom Perspective configured

### Scripts Written (2026-03-11)

| File | Status |
|---|---|
| `PLAN.md` | ✅ Written |
| `utils/neo4j_conn.py` | ✅ Written |
| `step1_ingest_web.py` | ✅ Written — awaiting WEB CSV download to run |
| `step2_references.py` | ✅ Written — awaiting Step 1 completion |
| `step3a_embeddings.py` | ✅ Written — awaiting Step 1 completion |
| `step3b_build_index.py` | ✅ Written — awaiting Step 3a completion |
| `step3c_knn.py` | ✅ Written — awaiting Step 3b ONLINE status |
| `step4_validate.py` | ✅ Written — run after all prior steps |
| `logs/step1_2026-03-11.md` | ✅ Written |

---

## Data Sources

| Dataset | URL | License |
|---|---|---|
| WEB Bible text | https://ebible.org/find/show.php?id=eng-web | Public Domain |
| OpenBible cross-references | https://a.openbible.info/data/cross-references.zip | CC0 |

**Do not use NIV text.** NIV is copyrighted by Biblica. Using it in a database,
generating embeddings from it, or serving it via an API is unauthorized redistribution.

---

## Cypher Patterns — Use These, Not Others

```cypher
-- ✅ Correct: undirected IS_SIMILAR (captures both directions)
MATCH (v:Verse {id: $id})-[:IS_SIMILAR]-(sim) WHERE sim.rank <= 5

-- ❌ Wrong: directed IS_SIMILAR (misses ~40% of semantic connections)
MATCH (v:Verse {id: $id})-[:IS_SIMILAR]->(sim)

-- ✅ Correct: safe expand with LIMIT backstop
MATCH (v:Verse {reference: $reference})
OPTIONAL MATCH (v)-[r1:REFERENCES]->(ref:Verse) WHERE r1.rank <= 5
OPTIONAL MATCH (v)-[r2:IS_SIMILAR]-(sim:Verse)  WHERE r2.rank <= 5
RETURN v, r1, ref, r2, sim
LIMIT 50

-- ✅ Correct: constraints before MERGE (never after)
CREATE CONSTRAINT verse_id_unique IF NOT EXISTS
FOR (v:Verse) REQUIRE v.id IS UNIQUE;
```

---

## Validation Queries (run after each pipeline step)

```cypher
-- Node count (expected: 31,102)
MATCH (v:Verse) RETURN count(v) AS total_verses;

-- REFERENCES min votes (expected: >= 10)
MATCH ()-[r:REFERENCES]->() RETURN min(r.votes) AS min_votes;

-- IS_SIMILAR max outgoing degree (expected: <= 5)
MATCH (v:Verse)-[:IS_SIMILAR]->()
WITH v, count(*) AS degree ORDER BY degree DESC LIMIT 1
RETURN v.reference, degree;

-- Isolated verse check (expected: 0)
MATCH (v:Verse)
WHERE NOT (v)-[:REFERENCES]-() AND NOT (v)-[:IS_SIMILAR]-()
RETURN count(v) AS isolated;
```

---

## Spot-Check Verses

When validating data quality, always inspect these three verses manually:

| Verse | Why |
|---|---|
| `Romans 8:28` | High-traffic NT verse, should have rich REFERENCES and IS_SIMILAR neighbors |
| `John 3:16` | Most cross-referenced verse — good stress test for hairball prevention |
| `Psalm 23:1` | OT anchor — validates OT→NT semantic edge discovery |

---

## Python File Conventions

| File | Purpose |
|---|---|
| `step1_ingest_web.py` | Parse WEB CSV, write Book/Chapter/Verse nodes |
| `step2_references.py` | Load OpenBible CSV, filter, write REFERENCES edges |
| `step3a_embeddings.py` | Generate embeddings with checkpoint resume |
| `step3b_build_index.py` | Create HNSW vector index, wait for ONLINE status |
| `step3c_knn.py` | Offline KNN, write IS_SIMILAR edges |
| `step4_validate.py` | Run all validation queries, print pass/fail |
| `utils/neo4j_conn.py` | Shared Neo4j driver (reads NEO4J_URI, NEO4J_PASSWORD from env) |

---

## Environment Variables

```bash
ANTHROPIC_API_KEY=...      # OpenAI-compatible; used for embeddings
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...         # Never log this value
```

---

## Hard Rules for All Automated Tasks

1. **Never modify existing REFERENCES edges on re-runs** — use `MERGE`, not `CREATE`, but do not update `votes` or `rank` if the edge already exists (source data is static)
2. **Never skip the constraint creation step** — if constraints are missing, `MERGE` degrades to full table scan silently
3. **Never run KNN before the vector index status is `ONLINE`** — partial index produces silently incorrect nearest-neighbor results
4. **Never expose `NEO4J_PASSWORD` or `ANTHROPIC_API_KEY` in logs or output files**
5. **Never touch files outside the project directory** — scope all file writes to `./` or subdirectories
6. **Always use `--allowedTools Read,Write` in headless CI** — never open Bash in automated pipelines unless explicitly required and scoped

---

## Neo4j Memory Configuration (required before Step 3b)

Add to `neo4j.conf` before building the vector index:

```
dbms.memory.heap.initial_size=2G
dbms.memory.heap.max_size=4G
dbms.memory.pagecache.size=1G
```

Restart Neo4j after this change. Skipping this causes OOM during HNSW index construction.

---

## Common Failure Modes

| Symptom | Likely Cause | Fix |
|---|---|---|
| `MERGE` creates 0 edges | OSIS ID mismatch between WEB and OpenBible | Run spot-check: print 5 verse IDs from each source and compare |
| Vector index stuck at < 100% | Heap too small | Increase heap, restart, rebuild index |
| IS_SIMILAR edges missing for short verses | `char_count < 25` filter working correctly | Expected behavior, not a bug |
| KNN returns verse itself as top result | Self-exclusion WHERE clause missing | Check `WHERE similar.id <> $verse_id` is present |
| Embedding job restarts from zero | Checkpoint file deleted or path wrong | Verify `CHECKPOINT_FILE` path; check file exists before starting |
