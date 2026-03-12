# Bible Graph Explorer
## Product Requirements & Technical Design Document v2.1

---

## I. Design Principles — What Changed and Why

The original document's 80/20 direction was correct, but three issues needed to be resolved before writing a single line of code, otherwise they would become hard blockers later:

| Problem | Original Approach | Revised Approach |
|---|---|---|
| Copyright | "Obtain NIV through compliant means" | Explicitly switch to WEB (World English Bible) |
| Hairball prevention | Delegate display limits to the frontend | Implement a three-layer truncation mechanism at the data layer |
| KNN computation | Mentioned but not specified | Explicitly use Neo4j's native HNSW vector index |

---

## II. Product Vision (Revised)

Build a Bible knowledge graph based on the **World English Bible (WEB)** — a complete modern English translation with no copyright restrictions, free for commercial use and redistribution. Through an interactive visual graph, reveal cross-references and deep semantic relationships between verses to support systematic theological research.

> **Why not NIV?**
> NIV is owned by Biblica. Storing its text in a database, generating embeddings from it, and serving it through a retrieval interface constitutes unauthorized redistribution and carries real legal risk. WEB is linguistically close to modern English and essentially equivalent to NIV for research purposes.

---

## III. Core Features (Maintaining the Original 80%)

1. **Hierarchical Search & Navigation**: Locate any Book / Chapter / Verse via search box
2. **Expand on Demand**: Double-click a node to load its strongest connected neighbors (Top N, controlled at the data layer)
3. **Dual Relationship Types**:
   - `[REFERENCES]`: Human-curated traditional cross-references from OpenBible (votes ≥ 10, ~100K high-quality edges)
   - `[IS_SIMILAR]`: Embedding-based semantic similarity (Top 5 per verse, ~155K edges)
4. **Text Reading Panel**: Click a node to display its WEB full text in a sidebar

---

## IV. Data Model (Graph Schema)

### Nodes

```
(:Book)    {name: "Romans", testament: "NT", book_order: 45}
(:Chapter) {number: 8, verse_count: 39}
(:Verse)   {
    id: "ROM.8.28",            ← Standardized ID for cross-dataset alignment
    reference: "Romans 8:28",
    text: "We know that all things work together ...",
    book: "Romans",
    chapter: 8,
    verse: 28,
    char_count: 142            ← Used to filter ultra-short verses (e.g. "Jesus wept")
}
```

> **Why the `id` field?**
> OpenBible uses OSIS format (e.g. `ROM.8.28`). The WEB text must be aligned to the same format. Without this, 90% of `MERGE` operations in Step 2 will silently fail due to key mismatches — no errors thrown, but no edges created.

### Edges

```
[:CONTAINS]    Book → Chapter → Verse  (structural hierarchy, no extra properties)
[:REFERENCES]  Verse → Verse  {votes: int, rank: int}
[:IS_SIMILAR]  Verse → Verse  {score: float, rank: int}
```

> **Why store `rank` on edges?**
> At write time, each edge is ranked among all same-type outgoing edges from its source node (1 = strongest). At query time, Cypher filters with `WHERE rank <= 5` directly — no in-memory sorting, consistently fast reads.

---

## V. Implementation — Four Steps with Full Detail

### Step 1: Data Ingestion (WEB Text)

**Data source**: Download WEB in USFM or CSV format from `https://ebible.org/find/show.php?id=eng-web`

```python
# Core logic: parse CSV into standardized nodes, generate OSIS IDs
import pandas as pd
from neo4j import GraphDatabase

def build_verse_id(book_abbr, chapter, verse):
    """Generate OSIS standard ID, aligned with OpenBible"""
    return f"{book_abbr.upper()}.{chapter}.{verse}"

# Create constraints BEFORE bulk import — otherwise MERGE triggers full table scans
CREATE CONSTRAINT verse_id_unique IF NOT EXISTS
FOR (v:Verse) REQUIRE v.id IS UNIQUE;

CREATE CONSTRAINT book_name_unique IF NOT EXISTS
FOR (b:Book) REQUIRE b.name IS UNIQUE;
```

> **Critical**: Create constraints first, then run `MERGE`. Reversed order turns a 10-second import into a 10-minute one.

---

### Step 2: Inject Static Cross-References (REFERENCES Edges)

**Data source**: `https://a.openbible.info/data/cross-references.zip` (~340K raw entries)

#### Quality Control: Three-Layer Filter

```python
df = pd.read_csv("cross_references.csv", sep="\t",
                 names=["from_verse", "to_verse", "votes"])

# Layer 1: Drop low-quality references (votes < 10 are academically unreliable)
df = df[df["votes"] >= 10]                  # ~340K → ~100K

# Layer 2: Drop self-references
df = df[df["from_verse"] != df["to_verse"]]

# Layer 3: Rank outgoing edges per node (keep only Top 20 per source verse)
df["rank"] = df.groupby("from_verse")["votes"].rank(
    ascending=False, method="first").astype(int)
df = df[df["rank"] <= 20]

# Write to Neo4j
CYPHER = """
UNWIND $rows AS row
MATCH (a:Verse {id: row.from_verse})
MATCH (b:Verse {id: row.to_verse})
MERGE (a)-[r:REFERENCES]->(b)
SET r.votes = row.votes, r.rank = row.rank
"""
```

> **Why `votes >= 10`?**
> OpenBible's voting mechanism works like Stack Overflow. A single-vote reference is essentially noise. The threshold of 10 is the widely accepted community minimum for credible citations. This value is configurable.

---

### Step 3: Compute Semantic Similarity Edges (IS_SIMILAR Edges)

#### 3a. Generate Embeddings (Cost Estimate)

| Model | Price | Total cost for 31,102 verses |
|---|---|---|
| `text-embedding-3-small` | $0.02 / 1M tokens | **~$0.80** (recommended) |
| `text-embedding-3-large` | $0.13 / 1M tokens | ~$5.20 (marginal quality gain) |

~30 tokens per verse on average × 31,102 verses ≈ 930K tokens. **Total cost under $1.**

```python
import openai, json, os
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

client = openai.OpenAI()
CHECKPOINT_FILE = "embedding_checkpoint.json"

# Exponential backoff: first wait 1s, max 60s, up to 6 retries
# Better than time.sleep(0.5): zero wait under normal conditions,
# automatically lengthens interval on 429 errors
@retry(
    wait=wait_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError))
)
def embed_batch(texts):
    response = client.embeddings.create(input=texts, model="text-embedding-3-small")
    return [e.embedding for e in response.data]

def get_embeddings_with_checkpoint(verse_ids, texts, batch_size=100):
    """
    Checkpoint-based resume: if the job crashes at verse 25,000,
    restart picks up from where it left off — no work lost.
    Completed batches are persisted to a JSON file.
    """
    checkpoint = {}
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            checkpoint = json.load(f)

    results = dict(checkpoint)

    for i in range(0, len(verse_ids), batch_size):
        batch_ids = verse_ids[i:i+batch_size]
        batch_texts = texts[i:i+batch_size]

        # Skip already-completed batches
        if all(vid in results for vid in batch_ids):
            continue

        embeddings = embed_batch(batch_texts)
        for vid, emb in zip(batch_ids, embeddings):
            results[vid] = emb

        # Persist checkpoint after every batch
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(results, f)

        print(f"Progress: {min(i+batch_size, len(verse_ids))}/{len(verse_ids)}")

    return results
```

> **Why `tenacity` instead of `time.sleep`?**
> A hardcoded `sleep(0.5)` wastes time on every batch under normal conditions, yet still isn't enough when a real rate-limit hit occurs. Tenacity's exponential backoff adapts: 0 wait normally, then 1s → 2s → 4s → 8s... automatically. The checkpoint system ensures a 31K-verse long-running job survives network interruptions.

#### 3b. Build HNSW Vector Index (Neo4j ≥ 5.11)

```cypher
-- Native vector index: no external Faiss or Annoy needed
CREATE VECTOR INDEX verse_embedding_idx IF NOT EXISTS
FOR (v:Verse) ON (v.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
  }
}
```

> **⚠️ Increase Neo4j memory BEFORE building the index.**
> 31,102 verses × 1536 dimensions × 4 bytes (float32) ≈ **190 MB** of raw vector data. On top of graph structure and page cache, Neo4j Desktop's default 1GB heap will trigger GC pauses or OOM during HNSW index construction.
>
> In Neo4j Desktop → Database Settings → `neo4j.conf`:
> ```
> dbms.memory.heap.initial_size=2G
> dbms.memory.heap.max_size=4G
> dbms.memory.pagecache.size=1G
> ```
> Restart the database after this change. Only run KNN after the index status shows `ONLINE`.

#### 3c. Offline KNN + Write IS_SIMILAR Edges

```python
# Query Top 6 neighbors per verse (+1 because the verse itself appears in results)
CYPHER_KNN = """
MATCH (v:Verse {id: $verse_id})
CALL db.index.vector.queryNodes('verse_embedding_idx', 6, v.embedding)
YIELD node AS similar, score
WHERE similar.id <> $verse_id    -- exclude self
WITH similar, score,
     rank() OVER (ORDER BY score DESC) AS rank
MERGE (v)-[r:IS_SIMILAR]->(similar)
SET r.score = round(score, 4), r.rank = rank
"""
```

> **On edge directionality: write directed, query undirected.**
> Semantic similarity is inherently symmetric (if A is like B, B is like A). But KNN results are directional — Romans 8:28's Top 5 may include Genesis 50:20, while Genesis 50:20's Top 5 may be occupied by other OT verses and never point back.
>
> **Solution**: Write directed edges (`MERGE (v)-[:IS_SIMILAR]->(similar)`) to preserve data lineage, but use **undirected matching** in all Bloom configurations and Cypher queries:
> ```cypher
> -- ✅ Undirected: captures relationships in both directions
> MATCH (v:Verse {reference: $ref})-[:IS_SIMILAR]-(similar)
> WHERE similar.rank <= 5
> RETURN similar
>
> -- ❌ Directed: misses ~40% of semantic connections
> MATCH (v:Verse {reference: $ref})-[:IS_SIMILAR]->(similar)
> ```
> This distinction is especially impactful for NT-quotes-OT patterns.

#### 3d. Filter Ultra-Short Verses

```python
# Ultra-short verses (e.g. John 11:35 "Jesus wept.") produce poor-quality embeddings.
# They become noisy similarity hubs — falsely high-degree nodes in the graph.
# Threshold: char_count < 25
#   - Filters genuine standalone one-liners (John 11:35 = 11 chars)
#   - Preserves substantive short commandments: "Do not steal." (Exodus 20:15 = 14 chars → KEPT)
#   - The old 40-char threshold was too strict and removed core commandments and proverbs
MIN_CHAR_COUNT = 25  # Configurable
df_verses = df_verses[df_verses["char_count"] >= MIN_CHAR_COUNT]
```

> **Why 25 and not 40?**
> The Ten Commandments are around 14 characters each but carry enormous theological weight. A 40-char cutoff incorrectly excludes commandments, proverbs, and prophetic short-form sayings. 25 chars precisely filters true standalone one-liners (typically < 20 chars) while keeping theologically dense short verses.

---

### Step 4: Three-Layer Hairball Prevention (Data Layer)

> The original doc delegated this to the frontend. That is not sufficient. Bloom's expand controls are coarse — opening John 3:16 (most cross-referenced verse) can trigger cascading expansion. **Truncation must be enforced at the data layer.**

**Three layers:**

| Layer | Strategy | Where |
|---|---|---|
| At write time | `REFERENCES`: max Top 20 outgoing edges per verse | Python script (Step 2) |
| At write time | `IS_SIMILAR`: max Top 5 outgoing edges per verse | Python script (Step 3) |
| At query time | Bloom search phrases filter `rank <= N` | Cypher query templates |

**Cypher query template (register as a Bloom Search Phrase):**

```cypher
-- Safe expand: undirected IS_SIMILAR match, hard LIMIT as backstop
MATCH (v:Verse {reference: $reference})
OPTIONAL MATCH (v)-[r1:REFERENCES]->(ref:Verse) WHERE r1.rank <= 5
OPTIONAL MATCH (v)-[r2:IS_SIMILAR]-(sim:Verse) WHERE r2.rank <= 5   -- note: undirected
RETURN v, r1, ref, r2, sim
LIMIT 50
```

---

## VI. Neo4j Bloom Configuration

### Where Bloom Works Well

- **Solo research**: Zero frontend code, fully sufficient
- **Small local team (2–5 people)**: Share via Neo4j Desktop, workable

### Bloom's Real Limitations

| Limitation | Impact |
|---|---|
| No shareable URLs | Cannot send a specific graph view to collaborators |
| Not embeddable | Cannot publish inside a website or blog |
| Steep learning curve for non-technical users | Not suitable for general audiences |

### If You Need Multi-User Access Later

Recommended fallback (**no React + D3.js required**):

**Streamlit + `streamlit-agraph`** — ~200 lines of Python to build a shareable web interface that connects directly to Neo4j and renders an interactive graph. Still 80/20, but with URL sharing.

**Three things Streamlit adds that Bloom cannot do:**

1. **Book dropdown selector**: Users pick from 66 books without knowing any Cypher syntax
2. **`rank` threshold slider**: Real-time toggle between "Top 3 strongest connections" and "Top 10 for broader exploration" — no query editing required
3. **One-click export**: Save the current view as PNG or export the node/edge list as CSV for papers or teaching materials

Each of these takes 10–20 lines of Python to implement.

---

## VII. Data Quality Validation (Absent from Original)

Run all of these after the full data pipeline completes:

```cypher
-- 1. Basic completeness check
MATCH (v:Verse) RETURN count(v) AS total_verses;
-- Expected: 31,102

-- 2. REFERENCES edge quality check
MATCH ()-[r:REFERENCES]->() RETURN count(r), avg(r.votes), min(r.votes);
-- min(votes) should be >= 10

-- 3. IS_SIMILAR degree distribution (find anomalously high-degree nodes)
MATCH (v:Verse)-[:IS_SIMILAR]->()
WITH v, count(*) AS degree
ORDER BY degree DESC LIMIT 10
RETURN v.reference, degree;
-- Each verse should have at most 5 outgoing edges; higher means a bug

-- 4. Isolated node check (should be 0)
MATCH (v:Verse)
WHERE NOT (v)-[:REFERENCES]-() AND NOT (v)-[:IS_SIMILAR]-()
RETURN count(v) AS isolated_verses;

-- 5. Spot-check: manual quality inspection
MATCH (v:Verse {reference: "Romans 8:28"})
OPTIONAL MATCH (v)-[r:REFERENCES]->(ref)
RETURN v.reference, collect({ref: ref.reference, votes: r.votes}) AS refs
ORDER BY r.votes DESC LIMIT 5;
```

---

## VIII. Implementation Checklist

### Environment Setup
- [ ] Install Neo4j Desktop (free, local)
- [ ] Confirm Neo4j version ≥ 5.11 (required for native vector index)
- [ ] Configure Neo4j memory: heap max 4G, pagecache 1G (must be done before building the index)
- [ ] Install Python dependencies: `pip install neo4j openai pandas tenacity`

### Data Preparation
- [ ] Download WEB text (ebible.org), confirm Public Domain license
- [ ] Download OpenBible cross-reference dataset
- [ ] Verify OSIS ID format alignment between WEB and OpenBible (spot-check 20 entries)

### Data Ingestion
- [ ] Create uniqueness constraints (before any MERGE operations)
- [ ] Import Book / Chapter / Verse nodes
- [ ] Import REFERENCES edges (filter votes < 10)
- [ ] Upload embeddings to Neo4j (test with 100 verses first to verify dimensions)
- [ ] Build HNSW vector index; wait for status `ONLINE` before running KNN
- [ ] Run offline KNN; write IS_SIMILAR edges

### Validation
- [ ] Run all validation queries from Section VII
- [ ] Manual spot-check: Romans 8:28 / John 3:16 / Psalm 23:1 neighbor quality
- [ ] Confirm single expand returns no more than 10 new nodes

### Bloom Configuration
- [ ] Register Search Phrases (Verse lookup template)
- [ ] Set node styling (color by Book)
- [ ] Save Perspective config file (for team sharing)

---

## IX. Phased Delivery Plan

The project can be delivered in three independent phases, each with a usable artifact:

| Phase | Content | Deliverable | Estimated Time |
|---|---|---|---|
| **Phase 1** | WEB ingestion + REFERENCES edges | Explorable cross-reference graph | 1 day |
| **Phase 2** | Embedding generation + IS_SIMILAR edges | Semantic similarity layer | Half day (+ API wait time) |
| **Phase 3** | Bloom configuration + validation | Complete research tool | Half day |

**Start with Phase 1.** OpenBible's human-curated cross-references are extremely high quality on their own — Phase 1 alone is a genuinely valuable research graph. Phase 2's semantic layer is an enhancement, not a prerequisite.

---

*Document version: v2.1 | Key improvements: copyright fix (WEB), three-layer data-layer hairball prevention, HNSW vector index path, tenacity exponential backoff + checkpoint resume, IS_SIMILAR undirected queries, short-verse threshold correction (40→25), Neo4j memory configuration, Streamlit fallback with feature detail*
