# BibleMind Graph Explorer

An interactive knowledge graph for exploring the Bible — connecting 31,000+ verses through human-curated cross-references and AI-powered semantic similarity.

![Graph Explorer](https://img.shields.io/badge/Neo4j-5.x-blue) ![Python](https://img.shields.io/badge/Python-3.9+-green) ![Dash](https://img.shields.io/badge/Dash-2.9+-red) ![License](https://img.shields.io/badge/License-MIT-yellow)

---
<video src="https://files.catbox.moe/2tzcb1.mov" width="600" controls></video>



## What It Does

BibleMind builds a graph database of the entire Bible (World English Bible, public domain) and lets you explore how verses connect to each other in two distinct ways:

- **Cross-references** — community-curated links from [OpenBible.info](https://openbible.info), ranked by vote count (39,000+ edges)
- **Semantic similarity** — OpenAI embeddings + HNSW nearest-neighbor search to find verses that share meaning even when the words differ (155,000+ edges)

The Dash-powered UI lets you search for any verse, visualize its neighborhood as an interactive graph, expand nodes, and get AI-generated explanations of individual verses and their connections.

---

## Graph at a Glance

| Stat | Value |
|---|---|
| Bible text | World English Bible (WEB), public domain |
| Verses | 31,098 across 66 books |
| REFERENCES edges | 39,361 (votes ≥ 10, rank ≤ 20 per verse) |
| IS_SIMILAR edges | 155,085 (cosine KNN, rank ≤ 5 per verse) |
| Embedding model | `text-embedding-3-small` (1536 dimensions) |
| Graph backend | Neo4j 5.x with native HNSW vector index |

---

## Features

- **Interactive graph** — force-directed layout (cose-bilkent) with label-aware node spacing to prevent text overlap
- **Two edge types** — blue solid arrows for cross-references, teal dashed for semantic similarity; each filterable and rank-limited via sliders
- **Search & browse** — search by verse reference (`Romans 8:28, John 3:16`) or browse by Book → Chapter → Verse
- **Node actions** — click any node to instantly see its verse text; then Expand (add its neighborhood), Explain (AI analysis), or Remove
- **AI explanations** — clicking Explain calls GPT-4o-mini to explain the verse's meaning and its connection to every neighbor visible in the current graph; response streams in with a typewriter effect
- **Physics toggle** — auto-layout on or off; drag nodes freely when physics is disabled

---

## Architecture

```
BibleMind/
├── app_dash.py          # Dash UI — layout, callbacks, OpenAI integration
├── queries.py           # Neo4j data layer — all Cypher queries
├── utils/
│   └── neo4j_conn.py    # Neo4j driver factory
├── step1_ingest_web.py  # Parse WEB Bible CSV → Book/Chapter/Verse nodes
├── step2_references.py  # OpenBible cross-references → REFERENCES edges
├── step3a_embeddings.py # OpenAI embeddings → Verse.embedding property
├── step3b_build_index.py# HNSW vector index on Verse.embedding
├── step3c_knn.py        # KNN → IS_SIMILAR edges
└── step4_validate.py    # Full validation suite
```

### Graph Schema

```
(Book)-[:CONTAINS]->(Chapter)-[:CONTAINS]->(Verse)

(Verse)-[:REFERENCES {rank, votes}]->(Verse)   # human-curated, directed
(Verse)-[:IS_SIMILAR {rank, score}]-(Verse)    # embedding KNN, queried undirected
```

---

## Prerequisites

- Python 3.9+
- Neo4j 5.11+ (Community Edition works)
- OpenAI API key (for embeddings + explanations)

---

## Setup

### 1. Install Neo4j

```bash
# macOS
brew install neo4j
brew services start neo4j

# Set password via Neo4j browser at http://localhost:7474
```

Before building the vector index, increase Neo4j's heap in `neo4j.conf`:

```
server.memory.heap.initial_size=2g
server.memory.heap.max_size=4g
server.memory.pagecache.size=1g
```

### 2. Install Python dependencies

```bash
pip install neo4j python-dotenv openai dash dash-cytoscape dash-bootstrap-components
```

### 3. Configure environment

Create a `.env` file in the project root:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
OPENAI_API_KEY=sk-...
```

---

## Building the Knowledge Graph

Run the pipeline steps in order. Each step is idempotent (safe to re-run).

### Step 1 — Ingest Bible text

Download the [World English Bible VPL](https://ebible.org/find/show.php?id=eng-web), convert to CSV, then:

```bash
python step1_ingest_web.py --csv web.csv
```

Creates 31,098 `Verse` nodes, 66 `Book` nodes, and `CONTAINS` edges.

### Step 2 — Import cross-references

Download [OpenBible cross-references](https://a.openbible.info/data/cross-references.zip), then:

```bash
python step2_references.py --zip cross-references.zip
```

Creates 39,000+ `REFERENCES` edges filtered to votes ≥ 10 and rank ≤ 20 per verse.

### Step 3a — Generate embeddings

```bash
python step3a_embeddings.py
```

Calls OpenAI `text-embedding-3-small` for every verse longer than 25 characters. Checkpointed — safe to interrupt and resume.

### Step 3b — Build vector index

```bash
python step3b_build_index.py
```

Creates an HNSW index on `Verse.embedding` and waits for `ONLINE` status before exiting.

### Step 3c — Compute KNN edges

```bash
python step3c_knn.py
```

Queries the vector index for each verse's top-5 nearest neighbors and writes `IS_SIMILAR` edges.

### Step 4 — Validate

```bash
python step4_validate.py
```

Runs the full validation suite: node counts, edge constraints, spot-checks on Romans 8:28, John 3:16, and Psalm 23:1.

---

## Running the UI

```bash
python app_dash.py
```

Open [http://localhost:8050](http://localhost:8050).

---

## How to Use the Explorer

1. **Search** — type a verse reference in the search box (e.g. `John 3:16`) or browse Book → Chapter → Verse in the sidebar
2. **Explore** — the graph renders with the focus verse in red, cross-reference neighbors in blue, and semantic neighbors in teal. Gold nodes appear in both categories.
3. **Click a node** — the verse text appears instantly in the right panel; an action bar offers three options:
   - **Expand** — adds that verse's own neighborhood to the graph
   - **Explain** — asks GPT-4o-mini to explain the verse and its connections to every currently visible neighbor
   - **Remove** — hides the node from the current view
4. **Filter** — use the edge sliders to limit how many cross-references or similar verses appear (rank 1 = strongest connection)
5. **Multi-verse** — search multiple verses at once: `Romans 8:28, John 3:16`

---

## Data Sources & Licenses

| Dataset | Source | License |
|---|---|---|
| Bible text | World English Bible (WEB) — ebible.org | Public Domain |
| Cross-references | OpenBible.info | CC0 |
| Embeddings | OpenAI `text-embedding-3-small` | OpenAI ToS |

**Note:** The WEB Bible is used specifically because it is public domain. Do not substitute copyrighted translations (NIV, ESV, etc.) — embedding or serving them via an API constitutes unauthorized redistribution.

---

## License

MIT
