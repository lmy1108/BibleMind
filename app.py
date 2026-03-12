"""
app.py — BibleMind Graph Explorer (Streamlit UI)
-------------------------------------------------
Interactive visualization of the Bible knowledge graph.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from neo4j import GraphDatabase
from streamlit_agraph import agraph, Node, Edge, Config

# ---------------------------------------------------------------------------
# Config & connection
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).parent / ".env")

NEO4J_URI      = os.environ.get("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.environ.get("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

COLOR_FOCUS = "#E63946"   # red   — focus / expanded
COLOR_REF   = "#457B9D"   # blue  — REFERENCES neighbors
COLOR_SIM   = "#2A9D8F"   # teal  — IS_SIMILAR neighbors
COLOR_BOTH  = "#E9C46A"   # gold  — both


@st.cache_resource
def get_driver():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    return driver


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def fetch_books(driver) -> list[str]:
    with driver.session() as s:
        return [r["name"] for r in s.run(
            "MATCH (b:Book) RETURN b.name AS name ORDER BY b.book_order"
        )]


def fetch_chapters(driver, book: str) -> list[int]:
    with driver.session() as s:
        return [r["n"] for r in s.run(
            "MATCH (:Book {name:$book})-[:CONTAINS]->(c:Chapter) "
            "RETURN c.number AS n ORDER BY c.number",
            book=book,
        )]


def fetch_verses_in_chapter(driver, book: str, chapter: int) -> list[dict]:
    with driver.session() as s:
        return [
            {"verse": r["verse"], "reference": r["reference"]}
            for r in s.run(
                "MATCH (:Book {name:$book})-[:CONTAINS]->(:Chapter {number:$ch})"
                "-[:CONTAINS]->(v:Verse) "
                "RETURN v.verse AS verse, v.reference AS reference ORDER BY v.verse",
                book=book, ch=chapter,
            )
        ]


def search_verse(driver, query: str) -> list[dict]:
    with driver.session() as s:
        return [
            {"id": r["id"], "reference": r["reference"]}
            for r in s.run(
                "MATCH (v:Verse) "
                "WHERE toLower(v.reference) CONTAINS toLower($q) "
                "   OR toLower(v.id) CONTAINS toLower($q) "
                "RETURN v.id AS id, v.reference AS reference "
                "ORDER BY v.id LIMIT 20",
                q=query,
            )
        ]


def get_verse(driver, verse_id: str) -> dict | None:
    with driver.session() as s:
        r = s.run(
            "MATCH (v:Verse {id:$id}) RETURN v.reference AS reference, v.text AS text",
            id=verse_id,
        ).single()
        return {"reference": r["reference"], "text": r["text"]} if r else None


def fetch_neighborhood(
    driver, verse_id: str, ref_rank: int, sim_rank: int, show_refs: bool, show_sim: bool,
) -> tuple[dict, list[tuple], dict[str, str]]:
    neighbors: dict[str, dict] = {}
    raw_edges: list[tuple] = []
    texts: dict[str, str] = {}

    with driver.session() as s:
        focus = s.run(
            "MATCH (v:Verse {id:$id}) RETURN v.reference AS ref, v.text AS text",
            id=verse_id,
        ).single()
        if not focus:
            return {}, [], {}

        texts[verse_id] = focus["text"]

        if show_refs:
            for row in s.run(
                "MATCH (v:Verse {id:$id})-[r:REFERENCES]->(n:Verse) "
                "WHERE r.rank <= $rank "
                "RETURN n.id AS nid, n.reference AS ref, n.text AS text, r.rank AS rank, r.votes AS votes "
                "ORDER BY r.rank",
                id=verse_id, rank=ref_rank,
            ):
                nid = row["nid"]
                texts[nid] = row["text"]
                entry = neighbors.setdefault(nid, {"ref": False, "sim": False,
                                                    "reference": row["ref"], "text": row["text"]})
                entry["ref"] = True
                raw_edges.append((verse_id, nid, "ref"))

        if show_sim:
            for row in s.run(
                "MATCH (v:Verse {id:$id})-[r:IS_SIMILAR]-(n:Verse) "
                "WHERE r.rank <= $rank "
                "RETURN DISTINCT n.id AS nid, n.reference AS ref, n.text AS text, "
                "       r.rank AS rank, r.score AS score "
                "ORDER BY r.rank LIMIT 20",
                id=verse_id, rank=sim_rank,
            ):
                nid = row["nid"]
                texts[nid] = row["text"]
                entry = neighbors.setdefault(nid, {"ref": False, "sim": False,
                                                    "reference": row["ref"], "text": row["text"]})
                entry["sim"] = True
                raw_edges.append((verse_id, nid, "sim"))

    return neighbors, raw_edges, texts


def build_graph(
    driver,
    focus_ids: set[str],
    expanded_ids: set[str],
    removed_ids: set[str],
    ref_rank: int,
    sim_rank: int,
    show_refs: bool,
    show_sim: bool,
) -> tuple[list[Node], list[Edge], dict[str, str]]:
    all_expand = (focus_ids | expanded_ids) - removed_ids

    nodes_map: dict[str, Node] = {}
    seen_edges: set[tuple] = set()
    edge_list: list[Edge] = []
    all_texts: dict[str, str] = {}

    per_id_neighbors: dict[str, dict] = {}

    for vid in all_expand:
        neighbors, raw_edges, texts = fetch_neighborhood(
            driver, vid, ref_rank, sim_rank, show_refs, show_sim
        )
        all_texts.update(texts)
        # Filter out removed nodes from neighbors
        neighbors = {k: v for k, v in neighbors.items() if k not in removed_ids}
        per_id_neighbors[vid] = neighbors

        for (src, tgt, etype) in raw_edges:
            if tgt in removed_ids:
                continue
            key = (src, tgt, etype)
            if key not in seen_edges:
                seen_edges.add(key)
                if etype == "ref":
                    edge_list.append(Edge(source=src, target=tgt, color="#457B9D", dashes=False))
                else:
                    edge_list.append(Edge(source=src, target=tgt, color="#2A9D8F", dashes=True))

    # Resolve focus node labels
    if all_expand:
        with driver.session() as s:
            for row in s.run(
                "MATCH (v:Verse) WHERE v.id IN $ids "
                "RETURN v.id AS id, v.reference AS ref, v.text AS text",
                ids=list(all_expand),
            ):
                all_texts[row["id"]] = row["text"]
                text = row["text"]
                snippet = text[:120] + "…" if len(text) > 120 else text
                nodes_map[row["id"]] = Node(
                    id=row["id"],
                    label=row["ref"],
                    title=snippet,
                    color=COLOR_FOCUS,
                    size=30,
                    font={"size": 14, "bold": True},
                )

    # Neighbor nodes
    for vid, neighbors in per_id_neighbors.items():
        for nid, info in neighbors.items():
            if nid in all_expand:
                continue
            if nid in nodes_map:
                # Merge ref/sim flags
                prev = nodes_map[nid]
                prev_ref = prev.color in (COLOR_REF, COLOR_BOTH)
                prev_sim = prev.color in (COLOR_SIM, COLOR_BOTH)
                is_ref = prev_ref or info["ref"]
                is_sim = prev_sim or info["sim"]
                color = COLOR_BOTH if (is_ref and is_sim) else (COLOR_REF if is_ref else COLOR_SIM)
                nodes_map[nid] = Node(
                    id=nid, label=info["reference"],
                    title=(info["text"][:120] + "…" if len(info["text"]) > 120 else info["text"]),
                    color=color, size=20,
                )
            else:
                color = (COLOR_BOTH if (info["ref"] and info["sim"])
                         else COLOR_REF if info["ref"] else COLOR_SIM)
                snippet = info["text"][:120] + "…" if len(info["text"]) > 120 else info["text"]
                nodes_map[nid] = Node(
                    id=nid, label=info["reference"], title=snippet, color=color, size=20,
                )

    return list(nodes_map.values()), edge_list, all_texts


# ---------------------------------------------------------------------------
# Streamlit layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="BibleMind Graph Explorer", page_icon="✝", layout="wide")
st.title("BibleMind Graph Explorer")
st.caption("Explore cross-references and semantic similarity across 31K Bible verses.")

driver = get_driver()

# Session state
for key, default in [
    ("expanded_ids", set()),
    ("removed_ids",  set()),
    ("last_focus_ids", set()),
    ("pending_node", None),      # node awaiting action
    ("explain_node", None),      # node whose text is shown
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Search")
    search_query = st.text_input(
        "Verse reference(s)", placeholder="e.g. Romans 8:28, John 3:16",
    )
    st.caption("Separate multiple verses with commas.")

    st.divider()
    st.subheader("Browse by Book / Chapter")
    books = fetch_books(driver)
    sel_book = st.selectbox("Book", ["— select —"] + books)

    browse_verse_id: str | None = None
    if sel_book != "— select —":
        chapters = fetch_chapters(driver, sel_book)
        sel_chapter = st.selectbox("Chapter", chapters)
        verses = fetch_verses_in_chapter(driver, sel_book, sel_chapter)
        sel_verse_label = st.selectbox("Verse", ["— select —"] + [v["reference"] for v in verses])
        if sel_verse_label != "— select —":
            matched = [v for v in verses if v["reference"] == sel_verse_label]
            if matched:
                browse_verse_id = matched[0]["reference"]

    st.divider()
    st.subheader("Edge filters")
    show_refs = st.checkbox("REFERENCES  (blue, solid)", value=True)
    show_sim  = st.checkbox("IS_SIMILAR  (teal, dashed)", value=True)
    ref_rank  = st.slider("Max REFERENCES rank", 1, 20, 5)
    sim_rank  = st.slider("Max IS_SIMILAR rank", 1, 5, 3)

    st.divider()
    physics_on = st.checkbox("Auto-layout (physics)", value=True,
                             help="On: nodes arrange automatically and edges are springy. "
                                  "Off: drag nodes freely, edges stretch without snapping back.")

    st.divider()
    st.caption("**Node colours**")
    st.markdown(
        "<span style='color:#E63946'>●</span> Focus / expanded &nbsp;"
        "<span style='color:#457B9D'>●</span> REFERENCES &nbsp;"
        "<span style='color:#2A9D8F'>●</span> IS_SIMILAR &nbsp;"
        "<span style='color:#E9C46A'>●</span> Both",
        unsafe_allow_html=True,
    )
    st.caption("Click a node to see **Expand / Explain / Remove** options.")

    if st.session_state.expanded_ids or st.session_state.removed_ids:
        if st.button("Reset graph"):
            st.session_state.expanded_ids = set()
            st.session_state.removed_ids  = set()
            st.session_state.pending_node = None
            st.session_state.explain_node = None
            st.rerun()

# ── Resolve focus IDs ────────────────────────────────────────────────────────
focus_ids: set[str] = set()
raw_queries = [q.strip() for q in search_query.split(",") if q.strip()] if search_query else []
if not raw_queries and browse_verse_id:
    raw_queries = [browse_verse_id]

for q in raw_queries:
    results = search_verse(driver, q)
    if not results:
        st.warning(f"No verse found for '{q}'")
        continue
    exact = [r for r in results if r["reference"].lower() == q.lower()]
    if exact:
        focus_ids.add(exact[0]["id"])
    elif len(results) == 1:
        focus_ids.add(results[0]["id"])
    else:
        options = {r["reference"]: r["id"] for r in results}
        chosen = st.selectbox(f"Multiple matches for '{q}':", list(options.keys()))
        focus_ids.add(options[chosen])

if focus_ids != st.session_state.last_focus_ids:
    st.session_state.expanded_ids = set()
    st.session_state.removed_ids  = set()
    st.session_state.pending_node = None
    st.session_state.explain_node = None
    st.session_state.last_focus_ids = focus_ids

# ── Node action bar ──────────────────────────────────────────────────────────
if st.session_state.pending_node:
    pn = st.session_state.pending_node
    pn_data = get_verse(driver, pn)
    pn_label = pn_data["reference"] if pn_data else pn

    st.markdown(f"**Selected:** {pn_label}")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 3])

    with c1:
        if st.button("1️⃣ Expand", use_container_width=True):
            all_expand = focus_ids | st.session_state.expanded_ids
            if pn not in all_expand:
                st.session_state.expanded_ids.add(pn)
            st.session_state.pending_node = None
            st.session_state.explain_node = None
            st.rerun()

    with c2:
        if st.button("2️⃣ Explain", use_container_width=True):
            st.session_state.explain_node = pn
            st.session_state.pending_node = None
            st.rerun()

    with c3:
        if st.button("3️⃣ Remove", use_container_width=True):
            st.session_state.removed_ids.add(pn)
            st.session_state.pending_node = None
            st.session_state.explain_node = None
            st.rerun()

    with c4:
        if st.button("✕ Cancel", use_container_width=False):
            st.session_state.pending_node = None
            st.rerun()

    st.divider()

# ── Main graph area ──────────────────────────────────────────────────────────
col_graph, col_text = st.columns([3, 1])

with col_graph:
    if not focus_ids:
        st.info("Search for one or more verses, or browse by Book / Chapter / Verse.")
    else:
        nodes, edges, texts = build_graph(
            driver,
            focus_ids=focus_ids,
            expanded_ids=st.session_state.expanded_ids,
            removed_ids=st.session_state.removed_ids,
            ref_rank=ref_rank,
            sim_rank=sim_rank,
            show_refs=show_refs,
            show_sim=show_sim,
        )

        stats = f"**{len(nodes)} nodes · {len(edges)} edges**"
        if st.session_state.expanded_ids:
            stats += f"  ·  {len(st.session_state.expanded_ids)} expanded"
        if st.session_state.removed_ids:
            stats += f"  ·  {len(st.session_state.removed_ids)} removed"
        st.markdown(stats)

        config = Config(
            width="100%",
            height=620,
            directed=True,
            physics=physics_on,
            hierarchical=False,
            nodeHighlightBehavior=True,
            highlightColor="#F7A072",
            collapsible=False,
            node={"labelProperty": "label"},
            link={"renderLabel": False},
        )

        clicked = agraph(nodes=nodes, edges=edges, config=config)

        # On click: set pending node (don't auto-expand)
        if clicked and clicked != st.session_state.pending_node:
            st.session_state.pending_node = clicked
            st.rerun()

# ── Text panel ───────────────────────────────────────────────────────────────
with col_text:
    explain_id = st.session_state.explain_node
    if explain_id:
        data = get_verse(driver, explain_id)
        if data:
            st.subheader("📖 Verse")
            st.markdown(f"**{data['reference']}**")
            st.write(data["text"])
    elif focus_ids and "texts" in dir():
        st.subheader("Focus verse(s)")
        for vid in sorted(focus_ids):
            if vid in texts:
                label_str = next((n.label for n in nodes if n.id == vid), vid)
                st.markdown(f"**{label_str}**")
                st.write(texts[vid])
                st.divider()
