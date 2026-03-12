"""
queries.py — Neo4j data layer for BibleMind Graph Explorer
-----------------------------------------------------------
All database queries used by the Dash UI.
"""

from __future__ import annotations

# Node colours (shared with app_dash.py via import)
COLOR_FOCUS = "#E63946"  # red   — focus / expanded
COLOR_REF   = "#457B9D"  # blue  — REFERENCES neighbors
COLOR_SIM   = "#2A9D8F"  # teal  — IS_SIMILAR neighbors
COLOR_BOTH  = "#E9C46A"  # gold  — both


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
    driver,
    verse_id: str,
    ref_rank: int,
    sim_rank: int,
    show_refs: bool,
    show_sim: bool,
) -> tuple[dict, list[tuple], dict[str, str]]:
    """
    Returns:
      neighbors: {id: {"ref": bool, "sim": bool, "reference": str, "text": str}}
      raw_edges: list of (source, target, type)  where type is "ref" or "sim"
      texts:     {id: text}
    """
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
                "RETURN n.id AS nid, n.reference AS ref, n.text AS text "
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
                "RETURN DISTINCT n.id AS nid, n.reference AS ref, n.text AS text, min(r.rank) AS rank "
                "ORDER BY rank LIMIT 20",
                id=verse_id, rank=sim_rank,
            ):
                nid = row["nid"]
                texts[nid] = row["text"]
                entry = neighbors.setdefault(nid, {"ref": False, "sim": False,
                                                    "reference": row["ref"], "text": row["text"]})
                entry["sim"] = True
                raw_edges.append((verse_id, nid, "sim"))

    return neighbors, raw_edges, texts


def build_graph_elements(
    driver,
    focus_ids: set[str],
    expanded_ids: set[str],
    removed_ids: set[str],
    ref_rank: int,
    sim_rank: int,
    show_refs: bool,
    show_sim: bool,
) -> tuple[list[dict], dict[str, str]]:
    """
    Build Cytoscape element dicts for all focus + expanded nodes and their neighbors.
    Returns (elements, texts) where:
      elements: list of {"data": {...}} dicts — nodes first, then edges
      texts:    {verse_id: text} for the verse panel
    """
    all_expand = (focus_ids | expanded_ids) - removed_ids

    node_data: dict[str, dict] = {}   # id → node data dict
    seen_edges: set[tuple] = set()
    edge_elements: list[dict] = []
    all_texts: dict[str, str] = {}

    # Fetch neighborhoods for all expanded IDs
    per_id_neighbors: dict[str, dict] = {}
    for vid in all_expand:
        neighbors, raw_edges, texts = fetch_neighborhood(
            driver, vid, ref_rank, sim_rank, show_refs, show_sim
        )
        all_texts.update(texts)
        neighbors = {k: v for k, v in neighbors.items() if k not in removed_ids}
        per_id_neighbors[vid] = neighbors

        for (src, tgt, etype) in raw_edges:
            if tgt in removed_ids:
                continue
            key = (src, tgt, etype)
            if key not in seen_edges:
                seen_edges.add(key)
                edge_elements.append({"data": {"source": src, "target": tgt, "type": etype}})

    # Resolve focus node details
    if all_expand:
        with driver.session() as s:
            for row in s.run(
                "MATCH (v:Verse) WHERE v.id IN $ids "
                "RETURN v.id AS id, v.reference AS ref, v.text AS text",
                ids=list(all_expand),
            ):
                all_texts[row["id"]] = row["text"]
                node_data[row["id"]] = {
                    "id": row["id"], "label": row["ref"],
                    "color": COLOR_FOCUS, "size": 30,
                }

    # Build neighbor nodes
    for vid, neighbors in per_id_neighbors.items():
        for nid, info in neighbors.items():
            if nid in all_expand:
                continue
            if nid in node_data:
                # Upgrade color if node already seen as neighbor of another focus
                prev_color = node_data[nid]["color"]
                prev_ref = prev_color in (COLOR_REF, COLOR_BOTH)
                prev_sim = prev_color in (COLOR_SIM, COLOR_BOTH)
                is_ref = prev_ref or info["ref"]
                is_sim = prev_sim or info["sim"]
                color = COLOR_BOTH if (is_ref and is_sim) else (COLOR_REF if is_ref else COLOR_SIM)
                node_data[nid]["color"] = color
            else:
                color = (COLOR_BOTH if (info["ref"] and info["sim"])
                         else COLOR_REF if info["ref"] else COLOR_SIM)
                node_data[nid] = {
                    "id": nid, "label": info["reference"],
                    "color": color, "size": 20,
                }

    # Nodes must come before edges
    node_elements = [{"data": d} for d in node_data.values()]
    return node_elements + edge_elements, all_texts
