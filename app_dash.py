"""
app_dash.py — BibleMind Graph Explorer (Dash + dash-cytoscape)
--------------------------------------------------------------
Run with:  python app_dash.py
Then open: http://localhost:8050
"""

from __future__ import annotations

import os
from pathlib import Path

import openai
import dash
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from dash import Input, Output, State, callback_context as ctx, dcc, html, no_update
from dotenv import load_dotenv

from queries import (
    build_graph_elements,
    fetch_books,
    fetch_chapters,
    fetch_verses_in_chapter,
    get_verse,
    search_verse,
)
from utils.neo4j_conn import get_driver

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).parent / ".env")
cyto.load_extra_layouts()  # loads cose-bilkent, cola, etc.

driver = get_driver()

EMPTY_GRAPH_STATE = {"focus_ids": [], "expanded_ids": [], "removed_ids": []}

# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------

STYLESHEET = [
    {
        "selector": "node",
        "style": {
            "background-color": "data(color)",
            "label": "data(label)",
            "font-size": "10px",
            "font-family": "Georgia, serif",
            "color": "#111111",
            "text-valign": "bottom",
            "text-halign": "center",
            "text-margin-y": "4px",
            "text-wrap": "wrap",
            "text-max-width": "90px",
            "width": "data(size)",
            "height": "data(size)",
            "border-width": "1.5px",
            "border-color": "#ffffff",
            "cursor": "pointer",
        },
    },
    {
        "selector": "node:selected",
        "style": {
            "border-width": "3px",
            "border-color": "#FFD700",
            "border-opacity": 1,
        },
    },
    {
        "selector": "edge[type = 'ref']",
        "style": {
            "line-color": "#457B9D",
            "target-arrow-color": "#457B9D",
            "target-arrow-shape": "triangle",
            "arrow-scale": 1.2,
            "curve-style": "bezier",
            "width": 1.5,
            "opacity": 0.7,
        },
    },
    {
        "selector": "edge[type = 'sim']",
        "style": {
            "line-color": "#2A9D8F",
            "line-style": "dashed",
            "line-dash-pattern": [6, 3],
            "curve-style": "bezier",
            "width": 1.5,
            "opacity": 0.6,
        },
    },
    {
        "selector": "edge:selected",
        "style": {"width": 3, "opacity": 1.0},
    },
]

# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def color_dot(color: str, label: str) -> html.Span:
    return html.Span([
        html.Span("●", style={"color": color, "marginRight": "4px"}),
        html.Span(label, style={"marginRight": "12px", "fontSize": "12px"}),
    ])


def build_sidebar() -> dbc.Col:
    books = fetch_books(driver)

    return dbc.Col([
        html.H6("Search", className="fw-bold mt-3"),
        dbc.Input(
            id="search-input",
            placeholder="e.g. Romans 8:28, John 3:16",
            debounce=True,
            className="mb-1",
        ),
        html.Small("Separate multiple verses with commas.", className="text-muted"),

        html.Hr(),
        html.H6("Browse", className="fw-bold"),
        dcc.Dropdown(
            id="book-select",
            options=[{"label": b, "value": b} for b in books],
            placeholder="Book…",
            clearable=True,
            className="mb-2",
        ),
        dcc.Dropdown(id="chapter-select", placeholder="Chapter…", clearable=True, className="mb-2"),
        dcc.Dropdown(id="verse-select",   placeholder="Verse…",   clearable=True, className="mb-2"),

        html.Hr(),
        html.H6("Edge filters", className="fw-bold"),
        dbc.Checklist(
            id="edge-types",
            options=[
                {"label": " REFERENCES  (blue, solid)",  "value": "refs"},
                {"label": " IS_SIMILAR  (teal, dashed)", "value": "sim"},
            ],
            value=["refs", "sim"],
            className="mb-2",
        ),
        html.Div([
            html.Small("REFERENCES rank (1 = strongest)", className="text-muted"),
            dcc.Slider(id="ref-rank",  min=1, max=20, step=1, value=5,
                       marks={1: "1", 5: "5", 10: "10", 20: "20"},
                       tooltip={"placement": "bottom", "always_visible": False}),
        ], className="mb-3"),
        html.Div([
            html.Small("IS_SIMILAR rank", className="text-muted"),
            dcc.Slider(id="sim-rank", min=1, max=5,  step=1, value=3,
                       marks={1: "1", 3: "3", 5: "5"},
                       tooltip={"placement": "bottom", "always_visible": False}),
        ], className="mb-3"),

        html.Hr(),
        dbc.Switch(id="physics-toggle", label="Auto-layout (physics)", value=True,
                   className="mb-2"),
        html.Small("Off: drag nodes freely, edges stretch.", className="text-muted d-block mb-3"),

        dbc.Button("Reset graph", id="reset-btn", color="secondary",
                   size="sm", outline=True, className="mb-3 w-100"),

        html.Hr(),
        html.Div([
            color_dot("#E63946", "Focus/expanded"),
            color_dot("#457B9D", "REFERENCES"),
            color_dot("#2A9D8F", "IS_SIMILAR"),
            color_dot("#E9C46A", "Both"),
        ], className="mt-1"),
        html.Small("Click a node → Expand / Explain / Remove",
                   className="text-muted d-block mt-2"),
    ], width=3, className="border-end pe-3")


def build_action_bar() -> dbc.Collapse:
    return dbc.Collapse(
        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col(html.Span(id="action-label", className="fw-semibold"), width="auto",
                        className="d-flex align-items-center"),
                dbc.Col(dbc.Button("Expand",  id="btn-expand",  color="primary",
                                   size="sm", className="me-1"), width="auto"),
                dbc.Col(dbc.Button("Explain", id="btn-explain", color="success",
                                   size="sm", className="me-1"), width="auto"),
                dbc.Col(dbc.Button("Remove",  id="btn-remove",  color="danger",
                                   size="sm", className="me-1"), width="auto"),
                dbc.Col(dbc.Button("✕ Cancel",   id="btn-cancel",  color="light",
                                   size="sm"), width="auto"),
            ], align="center"),
        ]), className="mt-2 mb-1"),
        id="action-bar",
        is_open=False,
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="BibleMind Graph Explorer",
    suppress_callback_exceptions=True,
)

app.layout = dbc.Container([
    # Stores
    dcc.Store(id="focus-store",         storage_type="session"),
    dcc.Store(id="graph-state",         storage_type="session", data=EMPTY_GRAPH_STATE),
    dcc.Store(id="selected-node-store", storage_type="session"),
    dcc.Store(id="explain-store",       storage_type="session"),
    dcc.Store(id="explain-full-text"),   # raw text from OpenAI
    dcc.Store(id="explain-char-pos",    data=0),  # typewriter position

    # Header
    dbc.Row(dbc.Col(html.H4("✝ BibleMind Graph Explorer", className="my-3"))),

    # Main
    dbc.Row([
        build_sidebar(),

        # Graph column
        dbc.Col([
            html.Small(id="stats-bar", className="text-muted"),
            cyto.Cytoscape(
                id="cytoscape",
                elements=[],
                stylesheet=STYLESHEET,
                layout={"name": "cose-bilkent", "nodeDimensionsIncludeLabels": True,
                        "animate": True, "padding": 30},
                style={"width": "100%", "height": "650px",
                       "border": "1px solid #dee2e6", "borderRadius": "4px"},
                minZoom=0.2,
                maxZoom=4.0,
                responsive=True,
            ),
            build_action_bar(),
        ], width=7),

        # Text panel
        dbc.Col([
            dcc.Interval(id="type-interval", interval=20, n_intervals=0, disabled=True),
            dcc.Loading(
                html.Div(id="verse-panel", className="mt-4 ps-2",
                         style={"fontSize": "13px", "lineHeight": "1.6"}),
                type="circle",
                color="#457B9D",
            ),
            html.Div(id="explain-panel", className="ps-2",
                     style={"fontSize": "12px", "lineHeight": "1.7", "marginTop": "4px"}),
        ], width=2),
    ]),
], fluid=True)

# ---------------------------------------------------------------------------
# Layout helper
# ---------------------------------------------------------------------------

def make_layout(n_nodes: int, physics_on: bool = True) -> dict:
    """Return cose-bilkent layout params scaled to node count, or preset if physics off."""
    if not physics_on:
        return {"name": "preset", "animate": False}
    # More nodes → more repulsion + longer edges so labels don't crowd
    repulsion   = max(5000,  min(16000, 600 * n_nodes))
    edge_length = max(120,   min(300,   18  * n_nodes))
    return {
        "name": "cose-bilkent",
        "animate": True,
        "animationDuration": 500,
        "nodeDimensionsIncludeLabels": True,   # key: layout accounts for label size
        "nodeRepulsion": repulsion,
        "idealEdgeLength": edge_length,
        "edgeElasticity": 0.45,
        "padding": 40,
        "fit": True,
        "randomize": False,
        "componentSpacing": 40,
    }


# ---------------------------------------------------------------------------
# OpenAI helper
# ---------------------------------------------------------------------------

def call_openai_explain(verse: dict, connected: list[dict]) -> str:
    """Call OpenAI to explain a verse and its connections to neighboring verses."""
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    conn_section = ""
    if connected:
        lines = []
        for c in connected[:5]:
            snippet = c["text"][:120] + "…" if len(c["text"]) > 120 else c["text"]
            lines.append(f'- **{c["reference"]}** ({c["type"]}): "{snippet}"')
        conn_section = (
            "\n\nVerses connected to it in the current graph:\n"
            + "\n".join(lines)
            + "\n\nFor each connected verse, give one sentence explaining why it relates to "
            + verse["reference"] + "."
        )

    prompt = (
        f'Verse: **{verse["reference"]}**\n"{verse["text"]}"\n\n'
        f"1. **Meaning**: In 2–3 sentences, explain this verse's meaning and theological significance.\n"
        f"{conn_section}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a concise Bible scholar. Use plain language."},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=600,
        temperature=0.3,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

# CB-1a: book → chapter options
@app.callback(
    Output("chapter-select", "options"),
    Output("chapter-select", "value"),
    Input("book-select", "value"),
)
def update_chapters(book):
    if not book:
        return [], None
    chapters = fetch_chapters(driver, book)
    return [{"label": str(c), "value": c} for c in chapters], None


# CB-1b: book + chapter → verse options
@app.callback(
    Output("verse-select", "options"),
    Output("verse-select", "value"),
    Input("book-select", "value"),
    Input("chapter-select", "value"),
)
def update_verses(book, chapter):
    if not book or not chapter:
        return [], None
    verses = fetch_verses_in_chapter(driver, book, int(chapter))
    return [{"label": v["reference"], "value": v["reference"]} for v in verses], None


# CB-2: Resolve focus IDs from search or browse
@app.callback(
    Output("focus-store",  "data"),
    Output("graph-state",  "data"),
    Input("search-input",  "value"),
    Input("verse-select",  "value"),
    Input("reset-btn",     "n_clicks"),
    State("graph-state",   "data"),
    prevent_initial_call=False,
)
def resolve_focus_ids(search_text, browse_ref, reset_clicks, graph_state):
    triggered = ctx.triggered_id

    if triggered == "reset-btn":
        return no_update, {**EMPTY_GRAPH_STATE,
                           "focus_ids": graph_state.get("focus_ids", [])}

    raw_queries = []
    if search_text:
        raw_queries = [q.strip() for q in search_text.split(",") if q.strip()]
    elif browse_ref:
        raw_queries = [browse_ref]

    if not raw_queries:
        return {"focus_ids": []}, EMPTY_GRAPH_STATE

    focus_ids = []
    for q in raw_queries:
        results = search_verse(driver, q)
        if not results:
            continue
        exact = [r for r in results if r["reference"].lower() == q.lower()]
        if exact:
            focus_ids.append(exact[0]["id"])
        else:
            focus_ids.append(results[0]["id"])

    prev_focus = set(graph_state.get("focus_ids", []))
    new_focus  = set(focus_ids)
    if new_focus != prev_focus:
        new_state = {"focus_ids": focus_ids, "expanded_ids": [], "removed_ids": []}
    else:
        new_state = graph_state

    return {"focus_ids": focus_ids}, new_state


# CB-3: Build Cytoscape elements
@app.callback(
    Output("cytoscape",      "elements"),
    Output("cytoscape",      "layout",   allow_duplicate=True),
    Output("stats-bar",      "children"),
    Output("verse-panel",    "children"),
    Input("focus-store",     "data"),
    Input("graph-state",     "data"),
    Input("edge-types",      "value"),
    Input("ref-rank",        "value"),
    Input("sim-rank",        "value"),
    State("physics-toggle",  "value"),
    prevent_initial_call=True,
)
def build_cytoscape(focus_store, graph_state, edge_types, ref_rank, sim_rank, physics_on):
    focus_ids    = set(focus_store.get("focus_ids", []))   if focus_store    else set()
    expanded_ids = set(graph_state.get("expanded_ids", [])) if graph_state   else set()
    removed_ids  = set(graph_state.get("removed_ids",  [])) if graph_state   else set()

    show_refs = "refs" in (edge_types or [])
    show_sim  = "sim"  in (edge_types or [])

    if not focus_ids:
        return [], make_layout(0, physics_on), "", html.P("Search for a verse to begin.", className="text-muted mt-4")

    elements, texts = build_graph_elements(
        driver, focus_ids, expanded_ids, removed_ids,
        ref_rank or 5, sim_rank or 3, show_refs, show_sim,
    )

    n_nodes = sum(1 for e in elements if "source" not in e["data"])
    n_edges = len(elements) - n_nodes
    layout  = make_layout(n_nodes, physics_on if physics_on is not None else True)

    stats_parts = [f"{n_nodes} nodes · {n_edges} edges"]
    if expanded_ids:
        stats_parts.append(f"{len(expanded_ids)} expanded")
    if removed_ids:
        stats_parts.append(f"{len(removed_ids)} removed")
    stats = "  ·  ".join(stats_parts)

    # Default verse panel: show focus verse(s)
    panel_items = []
    for vid in sorted(focus_ids):
        text = texts.get(vid, "")
        if text:
            label = next(
                (e["data"]["label"] for e in elements
                 if e["data"].get("id") == vid and "label" in e["data"]),
                vid,
            )
            panel_items.append(html.Div([
                html.Strong(label),
                html.P(text, className="mt-1 mb-3"),
            ]))

    return elements, layout, stats, panel_items or html.P("No text found.", className="text-muted")


# CB-4: Tap node → show verse text + action bar; clear any previous explanation
@app.callback(
    Output("selected-node-store", "data"),
    Output("action-bar",          "is_open"),
    Output("action-label",        "children"),
    Output("verse-panel",         "children",  allow_duplicate=True),
    Output("explain-panel",       "children",  allow_duplicate=True),
    Output("explain-full-text",   "data",      allow_duplicate=True),
    Output("type-interval",       "disabled",  allow_duplicate=True),
    Input("cytoscape",            "tapNode"),
    prevent_initial_call=True,
)
def store_selected_node(tap_node):
    if not tap_node:
        return no_update, False, "", no_update, no_update, no_update, no_update
    data = tap_node["data"]
    if "source" in data:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update
    label   = data.get("label", data["id"])
    node_id = data["id"]

    verse = get_verse(driver, node_id)
    panel = (
        html.Div([
            html.Strong(verse["reference"], className="d-block mb-1"),
            html.P(verse["text"], className="mt-1 mb-0"),
        ])
        if verse else no_update
    )
    # Clear previous explanation
    return {"id": node_id, "label": label}, True, f"Selected: {label}", panel, "", None, True


# CB-5: Action buttons
@app.callback(
    Output("graph-state",         "data",    allow_duplicate=True),
    Output("explain-store",       "data"),
    Output("action-bar",          "is_open", allow_duplicate=True),
    Input("btn-expand",           "n_clicks"),
    Input("btn-explain",          "n_clicks"),
    Input("btn-remove",           "n_clicks"),
    Input("btn-cancel",           "n_clicks"),
    State("selected-node-store",  "data"),
    State("graph-state",          "data"),
    prevent_initial_call=True,
)
def handle_node_action(expand_n, explain_n, remove_n, cancel_n, selected, graph_state):
    triggered = ctx.triggered_id
    if not triggered or not selected:
        return no_update, no_update, no_update

    node_id = selected["id"]
    state = dict(graph_state) if graph_state else dict(EMPTY_GRAPH_STATE)
    state = {
        "focus_ids":    list(state.get("focus_ids", [])),
        "expanded_ids": list(state.get("expanded_ids", [])),
        "removed_ids":  list(state.get("removed_ids", [])),
    }

    if triggered == "btn-expand":
        if node_id not in state["focus_ids"] and node_id not in state["expanded_ids"]:
            state["expanded_ids"].append(node_id)
        return state, no_update, False

    elif triggered == "btn-explain":
        return no_update, {"id": node_id}, False

    elif triggered == "btn-remove":
        if node_id not in state["removed_ids"]:
            state["removed_ids"].append(node_id)
        state["expanded_ids"] = [x for x in state["expanded_ids"] if x != node_id]
        return state, no_update, False

    elif triggered == "btn-cancel":
        return no_update, no_update, False

    return no_update, no_update, no_update


# CB-6: Explain verse — calls OpenAI, then feeds typewriter
@app.callback(
    Output("verse-panel",       "children",  allow_duplicate=True),
    Output("explain-full-text", "data",      allow_duplicate=True),
    Output("explain-char-pos",  "data",      allow_duplicate=True),
    Output("explain-panel",     "children",  allow_duplicate=True),
    Output("type-interval",     "disabled",  allow_duplicate=True),
    Input("explain-store",  "data"),
    State("cytoscape",      "elements"),
    prevent_initial_call=True,
)
def render_explained_verse(explain_data, elements):
    if not explain_data or not explain_data.get("id"):
        return no_update, no_update, no_update, no_update, no_update

    node_id = explain_data["id"]
    verse   = get_verse(driver, node_id)
    if not verse:
        return html.P("Verse not found.", className="text-muted"), None, 0, "", True

    verse_info = html.Div([
        html.Strong(verse["reference"], className="d-block mb-1"),
        html.P(verse["text"], className="text-muted mb-0",
               style={"fontStyle": "italic"}),
        html.Hr(className="my-2"),
    ])

    # Find neighbor IDs from current graph elements
    neighbor_ids: list[tuple] = []
    for el in (elements or []):
        d     = el["data"]
        etype = d.get("type", "ref")
        if "source" not in d:
            continue
        if d["source"] == node_id:
            neighbor_ids.append((d["target"], etype))
        elif d["target"] == node_id:
            neighbor_ids.append((d["source"], etype))

    # Batch-fetch neighbor texts (top 5)
    connected: list[dict] = []
    if neighbor_ids:
        top_ids  = [nid for nid, _ in neighbor_ids[:5]]
        etype_map = {nid: et for nid, et in neighbor_ids[:5]}
        with driver.session() as s:
            rows = s.run(
                "MATCH (v:Verse) WHERE v.id IN $ids "
                "RETURN v.id AS id, v.reference AS ref, v.text AS text",
                ids=top_ids,
            ).data()
        for r in rows:
            connected.append({
                "reference": r["ref"],
                "text":      r["text"],
                "type": "cross-reference" if etype_map.get(r["id"]) == "ref"
                        else "semantic similarity",
            })

    try:
        explanation = call_openai_explain(verse, connected)
    except Exception as exc:
        explanation = f"_(OpenAI error: {exc})_"

    # Start typewriter: reset position, clear panel, enable interval
    return verse_info, explanation, 0, "", False


# CB-6b: Typewriter — advance N characters per interval tick
CHARS_PER_TICK = 6   # lower = slower / more dramatic; raise for faster

@app.callback(
    Output("explain-panel",    "children",  allow_duplicate=True),
    Output("explain-char-pos", "data",      allow_duplicate=True),
    Output("type-interval",    "disabled",  allow_duplicate=True),
    Input("type-interval",     "n_intervals"),
    State("explain-full-text", "data"),
    State("explain-char-pos",  "data"),
    prevent_initial_call=True,
)
def type_step(_, full_text, pos):
    if not full_text:
        return no_update, no_update, True
    pos     = pos or 0
    new_pos = min(pos + CHARS_PER_TICK, len(full_text))
    partial = full_text[:new_pos]
    done    = new_pos >= len(full_text)
    return dcc.Markdown(partial, style={"fontSize": "12px"}), new_pos, done


# CB-7: Physics toggle (clientside — zero Python round-trip)
# Uses cose-bilkent with nodeDimensionsIncludeLabels so labels don't overlap.
# Node count is read from the current elements to scale repulsion dynamically.
app.clientside_callback(
    """
    function(physicsOn, elements) {
        if (!physicsOn) {
            return {name: 'preset', animate: false};
        }
        var nNodes = (elements || []).filter(function(e) {
            return !e.data.source;
        }).length;
        var repulsion   = Math.max(5000,  Math.min(16000, 600 * nNodes));
        var edgeLength  = Math.max(120,   Math.min(300,   18  * nNodes));
        return {
            name: 'cose-bilkent',
            animate: true,
            animationDuration: 500,
            nodeDimensionsIncludeLabels: true,
            nodeRepulsion: repulsion,
            idealEdgeLength: edgeLength,
            edgeElasticity: 0.45,
            padding: 40,
            fit: true,
            randomize: false,
            componentSpacing: 40,
        };
    }
    """,
    Output("cytoscape", "layout"),
    Input("physics-toggle", "value"),
    State("cytoscape", "elements"),
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=8050)
