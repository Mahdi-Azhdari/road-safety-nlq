"""
Road Safety Assistant — Streamlit interface.

Natural-language query interface for transportation safety analysis.
Renders maps, tables, temporal plots, and the compiled execution graph.

Run with:
    streamlit run app.py
"""

import io
import tempfile
import zipfile

import psycopg2
import streamlit as st
import streamlit.components.v1 as components

try:
    from core import AmbiguousLocationError, GEMINI_MODEL, OPENAI_MODEL
except ImportError:
    AmbiguousLocationError = None
    GEMINI_MODEL = "gemini-2.5-flash"
    OPENAI_MODEL = "gpt-4o"


# ── page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Road Safety Assistant",
    page_icon="🛣️",
    layout="wide",
)

for key, default in {
    "assistant":        None,
    "conn":             None,
    "connected":        False,
    "history":          [],
    "pending_prompt":   None,
    "location_options": None,
    "llm_provider":     "Gemini",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── helpers ────────────────────────────────────────────────────────────────────

def map_to_html(map_object) -> str:
    """Render a folium map to standalone HTML for embedding."""
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        map_object.save(f.name)
        return open(f.name, "r", encoding="utf-8").read()


def gdf_to_shapefile_zip(gdf, name: str) -> bytes:
    """Write a GeoDataFrame to a zipped shapefile in memory."""
    import geopandas as gpd
    with tempfile.TemporaryDirectory() as tmp:
        out = f"{tmp}/{name}.shp"
        gdf.to_crs(4326).to_file(out, driver="ESRI Shapefile")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for ext in ("shp", "shx", "dbf", "prj", "cpg"):
                p = f"{tmp}/{name}.{ext}"
                import os
                if os.path.exists(p):
                    zf.write(p, f"{name}.{ext}")
        return buf.getvalue()


def gdf_to_csv_bytes(gdf) -> bytes:
    """Export a GeoDataFrame as CSV with lat/lon columns (WGS84)."""
    df = gdf.to_crs(4326).copy()
    df["longitude"] = df.geometry.apply(lambda g: round(g.centroid.x, 6))
    df["latitude"]  = df.geometry.apply(lambda g: round(g.centroid.y, 6))
    df = df.drop(columns=["geometry"], errors="ignore")
    return df.to_csv(index=False).encode("utf-8")


def _role_label(role: str) -> str:
    """Human-readable label for a role name."""
    return role.replace("_", " ").title()


_OP_COLORS = {
    "LoadDatasetRegistry":       "#D3D1C7",
    "LoadEntitySpec":            "#D3D1C7",
    "ResolveReference":          "#D3D1C7",
    "BuildReferenceObject":      "#D3D1C7",
    "InitializeRoleQuery":       "#CECBF6",
    "ApplyNameFilter":           "#9FE1CB",
    "ApplySpatialConstraint":    "#9FE1CB",
    "ApplyScopeConstraint":      "#9FE1CB",
    "ApplyAttributeConstraints": "#9FE1CB",
    "ApplyRelationConstraint":   "#9FE1CB",
    "MatchSpatialSets":          "#F5C4B3",
    "Aggregate":                 "#F5C4B3",
    "Rank":                      "#F5C4B3",
    "MaterializeRole":           "#D3D1C7",
    "PrepareMap":                "#C0DD97",
    "PrepareTable":              "#C0DD97",
    "PrepareSummary":            "#C0DD97",
    "Unsupported":               "#F7C1C1",
}


def _execution_plan_to_dot(execution_plan):
    """Render an execution plan as a Graphviz DOT string."""
    lines = [
        "digraph ExecutionDAG {",
        "  rankdir=TB;",
        "  bgcolor=transparent;",
        "  splines=ortho;",
        "  nodesep=0.5;",
        "  ranksep=0.6;",
        '  node [fontname="Helvetica" fontsize=11 style=filled shape=box margin="0.15,0.08"];',
        '  edge [fontname="Helvetica" fontsize=9 color="#888780" arrowsize=0.7];',
    ]
    for node in execution_plan:
        nid    = node["node_id"]
        op     = node["op"]
        params = node.get("params", {})
        parts  = [op]
        if "role" in params:
            parts.append("role=" + str(params["role"]))
        elif "target_role" in params:
            parts.append("to " + str(params["target_role"]))
        if "entity" in params:
            parts.append("entity=" + str(params["entity"]))
        if params.get("names"):
            parts.append("names=" + str(params["names"]))
        if "distance_m" in params and params["distance_m"] is not None:
            parts.append("dist=" + str(params["distance_m"]) + "m")
        if "top_n" in params:
            parts.append("top_n=" + str(params["top_n"]))
        label = r"\n".join(parts)
        color = _OP_COLORS.get(op, "#F1EFE8")
        lines.append('  "' + nid + '" [label="' + label + '" fillcolor="' + color + '"];')
    for node in execution_plan:
        for src in node.get("inputs", []):
            lines.append('  "' + src + '" -> "' + node["node_id"] + '";')
    lines.append("}")
    return "\n".join(lines)


def _dag_to_svg(execution_plan):
    try:
        import graphviz
        return graphviz.Source(_execution_plan_to_dot(execution_plan)).pipe(format="svg"), None
    except Exception as e:
        return None, str(e)


def _dag_to_png(execution_plan):
    try:
        import graphviz
        return graphviz.Source(_execution_plan_to_dot(execution_plan)).pipe(format="png")
    except Exception:
        return None


def connect_and_build(
    llm_provider, llm_key, llm_model,
    resp_provider, response_key, response_model,
    db_host, db_port, db_name, db_user, db_pass,
):
    """Open the PostGIS connection and build the assistant."""
    from core import RoadSafetyAssistant
    conn = psycopg2.connect(
        host=db_host, port=db_port, dbname=db_name,
        user=db_user, password=db_pass,
    )
    assistant = RoadSafetyAssistant(
        conn=conn,
        llm_provider=llm_provider.lower(),
        llm_api_key=llm_key,
        llm_model=llm_model.strip(),
        response_provider=resp_provider.lower(),
        response_api_key=response_key or None,
        response_model=response_model.strip(),
    )
    st.session_state.conn      = conn
    st.session_state.assistant = assistant
    st.session_state.connected = True


def run_prompt(prompt, geocode_selection=None):
    assistant = st.session_state.assistant
    result    = assistant.run(prompt, geocode_selection=geocode_selection)
    st.session_state.history.append({"prompt": prompt, "result": result})


# ── sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🛣️ Road Safety Assistant")
    st.caption("Massachusetts · Gemini & OpenAI")

    st.subheader("LLM Provider")
    provider = st.radio(
        "Primary LLM",
        ["Gemini", "OpenAI"],
        index=0 if st.session_state.llm_provider == "Gemini" else 1,
        horizontal=True,
        disabled=st.session_state.connected,
    )
    st.session_state.llm_provider = provider

    if provider == "Gemini":
        llm_key       = st.text_input("Gemini API Key", type="password", placeholder="AIza…", disabled=st.session_state.connected)
        llm_model     = st.text_input("Gemini model", value=GEMINI_MODEL, disabled=st.session_state.connected)
        resp_provider = "Gemini"
        response_key  = st.text_input("Response-layer key (optional, Gemini)", type="password", placeholder="AIza…", disabled=st.session_state.connected)
        resp_model    = st.text_input("Response-layer model (optional)", value=GEMINI_MODEL, disabled=st.session_state.connected)
    else:
        llm_key       = st.text_input("OpenAI API Key", type="password", placeholder="sk-…", disabled=st.session_state.connected)
        llm_model     = st.text_input("OpenAI model", value=OPENAI_MODEL, disabled=st.session_state.connected)
        resp_provider = "OpenAI"
        response_key  = st.text_input("Response-layer key (optional, OpenAI)", type="password", placeholder="sk-…", disabled=st.session_state.connected)
        resp_model    = st.text_input("Response-layer model (optional)", value=OPENAI_MODEL, disabled=st.session_state.connected)

    st.divider()

    st.subheader("Database")
    db_host = st.text_input("Host",          value="localhost",  disabled=st.session_state.connected)
    db_port = st.number_input("Port",        value=5432, step=1, disabled=st.session_state.connected)
    db_name = st.text_input("Database name", value="roadsafety", disabled=st.session_state.connected)
    db_user = st.text_input("User",          value="postgres",   disabled=st.session_state.connected)
    db_pass = st.text_input("Password",      type="password",    disabled=st.session_state.connected)

    connect_btn = st.button(
        "Connect & Start", type="primary",
        use_container_width=True,
        disabled=st.session_state.connected,
    )

    if connect_btn:
        if not llm_key:
            st.error(f"{provider} API key is required.")
        elif not db_pass:
            st.error("Database password is required.")
        else:
            with st.spinner("Connecting…"):
                try:
                    connect_and_build(
                        provider, llm_key, llm_model,
                        resp_provider, response_key, resp_model,
                        db_host, int(db_port), db_name, db_user, db_pass,
                    )
                    st.success("Connected!")
                except Exception as e:
                    st.error(f"Connection failed: {e}")

    if st.session_state.connected:
        st.success(f"✅ Connected  ·  {st.session_state.llm_provider}")
        if st.button("Disconnect", use_container_width=True):
            try:
                if st.session_state.conn:
                    st.session_state.conn.close()
            except Exception:
                pass
            for k, v in {
                "connected": False, "assistant": None, "conn": None,
                "history": [], "pending_prompt": None, "location_options": None,
            }.items():
                st.session_state[k] = v
            st.rerun()

    st.divider()
    if st.session_state.connected and st.session_state.assistant:
        with st.expander("Example prompts"):
            st.code(st.session_state.assistant.help_message(), language=None)


# ── main area ──────────────────────────────────────────────────────────────────

if not st.session_state.connected:
    st.markdown(
        "## Road Safety Assistant\n\n"
        "Natural-language interface for transportation safety analysis. "
        "Supports **Gemini** and **OpenAI** as the primary LLM.\n\n"
        "Fill in credentials in the sidebar and click **Connect & Start**."
    )
    st.stop()

st.header("Road Safety Assistant")

with st.form("prompt_form", clear_on_submit=True):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_prompt = st.text_input(
            "Your query",
            placeholder='"show crashes in Quincy"  |  "top 10 schools by crashes within 500m"',
            label_visibility="collapsed",
        )
    with col_btn:
        submitted = st.form_submit_button("Ask ▶", use_container_width=True, type="primary")

if submitted and user_prompt.strip():
    st.session_state.pending_prompt   = None
    st.session_state.location_options = None
    with st.spinner("Running query…"):
        try:
            run_prompt(user_prompt.strip())
        except Exception as e:
            if AmbiguousLocationError and isinstance(e, AmbiguousLocationError):
                st.session_state.pending_prompt   = user_prompt.strip()
                st.session_state.location_options = e.options
                st.rerun()
            else:
                st.error(f"Unexpected error: {e}")

if st.session_state.get("location_options"):
    st.info("📍 Multiple locations found. Please select the correct one:")
    options = st.session_state.location_options
    labels  = [o["display_name"] for o in options]
    choice  = st.selectbox("Select location:", labels, key="location_select")
    col_confirm, col_cancel = st.columns([1, 1])
    with col_confirm:
        if st.button("✅ Confirm & Run", type="primary", use_container_width=True):
            idx    = labels.index(choice)
            prompt = st.session_state.pending_prompt
            st.session_state.location_options = None
            st.session_state.pending_prompt   = None
            with st.spinner("Running with selected location…"):
                try:
                    run_prompt(prompt, geocode_selection=idx)
                except Exception as e:
                    st.error(f"Error: {e}")
            st.rerun()
    with col_cancel:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.location_options = None
            st.session_state.pending_prompt   = None
            st.rerun()

if st.session_state.history:
    if st.button("🗑️ Clear history"):
        st.session_state.history = []
        st.rerun()


# ── result cards ───────────────────────────────────────────────────────────────

for item in reversed(st.session_state.history):
    prompt = item["prompt"]
    result = item["result"]

    with st.container(border=True):
        st.markdown(f"**🔍 {prompt}**")

        if result.narrative_answer:
            st.markdown(result.narrative_answer)
        elif result.summary:
            st.info(result.summary)

        if result.warnings:
            for w in result.warnings:
                st.warning(w)

        if result.map_object is not None:
            st.subheader("Map")
            components.html(map_to_html(result.map_object), height=520, scrolling=False)

        if hasattr(result, "temporal_plots") and result.temporal_plots:
            st.subheader("Temporal Distribution")
            for fig in result.temporal_plots:
                st.pyplot(fig)

        if result.tables:
            for name, df in result.tables.items():
                with st.expander(f"📋 Table: {name}  ({len(df):,} rows)"):
                    st.dataframe(df, use_container_width=True)

        # ── downloads ──────────────────────────────────────────────────────────
        has_tables = bool(result.tables)
        has_gdfs   = bool(getattr(result, "gdfs", {}))

        if has_tables or has_gdfs:
            with st.expander("⬇ Downloads", expanded=False):

                if has_tables:
                    st.markdown("**Tables (CSV)**")
                    dl_cols = st.columns(min(len(result.tables), 4))
                    for col, (name, df) in zip(dl_cols, result.tables.items()):
                        col.download_button(
                            label=f"📄 {name}.csv",
                            data=df.to_csv(index=False).encode("utf-8"),
                            file_name=f"{name}.csv",
                            mime="text/csv",
                            key=f"dlbar_csv_{id(item)}_{name}",
                            use_container_width=True,
                        )

                if has_gdfs:
                    st.markdown("**Spatial layers (Shapefile · CSV with lat/lon)**")
                    gdfs = result.gdfs
                    for role, gdf in gdfs.items():
                        label = _role_label(role)
                        c_shp, c_csv = st.columns(2)
                        try:
                            shp_bytes = gdf_to_shapefile_zip(gdf, role)
                            c_shp.download_button(
                                label=f"🗂 {label} — Shapefile (.zip)",
                                data=shp_bytes,
                                file_name=f"{role}.zip",
                                mime="application/zip",
                                key=f"dlbar_shp_{id(item)}_{role}",
                                use_container_width=True,
                            )
                        except Exception as e:
                            c_shp.caption(f"Shapefile unavailable: {e}")

                        try:
                            csv_bytes = gdf_to_csv_bytes(gdf)
                            c_csv.download_button(
                                label=f"📄 {label} — CSV with lat/lon",
                                data=csv_bytes,
                                file_name=f"{role}.csv",
                                mime="text/csv",
                                key=f"dlbar_csv2_{id(item)}_{role}",
                                use_container_width=True,
                            )
                        except Exception as e:
                            c_csv.caption(f"CSV unavailable: {e}")

        with st.expander("🔧 Debug — semantic frame & execution plan"):
            tab_sf, tab_plan, tab_dag = st.tabs([
                "Semantic Frame",
                "Execution Plan (JSON)",
                "Execution DAG (graph)",
            ])

            with tab_sf:
                frame = (result.debug or {}).get(
                    "validated_semantic_frame", result.semantic_frame
                )
                st.json(frame)

            with tab_plan:
                st.json(result.execution_plan)

            with tab_dag:
                if not result.execution_plan:
                    st.caption("No execution plan available.")
                else:
                    svg_bytes, err = _dag_to_svg(result.execution_plan)
                    if err:
                        st.warning(
                            f"Graph rendering failed: {err}\n\n"
                            "Install the Graphviz system package and "
                            "`pip install graphviz` to enable this view."
                        )
                    else:
                        svg_str = svg_bytes.decode("utf-8")
                        components.html(
                            '<div style="overflow:auto;background:white;'
                            'border:1px solid #e0ddd6;border-radius:6px;padding:12px;">'
                            + svg_str + "</div>",
                            height=600,
                            scrolling=True,
                        )
                        png_bytes = _dag_to_png(result.execution_plan)
                        if png_bytes:
                            st.download_button(
                                label="⬇ Download PNG",
                                data=png_bytes,
                                file_name=f"dag_{prompt[:40].replace(' ', '_')}.png",
                                mime="image/png",
                                key=f"dag_png_{id(item)}",
                            )
                        dot_str = _execution_plan_to_dot(result.execution_plan)
                        st.download_button(
                            label="⬇ Download DOT source",
                            data=dot_str.encode("utf-8"),
                            file_name=f"dag_{prompt[:40].replace(' ', '_')}.dot",
                            mime="text/plain",
                            key=f"dag_dot_{id(item)}",
                        )
