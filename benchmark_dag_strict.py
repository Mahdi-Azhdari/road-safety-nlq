"""
Road Safety Assistant — benchmark runner.

Runs the 80-query evaluation suite against the semantic frame ground truth
and reports intent completeness, repair rates, execution success, and
DAG structure metrics.

Outputs are written to benchmark_out_<provider>/
  results.xlsx       — full results and per-group summary
  debug.json         — per-query frame, DAG, and repair details
  execution_log.json — lightweight per-query log
  summary.txt        — printed summary
"""

import re

import os, sys, json, time, traceback
import pandas as pd
import psycopg2

# ── CONFIGURATION ─────────────────────────────────────────
LLM_PROVIDER = "gemini"           # "gemini" | "openai"
LLM_API_KEY  = ""                 # set your key here or via env var
LLM_MODEL    = "gemini-2.5-flash" # or "gpt-4o"

# ── DATABASE ──────────────────────────────────────────────
os.environ.setdefault("DB_HOST",     "localhost")
os.environ.setdefault("DB_PORT",     "5432")
os.environ.setdefault("DB_NAME",     "roadsafety")
os.environ.setdefault("DB_USER",     "postgres")
os.environ.setdefault("DB_PASSWORD", "")   # set via environment or edit here

APP_DIR    = r"."
TIMEOUT_MS = 900_000
OUT_DIR    = f"benchmark_out_{LLM_PROVIDER}"

if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
os.chdir(APP_DIR)

print(f"Working directory: {os.getcwd()}")
print(f"Provider: {LLM_PROVIDER}  Model: {LLM_MODEL}\n")

import core
from core import RoadSafetyAssistant

try:
    display
except NameError:
    from IPython.display import display

from benchmark_ground_truth_strict import PROMPTS, GROUND_TRUTH

assert len(PROMPTS) == 80
print(f"Loaded {len(PROMPTS)} prompts, {len(GROUND_TRUTH)} ground truth entries.\n")

os.makedirs(OUT_DIR, exist_ok=True)

conn = psycopg2.connect(
    host=os.environ["DB_HOST"],
    port=int(os.environ["DB_PORT"]),
    dbname=os.environ["DB_NAME"],
    user=os.environ["DB_USER"],
    password=os.environ["DB_PASSWORD"],
)

assistant = RoadSafetyAssistant(
    conn=conn,
    llm_provider=LLM_PROVIDER,
    llm_api_key=LLM_API_KEY,
    llm_model=LLM_MODEL,
    response_api_key=None,
)
print(f"Assistant ready  [provider={LLM_PROVIDER}]\n")

GROUP_NAMES = {
    "G1": "Entity Retrieval",
    "G2": "Spatial Scoping",
    "G3": "Attribute Filtering",
    "G4": "Temporal Filtering",
    "G5": "Spatial Relationships",
    "G6": "Infrastructure Ranking",
    "G7": "Town Ranking",
    "G8": "Road Segment Ranking",
    "G9": "Combined Multi-Constraint",
}


# =========================================================
# DAG ANALYSIS HELPERS
# =========================================================

def analyse_dag(execution_plan: list[dict]) -> dict:
    """
    Compute DAG-specific metrics from the serialised execution plan.
    Returns a dict ready to merge into the results row.
    """
    if not execution_plan:
        return {
            "dag_node_count": 0,
            "dag_depth": 0,
            "dag_root_count": 0,
            "dag_leaf_count": 0,
            "dag_parallel_pairs": 0,
            "dag_valid": False,
            "dag_ops": "",
        }

    nodes = {n["node_id"]: n for n in execution_plan}

    # Build adjacency: node → set of nodes it directly depends on
    deps  = {nid: set(n.get("inputs", [])) for nid, n in nodes.items()}
    # Build reverse: node → set of dependents
    rdeps = {nid: set() for nid in nodes}
    for nid, inputs in deps.items():
        for src in inputs:
            if src in rdeps:
                rdeps[src].add(nid)

    roots  = [nid for nid, d in deps.items()  if not d]
    leaves = [nid for nid, d in rdeps.items() if not d]

    # Longest path from any root to any leaf (DAG depth)
    dist: dict[str, int] = {r: 0 for r in roots}
    for nid in [n["node_id"] for n in execution_plan]:   # already in topo order
        for succ in rdeps.get(nid, []):
            dist[succ] = max(dist.get(succ, 0), dist.get(nid, 0) + 1)
    depth = max(dist.values()) + 1 if dist else 0

    # Parallel pairs: pairs of nodes that share no ancestor/descendant relationship
    # (i.e. could in principle run concurrently). Computed as:
    # all_pairs - pairs_with_ordering
    nids = list(nodes.keys())

    def ancestors(nid: str) -> set:
        result, stack = set(), [nid]
        while stack:
            cur = stack.pop()
            for src in deps.get(cur, []):
                if src not in result:
                    result.add(src)
                    stack.append(src)
        return result

    # Cache ancestors for each node
    anc = {nid: ancestors(nid) for nid in nids}

    parallel_pairs = 0
    for i in range(len(nids)):
        for j in range(i + 1, len(nids)):
            a, b = nids[i], nids[j]
            # They are ordered if a is ancestor of b or vice versa
            if a not in anc[b] and b not in anc[a]:
                parallel_pairs += 1

    # Structural validity: every input resolves, no obvious structural problem
    dag_valid = all(
        src in nodes
        for n in execution_plan
        for src in n.get("inputs", [])
    )

    ops = sorted({n["op"] for n in execution_plan})

    return {
        "dag_node_count":     len(nodes),
        "dag_depth":          depth,
        "dag_root_count":     len(roots),
        "dag_leaf_count":     len(leaves),
        "dag_parallel_pairs": parallel_pairs,
        "dag_valid":          dag_valid,
        "dag_ops":            ", ".join(ops),
    }


# =========================================================
# REPAIR DIFF  (unchanged from original benchmark)
# =========================================================

def _norm(obj):
    if isinstance(obj, dict):
        return {k: _norm(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [_norm(v) for v in obj]
    if isinstance(obj, str):
        return obj.strip().lower()
    return obj


def compute_repair_diff(raw_sf, validated_sf):
    repairs = []
    if not raw_sf or not validated_sf:
        return repairs

    raw_t = {
        (t.get("entity", ""), t.get("role", ""))
        for t in (raw_sf.get("targets") or [])
        if isinstance(t, dict)
    }
    val_t = {
        (t.get("entity", ""), t.get("role", ""))
        for t in (validated_sf.get("targets") or [])
        if isinstance(t, dict)
    }
    for ent, role in val_t - raw_t:
        repairs.append(f"added_{role}_{ent}")
    for ent, role in raw_t - val_t:
        repairs.append(f"removed_{role}_{ent}")

    raw_a = raw_sf.get("attribute_constraints") or []
    val_a = validated_sf.get("attribute_constraints") or []
    raw_f = {(a.get("target_role",""), a.get("field","")) for a in raw_a if isinstance(a, dict)}
    val_f = {(a.get("target_role",""), a.get("field","")) for a in val_a if isinstance(a, dict)}
    for role, field in val_f - raw_f:
        repairs.append(f"added_attr_{role}_{field}")
    for role, field in raw_f - val_f:
        repairs.append(f"removed_attr_{role}_{field}")

    raw_rm = {a.get("field",""): a.get("target_role","") for a in raw_a if isinstance(a, dict)}
    val_rm = {a.get("field",""): a.get("target_role","") for a in val_a if isinstance(a, dict)}
    for field in set(raw_rm) & set(val_rm):
        if raw_rm[field] != val_rm[field]:
            repairs.append(f"retargeted_{field}_{raw_rm[field]}_to_{val_rm[field]}")

    raw_sc = raw_sf.get("spatial_constraints") or []
    val_sc = validated_sf.get("spatial_constraints") or []
    if len(raw_sc) != len(val_sc):
        repairs.append(f"spatial_constraints_changed_{len(raw_sc)}_to_{len(val_sc)}")

    if raw_sf.get("ranking") is None and validated_sf.get("ranking") is not None:
        repairs.append("added_ranking")
    if raw_sf.get("ranking") is not None and validated_sf.get("ranking") is None:
        repairs.append("removed_ranking")

    if _norm(raw_sf) != _norm(validated_sf) and not repairs:
        repairs.append("other_modification")

    return repairs


# =========================================================
# FRAME ACCESSORS  (unchanged from original benchmark)
# =========================================================

def get_targets_by_role(targets, role):
    return [t for t in targets if isinstance(t, dict) and t.get("role") == role]

def get_entity_names(targets, role):
    names = []
    for t in get_targets_by_role(targets, role):
        for n in (t.get("names") or []):
            if n:
                names.append(str(n).lower())
    return names

def get_entities_by_role(targets, role):
    return [t.get("entity","") for t in get_targets_by_role(targets, role)]

def get_attr_fields(attrs):
    return {a.get("field","") for a in attrs if isinstance(a, dict)}

def get_distances(sp_cons):
    distances = []
    for sc in sp_cons:
        if not isinstance(sc, dict):
            continue
        for key in ("distance_m","distance","radius_m","radius"):
            v = sc.get(key)
            if v is not None:
                try:
                    distances.append(float(v))
                except Exception:
                    pass
    return distances


# =========================================================
# INTENT COMPLETENESS CHECKER  (unchanged from original)
# =========================================================

def check_intent_completeness(query_id: str, validated_sf: dict) -> list[str]:
    """
    Three-level intent check against the strict ground truth.

    LEVEL 1: required elements must be present
    LEVEL 2: forbidden elements must be absent
    LEVEL 3: DAG structural expectations (passed in separately via dag_metrics)
    """
    gt = GROUND_TRUTH.get(query_id)
    if gt is None:
        return []
    missing = []

    targets  = validated_sf.get("targets") or []
    attrs    = validated_sf.get("attribute_constraints") or []
    sp_cons  = validated_sf.get("spatial_constraints") or []
    refs     = validated_sf.get("references") or []
    ranking  = validated_sf.get("ranking")

    primary_entities = get_entities_by_role(targets, "primary")
    support_entities = get_entities_by_role(targets, "support")
    filter_entities  = get_entities_by_role(targets, "filter")
    scope_names      = get_entity_names(targets, "scope")
    attr_fields      = get_attr_fields(attrs)
    distances        = get_distances(sp_cons)

    # ── LEVEL 1: required ────────────────────────────────────────────────

    if gt.get("primary_entity") and gt["primary_entity"] not in primary_entities:
        missing.append(f"L1_primary_entity: expected '{gt['primary_entity']}', got {primary_entities}")

    if gt.get("support_entity") and gt["support_entity"] not in support_entities:
        missing.append(f"L1_support_entity: expected '{gt['support_entity']}', got {support_entities}")

    if gt.get("filter_entity") and gt["filter_entity"] not in filter_entities:
        missing.append(f"L1_filter_entity: expected '{gt['filter_entity']}' in filter role, got {filter_entities}")

    if gt.get("scope_town"):
        for town in gt["scope_town"]:
            if not any(town.lower() in n for n in scope_names):
                missing.append(f"L1_scope_town: '{town}' missing from scope names {scope_names}")

    if gt.get("has_anchor"):
        has_ref = len(refs) > 0
        has_target = any(t.get("role") == "anchor" for t in targets if isinstance(t, dict))
        if not has_ref and not has_target:
            missing.append("L1_has_anchor: no anchor reference or target in frame")

    if gt.get("spatial_distance_m") is not None:
        expected_m = float(gt["spatial_distance_m"])
        if not any(abs(d - expected_m) < 1.0 for d in distances):
            missing.append(f"L1_spatial_distance_m: expected {expected_m}m, frame has {distances}")

    if gt.get("spatial_relation"):
        expected_rel = gt["spatial_relation"]
        found_rels = [sc.get("relation") for sc in sp_cons if isinstance(sc, dict)]
        if expected_rel not in found_rels:
            missing.append(f"L1_spatial_relation: expected '{expected_rel}', found {found_rels}")

    if gt.get("has_ranking") and ranking is None:
        missing.append("L1_has_ranking: ranking spec missing from frame")

    if gt.get("ranking_limit") is not None and ranking is not None:
        top_n = ranking.get("top_n") or ranking.get("limit") or ranking.get("n")
        try:
            top_n = int(top_n)
        except Exception:
            top_n = None
        if top_n != gt["ranking_limit"]:
            missing.append(f"L1_ranking_limit: expected {gt['ranking_limit']}, got {top_n}")

    if gt.get("ranking_order") and ranking is not None:
        order = ranking.get("order", "")
        if gt["ranking_order"] not in order.lower():
            missing.append(f"L1_ranking_order: expected '{gt['ranking_order']}', got '{order}'")

    if gt.get("attr_severity"):
        if not (attr_fields & {core.CRASH_SEVE_COL, "crash_seve"}):
            missing.append(f"L1_attr_severity '{gt['attr_severity']}': field missing from frame")

    if gt.get("attr_hrmf"):
        if not (attr_fields & {core.FIRST_HARM_COL, "first_hrmf"}):
            missing.append(f"L1_attr_hrmf '{gt['attr_hrmf']}': field missing from frame")

    if gt.get("attr_sidewalk"):
        if not (attr_fields & {core.DERIVED_SIDEWALK_STATUS, core.LT_SIDEWALK_COL,
                                core.RT_SIDEWALK_COL, "sidewalk_status"}):
            missing.append("L1_attr_sidewalk: sidewalk field missing from frame")

    if gt.get("attr_speed_above") is not None:
        if not (attr_fields & {core.SPEED_LIM_COL, "speed_lim"}):
            missing.append(f"L1_attr_speed_above {gt['attr_speed_above']}: field missing from frame")

    if gt.get("attr_junction"):
        if not (attr_fields & {core.RDWY_JNCT_COL, "rdwy_jnct_"}):
            missing.append(f"L1_attr_junction '{gt['attr_junction']}': field missing from frame")

    if gt.get("attr_time_start") is not None:
        if not (attr_fields & {core.DERIVED_CRASH_TIME_MINUTES, "crash_time_minutes",
                                core.CRASH_TIME_COL, "crash_time"}):
            missing.append("L1_attr_time: time field missing from frame")

    if gt.get("attr_date_start") is not None:
        if not (attr_fields & {core.DERIVED_CRASH_DATE_VALUE, "crash_date_value",
                                core.CRASH_DATE_COL, "crash_date"}):
            missing.append("L1_attr_date: date field missing from frame")

    # ── LEVEL 2: forbidden ───────────────────────────────────────────────

    if gt.get("no_anchor"):
        has_ref = len(refs) > 0
        has_anchor_target = any(t.get("role") == "anchor" for t in targets if isinstance(t, dict))
        if has_ref or has_anchor_target:
            missing.append(
                f"L2_no_anchor: spurious anchor found (refs={refs}, anchor_target={has_anchor_target})"
            )

    max_sc = gt.get("no_extra_spatial")
    if max_sc is not None and len(sp_cons) > max_sc:
        missing.append(
            f"L2_no_extra_spatial: expected <= {max_sc} spatial constraints, got {len(sp_cons)}"
        )

    if gt.get("no_spurious_filter"):
        if filter_entities:
            missing.append(
                f"L2_no_spurious_filter: filter role must be empty, got {filter_entities}"
            )

    if gt.get("no_support"):
        if support_entities:
            missing.append(
                f"L2_no_support: support role must be empty, got {support_entities}"
            )

    max_ac = gt.get("max_attribute_constraints")
    if max_ac is not None and len(attrs) > max_ac:
        missing.append(
            f"L2_max_attribute_constraints: expected <= {max_ac} attr constraints, got {len(attrs)}"
        )

    return missing


def check_dag_structure(query_id: str, dag_metrics: dict) -> list[str]:
    """
    LEVEL 3: DAG structural checks against ground truth expectations.
    Separated from intent check so results can be reported independently.
    """
    gt = GROUND_TRUTH.get(query_id)
    if gt is None:
        return []
    issues = []

    ops = set(dag_metrics.get("dag_ops", "").split(", ")) if dag_metrics.get("dag_ops") else set()
    n   = dag_metrics.get("dag_node_count", 0)
    d   = dag_metrics.get("dag_depth", 0)

    if gt.get("dag_has_ranking") and not ({"Aggregate", "Rank"} & ops):
        issues.append("L3_dag_has_ranking: Aggregate/Rank nodes missing from DAG")

    if gt.get("dag_has_match") and "MatchSpatialSets" not in ops:
        issues.append("L3_dag_has_match: MatchSpatialSets node missing from DAG")

    if gt.get("dag_has_scope") and "ApplyScopeConstraint" not in ops:
        issues.append("L3_dag_has_scope: ApplyScopeConstraint node missing from DAG")

    if gt.get("dag_has_anchor") and "ResolveReference" not in ops:
        issues.append("L3_dag_has_anchor: ResolveReference node missing from DAG")

    if gt.get("dag_min_nodes") is not None and n < gt["dag_min_nodes"]:
        issues.append(f"L3_dag_node_count: expected >= {gt['dag_min_nodes']}, got {n}")

    if gt.get("dag_max_nodes") is not None and n > gt["dag_max_nodes"]:
        issues.append(f"L3_dag_node_count: expected <= {gt['dag_max_nodes']}, got {n}")

    if gt.get("dag_min_depth") is not None and d < gt["dag_min_depth"]:
        issues.append(f"L3_dag_depth: expected >= {gt['dag_min_depth']}, got {d}")

    if gt.get("dag_max_depth") is not None and d > gt["dag_max_depth"]:
        issues.append(f"L3_dag_depth: expected <= {gt['dag_max_depth']}, got {d}")

    return issues


# =========================================================
# RUN WRAPPER
# =========================================================

def safe_run(assistant, prompt, max_retries=2):
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return assistant.run(prompt, geocode_selection=None)
        except core.AmbiguousLocationError:
            try:
                return assistant.run(prompt, geocode_selection=0)
            except Exception as e:
                last_exc = e
                break
        except Exception as e:
            last_exc = e
            err_str = str(e).lower()
            is_transient = any(k in err_str for k in [
                "none","timeout","503","429","rate limit","empty","transient"
            ])
            if is_transient and attempt < max_retries:
                wait = 5 * (attempt + 1)
                print(f"   [retry {attempt+1}/{max_retries}] transient, waiting {wait}s…")
                time.sleep(wait)
                continue
            break
    raise last_exc or RuntimeError("exhausted retries")


def is_response_agent_failure(warnings):
    return any(
        "response-agent" in str(w).lower() or "api key" in str(w).lower()
        for w in (warnings or [])
    )


def execution_ok(result):
    tables = result.tables or {}
    return bool(
        tables
        or getattr(result, "map_object", None) is not None
        or getattr(result, "temporal_plots", None)
        or getattr(result, "summary", None)
        or getattr(result, "narrative_answer", None)
    )


def row_count(result):
    """
    Count result rows for the benchmark.

    Priority:
    1. result.tables — populated when "table" is in sf.outputs (ranking queries
       always have it; simple retrieval queries may not depending on LLM output).
    2. Parse from result.summary — summarize_result() always writes
       "<Entity> (<role>) selected count: N" lines, so we sum the primary
       and support selected counts as a reliable fallback.
    3. Zero if neither is available.
    """
    # Try tables first (ranking / explicit table output)
    total_from_tables = 0
    for _, df in (result.tables or {}).items():
        try:
            total_from_tables += len(df)
        except Exception:
            pass
    if total_from_tables > 0:
        return total_from_tables

    # Fallback: parse "selected count: N" from summary
    summary = getattr(result, "summary", "") or ""
    # Extract per-role counts; prefer primary role if present, else sum all
    primary_match = re.findall(r'(?:primary)[^\n]*selected count:\s*(\d+)', summary, re.IGNORECASE)
    if primary_match:
        return int(primary_match[0])
    counts = re.findall(r'selected count:\s*(\d+)', summary, re.IGNORECASE)
    if counts:
        # Sum all roles reported — gives total across primary/support/etc.
        return sum(int(c) for c in counts)

    return 0


# =========================================================
# MAIN BENCHMARK LOOP
# =========================================================

rows = []
debug_dump = []
execution_log = []
t_all = time.time()

for i, p in enumerate(PROMPTS, 1):
    pid, prompt, cat = p["id"], p["prompt"], p["cat"]
    print(f"[{i:02d}/{len(PROMPTS)}] {pid}: {prompt}")
    t0 = time.time()

    try:
        try:
            conn.rollback()
        except Exception:
            pass
        with conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = {TIMEOUT_MS};")
        conn.commit()

        result = safe_run(assistant, prompt)
        dt = round(time.time() - t0, 2)

        dbg          = result.debug or {}
        raw_sf       = dbg.get("raw_semantic_frame") or {}
        validated_sf = dbg.get("validated_semantic_frame") or result.semantic_frame or {}
        warnings     = result.warnings or []

        # ── DAG metrics first (needed by check_dag_structure) ──
        dag_metrics = analyse_dag(result.execution_plan or [])

        # ── semantic frame analysis ──────────────────────────
        repair_list        = compute_repair_diff(raw_sf, validated_sf)
        was_repaired       = len(repair_list) > 0
        n_rows             = row_count(result)
        resp_agent_fail    = is_response_agent_failure(warnings)
        exec_ok            = execution_ok(result) or resp_agent_fail
        missing_constraints = check_intent_completeness(pid, validated_sf)
        dag_issues          = check_dag_structure(pid, dag_metrics)
        all_issues          = missing_constraints + dag_issues
        intent_complete     = len(all_issues) == 0
        intent_issues       = "; ".join(all_issues)
        l1_issues           = [x for x in missing_constraints if x.startswith("L1_")]
        l2_issues           = [x for x in missing_constraints if x.startswith("L2_")]
        l3_issues           = dag_issues

        targets = validated_sf.get("targets") or []
        attrs   = validated_sf.get("attribute_constraints") or []
        ranking = validated_sf.get("ranking")
        attr_fields_set = get_attr_fields(attrs)

        primary_entity = (get_entities_by_role(targets, "primary") or [None])[0]
        has_ranking  = ranking is not None
        has_scope    = bool(get_targets_by_role(targets, "scope"))
        has_anchor   = (
            len(validated_sf.get("references") or []) > 0
            or bool(get_targets_by_role(targets, "anchor"))
        )
        has_temporal = bool(attr_fields_set & {
            core.DERIVED_CRASH_TIME_MINUTES, core.DERIVED_CRASH_DATE_VALUE,
            "crash_time_minutes", "crash_date_value",
        })
        has_hrmf     = bool(attr_fields_set & {core.FIRST_HARM_COL, "first_hrmf"})
        has_sev      = bool(attr_fields_set & {core.CRASH_SEVE_COL, "crash_seve"})
        has_sidewalk = bool(attr_fields_set & {
            core.DERIVED_SIDEWALK_STATUS, core.LT_SIDEWALK_COL,
            core.RT_SIDEWALK_COL, "sidewalk_status",
        })
        has_speed    = bool(attr_fields_set & {core.SPEED_LIM_COL, "speed_lim"})
        has_junction = bool(attr_fields_set & {core.RDWY_JNCT_COL, "rdwy_jnct_"})

        exec_error = ""
        if not exec_ok:
            non_ra = [w for w in warnings
                      if "response-agent" not in str(w).lower()
                      and "api key" not in str(w).lower()]
            exec_error = "; ".join(non_ra)

        rows.append({
            "id":           pid,
            "prompt":       prompt,
            "category":     cat,
            "llm_provider": LLM_PROVIDER,
            "llm_model":    LLM_MODEL,
            "execution_mode": "schema_grounded",
            "row_count":    n_rows,
            "runtime_sec":  dt,
            "timed_out":    False,
            "exec_error":   exec_error,
            "response_agent_fail": resp_agent_fail,
            # intent
            "intent_complete": intent_complete,
            "intent_issues":   intent_issues,
            "n_missing":       len(all_issues),
            "n_l1_missing":    len(l1_issues),
            "n_l2_spurious":   len(l2_issues),
            "n_l3_dag":        len(l3_issues),
            "l1_issues":       "; ".join(l1_issues),
            "l2_issues":       "; ".join(l2_issues),
            "l3_issues":       "; ".join(l3_issues),
            # repair
            "was_repaired":  was_repaired,
            "repair_count":  len(repair_list),
            "repairs":       "; ".join(repair_list) if repair_list else "",
            # frame features
            "primary_entity":      primary_entity,
            "scope_towns":         ", ".join(get_entity_names(targets, "scope")),
            "has_ranking":         has_ranking,
            "has_scope":           has_scope,
            "has_anchor":          has_anchor,
            "has_temporal":        has_temporal,
            "has_hrmf_filter":     has_hrmf,
            "has_sev_filter":      has_sev,
            "has_sidewalk_filter": has_sidewalk,
            "has_speed_filter":    has_speed,
            "has_junction_filter": has_junction,
            "summary": (result.summary or "")[:200],
            # DAG metrics
            **dag_metrics,
        })

        debug_dump.append({
            "id":                    pid,
            "prompt":                prompt,
            "execution_ok":          exec_ok,
            "sec":                   dt,
            "row_count":             n_rows,
            "raw_semantic_frame":    raw_sf,
            "validated_semantic_frame": validated_sf,
            "execution_plan":        result.execution_plan,
            "dag_metrics":           dag_metrics,
            "repairs":               repair_list,
            "was_repaired":          was_repaired,
            "intent_complete":       intent_complete,
            "intent_issues":         all_issues,
            "l1_issues":             l1_issues,
            "l2_issues":             l2_issues,
            "l3_issues":             l3_issues,
            "resolved_anchors":      dbg.get("resolved_anchors", {}),
            "summary":               result.summary,
            "warnings":              warnings,
        })

        execution_log.append({
            "id":             pid,
            "prompt":         prompt,
            "category":       cat,
            "execution_ok":   exec_ok,
            "row_count":      n_rows,
            "runtime_sec":    dt,
            "intent_complete": intent_complete,
            **dag_metrics,
        })

        intent_flag = "✓ complete" if intent_complete else f"MISSING: {intent_issues[:60]}"
        print(
            f"   {'OK  ' if exec_ok else 'WARN'}  {dt}s  "
            f"rows={n_rows:,}  repaired={was_repaired}  "
            f"nodes={dag_metrics['dag_node_count']}  "
            f"depth={dag_metrics['dag_depth']}  "
            f"intent={intent_flag}"
        )

    except Exception as e:
        dt = round(time.time() - t0, 2)
        err = str(e)
        is_timeout = "statement timeout" in err.lower() or "query_canceled" in err.lower()
        try:
            conn.rollback()
        except Exception:
            pass

        rows.append({
            "id":           pid,
            "prompt":       prompt,
            "category":     cat,
            "llm_provider": LLM_PROVIDER,
            "llm_model":    LLM_MODEL,
            "execution_mode": "schema_grounded",
            "executed_ok":  False,
            "row_count":    0,
            "runtime_sec":  dt,
            "timed_out":    is_timeout,
            "exec_error":   err[:300],
            "response_agent_fail": False,
            "intent_complete": False,
            "intent_issues":   "EXCEPTION",
            "n_missing":       -1,
            "was_repaired":    False,
            "repair_count":    0,
            "repairs":         "",
            "primary_entity":      "",
            "scope_towns":         "",
            "has_ranking":         False,
            "has_scope":           False,
            "has_anchor":          False,
            "has_temporal":        False,
            "has_hrmf_filter":     False,
            "has_sev_filter":      False,
            "has_sidewalk_filter": False,
            "has_speed_filter":    False,
            "has_junction_filter": False,
            "summary":             "",
            "dag_node_count":      0,
            "dag_depth":           0,
            "dag_root_count":      0,
            "dag_leaf_count":      0,
            "dag_parallel_pairs":  0,
            "dag_valid":           False,
            "dag_ops":             "",
        })
        debug_dump.append({
            "id":    pid,
            "prompt": prompt,
            "execution_ok": False,
            "sec":   dt,
            "error": err,
            "traceback": traceback.format_exc(),
        })
        execution_log.append({
            "id": pid, "prompt": prompt, "category": cat,
            "execution_ok": False, "row_count": 0,
            "runtime_sec": dt, "error": err[:200],
        })
        print(f"   {'TIMEOUT' if is_timeout else 'FAIL '}: {err[:80]}")

    finally:
        try:
            conn.rollback()
            with conn.cursor() as cur:
                cur.execute("SET statement_timeout = 0;")
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass

total_sec = round(time.time() - t_all, 1)


# =========================================================
# SAVE RESULTS
# =========================================================

df = pd.DataFrame(rows)
df["group"] = df["id"].str.extract(r"(G\d+)")

ok           = df[df["executed_ok"]]
n_ok         = len(ok)
n_timeout    = int(df["timed_out"].sum())
n_fail       = len(df) - n_ok - n_timeout
n_repaired   = int(ok["was_repaired"].sum()) if n_ok else 0
n_intent_ok  = int(df["intent_complete"].sum())
n_intent_fail = len(df) - n_intent_ok

with pd.ExcelWriter(f"{OUT_DIR}/results.xlsx", engine="openpyxl") as writer:

    df.to_excel(writer, sheet_name="All Results", index=False)

    # ── per-group summary ──────────────────────────────────
    gs = (
        df.groupby("group")
        .agg(
            total=        ("executed_ok",  "count"),
            executed_ok=  ("executed_ok",  "sum"),
            intent_ok=    ("intent_complete","sum"),
            n_missing_avg=("n_missing",    "mean"),
            timed_out=    ("timed_out",    "sum"),
            avg_sec=      ("runtime_sec",  "mean"),
            max_sec=      ("runtime_sec",  "max"),
            avg_rows=     ("row_count",    "mean"),
            repaired=     ("was_repaired", "sum"),
            avg_nodes=    ("dag_node_count","mean"),
            avg_depth=    ("dag_depth",    "mean"),
            avg_parallel= ("dag_parallel_pairs","mean"),
        )
        .assign(
            execution_fail= lambda x: x["total"] - x["executed_ok"] - x["timed_out"],
            intent_fail=    lambda x: x["total"] - x["intent_ok"],
            avg_sec=        lambda x: x["avg_sec"].round(1),
            max_sec=        lambda x: x["max_sec"].round(1),
            avg_rows=       lambda x: x["avg_rows"].round(0).astype(int),
            n_missing_avg=  lambda x: x["n_missing_avg"].round(2),
            avg_nodes=      lambda x: x["avg_nodes"].round(1),
            avg_depth=      lambda x: x["avg_depth"].round(1),
            avg_parallel=   lambda x: x["avg_parallel"].round(1),
        )
        .reset_index()
    )
    gs["group_name"] = gs["group"].map(GROUP_NAMES)
    gs.to_excel(writer, sheet_name="By Group", index=False)

    # ── DAG structure sheet ────────────────────────────────
    dag_cols = [
        "id","prompt","category",
        "dag_node_count","dag_depth","dag_root_count","dag_leaf_count",
        "dag_parallel_pairs","dag_valid","dag_ops",
        "executed_ok","runtime_sec",
    ]
    df[dag_cols].to_excel(writer, sheet_name="DAG Structure", index=False)

    # ── intent failures ────────────────────────────────────
    incomplete = df[~df["intent_complete"]][[
        "id","prompt","category","intent_issues","n_missing",
        "executed_ok","row_count","was_repaired",
    ]]
    incomplete.to_excel(writer, sheet_name="Intent Failures", index=False)

    # ── repair analysis ────────────────────────────────────
    ok[[
        "id","prompt","category","was_repaired","repair_count","repairs"
    ]].to_excel(writer, sheet_name="Repair Analysis", index=False)

    # ── execution results ──────────────────────────────────
    df[[
        "id","prompt","category","executed_ok","row_count","runtime_sec","summary"
    ]].to_excel(writer, sheet_name="Execution Results", index=False)

    # ── failures ──────────────────────────────────────────
    fails = df[~df["executed_ok"]]
    if len(fails):
        fails[[
            "id","prompt","category","timed_out","runtime_sec","exec_error"
        ]].to_excel(writer, sheet_name="Exec Failures", index=False)

    # ── feature flags ──────────────────────────────────────
    df[[
        "id","prompt","category",
        "has_ranking","has_scope","has_anchor","has_temporal",
        "has_hrmf_filter","has_sev_filter","has_sidewalk_filter",
        "has_speed_filter","has_junction_filter",
        "intent_complete","n_missing",
    ]].to_excel(writer, sheet_name="Feature Flags", index=False)


def _safe_json(o):
    try:
        json.dumps(o)
        return o
    except Exception:
        if isinstance(o, dict):
            return {str(k): _safe_json(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_safe_json(v) for v in o]
        return str(o)


with open(f"{OUT_DIR}/debug.json", "w", encoding="utf-8") as f:
    json.dump(_safe_json(debug_dump), f, ensure_ascii=False, indent=2)

with open(f"{OUT_DIR}/execution_log.json", "w", encoding="utf-8") as f:
    json.dump(_safe_json(execution_log), f, ensure_ascii=False, indent=2)


# =========================================================
# SUMMARY
# =========================================================

all_repairs = []
for r in ok["repairs"]:
    if r:
        all_repairs.extend(r.split("; "))
repair_counts = pd.Series(all_repairs).value_counts()

all_intent = []
for r in df["intent_issues"]:
    if r and r != "EXCEPTION":
        all_intent.extend(r.split("; "))
intent_counts = pd.Series(all_intent).value_counts()

summary_lines = [
    f"Road Safety Benchmark  [{LLM_PROVIDER.upper()}  {LLM_MODEL}]",
    "=" * 60,
    f"Total: {len(df)}  |  Execution OK: {n_ok}  |  Timeout: {n_timeout}  |  Failed: {n_fail}",
    f"Total runtime: {total_sec}s",
    "",
    (f"Runtime (OK):  Median {ok['runtime_sec'].median():.1f}s  "
     f"Mean {ok['runtime_sec'].mean():.1f}s  "
     f"Max {ok['runtime_sec'].max():.1f}s") if n_ok else "",
    "",
    "DAG Structure (executed queries):",
    (f"  Avg nodes:     {ok['dag_node_count'].mean():.1f}") if n_ok else "",
    (f"  Avg depth:     {ok['dag_depth'].mean():.1f}") if n_ok else "",
    (f"  Avg parallel pairs: {ok['dag_parallel_pairs'].mean():.1f}") if n_ok else "",
    (f"  All DAGs valid: {ok['dag_valid'].all()}") if n_ok else "",
    "",
    "Intent Completeness:",
    f"  Complete:   {n_intent_ok}/{len(df)} ({100*n_intent_ok/len(df):.0f}%)",
    f"  Incomplete: {n_intent_fail}/{len(df)} ({100*n_intent_fail/len(df):.0f}%)",
    "",
    "Repair Analysis:",
    (f"  Frames repaired: {n_repaired}/{n_ok} ({100*n_repaired/n_ok:.0f}%)") if n_ok else "",
    f"  Total individual repairs: {int(ok['repair_count'].sum()) if n_ok else 0}",
]

if len(repair_counts):
    summary_lines.append("  Repair type breakdown (top 15):")
    for rtype, cnt in repair_counts.head(15).items():
        summary_lines.append(f"    {rtype}: {cnt}")

if len(intent_counts):
    summary_lines.append("\nIntent failure breakdown (top 15):")
    for itype, cnt in intent_counts.head(15).items():
        summary_lines.append(f"    {itype}: {cnt}")

summary_text = "\n".join(summary_lines)
with open(f"{OUT_DIR}/summary.txt", "w", encoding="utf-8") as f:
    f.write(summary_text)


# =========================================================
# DISPLAY
# =========================================================

print(f"\n{summary_text}")
print(f"\n{'='*60}")
print(f"Saved: {OUT_DIR}/")

print("\n── By Group ──")
display(gs[[
    "group","group_name","total","executed_ok","execution_fail",
    "intent_ok","intent_fail","timed_out",
    "avg_sec","max_sec","avg_rows","repaired",
    "avg_nodes","avg_depth","avg_parallel",
]])

if n_fail > 0 or n_timeout > 0:
    print("\n── Execution Failures ──")
    display(df[~df["executed_ok"]][[
        "id","prompt","category","timed_out","runtime_sec","exec_error"
    ]])

print(f"\n── Intent Completeness Failures ({n_intent_fail}) ──")
if len(incomplete):
    display(incomplete.sort_values("n_missing", ascending=False))
else:
    print("  None — all frames captured complete intent.")

print(f"\n── Repair Analysis ({n_repaired}/{n_ok} frames modified) ──")
rep_rows = ok[ok["was_repaired"]][["id","prompt","repair_count","repairs"]]
if len(rep_rows):
    display(rep_rows)
else:
    print("  No repairs needed.")

print("\n── DAG Structure by Group ──")
display(gs[["group","group_name","avg_nodes","avg_depth","avg_parallel"]].round(1))

print("\n── G9 Combined ──")
g9 = df[df["group"]=="G9"][[
    "id","prompt","executed_ok","intent_complete","n_missing",
    "runtime_sec","dag_node_count","dag_depth","dag_parallel_pairs",
]]
display(g9)

print("\n── All Results ──")
display(df[[
    "id","prompt","category","executed_ok","row_count","runtime_sec",
    "intent_complete","n_missing","was_repaired","repair_count",
    "dag_node_count","dag_depth","dag_parallel_pairs",
]])
