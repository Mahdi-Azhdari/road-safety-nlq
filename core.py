"""
Road Safety Assistant — core pipeline.

Natural-language interface for transportation safety analysis against a
PostGIS database. The pipeline interprets user queries into a semantic frame,
validates and repairs the frame against the supported schema, compiles it
into a typed DAG of spatial operations, and executes it.
"""

import os
import re
import json
from datetime import datetime, date
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import requests
import psycopg2
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point
import folium

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/notebook use
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    from IPython.display import display as ipy_display
except Exception:
    ipy_display = None

from branca.element import MacroElement
from jinja2 import Template


# =========================================================
# 1) CONFIG AND CONSTANTS
# =========================================================
# Credentials default to empty; the Streamlit app supplies them at runtime.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
DB_PASSWORD = os.getenv("DB_PASSWORD", "").strip()
RESPONSE_GEMINI_API_KEY = os.getenv("RESPONSE_GEMINI_API_KEY", "").strip()

GEMINI_MODEL = "gemini-2.5-flash"
RESPONSE_GEMINI_MODEL = os.getenv("RESPONSE_GEMINI_MODEL", GEMINI_MODEL).strip() or GEMINI_MODEL

STATE_SUFFIX = "Massachusetts"
DEFAULT_RADIUS_M = 200.0
DEFAULT_TOP_N = 20
MAX_TOP_N = 100
GEOCODE_LIMIT = 5

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_UA = "RoadSafetyAssistant/1.0 (research)"


class AmbiguousLocationError(Exception):
    """Raised when geocoding returns multiple results and user must choose."""
    def __init__(self, place_name: str, options: list[dict]):
        self.place_name = place_name
        self.options = options  # list of {display_name, lat, lon}
        labels = [o["display_name"] for o in options]
        super().__init__(f"Multiple locations found for '{place_name}': {labels}")

CRASH_TABLE = 'public."Crash"'
ROAD_TABLE = 'public."Road_Inventory_2025"'
SCHOOL_TABLE = 'public."SCHOOLS_PT"'
TOWN_TABLE = 'public."Towns.Mass"'

# Bus stop table (edit if your schema differs)
BUS_STOP_TABLE = 'public."Bus_stops"'
BUS_STOP_ID_COL = "stop_id"
BUS_STOP_NAME_COL = "stop_name"
BUS_STOP_GEOM_COL = "geom"

# Crosswalk table (edit if your schema differs)
CROSSWALK_TABLE = 'public."Crosswalks.shp"'
CROSSWALK_ID_COL = "id"
CROSSWALK_GEOM_COL = "geom"

CRASH_ID_COL = "id"
CRASH_GEOM_COL = "geom"
CRASH_SEVE_COL = "crash_seve"
CRASH_DATE_COL = "crash_date"
CRASH_TIME_COL = "crash_time"
FIRST_HARM_COL = "first_hrmf"

SCHOOL_NAME_COL = "name"
SCHOOL_GEOM_COL = "geom"

TOWN_NAME_COL = "namelsad20"
TOWN_GEOM_COL = "geom"

LT_SIDEWALK_COL = "lt_sidewlk"
RT_SIDEWALK_COL = "rt_sidewlk"
SPEED_LIM_COL = "speed_lim"
OP_DIR_SPEED_LIM_COL = "op_dir_sl"
RDWY_JNCT_COL = "rdwy_jnct_"
ROAD_GEOM_COL_FALLBACK = "geom"

DERIVED_CRASH_DATE_VALUE = "crash_date_value"
DERIVED_CRASH_TIME_MINUTES = "crash_time_minutes"
DERIVED_SIDEWALK_STATUS = "sidewalk_status"
DERIVED_NAME_MATCH_KEY = "name_match_key"

SNAP_CRASH_TO_ROAD_M = 15.0
ROAD_RANKING_SNAP_M = 50.0  # snap distance for ranking road segments by crash count
MAX_RENDER_ROADS = 500000
MAX_RENDER_CRASHES = 200000
MAX_RENDER_SCHOOLS = 50000
MAX_RENDER_BUS_STOPS = 50000
MAX_RENDER_CROSSWALKS = 110000
MAX_RENDER_TOWNS = 10000

SEV_PDO = "Property damage only (none injured)"
SEV_NONFATL = "Non-fatal injury"
SEV_FATL = "Fatal injury"
SEV_UNK = "Unknown"
CRASH_SEVE_CANONICAL = [SEV_PDO, SEV_NONFATL, SEV_FATL, SEV_UNK]

FIRST_HARM_LEVELS = [
    "Collision with animal - deer",
    "Collision with animal - other",
    "Collision with bridge",
    "Collision with bridge overhead structure",
    "Collision with curb",
    "Collision with cyclist",
    "Collision with ditch",
    "Collision with embankment",
    "Collision with guardrail",
    "Collision with median barrier",
    "Collision with motor vehicle in traffic",
    "Collision with other light pole or other post/support",
    "Collision with other movable object",
    "Collision with Other Vulnerable User",
    "Collision with parked motor vehicle",
    "Collision with pedestrian",
    "Collision with railway vehicle (e.g., train, engine)",
    "Collision with tree",
    "Collision with unknown fixed object",
    "Collision with utility pole",
    "Collision with work zone maintenance equipment",
    "Collison with moped",
    "Fell/Jumped From Motor Vehicle",
    "Jackknife",
    "Not reported",
    "Other",
    "Other non-collision",
    "Overturn/rollover",
    "Reported but invalid",
    "Unknown",
    "Unknown non-collision",
]

MONTH_NAME_TO_NUM = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

# Trailing suffixes stripped during Town normalization
TOWN_SUFFIX_VARIANTS = [
    "town",
    "city",
]

CRASH_SPECIFIC_FIELDS = {
    CRASH_SEVE_COL,
    CRASH_DATE_COL,
    CRASH_TIME_COL,
    FIRST_HARM_COL,
    LT_SIDEWALK_COL,
    RT_SIDEWALK_COL,
    SPEED_LIM_COL,
    RDWY_JNCT_COL,
    DERIVED_SIDEWALK_STATUS,
    DERIVED_CRASH_DATE_VALUE,
    DERIVED_CRASH_TIME_MINUTES,
}

# Fields that are ONLY on Road (not merged into Crash)
ROAD_ONLY_FIELDS = {
    OP_DIR_SPEED_LIM_COL,
}

# Fields that exist on both Road and Crash (shared from merged inventory)
SHARED_ROAD_CRASH_FIELDS = {
    LT_SIDEWALK_COL,
    RT_SIDEWALK_COL,
    SPEED_LIM_COL,
    DERIVED_SIDEWALK_STATUS,
}

BUS_STOP_SPECIFIC_FIELDS = {
    BUS_STOP_ID_COL,
    BUS_STOP_NAME_COL,
}

CROSSWALK_SPECIFIC_FIELDS = {
    CROSSWALK_ID_COL,
}

TOWN_SPECIFIC_FIELDS = {
    TOWN_NAME_COL,
    DERIVED_NAME_MATCH_KEY,
}

SUPPORTED_ROLE_NAMES = {"primary", "support", "anchor", "scope", "filter"}
SUPPORTED_TARGET_ENTITIES = {"Crash", "Road", "School", "BusStop", "Crosswalk", "Town"}
SUPPORTED_SCOPE_ENTITIES = {"Town"}
SUPPORTED_REFERENCE_ENTITIES = {"Place", "School"}
SUPPORTED_OUTPUTS = {"map", "summary", "table"}
SUPPORTED_SPATIAL_RELATIONS = {"within_distance", "intersects", "contains", "nearest_to"}
SUPPORTED_RELATIONS = {"snap_match"}
SUPPORTED_ATTRIBUTE_OPERATORS = {"eq", "in", "gt", "gte", "lt", "lte", "between", "is_null", "not_null"}
SUPPORTED_RANK_METRICS = {"crash_count"}

GENERIC_SET_MATCH_RELATIONS = {"within_distance", "intersects", "contains"}
SCOPE_FILTER_RELATION_DEFAULT = "intersects"
RELATION_DISTANCE_DEFAULTS = {
    "within_distance": DEFAULT_RADIUS_M,
    "snap_match": SNAP_CRASH_TO_ROAD_M,
}

RESPONSE_LAYER_VERSION = "phase1_phase2_v1"
RESPONSE_AGENT_MAX_CHARS = 1200


# =========================================================
# 2) DATACLASSES / TYPED OBJECTS
# =========================================================
@dataclass
class TargetSpec:
    entity: Optional[str] = None
    role: Optional[str] = None
    names: list[str] = field(default_factory=list)


@dataclass
class ReferenceSpec:
    entity: Optional[str] = None
    role: Optional[str] = None
    name: Optional[str] = None


@dataclass
class SpatialConstraint:
    relation: Optional[str] = None
    target_role: Optional[str] = None
    reference_role: Optional[str] = None
    distance_m: Optional[float] = None


@dataclass
class AttributeConstraint:
    target_role: Optional[str] = None
    field: Optional[str] = None
    operator: Optional[str] = None
    value: Any = None


@dataclass
class RelationConstraint:
    relation: Optional[str] = None
    source_role: Optional[str] = None
    target_role: Optional[str] = None
    distance_m: Optional[float] = None


@dataclass
class RankingSpec:
    metric: Optional[str] = None
    target_role: Optional[str] = None
    order: Optional[str] = None
    top_n: Optional[int] = None


@dataclass
class SemanticFrame:
    supported: bool
    targets: list[TargetSpec] = field(default_factory=list)
    references: list[ReferenceSpec] = field(default_factory=list)
    spatial_constraints: list[SpatialConstraint] = field(default_factory=list)
    attribute_constraints: list[AttributeConstraint] = field(default_factory=list)
    relations: list[RelationConstraint] = field(default_factory=list)
    ranking: Optional[RankingSpec] = None
    outputs: list[str] = field(default_factory=lambda: ["map", "summary"])
    notes: Optional[str] = None


@dataclass
class ResolverResult:
    entity: str
    role: str
    name: str
    display_name: str
    lat: float
    lon: float
    dist_m: float
    wkt_26986: str


@dataclass
class ReferenceObject:
    role: str
    entity: str
    df: pd.DataFrame


@dataclass
class DatasetSpec:
    entity: str
    table: str
    geometry_column: str
    geometry_family: str
    primary_key: Optional[str]
    label_field: Optional[str]
    display_fields: list[str]
    fields: dict[str, dict]
    derived_fields: dict[str, dict] = field(default_factory=dict)
    relation_capabilities: list[str] = field(default_factory=list)
    scope_capable: bool = False
    name_match_field: Optional[str] = None
    name_match_strip_suffixes: list[str] = field(default_factory=list)
    display_geometry_mode: str = "native"


@dataclass
class RoleBinding:
    role: str
    entity: str


@dataclass
class DAGNode:
    """
    Single typed operation in the execution DAG.

    Each node carries a stable unique node_id, a typed op (drawn from the same
    handler vocabulary the linear plan used), a params dict, and an explicit
    list of input node_ids representing data dependencies.
    """
    node_id: str
    op: str
    params: dict[str, Any] = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)


@dataclass
class DAGPlan:
    """
    Typed directed acyclic execution plan.

    Replaces LinearExecutionPlan. Holds nodes keyed by node_id, plus a cached
    deterministic topological order computed at compile time. The executor
    iterates `order` to dispatch nodes; equivalent to the old linear sequence
    but the structure is now explicit and ready for branching extensions.
    """
    nodes: dict[str, DAGNode] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    roots: list[str] = field(default_factory=list)
    leaves: list[str] = field(default_factory=list)

    @property
    def steps(self) -> list[DAGNode]:
        """
        Backward-compatible accessor — returns nodes in topological order.

        Existing call sites that iterated `plan.steps` (debug rendering,
        response bundle building) keep working without modification.
        """
        return [self.nodes[nid] for nid in self.order]


@dataclass
class MatchSpec:
    left_role: str
    right_role: str
    relation: str
    distance_m: Optional[float] = None
    left_keep: bool = True
    right_keep: bool = True
    left_tag_field: Optional[str] = None
    right_tag_field: Optional[str] = None


@dataclass
class AggregateSpec:
    metric: str
    group_role: str
    measure_role: str
    relation: str
    distance_m: Optional[float] = None
    output_name: str = "aggregate_table"
    value_column: str = "metric_value"


@dataclass
class MatchMetadata:
    left_role: str
    right_role: str
    left_entity: str
    right_entity: str
    relation: str
    distance_m: Optional[float]
    left_key_field: Optional[str] = None
    right_key_field: Optional[str] = None
    left_tag_field: Optional[str] = None
    right_tag_field: Optional[str] = None
    matched_left_count: Optional[int] = None
    matched_right_count: Optional[int] = None
    pair_count: Optional[int] = None


@dataclass
class RoleData:
    role: str
    entity: str
    table: str
    geometry_column: str
    geometry_family: str
    primary_key: Optional[str]
    label_field: Optional[str]
    sql_base: Optional[str] = None
    params: dict[str, Any] = field(default_factory=dict)
    selected_count: int = 0
    gdf: Optional[gpd.GeoDataFrame] = None
    render_limit: Optional[int] = None
    display_fields: list[str] = field(default_factory=list)
    selected_names: list[str] = field(default_factory=list)
    applied_name_filters: list[dict[str, Any]] = field(default_factory=list)
    applied_attribute_constraints: list[dict[str, Any]] = field(default_factory=list)
    applied_spatial_constraints: list[dict[str, Any]] = field(default_factory=list)
    applied_relations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TableObject:
    name: str
    df: pd.DataFrame


def make_empty_analysis_state() -> dict[str, Any]:
    return {
        "aggregate_table": None,
        "ranking_table": None,
        "primary_display_gdf": None,
        "support_display_gdf": None,
    }


@dataclass
class ExecutionState:
    user_prompt: str
    semantic_frame: Optional[SemanticFrame] = None
    road_geom_col: Optional[str] = None
    role_bindings: dict[str, RoleBinding] = field(default_factory=dict)
    dataset_specs_by_role: dict[str, DatasetSpec] = field(default_factory=dict)
    resolver_results: dict[str, ResolverResult] = field(default_factory=dict)
    references_by_role: dict[str, ReferenceObject] = field(default_factory=dict)
    role_data: dict[str, RoleData] = field(default_factory=dict)
    match_metadata_by_name: dict[str, MatchMetadata] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=make_empty_analysis_state)
    summary_text: Optional[str] = None
    map_object: Optional[folium.Map] = None
    tables: list[TableObject] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)


# -------------------------
# Response-layer dataclasses
# -------------------------
@dataclass
class ResponseBundleTableSummary:
    name: str
    row_count: int
    columns: list[str] = field(default_factory=list)
    preview_rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ResponseBundleRoleSummary:
    role: str
    entity: str
    selected_count: int
    selected_names: list[str] = field(default_factory=list)
    has_materialized_gdf: bool = False
    render_row_count: int = 0
    label_field: Optional[str] = None
    primary_key: Optional[str] = None


@dataclass
class ResponseBundleMatchSummary:
    name: str
    left_role: str
    right_role: str
    left_entity: str
    right_entity: str
    relation: str
    distance_m: Optional[float] = None
    matched_left_count: Optional[int] = None
    matched_right_count: Optional[int] = None
    pair_count: Optional[int] = None


@dataclass
class ResponseBundleDownloadItem:
    name: str
    kind: str
    available: bool = False
    filename: Optional[str] = None
    note: Optional[str] = None


@dataclass
class ResponseBundleConversationContext:
    prior_context_available: bool = False
    compact_context: Optional[str] = None


@dataclass
class StructuredResponseBundle:
    bundle_version: str
    original_user_prompt: str
    supported: bool
    success: bool

    semantic_frame: dict[str, Any] = field(default_factory=dict)
    execution_plan: list[dict[str, Any]] = field(default_factory=list)

    primary_entity: Optional[str] = None
    primary_role: Optional[str] = None
    support_entity: Optional[str] = None
    support_role: Optional[str] = None
    scope_entity: Optional[str] = None
    scope_names: list[str] = field(default_factory=list)

    anchor_descriptions: list[str] = field(default_factory=list)
    spatial_constraint_descriptions: list[str] = field(default_factory=list)
    attribute_constraint_descriptions: list[str] = field(default_factory=list)
    relation_descriptions: list[str] = field(default_factory=list)
    ranking_description: Optional[str] = None

    selected_counts_by_role: dict[str, int] = field(default_factory=dict)
    role_summaries: list[ResponseBundleRoleSummary] = field(default_factory=list)
    match_summaries: list[ResponseBundleMatchSummary] = field(default_factory=list)

    aggregate_exists: bool = False
    aggregate_row_count: int = 0
    ranking_exists: bool = False
    ranking_row_count: int = 0
    map_exists: bool = False
    table_count: int = 0

    tables: list[ResponseBundleTableSummary] = field(default_factory=list)

    empty_result: bool = False
    partial_success: bool = False

    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    empty_result_note: Optional[str] = None

    failure_type: Optional[str] = None
    error_message: Optional[str] = None

    deterministic_summary: Optional[str] = None

    downloads: list[ResponseBundleDownloadItem] = field(default_factory=list)
    conversation_context: Optional[ResponseBundleConversationContext] = None


@dataclass
class NarrativeResponse:
    available: bool
    used_response_agent: bool
    text: Optional[str]
    fallback_reason: Optional[str] = None
    raw_text: Optional[str] = None


@dataclass
class PublicResult:
    semantic_frame: dict[str, Any]
    execution_plan: list[dict[str, Any]]
    summary: Optional[str]
    map_object: Optional[folium.Map]
    tables: dict[str, pd.DataFrame]
    debug: dict[str, Any]
    response_bundle: Optional[dict[str, Any]] = None
    narrative_answer: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    downloads: list[dict[str, Any]] = field(default_factory=list)
    temporal_plots: list[Any] = field(default_factory=list)  # matplotlib Figure objects
    gdfs: dict[str, gpd.GeoDataFrame] = field(default_factory=dict)  # spatial layers by role


# =========================================================
# 3) DATASET REGISTRY
# =========================================================
DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "Crash": DatasetSpec(
        entity="Crash",
        table=CRASH_TABLE,
        geometry_column=CRASH_GEOM_COL,
        geometry_family="point",
        primary_key=CRASH_ID_COL,
        label_field=CRASH_ID_COL,
        display_fields=[CRASH_ID_COL, CRASH_SEVE_COL, CRASH_DATE_COL, CRASH_TIME_COL, FIRST_HARM_COL, LT_SIDEWALK_COL, RT_SIDEWALK_COL, SPEED_LIM_COL, RDWY_JNCT_COL],
        fields={
            CRASH_ID_COL: {"type": "id"},
            CRASH_SEVE_COL: {"type": "categorical"},
            CRASH_DATE_COL: {"type": "text"},
            CRASH_TIME_COL: {"type": "text"},
            FIRST_HARM_COL: {"type": "categorical"},
            LT_SIDEWALK_COL: {"type": "numeric"},
            RT_SIDEWALK_COL: {"type": "numeric"},
            SPEED_LIM_COL: {"type": "numeric"},
            RDWY_JNCT_COL: {"type": "categorical"},
        },
        derived_fields={
            DERIVED_CRASH_DATE_VALUE: {
                "type": "date",
                "depends_on": [CRASH_DATE_COL],
            },
            DERIVED_CRASH_TIME_MINUTES: {
                "type": "numeric",
                "depends_on": [CRASH_TIME_COL],
            },
            DERIVED_SIDEWALK_STATUS: {
                "type": "categorical",
                "depends_on": [LT_SIDEWALK_COL, RT_SIDEWALK_COL],
            },
        },
        relation_capabilities=["snap_match"],
        scope_capable=False,
        name_match_field=None,
        name_match_strip_suffixes=[],
        display_geometry_mode="native",
    ),
    "Road": DatasetSpec(
        entity="Road",
        table=ROAD_TABLE,
        geometry_column=ROAD_GEOM_COL_FALLBACK,
        geometry_family="line",
        primary_key="objectid",
        label_field="objectid",
        display_fields=["objectid", SPEED_LIM_COL, OP_DIR_SPEED_LIM_COL, LT_SIDEWALK_COL, RT_SIDEWALK_COL],
        fields={
            SPEED_LIM_COL: {"type": "numeric"},
            OP_DIR_SPEED_LIM_COL: {"type": "numeric"},
            LT_SIDEWALK_COL: {"type": "numeric"},
            RT_SIDEWALK_COL: {"type": "numeric"},
        },
        derived_fields={
            DERIVED_SIDEWALK_STATUS: {
                "type": "categorical",
                "depends_on": [LT_SIDEWALK_COL, RT_SIDEWALK_COL],
            }
        },
        relation_capabilities=["snap_match"],
        scope_capable=False,
        name_match_field=None,
        name_match_strip_suffixes=[],
        display_geometry_mode="native",
    ),
    "School": DatasetSpec(
        entity="School",
        table=SCHOOL_TABLE,
        geometry_column=SCHOOL_GEOM_COL,
        geometry_family="point",
        primary_key=SCHOOL_NAME_COL,
        label_field=SCHOOL_NAME_COL,
        display_fields=[SCHOOL_NAME_COL],
        fields={
            SCHOOL_NAME_COL: {"type": "text"},
        },
        relation_capabilities=[],
        scope_capable=False,
        name_match_field=SCHOOL_NAME_COL,
        name_match_strip_suffixes=[],
        display_geometry_mode="native",
    ),
    "BusStop": DatasetSpec(
        entity="BusStop",
        table=BUS_STOP_TABLE,
        geometry_column=BUS_STOP_GEOM_COL,
        geometry_family="point",
        primary_key=BUS_STOP_ID_COL,
        label_field=BUS_STOP_NAME_COL,
        display_fields=[BUS_STOP_ID_COL, BUS_STOP_NAME_COL],
        fields={
            BUS_STOP_ID_COL: {"type": "id"},
            BUS_STOP_NAME_COL: {"type": "text"},
        },
        relation_capabilities=[],
        scope_capable=False,
        name_match_field=BUS_STOP_NAME_COL,
        name_match_strip_suffixes=[],
        display_geometry_mode="native",
    ),
    "Crosswalk": DatasetSpec(
        entity="Crosswalk",
        table=CROSSWALK_TABLE,
        geometry_column=CROSSWALK_GEOM_COL,
        geometry_family="polygon",
        primary_key=CROSSWALK_ID_COL,
        label_field=CROSSWALK_ID_COL,
        display_fields=[CROSSWALK_ID_COL],
        fields={
            CROSSWALK_ID_COL: {"type": "id"},
        },
        relation_capabilities=[],
        scope_capable=False,
        name_match_field=None,
        name_match_strip_suffixes=[],
        display_geometry_mode="centroid_point",
    ),
    "Town": DatasetSpec(
        entity="Town",
        table=TOWN_TABLE,
        geometry_column=TOWN_GEOM_COL,
        geometry_family="polygon",
        primary_key=TOWN_NAME_COL,
        label_field=TOWN_NAME_COL,
        display_fields=[TOWN_NAME_COL],
        fields={
            TOWN_NAME_COL: {"type": "text"},
        },
        derived_fields={
            DERIVED_NAME_MATCH_KEY: {
                "type": "text",
                "depends_on": [TOWN_NAME_COL],
            }
        },
        relation_capabilities=[],
        scope_capable=True,
        name_match_field=TOWN_NAME_COL,
        name_match_strip_suffixes=list(TOWN_SUFFIX_VARIANTS),
        display_geometry_mode="native",
    ),
    "Place": DatasetSpec(
        entity="Place",
        table="",
        geometry_column="geom",
        geometry_family="point",
        primary_key=None,
        label_field="display_name",
        display_fields=["name", "display_name"],
        fields={
            "name": {"type": "text"},
            "display_name": {"type": "text"},
        },
        relation_capabilities=[],
        scope_capable=False,
        name_match_field="display_name",
        name_match_strip_suffixes=[],
        display_geometry_mode="native",
    ),
}


# =========================================================
# 4) GEMINI CLIENTS
# =========================================================
class GeminiClient:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.backend = None
        self.client = None
        self.model_obj = None

        try:
            from google import genai
            self.client = genai.Client(api_key=api_key)
            self.backend = "google-genai"
            return
        except Exception:
            pass

        try:
            import google.generativeai as genai_old
            genai_old.configure(api_key=api_key)
            self.model_obj = genai_old.GenerativeModel(model)
            self.backend = "google-generativeai"
            return
        except Exception as e:
            raise ImportError(
                "Could not import Gemini SDK.\n"
                "Install one of:\n"
                "  pip install -U google-genai\n"
                "or\n"
                "  pip install -U google-generativeai\n\n"
                f"Original error: {e}"
            )

    def generate_text(self, prompt: str, temperature: float = 0.0) -> str:
        if self.backend == "google-genai":
            resp = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={"temperature": temperature},
            )
            return (resp.text or "").strip()

        resp = self.model_obj.generate_content(
            prompt,
            generation_config={"temperature": temperature},
        )
        return (getattr(resp, "text", "") or "").strip()


# Default model identifiers — exposed for app.py convenience
OPENAI_MODEL = "gpt-4o"


class OpenAIClient:
    """
    Thin wrapper around the OpenAI chat completions API.
    Implements the same generate_text(prompt, temperature) interface as
    GeminiClient so it can replace it anywhere in the pipeline.
    """
    def __init__(self, api_key: str, model: str = OPENAI_MODEL):
        self.api_key = api_key
        self.model = model
        try:
            from openai import OpenAI as _OpenAI
            self._client = _OpenAI(api_key=api_key)
        except ImportError as exc:
            raise ImportError(
                "openai package not found. Install with:\n"
                "  pip install openai\n\n"
                f"Original error: {exc}"
            )

    def generate_text(self, prompt: str, temperature: float = 0.0) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return (response.choices[0].message.content or "").strip()


def make_llm_client(provider: str, api_key: str, model: str):
    """
    Factory: return the right LLM client based on provider string.
    provider: "openai" | "gemini"  (case-insensitive)
    Both returned objects expose generate_text(prompt, temperature) -> str.
    """
    if provider.lower() == "openai":
        return OpenAIClient(api_key=api_key, model=model)
    return GeminiClient(api_key=api_key, model=model)


class OptionalGeminiClient:
    """
    Lightweight optional wrapper for the response-layer LLM.
    Supports both Gemini and OpenAI via make_llm_client().
    If the key is missing, the client remains disabled and downstream code
    must fall back to deterministic summaries without raising.
    """
    def __init__(self, api_key: str, model: str, provider: str = "gemini"):
        self.api_key = (api_key or "").strip()
        self.model = model
        self.provider = provider.lower()
        self.enabled = bool(self.api_key)
        self._client = None
        self.init_error: Optional[str] = None

        if not self.enabled:
            return

        try:
            self._client = make_llm_client(self.provider, self.api_key, self.model)
        except Exception as e:
            self.enabled = False
            self.init_error = str(e)
            self._client = None

    def generate_text(self, prompt: str, temperature: float = 0.0) -> str:
        if not self.enabled or self._client is None:
            raise ValueError("Optional response LLM client is not enabled.")
        return self._client.generate_text(prompt, temperature=temperature)


# =========================================================
# 5) DB HELPERS
# =========================================================
def fetch_df(conn, sql: str, params: dict | None = None) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params or {})
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
    return pd.DataFrame(rows, columns=cols)


def fetch_scalar(conn, sql: str, params: dict | None = None):
    with conn.cursor() as cur:
        cur.execute(sql, params or {})
        row = cur.fetchone()
    return None if row is None else row[0]


def detect_road_geom_col(conn) -> str:
    sql = """
    SELECT f_geometry_column
    FROM public.geometry_columns
    WHERE f_table_schema = 'public'
      AND f_table_name = 'Road_Inventory_2025'
    LIMIT 1;
    """
    try:
        df = fetch_df(conn, sql)
        if not df.empty and df.iloc[0, 0]:
            return str(df.iloc[0, 0])
    except Exception:
        pass
    return ROAD_GEOM_COL_FALLBACK


# =========================================================
# 6) NORMALIZATION HELPERS
# =========================================================
def normalize_severity(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    s_low = s.lower()

    if s_low in {"all", "all crashes", "any", "any severity", "overall", "no filter", "none", "null"}:
        return None

    for v in CRASH_SEVE_CANONICAL:
        if s_low == v.lower():
            return v

    if "nonfatal" in s_low or "non-fatal" in s_low:
        return SEV_NONFATL
    if "pdo" in s_low or "property damage" in s_low or "damage only" in s_low:
        return SEV_PDO
    if "unknown" in s_low:
        return SEV_UNK
    if "fatal" in s_low:
        return SEV_FATL
    if "injury" in s_low:
        return SEV_NONFATL
    return s


def match_harm_levels(query: str) -> list[str]:
    """
    Fuzzy keyword matching: map natural language to canonical FIRST_HARM_LEVELS.
    Tokenizes query, keeps levels whose lowered text contains ALL query tokens.
    """
    if not query or not query.strip():
        return []
    q = query.strip().lower()
    q_tokens = q.split()
    for level in FIRST_HARM_LEVELS:
        if level.lower() == q:
            return [level]
    matches = []
    for level in FIRST_HARM_LEVELS:
        level_low = level.lower()
        if all(tok in level_low for tok in q_tokens):
            matches.append(level)
    return matches


def normalize_top_n(raw: object | None, default: int = DEFAULT_TOP_N) -> int:
    try:
        if raw is None:
            return int(default)
        n = int(raw)
        if n <= 0:
            return int(default)
        return min(n, MAX_TOP_N)
    except Exception:
        return int(default)


def normalize_order(raw: str | None) -> str:
    if raw is None:
        return "highest"
    s = str(raw).strip().lower()
    if s in {"lowest", "least", "bottom", "min", "minimum", "smallest", "fewest"}:
        return "lowest"
    return "highest"


def normalize_output_modes(raw) -> list[str]:
    if raw is None:
        return ["map", "summary"]
    if isinstance(raw, str):
        raw = [raw]
    out = []
    for x in raw:
        s = str(x).strip().lower()
        if s in SUPPORTED_OUTPUTS and s not in out:
            out.append(s)
    return out or ["map", "summary"]


def normalize_operator(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    mapping = {
        "=": "eq",
        "==": "eq",
        "eq": "eq",
        "in": "in",
        ">": "gt",
        "gt": "gt",
        ">=": "gte",
        "gte": "gte",
        "<": "lt",
        "lt": "lt",
        "<=": "lte",
        "lte": "lte",
        "between": "between",
        "is_null": "is_null",
        "null": "is_null",
        "not_null": "not_null",
        "not null": "not_null",
    }
    return mapping.get(s, s)


def normalize_entity_name(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    mapping = {
        "crash": "Crash",
        "crashes": "Crash",
        "road": "Road",
        "roads": "Road",
        "road segment": "Road",
        "road segments": "Road",
        "street": "Road",
        "streets": "Road",
        "school": "School",
        "schools": "School",
        "bus stop": "BusStop",
        "bus stops": "BusStop",
        "busstop": "BusStop",
        "busstops": "BusStop",
        "stop": "BusStop",
        "stops": "BusStop",
        "crosswalk": "Crosswalk",
        "crosswalks": "Crosswalk",
        "cross walk": "Crosswalk",
        "cross walks": "Crosswalk",
        "town": "Town",
        "towns": "Town",
        "city": "Town",
        "cities": "Town",
        "municipality": "Town",
        "municipalities": "Town",
        "place": "Place",
        "location": "Place",
        "poi": "Place",
    }
    return mapping.get(s, raw if raw in DATASET_REGISTRY else None)


def normalize_relation_name(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    mapping = {
        "within_distance": "within_distance",
        "within buffer": "within_distance",
        "buffer": "within_distance",
        "intersects": "intersects",
        "contains": "contains",
        "nearest_to": "nearest_to",
        "snap_match": "snap_match",
        "match_to": "snap_match",
    }
    return mapping.get(s, s)


def normalize_role_name(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    return s if s in SUPPORTED_ROLE_NAMES else None


def normalize_field_name(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    mapping = {
        "severity": CRASH_SEVE_COL,
        "crash severity": CRASH_SEVE_COL,
        "crash_severity": CRASH_SEVE_COL,
        CRASH_SEVE_COL.lower(): CRASH_SEVE_COL,

        "date": CRASH_DATE_COL,
        "crash date": CRASH_DATE_COL,
        "crash_date": CRASH_DATE_COL,
        DERIVED_CRASH_DATE_VALUE.lower(): DERIVED_CRASH_DATE_VALUE,

        "time": CRASH_TIME_COL,
        "crash time": CRASH_TIME_COL,
        "crash_time": CRASH_TIME_COL,
        DERIVED_CRASH_TIME_MINUTES.lower(): DERIVED_CRASH_TIME_MINUTES,

        "speed limit": SPEED_LIM_COL,
        "speed_lim": SPEED_LIM_COL,
        "op_dir_sl": OP_DIR_SPEED_LIM_COL,
        "opposite direction speed limit": OP_DIR_SPEED_LIM_COL,
        "left sidewalk": LT_SIDEWALK_COL,
        "right sidewalk": RT_SIDEWALK_COL,
        "lt_sidewlk": LT_SIDEWALK_COL,
        "rt_sidewlk": RT_SIDEWALK_COL,
        "sidewalk status": DERIVED_SIDEWALK_STATUS,
        "sidewalk_status": DERIVED_SIDEWALK_STATUS,

        "junction": RDWY_JNCT_COL,
        "junction type": RDWY_JNCT_COL,
        "roadway junction": RDWY_JNCT_COL,
        "rdwy_jnct_": RDWY_JNCT_COL,
        "intersection type": RDWY_JNCT_COL,
        "intersection": RDWY_JNCT_COL,

        "bus stop id": BUS_STOP_ID_COL,
        "stop id": BUS_STOP_ID_COL,
        BUS_STOP_ID_COL.lower(): BUS_STOP_ID_COL,
        "bus stop name": BUS_STOP_NAME_COL,
        "stop name": BUS_STOP_NAME_COL,
        BUS_STOP_NAME_COL.lower(): BUS_STOP_NAME_COL,

        "crosswalk id": CROSSWALK_ID_COL,
        CROSSWALK_ID_COL.lower(): CROSSWALK_ID_COL,

        "first harmful event": FIRST_HARM_COL,
        "first harmful": FIRST_HARM_COL,
        "first harm": FIRST_HARM_COL,
        "first_hrmf": FIRST_HARM_COL,
        "harm event": FIRST_HARM_COL,
        "harmful event": FIRST_HARM_COL,

        "town name": TOWN_NAME_COL,
        "city name": TOWN_NAME_COL,
        "municipality name": TOWN_NAME_COL,
        TOWN_NAME_COL.lower(): TOWN_NAME_COL,
        "name match key": DERIVED_NAME_MATCH_KEY,
        "name_match_key": DERIVED_NAME_MATCH_KEY,
    }
    return mapping.get(s, raw)


def normalize_names_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    out = []
    for x in raw:
        s = str(x).strip()
        if s and s not in out:
            out.append(s)
    return out


def classify_sidewalk_state(lt, rt) -> str:
    try:
        lt = 0 if pd.isna(lt) or str(lt).strip() == '' else float(lt)
    except (ValueError, TypeError):
        lt = 0
    try:
        rt = 0 if pd.isna(rt) or str(rt).strip() == '' else float(rt)
    except (ValueError, TypeError):
        rt = 0
    if lt > 0 and rt > 0:
        return "both"
    if lt == 0 and rt == 0:
        return "none"
    if lt > 0 and rt == 0:
        return "left_only"
    if lt == 0 and rt > 0:
        return "right_only"
    return "partial"


def point_4326_to_26986(lat: float, lon: float):
    g = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(26986)
    return g.iloc[0]


def geodf_from_wkt_df(
    df: pd.DataFrame,
    geom_col: str = "wkt",
    crs_from: int = 26986,
    crs_to: int = 4326,
) -> gpd.GeoDataFrame:
    if df.empty:
        empty = df.copy()
        return gpd.GeoDataFrame(empty, geometry=gpd.GeoSeries([], crs=f"EPSG:{crs_to}"), crs=f"EPSG:{crs_to}")
    gdf = gpd.GeoDataFrame(df.copy(), geometry=df[geom_col].apply(wkt.loads), crs=f"EPSG:{crs_from}")
    return gdf.to_crs(crs_to)


def finalize_roads_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    if df.empty:
        empty = df.copy()
        return gpd.GeoDataFrame(empty, geometry=gpd.GeoSeries([], crs="EPSG:4326"), crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(df.copy(), geometry=df["wkt"].apply(wkt.loads), crs="EPSG:26986")
    if LT_SIDEWALK_COL in gdf.columns and RT_SIDEWALK_COL in gdf.columns:
        gdf[DERIVED_SIDEWALK_STATUS] = gdf.apply(
            lambda r: classify_sidewalk_state(r.get(LT_SIDEWALK_COL), r.get(RT_SIDEWALK_COL)),
            axis=1,
        )
    return gdf.to_crs(4326)


def finalize_centroid_display_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    if df.empty:
        empty = df.copy()
        return gpd.GeoDataFrame(empty, geometry=gpd.GeoSeries([], crs="EPSG:4326"), crs="EPSG:4326")
    gdf_poly = gpd.GeoDataFrame(df.copy(), geometry=df["wkt"].apply(wkt.loads), crs="EPSG:26986")
    gdf_poly["geometry"] = gdf_poly.geometry.centroid
    return gdf_poly.to_crs(4326)


def get_default_render_limit(entity: str) -> int:
    if entity == "Road":
        return MAX_RENDER_ROADS
    if entity == "Crash":
        return MAX_RENDER_CRASHES
    if entity == "School":
        return MAX_RENDER_SCHOOLS
    if entity == "BusStop":
        return MAX_RENDER_BUS_STOPS
    if entity == "Crosswalk":
        return MAX_RENDER_CROSSWALKS
    if entity == "Town":
        return MAX_RENDER_TOWNS
    return 50000


def get_dataset_key_field(ds: DatasetSpec) -> Optional[str]:
    return ds.primary_key


def get_dataset_label_field(ds: DatasetSpec) -> Optional[str]:
    return ds.label_field or ds.primary_key


def _clean_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _strip_punctuation_edges(text: str) -> str:
    return re.sub(r"^[\s,;:.\-]+|[\s,;:.\-]+$", "", str(text or "").strip())


def normalize_text_key(text: str) -> str:
    s = str(text or "").lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"_", " ", s)
    s = _clean_space(s)
    return s


def normalize_town_base_name(text: str, suffixes: Optional[list[str]] = None) -> str:
    """
    Normalize a user or database Town name down to a base comparison key.

    Examples:
      'Amherst' -> 'amherst'
      'Amherst town' -> 'amherst'
      'Quincy city' -> 'quincy'
      '  Quincy, city  ' -> 'quincy'
    """
    suffixes = suffixes or list(TOWN_SUFFIX_VARIANTS)
    s = normalize_text_key(text)
    if not s:
        return s

    suffixes_norm = [
        normalize_text_key(x) for x in suffixes
        if str(x).strip()
    ]
    suffixes_norm = sorted(set([x for x in suffixes_norm if x]), key=len, reverse=True)

    changed = True
    while changed and s:
        changed = False
        for suf in suffixes_norm:
            if s == suf:
                s = ""
                changed = True
                break
            if s.endswith(" " + suf):
                s = _clean_space(s[: -(len(suf) + 1)])
                changed = True
                break

    return _clean_space(s)


def strip_suffix_variants(text: str, suffixes: list[str]) -> str:
    return normalize_town_base_name(text, suffixes=suffixes)


def build_name_match_tokens(name: str, suffixes: list[str]) -> dict[str, str]:
    full_norm = normalize_text_key(name)
    core_norm = normalize_town_base_name(name, suffixes=suffixes)
    if not core_norm:
        core_norm = full_norm
    return {
        "raw": str(name),
        "full_norm": full_norm,
        "core_norm": core_norm,
    }


def normalize_name_match_inputs(names: list[str], suffixes: list[str]) -> tuple[list[str], list[str]]:
    full_norms = []
    core_norms = []
    for name in normalize_names_list(names):
        tokens = build_name_match_tokens(name, suffixes=suffixes)
        if tokens["full_norm"] and tokens["full_norm"] not in full_norms:
            full_norms.append(tokens["full_norm"])
        if tokens["core_norm"] and tokens["core_norm"] not in core_norms:
            core_norms.append(tokens["core_norm"])
    return full_norms, core_norms


def normalize_town_names_debug_payload(names: list[str], suffixes: list[str]) -> list[dict[str, str]]:
    out = []
    for name in normalize_names_list(names):
        out.append(build_name_match_tokens(name, suffixes=suffixes))
    return out


def parse_date_text(value: Any) -> date:
    if value is None:
        raise ValueError("Date value is missing.")
    s = _clean_space(str(value))
    s = s.replace(",", " ")
    s = s.replace("/", " ")
    s = _clean_space(s)

    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", s):
        return datetime.strptime(s, "%Y-%m-%d").date()

    if re.fullmatch(r"\d{1,2}\s+\d{1,2}\s+\d{4}", s):
        return datetime.strptime(s, "%m %d %Y").date()

    parts = s.lower().split()
    if len(parts) == 3 and parts[0] in MONTH_NAME_TO_NUM:
        mm = MONTH_NAME_TO_NUM[parts[0]]
        dd = int(parts[1])
        yyyy = int(parts[2])
        return date(yyyy, mm, dd)

    for fmt in ("%B %d %Y", "%b %d %Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass

    raise ValueError(f"Could not parse date value: {value}")


def normalize_stored_date_string(value: str) -> str:
    d = parse_date_text(value)
    return d.strftime("%m %d %Y")


def normalize_iso_date_string(value: str) -> str:
    d = parse_date_text(value)
    return d.strftime("%Y-%m-%d")


def parse_time_text_to_minutes(value: Any) -> int:
    if value is None:
        raise ValueError("Time value is missing.")

    s = _clean_space(str(value)).lower()
    s = s.replace(".", "")
    s = s.replace("a m", "am").replace("p m", "pm")
    s = s.replace("am ", "am").replace("pm ", "pm")
    s = s.replace(" am", "am").replace(" pm", "pm")
    s = _clean_space(s)

    if re.fullmatch(r"\d{1,2}", s):
        h = int(s)
        if not (0 <= h <= 23):
            raise ValueError(f"Hour out of range: {value}")
        return h * 60

    if re.fullmatch(r"\d{1,2}:\d{2}", s):
        h, m = s.split(":")
        h = int(h)
        m = int(m)
        if not (0 <= h <= 23 and 0 <= m <= 59):
            raise ValueError(f"Time out of range: {value}")
        return h * 60 + m

    if re.fullmatch(r"\d{1,2}(am|pm)", s):
        dt = datetime.strptime(s, "%I%p")
        return dt.hour * 60 + dt.minute

    if re.fullmatch(r"\d{1,2}:\d{2}(am|pm)", s):
        dt = datetime.strptime(s, "%I:%M%p")
        return dt.hour * 60 + dt.minute

    raise ValueError(f"Could not parse time value: {value}")


def minutes_to_label(minutes: int) -> str:
    minutes = int(minutes)
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"


def normalize_date_constraint_value(operator: str, value: Any) -> tuple[str, Any]:
    op = normalize_operator(operator)
    if op == "eq":
        return DERIVED_CRASH_DATE_VALUE, normalize_iso_date_string(value)
    if op == "in":
        vals = value if isinstance(value, list) else [value]
        return DERIVED_CRASH_DATE_VALUE, [normalize_iso_date_string(v) for v in vals]
    if op == "between":
        if isinstance(value, dict):
            v1 = value.get("min")
            v2 = value.get("max")
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            v1, v2 = value[0], value[1]
        else:
            raise ValueError("Date between requires two values.")
        d1 = parse_date_text(v1)
        d2 = parse_date_text(v2)
        if d1 > d2:
            d1, d2 = d2, d1
        return DERIVED_CRASH_DATE_VALUE, [d1.strftime("%Y-%m-%d"), d2.strftime("%Y-%m-%d")]
    return CRASH_DATE_COL, value


def normalize_time_constraint_value(operator: str, value: Any) -> tuple[str, Any]:
    op = normalize_operator(operator)
    if op == "eq":
        return DERIVED_CRASH_TIME_MINUTES, int(parse_time_text_to_minutes(value))
    if op == "in":
        vals = value if isinstance(value, list) else [value]
        return DERIVED_CRASH_TIME_MINUTES, [int(parse_time_text_to_minutes(v)) for v in vals]
    if op == "between":
        if isinstance(value, dict):
            v1 = value.get("min")
            v2 = value.get("max")
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            v1, v2 = value[0], value[1]
        else:
            raise ValueError("Time between requires two values.")
        t1 = int(parse_time_text_to_minutes(v1))
        t2 = int(parse_time_text_to_minutes(v2))
        if t1 > t2:
            t1, t2 = t2, t1
        return DERIVED_CRASH_TIME_MINUTES, [t1, t2]
    if op in {"gt", "gte", "lt", "lte"}:
        return DERIVED_CRASH_TIME_MINUTES, int(parse_time_text_to_minutes(value))
    return CRASH_TIME_COL, value


def normalize_field_value_by_name(field: str, value: Any) -> Any:
    if field == CRASH_SEVE_COL:
        if isinstance(value, list):
            return [normalize_severity(v) for v in value]
        return normalize_severity(value)
    if field == FIRST_HARM_COL:
        return value
    return value
def _looks_like_date_text(text: str) -> bool:
    s = str(text or "").strip().lower()
    if any(month in s for month in MONTH_NAME_TO_NUM):
        return True
    if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b", s):
        return True
    if re.search(r"\b\d{1,2}\s+\d{1,2}\s+\d{4}\b", s):
        return True
    if re.search(r"\b\d{4}-\d{1,2}-\d{1,2}\b", s):
        return True
    return False


def _looks_like_time_text(text: str) -> bool:
    s = str(text or "").strip().lower()
    if re.fullmatch(r"\d{1,2}", s):
        return True
    if re.fullmatch(r"\d{1,2}:\d{2}", s):
        return True
    if re.fullmatch(r"\d{1,2}\s*(am|pm)", s):
        return True
    if re.fullmatch(r"\d{1,2}:\d{2}\s*(am|pm)", s):
        return True
    return False


def _detect_entity_role(role_map: dict[str, str], entity: str, preferred_role: Optional[str] = None) -> Optional[str]:
    if preferred_role in {"primary", "support", "scope"} and role_map.get(preferred_role) == entity:
        return preferred_role
    if role_map.get("primary") == entity:
        return "primary"
    if role_map.get("support") == entity:
        return "support"
    if role_map.get("scope") == entity:
        return "scope"
    return preferred_role


def _detect_crash_role(role_map: dict[str, str], preferred_role: Optional[str] = None) -> Optional[str]:
    return _detect_entity_role(role_map, "Crash", preferred_role)


def _extract_date_ranges_from_prompt(user_prompt: str) -> list[list[str]]:
    text = _clean_space(user_prompt)

    patterns = [
        r"\bbetween\s+([A-Za-z]+\s+\d{1,2}\s+\d{4})\s+and\s+([A-Za-z]+\s+\d{1,2}\s+\d{4})\b",
        r"\bfrom\s+([A-Za-z]+\s+\d{1,2}\s+\d{4})\s+to\s+([A-Za-z]+\s+\d{1,2}\s+\d{4})\b",
        r"\bbetween\s+(\d{1,2}[ /-]\d{1,2}[ /-]\d{4})\s+and\s+(\d{1,2}[ /-]\d{1,2}[ /-]\d{4})\b",
        r"\bfrom\s+(\d{1,2}[ /-]\d{1,2}[ /-]\d{4})\s+to\s+(\d{1,2}[ /-]\d{1,2}[ /-]\d{4})\b",
        r"\bbetween\s+(\d{4}-\d{1,2}-\d{1,2})\s+and\s+(\d{4}-\d{1,2}-\d{1,2})\b",
        r"\bfrom\s+(\d{4}-\d{1,2}-\d{1,2})\s+to\s+(\d{4}-\d{1,2}-\d{1,2})\b",
    ]

    found = []
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            a = _strip_punctuation_edges(m.group(1))
            b = _strip_punctuation_edges(m.group(2))
            try:
                d1 = parse_date_text(a)
                d2 = parse_date_text(b)
                if d1 > d2:
                    d1, d2 = d2, d1
                found.append([d1.strftime("%Y-%m-%d"), d2.strftime("%Y-%m-%d")])
            except Exception:
                pass

    dedup = []
    seen = set()
    for pair in found:
        tup = tuple(pair)
        if tup not in seen:
            dedup.append(pair)
            seen.add(tup)
    return dedup


def _extract_time_ranges_from_prompt(user_prompt: str) -> list[list[int]]:
    text = _clean_space(user_prompt)

    time_token = r"(\d{1,2}(?::\d{2})?\s*(?:am|pm)?|\d{1,2})"
    patterns = [
        rf"\bbetween\s+{time_token}\s+and\s+{time_token}\b",
        rf"\bfrom\s+{time_token}\s+to\s+{time_token}\b",
    ]

    found = []
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            a = _strip_punctuation_edges(m.group(1))
            b = _strip_punctuation_edges(m.group(2))
            try:
                if not (_looks_like_time_text(a) and _looks_like_time_text(b)):
                    continue
                t1 = int(parse_time_text_to_minutes(a))
                t2 = int(parse_time_text_to_minutes(b))
                if t1 > t2:
                    t1, t2 = t2, t1
                found.append([t1, t2])
            except Exception:
                pass

    dedup = []
    seen = set()
    for pair in found:
        tup = tuple(pair)
        if tup not in seen:
            dedup.append(pair)
            seen.add(tup)
    return dedup


def _has_constraint_for_field(attribute_constraints: list[AttributeConstraint], fields: set[str]) -> bool:
    for ac in attribute_constraints:
        if ac.field in fields:
            return True
    return False


def _get_role_map(targets: list[TargetSpec]) -> dict[str, str]:
    return {t.role: t.entity for t in targets if t.role and t.entity}


def _find_target(targets: list[TargetSpec], role: str) -> Optional[TargetSpec]:
    return next((t for t in targets if t.role == role), None)


def _ensure_target_entity(targets: list[TargetSpec], role: str, entity: str, names: Optional[list[str]] = None):
    t = _find_target(targets, role)
    if t is None:
        targets.append(TargetSpec(entity=entity, role=role, names=normalize_names_list(names)))
    else:
        t.entity = entity
        if names:
            for n in normalize_names_list(names):
                if n not in t.names:
                    t.names.append(n)


def _ensure_support_entity(targets: list[TargetSpec], entity: str):
    t = _find_target(targets, "support")
    if t is None:
        targets.append(TargetSpec(entity=entity, role="support"))
    elif t.entity != entity:
        t.entity = entity


def _ensure_scope_entity(targets: list[TargetSpec], entity: str, names: Optional[list[str]] = None):
    t = _find_target(targets, "scope")
    if t is None:
        targets.append(TargetSpec(entity=entity, role="scope", names=normalize_names_list(names)))
    else:
        t.entity = entity
        if names:
            for n in normalize_names_list(names):
                if n not in t.names:
                    t.names.append(n)


def _ensure_spatial_constraint(
    spatial_constraints: list[SpatialConstraint],
    relation: str,
    target_role: str,
    reference_role: str,
    distance_m: Optional[float],
):
    for sc in spatial_constraints:
        if sc.relation == relation and sc.target_role == target_role and sc.reference_role == reference_role:
            if sc.distance_m is None and distance_m is not None:
                sc.distance_m = distance_m
            return
    spatial_constraints.append(
        SpatialConstraint(
            relation=relation,
            target_role=target_role,
            reference_role=reference_role,
            distance_m=distance_m,
        )
    )


def _normalize_temporal_attribute_constraints(
    attribute_constraints: list[AttributeConstraint],
    role_map: dict[str, str],
    user_prompt: str,
) -> list[AttributeConstraint]:
    out: list[AttributeConstraint] = []

    for ac in attribute_constraints:
        field = normalize_field_name(ac.field)
        role = _detect_crash_role(role_map, ac.target_role) if field in CRASH_SPECIFIC_FIELDS else ac.target_role
        op = normalize_operator(ac.operator)
        val = ac.value

        if field in {CRASH_DATE_COL, DERIVED_CRASH_DATE_VALUE}:
            try:
                norm_field, norm_val = normalize_date_constraint_value(op, val)
                out.append(AttributeConstraint(target_role=role, field=norm_field, operator=op, value=norm_val))
            except Exception:
                out.append(AttributeConstraint(target_role=role, field=field, operator=op, value=val))
            continue

        if field in {CRASH_TIME_COL, DERIVED_CRASH_TIME_MINUTES}:
            try:
                norm_field, norm_val = normalize_time_constraint_value(op, val)
                out.append(AttributeConstraint(target_role=role, field=norm_field, operator=op, value=norm_val))
            except Exception:
                out.append(AttributeConstraint(target_role=role, field=field, operator=op, value=val))
            continue

        out.append(AttributeConstraint(target_role=role, field=field, operator=op, value=val))

    crash_role = _detect_crash_role(role_map, None)
    if crash_role is not None:
        if not _has_constraint_for_field(out, {DERIVED_CRASH_DATE_VALUE, CRASH_DATE_COL}):
            for rng in _extract_date_ranges_from_prompt(user_prompt):
                out.append(
                    AttributeConstraint(
                        target_role=crash_role,
                        field=DERIVED_CRASH_DATE_VALUE,
                        operator="between",
                        value=rng,
                    )
                )

        if not _has_constraint_for_field(out, {DERIVED_CRASH_TIME_MINUTES, CRASH_TIME_COL}):
            for rng in _extract_time_ranges_from_prompt(user_prompt):
                out.append(
                    AttributeConstraint(
                        target_role=crash_role,
                        field=DERIVED_CRASH_TIME_MINUTES,
                        operator="between",
                        value=rng,
                    )
                )

    return out


# =========================================================
# 7) RESOLVER FUNCTIONS
# =========================================================
def make_reference_row_from_point(role: str, anchor_name: str, lat: float, lon: float, display_name: str) -> pd.DataFrame:
    geom_26986 = point_4326_to_26986(lat, lon)
    return pd.DataFrame(
        {
            "role": [role],
            "name": [anchor_name],
            "display_name": [display_name],
            "reference_type": ["place"],
            "dist_m": [0.0],
            "wkt": [geom_26986.wkt],
        }
    )


def find_nearest_school(conn, lat: float, lon: float) -> pd.DataFrame:
    sql = f"""
    SELECT
      s."{SCHOOL_NAME_COL}" AS name,
      s."{SCHOOL_NAME_COL}" AS display_name,
      'school' AS reference_type,
      ST_AsText(s."{SCHOOL_GEOM_COL}") AS wkt,
      ST_Distance(
        s."{SCHOOL_GEOM_COL}",
        ST_Transform(ST_SetSRID(ST_Point(%(lon)s,%(lat)s),4326),26986)
      ) AS dist_m
    FROM {SCHOOL_TABLE} s
    ORDER BY s."{SCHOOL_GEOM_COL}" <-> ST_Transform(ST_SetSRID(ST_Point(%(lon)s,%(lat)s),4326),26986)
    LIMIT 1;
    """
    df = fetch_df(conn, sql, {"lat": lat, "lon": lon})
    if df.empty:
        raise ValueError(f'No school feature found in {SCHOOL_TABLE}.')
    return df


def geocode_place_nominatim(place_name: str, selection_index: Optional[int] = None):
    query = f"{place_name}, {STATE_SUFFIX}"
    params = {"q": query, "format": "json", "limit": GEOCODE_LIMIT, "addressdetails": 1}
    headers = {"User-Agent": NOMINATIM_UA}
    resp = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=20)
    resp.raise_for_status()
    results = resp.json()

    if not results:
        raise ValueError(f"No geocoding results for: {query}")

    if len(results) > 1 and selection_index is None:
        # Raise ambiguity error so the UI can present options
        options = [
            {"display_name": r.get("display_name", ""), "lat": float(r["lat"]), "lon": float(r["lon"])}
            for r in results
        ]
        raise AmbiguousLocationError(place_name, options)

    if selection_index is not None and 0 <= selection_index < len(results):
        best = results[selection_index]
    else:
        best = results[0]

    lat = float(best["lat"])
    lon = float(best["lon"])
    display_name = best.get("display_name", "")
    return lat, lon, display_name


def resolve_reference(conn, reference: ReferenceSpec, geocode_selection: Optional[int] = None) -> ResolverResult:
    if reference.entity not in SUPPORTED_REFERENCE_ENTITIES:
        raise ValueError(f"Unsupported reference entity for resolver: {reference.entity}")

    if not reference.name:
        raise ValueError("Reference name is missing.")

    lat, lon, display_name = geocode_place_nominatim(reference.name, selection_index=geocode_selection)

    if reference.entity == "School":
        school_df = find_nearest_school(conn, lat=lat, lon=lon)
        return ResolverResult(
            entity="School",
            role=str(reference.role),
            name=str(school_df["name"].iloc[0]),
            display_name=str(school_df["display_name"].iloc[0]),
            lat=lat,
            lon=lon,
            dist_m=float(school_df["dist_m"].iloc[0]),
            wkt_26986=str(school_df["wkt"].iloc[0]),
        )

    geom_26986 = point_4326_to_26986(lat, lon)
    return ResolverResult(
        entity="Place",
        role=str(reference.role),
        name=str(reference.name),
        display_name=str(display_name),
        lat=lat,
        lon=lon,
        dist_m=0.0,
        wkt_26986=geom_26986.wkt,
    )


# =========================================================
# 8) SEMANTIC FRAME EXTRACTION PROMPT AND PARSER
# =========================================================
def build_semantic_frame_prompt(road_geom_col: str) -> str:
    return f"""
You are a semantic planner for a Road Safety GIS assistant.
Return JSON only.
Do not return SQL.
Do not return Python.
Do not return execution steps.
Do not return operations lists.
Do not return intent families.

Database entities and fields:
- Crash: [{CRASH_ID_COL}, {CRASH_GEOM_COL}, {CRASH_SEVE_COL}, {CRASH_DATE_COL}, {CRASH_TIME_COL}, {FIRST_HARM_COL}, {LT_SIDEWALK_COL}, {RT_SIDEWALK_COL}, {SPEED_LIM_COL}, {RDWY_JNCT_COL}]
- Crash derived filter fields: [{DERIVED_CRASH_DATE_VALUE}, {DERIVED_CRASH_TIME_MINUTES}, {DERIVED_SIDEWALK_STATUS}]
- Crash has sidewalk and speed limit columns from merged road inventory. These fields can be filtered directly on Crash without needing a Road entity or snap_match.
- {RDWY_JNCT_COL} is a categorical field for junction type. Values include: "Driveway", "Four-way intersection", "T-intersection", "Not at junction", "Traffic circle", "Y-intersection", "Railway grade crossing", "Off-ramp", "On-ramp", "Five-point or more", "Not reported", "Unknown". Use operator "eq" for single value or "in" for multiple.
- Crash derived filter fields for semantics: [{DERIVED_CRASH_DATE_VALUE}, {DERIVED_CRASH_TIME_MINUTES}]
- Road: [{road_geom_col}, {SPEED_LIM_COL}, {OP_DIR_SPEED_LIM_COL}, {LT_SIDEWALK_COL}, {RT_SIDEWALK_COL}]
- Road derived filter fields for semantics: [{DERIVED_SIDEWALK_STATUS}]
- School: [{SCHOOL_NAME_COL}, {SCHOOL_GEOM_COL}]
- BusStop: [{BUS_STOP_ID_COL}, {BUS_STOP_NAME_COL}, {BUS_STOP_GEOM_COL}]
- Crosswalk: [{CROSSWALK_ID_COL}, {CROSSWALK_GEOM_COL}]
- Town: [{TOWN_NAME_COL}, {TOWN_GEOM_COL}]
- Place: external geocoded place reference

CRS is EPSG:26986 in meters.

Supported semantic frame schema:
{{
  "supported": true,
  "targets": [
    {{
      "entity": "Crash" | "Road" | "School" | "BusStop" | "Crosswalk" | "Town",
      "role": "primary" | "support" | "scope",
      "names": [string]
    }}
  ],
  "references": [
    {{"entity": "Place" | "School", "role": "anchor", "name": string}}
  ],
  "spatial_constraints": [
    {{
      "relation": "within_distance" | "intersects" | "contains" | "nearest_to",
      "target_role": "primary" | "support" | "scope",
      "reference_role": "anchor" | "primary" | "support" | "scope",
      "distance_m": number | null
    }}
  ],
  "attribute_constraints": [
    {{
      "target_role": "primary" | "support" | "scope",
      "field": string,
      "operator": "eq" | "in" | "gt" | "gte" | "lt" | "lte" | "between" | "is_null" | "not_null",
      "value": any
    }}
  ],
  "relations": [
    {{
      "relation": "snap_match",
      "source_role": "primary" | "support",
      "target_role": "primary" | "support",
      "distance_m": number | null
    }}
  ],
  "ranking": {{
    "metric": "crash_count",
    "target_role": "primary" | "support",
    "order": "highest" | "lowest",
    "top_n": integer
  }} | null,
  "outputs": ["map","summary","table"],
  "notes": string | null
}}

Role meanings:
- primary = main entity returned or ranked
- support = secondary entity used for matching / aggregation / relation logic
- anchor = external geocoded reference
- scope = polygon boundary restricting the analysis area

Rules:
- The LLM must express meaning only, not execution.
- Dataset entities already represent sets by default.
- Scope is a polygon boundary role. Prefer Town as scope when the user names towns/cities such as Quincy, Lenox, Amherst, Hadley, Northampton.
- For named towns/cities, put them in targets with role="scope", entity="Town", and names=[...].
- The scope role should not replace the main query meaning. It only restricts the analysis area.
- When the user says "show towns", primary should be Town.
- When the user says "show Quincy", prefer primary Town with names ["Quincy"].
- When the user says "show Quincy city", prefer primary Town with names ["Quincy city"].
- When the user says "show crashes in Quincy", use Crash primary plus Town scope with names ["Quincy"].
- When the user says "show roads in Lenox", use Road primary plus Town scope with names ["Lenox"].
- When the user says "show schools in Quincy city", use School primary plus Town scope with names ["Quincy city"].
- When the user says "show bus stops in Lenox town", use BusStop primary plus Town scope with names ["Lenox town"].
- When the user says "show crosswalks in Quincy", use Crosswalk primary plus Town scope with names ["Quincy"].
- When the user says "show crashes in Quincy and Lenox", use Crash primary plus Town scope with names ["Quincy", "Lenox"].
- When the user says "show roads in Amherst, Hadley, and Northampton", use Road primary plus Town scope with names ["Amherst", "Hadley", "Northampton"].
- When a scope exists, do not also model it as an anchor unless the user clearly asked for a geocoded place.
- For "show crashes", target is Crash primary, no references, no relations.
- For "show roads", target is Road primary, no references, no relations.
- For "show schools", target is School primary, no references, no relations.
- For "show bus stops", target is BusStop primary, no references, no relations.
- For "show crosswalks", target is Crosswalk primary, no references, no relations.
- For "show towns", target is Town primary, no references, no relations.
- For "show roads with no sidewalk statewide", use Road primary plus attribute constraint on field sidewalk_status eq "none".
- For "show sidewalk presence on roads around Amherst CVS", use Road primary, Place anchor, within_distance, and note that sidewalk_status may be relevant in notes.
- For "show roads with speed limits above 30 around Amherst CVS", use Road primary, Place anchor, within_distance, and attribute constraint speed_lim gt 30.
- For "show crashes on roads with speed limits higher than 30 around Amherst CVS", use Crash primary, Place anchor, within_distance, and attribute constraint speed_lim gt 30 on primary. No Road entity needed — crash data has speed_lim from merged road inventory.
- For "show crashes with speed limit above 40 in Quincy", use Crash primary, Town scope, attribute constraint speed_lim gt 40 on primary. No Road entity needed.
- For filtering crashes by speed limit, sidewalk conditions, junction type, severity, first harmful event, or temporal range, always use attribute constraints directly on the Crash role. No Road entity or snap_match is needed for any of these.
- For "show crashes near crosswalks", use Crash primary, Crosswalk support, and spatial constraint with target_role primary and reference_role support.
- For "show roads intersecting crosswalks", use Road primary, Crosswalk support, and spatial constraint with target_role primary and reference_role support relation intersects.
- For "show schools near crosswalks in Quincy", use School primary, Crosswalk support, Town scope ["Quincy"], and spatial constraint with target_role primary and reference_role support.
- For ranking queries like "top 20 schools by crashes within 200m", use School primary, Crash support, within_distance where support references primary, and ranking metric crash_count on primary.
- For ranking queries like "top 10 bus stops by crashes within 500m", use BusStop primary, Crash support, within_distance where support references primary, and ranking metric crash_count on primary.
- For ranking queries like "top 10 road segments by crashes in Amherst", use Road primary, Crash support, Town scope ["Amherst"], within_distance where support references primary, and ranking metric crash_count on primary. Road ranking uses a tight snap distance (50m) to associate crashes with nearby road segments.
- For ranking queries like "top 10 road segments by pedestrian crashes in Quincy", use Road primary, Crash support, Town scope ["Quincy"], within_distance where support references primary, attribute constraint first_hrmf eq "pedestrian" on support, and ranking crash_count on primary.
- For ranking queries like "top 10 road segments by crashes within 1km of Amherst Center", use Road primary, Crash support, Place anchor "Amherst Center" with 1000m buffer, within_distance where support references primary, and ranking crash_count on primary.
- For "road segments" or "roads" in ranking context, always use Road as primary entity with Crash as support. The system will snap crashes to nearby road segments for counting.
- For ranking queries like "top 20 towns by crashes", use Town primary, Crash support, intersects where support references primary, and ranking metric crash_count on primary. Town uses intersects (not within_distance) because crashes are inside town polygons.
- For ranking queries like "top 20 towns by fatal crashes", use Town primary, Crash support, intersects where support references primary, attribute constraint crash_seve eq "Fatal injury" on support, and ranking crash_count on primary.
- For ranking queries like "top 20 towns by crashes involving pedestrian", use Town primary, Crash support, intersects where support references primary, attribute constraint first_hrmf eq "pedestrian" on support, and ranking crash_count on primary.
- For ranking queries with scope, keep the ranking structure and add Town scope.
- For ranking queries that combine town ranking with infrastructure proximity, use a filter role. The filter entity spatially pre-filters the crash support before town-level aggregation. It is not displayed on the map.
- For "top 10 towns by crashes within 500m of schools", use Town primary, Crash support, School filter. Spatial constraints: support intersects primary (crash inside town), support within_distance filter at 500m (crash near school). Ranking crash_count on primary.
- For "top 10 towns by crashes near bus stops", use Town primary, Crash support, BusStop filter. Spatial constraints: support intersects primary, support within_distance filter at 200m. Ranking crash_count on primary.
- The filter role can pre-filter crashes by proximity to infrastructure for ranking. For "top 10 schools by crashes near crosswalks within 500m", use School primary, Crash support, Crosswalk filter. Spatial constraints: support within_distance primary at 500m, support within_distance filter at 200m. Ranking crash_count on primary.
- For sidewalk-based crash filtering in rankings, use sidewalk_status directly on Crash support — no filter role needed. Example: "top 10 schools by crashes without sidewalks within 500m" = School primary, Crash support with sidewalk_status eq "none", within_distance 500m.
- Example: "top 10 schools by crashes within 500m in Quincy" means School primary, Crash support, Town scope ["Quincy"], within_distance with support relative to primary, and ranking crash_count on primary.
- For non-ranking set queries like "show crashes within 500m of all schools", use Crash primary, School support, and spatial constraint with target_role primary and reference_role support.
- For non-ranking set queries like "show crashes within 500m of all bus stops", use Crash primary, BusStop support, and spatial constraint with target_role primary and reference_role support.
- For non-ranking set queries like "show crashes within 500m of all crosswalks", use Crash primary, Crosswalk support, and spatial constraint with target_role primary and reference_role support.
- For non-ranking set queries like "show roads without sidewalks within 500m of all schools", use Road primary, School support, primary sidewalk filter, and spatial constraint with primary relative to support.
- Crosswalk is a normal infrastructure entity, not a scope layer.
- Crosswalk geometry remains polygon for analysis.
- Crosswalk may be primary or support, but not scope.
- For crash temporal filters, use crash-specific fields on the crash role:
  - severity: crash_seve
  - date: crash_date or crash_date_value
  - time: crash_time or crash_time_minutes
- For date ranges like "between January 6 2025 and February 5 2025", prefer operator "between".
- For time-of-day ranges like "between 12 and 14", "between 5am and 4 pm", or "between 7:30 AM and 9:00 AM", prefer operator "between".
- For ranking queries with crash filters, put crash filters on support if support is Crash.
- Convert miles/km/feet to meters.
- Default distance may be null if omitted; Python will repair it.
- For severity filters use field crash_seve.
- For first harmful event filters use field first_hrmf with operator "eq" or "in". Pass natural language values (the system will resolve them). Example: "pedestrian" or "animal" or "motor vehicle".
- For null-style sidewalk queries use derived field sidewalk_status.
- Crash data includes sidewalk columns (lt_sidewlk, rt_sidewlk) from merged road inventory. For crash sidewalk filtering like "show crashes on roads without sidewalks" or "show crashes with no sidewalk", use sidewalk_status eq "none" directly on the Crash role. No Road entity or snap_match is needed for sidewalk filtering on crashes.
- For "show crashes on roads without sidewalks in Quincy", use Crash primary, Town scope, attribute constraint sidewalk_status eq "none" on primary. Do NOT add Road entity.
- For "show crashes with no sidewalk around Amherst Center within 1km", use Crash primary, Place anchor, attribute constraint sidewalk_status eq "none" on primary.
- For ranking like "top 10 schools by crashes without sidewalks within 500m", use School primary, Crash support with sidewalk_status eq "none" on support. No Road entity needed.
- For "top 10 towns by crashes without sidewalks", use Town primary, Crash support with sidewalk_status eq "none" on support, intersects. No Road entity needed.
- Road entity is still needed when the query is specifically about roads: "show roads without sidewalks", "show sidewalk presence on roads", "show roads with speed limit above 30".
- IMPORTANT DISAMBIGUATION: When the user says "show roads with..." or "show roads without...", they want to see ROAD segments, so use Road primary. When they say "show crashes on roads with..." or "show crashes without...", they want CRASH points filtered by road attributes, so use Crash primary with attribute filters. The word "roads" alone does NOT mean Road entity — only "show roads" or "roads in [town]" means Road primary.
- Examples: "show roads without sidewalks in Amherst" = Road primary. "show crashes without sidewalks in Amherst" = Crash primary. "show roads with speed limit above 30" = Road primary. "show crashes with speed limit above 30" = Crash primary.
- IMPORTANT: when the query mentions "crashes on roads without sidewalks" or "crashes with no sidewalk" or "crashes on no-sidewalk roads", do NOT use Road entity, do NOT use snap_match. Just filter Crash by sidewalk_status eq "none". The phrase "on roads" in a crash+sidewalk context means "at locations without sidewalks", not "join with Road table".
- snap_match with Road is generally NOT needed for crash filtering. Crash data has speed_lim, lt_sidewlk, rt_sidewlk, and rdwy_jnct_ columns from merged road inventory. Use these directly as Crash attribute filters.
- Road entity with snap_match is ONLY needed when the user asks to see the actual Road segments (e.g., "show roads with speed limits above 30"), NOT when filtering crashes by road attributes.
- IMPORTANT: "road segments" or "roads" in a RANKING context (e.g., "top 10 road segments by crashes") means Road is the PRIMARY entity ranked by crash count. This is different from "show roads" (display) or crash attribute filtering. For road ranking: Road primary, Crash support, within_distance at 50m.
- Do NOT use snap_match for road ranking queries. Use within_distance with 50m distance between Crash (support) and Road (primary).
- If unsupported return supported=false and minimal valid JSON.

Examples:

User: show roads with speed limits above 30 around Amherst CVS
JSON:
{{
  "supported": true,
  "targets": [
    {{"entity": "Road", "role": "primary", "names": []}}
  ],
  "references": [
    {{"entity": "Place", "role": "anchor", "name": "Amherst CVS"}}
  ],
  "spatial_constraints": [
    {{
      "relation": "within_distance",
      "target_role": "primary",
      "reference_role": "anchor",
      "distance_m": 200
    }}
  ],
  "attribute_constraints": [
    {{
      "target_role": "primary",
      "field": "speed_lim",
      "operator": "gt",
      "value": 30
    }}
  ],
  "relations": [],
  "ranking": null,
  "outputs": ["map","summary"],
  "notes": null
}}

User: show crashes in Quincy
JSON:
{{
  "supported": true,
  "targets": [
    {{"entity": "Crash", "role": "primary", "names": []}},
    {{"entity": "Town", "role": "scope", "names": ["Quincy"]}}
  ],
  "references": [],
  "spatial_constraints": [],
  "attribute_constraints": [],
  "relations": [],
  "ranking": null,
  "outputs": ["map","summary"],
  "notes": null
}}

User: show crosswalks around Amherst Center within 500m
JSON:
{{
  "supported": true,
  "targets": [
    {{"entity": "Crosswalk", "role": "primary", "names": []}}
  ],
  "references": [
    {{"entity": "Place", "role": "anchor", "name": "Amherst Center"}}
  ],
  "spatial_constraints": [
    {{
      "relation": "within_distance",
      "target_role": "primary",
      "reference_role": "anchor",
      "distance_m": 500
    }}
  ],
  "attribute_constraints": [],
  "relations": [],
  "ranking": null,
  "outputs": ["map","summary"],
  "notes": null
}}

User: show roads intersecting crosswalks
JSON:
{{
  "supported": true,
  "targets": [
    {{"entity": "Road", "role": "primary", "names": []}},
    {{"entity": "Crosswalk", "role": "support", "names": []}}
  ],
  "references": [],
  "spatial_constraints": [
    {{
      "relation": "intersects",
      "target_role": "primary",
      "reference_role": "support",
      "distance_m": null
    }}
  ],
  "attribute_constraints": [],
  "relations": [],
  "ranking": null,
  "outputs": ["map","summary"],
  "notes": null
}}

User: top 10 schools by crashes within 500m in Quincy city
JSON:
{{
  "supported": true,
  "targets": [
    {{"entity": "School", "role": "primary", "names": []}},
    {{"entity": "Crash", "role": "support", "names": []}},
    {{"entity": "Town", "role": "scope", "names": ["Quincy city"]}}
  ],
  "references": [],
  "spatial_constraints": [
    {{
      "relation": "within_distance",
      "target_role": "support",
      "reference_role": "primary",
      "distance_m": 500
    }}
  ],
  "attribute_constraints": [],
  "relations": [],
  "ranking": {{
    "metric": "crash_count",
    "target_role": "primary",
    "order": "highest",
    "top_n": 10
  }},
  "outputs": ["map","summary","table"],
  "notes": null
}}

User: top 10 road segments by crashes in Amherst
JSON:
{{
  "supported": true,
  "targets": [
    {{"entity": "Road", "role": "primary", "names": []}},
    {{"entity": "Crash", "role": "support", "names": []}},
    {{"entity": "Town", "role": "scope", "names": ["Amherst"]}}
  ],
  "references": [],
  "spatial_constraints": [
    {{
      "relation": "within_distance",
      "target_role": "support",
      "reference_role": "primary",
      "distance_m": 50
    }}
  ],
  "attribute_constraints": [],
  "relations": [],
  "ranking": {{
    "metric": "crash_count",
    "target_role": "primary",
    "order": "highest",
    "top_n": 10
  }},
  "outputs": ["map","summary","table"],
  "notes": null
}}

User: top 20 towns by crashes
JSON:
{{
  "supported": true,
  "targets": [
    {{"entity": "Town", "role": "primary", "names": []}},
    {{"entity": "Crash", "role": "support", "names": []}}
  ],
  "references": [],
  "spatial_constraints": [
    {{
      "relation": "intersects",
      "target_role": "support",
      "reference_role": "primary",
      "distance_m": null
    }}
  ],
  "attribute_constraints": [],
  "relations": [],
  "ranking": {{
    "metric": "crash_count",
    "target_role": "primary",
    "order": "highest",
    "top_n": 20
  }},
  "outputs": ["map","summary","table"],
  "notes": null
}}

User: top 10 towns by fatal crashes
JSON:
{{
  "supported": true,
  "targets": [
    {{"entity": "Town", "role": "primary", "names": []}},
    {{"entity": "Crash", "role": "support", "names": []}}
  ],
  "references": [],
  "spatial_constraints": [
    {{
      "relation": "intersects",
      "target_role": "support",
      "reference_role": "primary",
      "distance_m": null
    }}
  ],
  "attribute_constraints": [
    {{
      "target_role": "support",
      "field": "crash_seve",
      "operator": "eq",
      "value": "Fatal injury"
    }}
  ],
  "relations": [],
  "ranking": {{
    "metric": "crash_count",
    "target_role": "primary",
    "order": "highest",
    "top_n": 10
  }},
  "outputs": ["map","summary","table"],
  "notes": null
}}

User: top 10 towns by crashes within 500m of schools
JSON:
{{
  "supported": true,
  "targets": [
    {{"entity": "Town", "role": "primary", "names": []}},
    {{"entity": "Crash", "role": "support", "names": []}},
    {{"entity": "School", "role": "filter", "names": []}}
  ],
  "references": [],
  "spatial_constraints": [
    {{
      "relation": "intersects",
      "target_role": "support",
      "reference_role": "primary",
      "distance_m": null
    }},
    {{
      "relation": "within_distance",
      "target_role": "support",
      "reference_role": "filter",
      "distance_m": 500
    }}
  ],
  "attribute_constraints": [],
  "relations": [],
  "ranking": {{
    "metric": "crash_count",
    "target_role": "primary",
    "order": "highest",
    "top_n": 10
  }},
  "outputs": ["map","summary","table"],
  "notes": null
}}

User prompt:
"""


def extract_semantic_frame(llm: GeminiClient, user_prompt: str, road_geom_col: str) -> dict:
    prompt = build_semantic_frame_prompt(road_geom_col) + user_prompt
    text = llm.generate_text(prompt, temperature=0.0)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise ValueError(f"LLM did not return valid JSON. Got:\n{text}")
        return json.loads(m.group(0))


# =========================================================
# 9) SEMANTIC FRAME VALIDATION AND REPAIR
# =========================================================
def _extract_scope_names_from_prompt(prompt: str) -> list[str]:
    text = _clean_space(prompt)
    found: list[str] = []

    patterns = [
        r"\bin\s+([A-Za-z][A-Za-z\s,]*(?:\band\b\s*[A-Za-z][A-Za-z\s,]*)?)$",
        r"\bwithin\s+([A-Za-z][A-Za-z\s,]*(?:\band\b\s*[A-Za-z][A-Za-z\s,]*)?)$",
    ]

    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            chunk = _strip_punctuation_edges(m.group(1))
            chunk = re.sub(
                r"\bwithin\s+\d+(?:\.\d+)?\s*(m|meter|meters|km|kilometer|kilometers|ft|feet|mile|miles)\b",
                "",
                chunk,
                flags=re.IGNORECASE,
            )
            chunk = _clean_space(chunk)
            if chunk:
                parts = re.split(r"\s*,\s*|\s+\band\b\s+|\s+\&\s+", chunk, flags=re.IGNORECASE)
                for p in parts:
                    name = _strip_punctuation_edges(p)
                    if name and name.lower() not in {
                        "all schools",
                        "all bus stops",
                        "all stops",
                        "all roads",
                        "all crashes",
                        "all crosswalks",
                    }:
                        if name not in found:
                            found.append(name)
    return found


def _repair_targets(raw_targets, prompt_low: str, user_prompt: str) -> list[TargetSpec]:
    out = []
    if isinstance(raw_targets, list):
        for x in raw_targets:
            if not isinstance(x, dict):
                continue
            entity = normalize_entity_name(x.get("entity"))
            role = normalize_role_name(x.get("role"))
            names = normalize_names_list(x.get("names"))
            if entity in SUPPORTED_TARGET_ENTITIES and role in {"primary", "support", "scope", "filter"}:
                if role == "scope" and entity not in SUPPORTED_SCOPE_ENTITIES:
                    continue
                out.append(TargetSpec(entity=entity, role=role, names=names))

    roles_present = {t.role for t in out}
    if "primary" not in roles_present:
        ranking_words = ["top", "highest", "lowest", "most", "least", "fewest"]
        if any(k in prompt_low for k in ["bus stop", "bus stops", "busstop", "busstops", "stop", "stops"]) and any(k in prompt_low for k in ranking_words):
            out.insert(0, TargetSpec(entity="BusStop", role="primary"))
        elif "school" in prompt_low and any(k in prompt_low for k in ranking_words):
            out.insert(0, TargetSpec(entity="School", role="primary"))
        elif any(k in prompt_low for k in ["crosswalk", "crosswalks", "cross walk", "cross walks"]) and any(k in prompt_low for k in ranking_words):
            out.insert(0, TargetSpec(entity="Crosswalk", role="primary"))
        elif "show towns" in prompt_low or "show town" in prompt_low or "show cities" in prompt_low or "show city" in prompt_low:
            out.insert(0, TargetSpec(entity="Town", role="primary"))
        elif "crash" in prompt_low or "fatal" in prompt_low:
            out.insert(0, TargetSpec(entity="Crash", role="primary"))
        elif any(k in prompt_low for k in ["road", "roads", "street", "streets", "sidewalk"]):
            out.insert(0, TargetSpec(entity="Road", role="primary"))
        elif any(k in prompt_low for k in ["bus stop", "bus stops", "busstop", "busstops", "stop", "stops"]):
            out.insert(0, TargetSpec(entity="BusStop", role="primary"))
        elif any(k in prompt_low for k in ["crosswalk", "crosswalks", "cross walk", "cross walks"]):
            out.insert(0, TargetSpec(entity="Crosswalk", role="primary"))
        elif "school" in prompt_low:
            out.insert(0, TargetSpec(entity="School", role="primary"))

    if any(t.role == "primary" and t.entity == "Town" for t in out):
        primary_town = next((t for t in out if t.role == "primary" and t.entity == "Town"), None)
        if primary_town is not None and not primary_town.names:
            guessed_names = _extract_scope_names_from_prompt(user_prompt)
            if guessed_names:
                primary_town.names = guessed_names

    if not any(t.role == "scope" for t in out):
        guessed_scope_names = _extract_scope_names_from_prompt(user_prompt)
        if guessed_scope_names and not any(r in prompt_low for r in ["around ", "near "]):
            primary_is_plain_town_show = any(
                t.role == "primary" and t.entity == "Town" for t in out
            )
            if not primary_is_plain_town_show:
                out.append(TargetSpec(entity="Town", role="scope", names=guessed_scope_names))

    dedup = {}
    for t in out:
        if t.role not in dedup:
            dedup[t.role] = t
        else:
            if t.entity:
                dedup[t.role].entity = t.entity
            for n in t.names:
                if n not in dedup[t.role].names:
                    dedup[t.role].names.append(n)
    return list(dedup.values())


def _repair_references(raw_refs) -> list[ReferenceSpec]:
    out = []
    if isinstance(raw_refs, list):
        for x in raw_refs:
            if not isinstance(x, dict):
                continue
            entity = normalize_entity_name(x.get("entity"))
            role = normalize_role_name(x.get("role"))
            name = x.get("name")
            if entity in SUPPORTED_REFERENCE_ENTITIES and role == "anchor" and name:
                out.append(ReferenceSpec(entity=entity, role=role, name=str(name)))
    return out


def _repair_spatial_constraints(raw_spatial) -> list[SpatialConstraint]:
    out = []
    if isinstance(raw_spatial, list):
        for x in raw_spatial:
            if not isinstance(x, dict):
                continue
            relation = normalize_relation_name(x.get("relation"))
            target_role = normalize_role_name(x.get("target_role"))
            reference_role = normalize_role_name(x.get("reference_role"))
            distance_m = x.get("distance_m")
            try:
                distance_m = None if distance_m is None else float(distance_m)
            except Exception:
                distance_m = None
            if relation in SUPPORTED_SPATIAL_RELATIONS and target_role in {"primary", "support", "scope", "filter"} and reference_role in {"primary", "support", "anchor", "scope", "filter"}:
                out.append(
                    SpatialConstraint(
                        relation=relation,
                        target_role=target_role,
                        reference_role=reference_role,
                        distance_m=distance_m,
                    )
                )
    return out


def _repair_attribute_constraints(raw_attrs) -> list[AttributeConstraint]:
    out = []
    if isinstance(raw_attrs, list):
        for x in raw_attrs:
            if not isinstance(x, dict):
                continue
            role = normalize_role_name(x.get("target_role"))
            field = normalize_field_name(x.get("field"))
            operator = normalize_operator(x.get("operator"))
            value = normalize_field_value_by_name(field, x.get("value")) if field else x.get("value")
            if role in {"primary", "support", "scope", "filter"} and field and operator in SUPPORTED_ATTRIBUTE_OPERATORS:
                out.append(AttributeConstraint(target_role=role, field=str(field), operator=operator, value=value))
    return out


def _repair_relations(raw_relations) -> list[RelationConstraint]:
    out = []
    if isinstance(raw_relations, list):
        for x in raw_relations:
            if not isinstance(x, dict):
                continue
            relation = normalize_relation_name(x.get("relation"))
            source_role = normalize_role_name(x.get("source_role"))
            target_role = normalize_role_name(x.get("target_role"))
            distance_m = x.get("distance_m")
            try:
                distance_m = None if distance_m is None else float(distance_m)
            except Exception:
                distance_m = None
            if relation in SUPPORTED_RELATIONS and source_role in {"primary", "support"} and target_role in {"primary", "support"}:
                out.append(
                    RelationConstraint(
                        relation=relation,
                        source_role=source_role,
                        target_role=target_role,
                        distance_m=distance_m,
                    )
                )
    return out


def _repair_ranking(raw_ranking) -> Optional[RankingSpec]:
    if not isinstance(raw_ranking, dict):
        return None
    metric = raw_ranking.get("metric")
    target_role = normalize_role_name(raw_ranking.get("target_role")) or "primary"
    order = normalize_order(raw_ranking.get("order"))
    top_n = normalize_top_n(raw_ranking.get("top_n"))
    if metric in SUPPORTED_RANK_METRICS and target_role in {"primary", "support"}:
        return RankingSpec(metric=metric, target_role=target_role, order=order, top_n=top_n)
    return None


def _repair_ranking_dependencies(
    targets: list[TargetSpec],
    spatial_constraints: list[SpatialConstraint],
    ranking: Optional[RankingSpec],
):
    if ranking is None:
        return

    role_map = _get_role_map(targets)
    ranking_role = ranking.target_role if ranking.target_role in {"primary", "support"} else "primary"
    ranking_entity = role_map.get(ranking_role)

    if ranking_entity is None and role_map.get("primary") is not None:
        ranking.target_role = "primary"
        ranking_role = "primary"
        ranking_entity = role_map.get("primary")

    if ranking.metric == "crash_count":
        if ranking_entity == "Crash":
            return

        crash_role = _detect_crash_role(role_map, preferred_role="support")
        if crash_role is None:
            if ranking_role == "primary":
                _ensure_support_entity(targets, "Crash")
            else:
                if _find_target(targets, "primary") is None:
                    _ensure_target_entity(targets, "primary", "Crash")
                else:
                    _ensure_support_entity(targets, "Crash")

        role_map = _get_role_map(targets)
        crash_role = _detect_crash_role(role_map, preferred_role="support")
        if crash_role is not None and crash_role != ranking_role:
            # Town uses intersects (crash inside polygon), others use within_distance
            if ranking_entity == "Town":
                _ensure_spatial_constraint(
                    spatial_constraints=spatial_constraints,
                    relation="intersects",
                    target_role=crash_role,
                    reference_role=ranking_role,
                    distance_m=None,
                )
            elif ranking_entity == "Road":
                _ensure_spatial_constraint(
                    spatial_constraints=spatial_constraints,
                    relation="within_distance",
                    target_role=crash_role,
                    reference_role=ranking_role,
                    distance_m=ROAD_RANKING_SNAP_M,
                )
            else:
                _ensure_spatial_constraint(
                    spatial_constraints=spatial_constraints,
                    relation="within_distance",
                    target_role=crash_role,
                    reference_role=ranking_role,
                    distance_m=DEFAULT_RADIUS_M,
                )


def _ensure_filter_entity(targets: list[TargetSpec], entity: str):
    """Ensure a filter role exists with the given entity."""
    t = _find_target(targets, "filter")
    if t is None:
        targets.append(TargetSpec(entity=entity, role="filter"))
    elif t.entity != entity:
        t.entity = entity


def _repair_filter_dependencies(
    targets: list[TargetSpec],
    spatial_constraints: list[SpatialConstraint],
    ranking: Optional[RankingSpec],
    prompt_low: str,
):
    """
    Detect when a ranking query mentions an infrastructure entity that should
    spatially pre-filter the crash support before aggregation.

    Patterns detected:
      - "top N towns by crashes near/within Xm of schools" → filter=School
      - "top N towns by crashes near bus stops" → filter=BusStop
      - "top N towns by crashes near crosswalks" → filter=Crosswalk
      - "top N schools by crashes on roads without sidewalks" → filter=Road

    The filter role pre-narrows Crash through a spatial join before the
    main aggregation. It is never materialized or displayed.

    Also fixes the common LLM mistake where it puts the infrastructure entity
    as support (dropping Crash entirely).
    """
    if ranking is None or ranking.metric != "crash_count":
        return

    role_map = _get_role_map(targets)
    ranking_entity = role_map.get(ranking.target_role or "primary")
    if ranking_entity is None:
        return

    # Don't apply if ranking entity itself is Crash
    if ranking_entity == "Crash":
        return

    # Detect infrastructure entity in prompt that should be a filter
    filter_entity = None
    filter_relation = "within_distance"
    filter_distance = DEFAULT_RADIUS_M

    if ranking_entity == "Town":
        # For town ranking: look for school/busstop/crosswalk mentions
        if any(k in prompt_low for k in ["school", "schools"]):
            filter_entity = "School"
        elif any(k in prompt_low for k in ["bus stop", "bus stops", "busstop", "busstops"]):
            filter_entity = "BusStop"
        elif any(k in prompt_low for k in ["crosswalk", "crosswalks"]):
            filter_entity = "Crosswalk"
    elif ranking_entity in {"School", "BusStop", "Crosswalk"}:
        # For school/busstop ranking: look for "on roads" pattern
        # Only create Road filter for speed limit queries, NOT sidewalk
        # (crashes have sidewalk data directly — no Road join needed)
        speed_kw = ["speed", "speed limit", "speed_lim"]
        if any(k in prompt_low for k in ["on road", "on roads"]) and any(k in prompt_low for k in speed_kw):
            filter_entity = "Road"
            filter_relation = "snap_match"
            filter_distance = SNAP_CRASH_TO_ROAD_M

    if filter_entity is None:
        return

    # Don't create filter if the entity is already the ranking entity
    if filter_entity == ranking_entity:
        return

    # Extract distance from prompt if specified
    import re as _re
    dist_patterns = [
        r'(\d+)\s*(?:m|meters?)\s+(?:of|from|around|near)\s+(?:school|bus\s*stop|crosswalk)',
        r'within\s+(\d+)\s*(?:m|meters?)\s+of\s+(?:school|bus\s*stop|crosswalk)',
    ]
    for pat in dist_patterns:
        dist_match = _re.search(pat, prompt_low)
        if dist_match:
            filter_distance = float(dist_match.group(1))
            break

    # Case 1: LLM wrongly put the filter entity as support (dropping Crash)
    if role_map.get("support") == filter_entity:
        crash_role = _detect_crash_role(role_map, None)
        if crash_role is None:
            # Crash is missing — move the wrong support to filter, add Crash as support
            support_t = _find_target(targets, "support")
            if support_t is not None:
                support_t.role = "filter"
            _ensure_support_entity(targets, "Crash")

            # Clean up stale spatial constraints (e.g., support↔support)
            spatial_constraints[:] = [
                sc for sc in spatial_constraints
                if not (sc.target_role == sc.reference_role)
            ]

    # Case 2: Filter entity not in frame at all — add it
    role_map = _get_role_map(targets)
    if role_map.get("filter") != filter_entity:
        if filter_entity not in [role_map.get(r) for r in ("primary", "support", "scope")]:
            _ensure_filter_entity(targets, filter_entity)

    # Ensure Crash is support
    role_map = _get_role_map(targets)
    if _detect_crash_role(role_map, "support") is None:
        _ensure_support_entity(targets, "Crash")

    # Ensure spatial constraint: support(Crash) ↔ filter(infrastructure)
    role_map = _get_role_map(targets)
    if role_map.get("filter") == filter_entity:
        _ensure_spatial_constraint(
            spatial_constraints=spatial_constraints,
            relation=filter_relation,
            target_role="support",
            reference_role="filter",
            distance_m=filter_distance,
        )

    # Ensure spatial constraint: support(Crash) ↔ primary(ranking entity) for aggregation
    if ranking_entity == "Town":
        _ensure_spatial_constraint(
            spatial_constraints=spatial_constraints,
            relation="intersects",
            target_role="support",
            reference_role="primary",
            distance_m=None,
        )


def _repair_attribute_role_dependencies(
    targets: list[TargetSpec],
    attribute_constraints: list[AttributeConstraint],
    prompt_low: str,
):
    role_map = _get_role_map(targets)

    if any(ac.field in CRASH_SPECIFIC_FIELDS for ac in attribute_constraints):
        crash_role = _detect_crash_role(role_map, None)
        if crash_role is None:
            if "primary" not in role_map:
                _ensure_target_entity(targets, "primary", "Crash")
            else:
                if role_map.get("primary") in {"School", "BusStop", "Crosswalk"}:
                    _ensure_support_entity(targets, "Crash")
                else:
                    _ensure_target_entity(targets, "primary", "Crash")

    role_map = _get_role_map(targets)

    if any(ac.target_role == "primary" and ac.field in ROAD_ONLY_FIELDS for ac in attribute_constraints):
        if role_map.get("primary") is None:
            _ensure_target_entity(targets, "primary", "Road")
        elif role_map.get("primary") != "Crash" or "crash" not in prompt_low:
            _ensure_target_entity(targets, "primary", "Road")

    if any(ac.target_role == "support" and ac.field in ROAD_ONLY_FIELDS for ac in attribute_constraints):
        _ensure_support_entity(targets, "Road")


def _repair_relation_dependencies(targets: list[TargetSpec], relations: list[RelationConstraint]):
    if any(r.relation == "snap_match" for r in relations):
        if _find_target(targets, "primary") is None:
            _ensure_target_entity(targets, "primary", "Crash")
        if _find_target(targets, "support") is None:
            _ensure_support_entity(targets, "Road")


def _repair_scope_dependencies(targets: list[TargetSpec], prompt_low: str):
    scope_t = _find_target(targets, "scope")
    if scope_t is not None and scope_t.entity is None:
        scope_t.entity = "Town"

    if scope_t is not None and scope_t.entity == "Town" and not scope_t.names:
        guessed = _extract_scope_names_from_prompt(prompt_low)
        if guessed:
            scope_t.names = guessed

    primary_t = _find_target(targets, "primary")
    if primary_t is not None and primary_t.entity == "Town" and not primary_t.names:
        if any(k in prompt_low for k in ["show lenox", "show quincy", "show amherst", "show hadley", "show northampton"]):
            guessed = _extract_scope_names_from_prompt(prompt_low)
            if guessed:
                primary_t.names = guessed


def _repair_prompt_sidewalk_defaults(
    targets: list[TargetSpec],
    attribute_constraints: list[AttributeConstraint],
    prompt_low: str,
):
    role_map = _get_role_map(targets)
    sidewalk_keywords = ["no sidewalk", "no sidewalks", "without sidewalk", "without sidewalks"]

    if any(k in prompt_low for k in sidewalk_keywords):
        primary_entity = role_map.get("primary")

        # If primary is Road, add sidewalk filter on primary (existing behavior)
        if primary_entity == "Road":
            if not any(ac.target_role == "primary" and ac.field == DERIVED_SIDEWALK_STATUS for ac in attribute_constraints):
                attribute_constraints.append(
                    AttributeConstraint(
                        target_role="primary",
                        field=DERIVED_SIDEWALK_STATUS,
                        operator="eq",
                        value="none",
                    )
                )

        # If primary is Crash, add sidewalk filter directly on Crash
        # (crash data has lt_sidewlk/rt_sidewlk columns — no Road join needed)
        elif primary_entity == "Crash":
            crash_role = "primary"
            if not any(ac.target_role == crash_role and ac.field == DERIVED_SIDEWALK_STATUS for ac in attribute_constraints):
                attribute_constraints.append(
                    AttributeConstraint(
                        target_role=crash_role,
                        field=DERIVED_SIDEWALK_STATUS,
                        operator="eq",
                        value="none",
                    )
                )

        # If Crash is support (e.g., ranking), add sidewalk filter on support
        elif _detect_crash_role(role_map, None) == "support":
            if not any(ac.target_role == "support" and ac.field == DERIVED_SIDEWALK_STATUS for ac in attribute_constraints):
                attribute_constraints.append(
                    AttributeConstraint(
                        target_role="support",
                        field=DERIVED_SIDEWALK_STATUS,
                        operator="eq",
                        value="none",
                    )
                )


def _repair_prompt_anchor_defaults(
    references: list[ReferenceSpec],
    spatial_constraints: list[SpatialConstraint],
    role_map: dict[str, str],
    user_prompt: str,
):
    prompt_low = user_prompt.lower()

    if role_map.get("scope") == "Town":
        return

    if not references and any(k in prompt_low for k in ["around ", "near ", "within ", "around"]) and role_map.get("primary") in SUPPORTED_TARGET_ENTITIES:
        place_match = re.search(r"(?:around|near)\s+(.+)$", user_prompt, flags=re.IGNORECASE)
        if place_match:
            place_name = place_match.group(1).strip()
            if place_name:
                references.append(ReferenceSpec(entity="Place", role="anchor", name=place_name))
                _ensure_spatial_constraint(
                    spatial_constraints=spatial_constraints,
                    relation="within_distance",
                    target_role="primary",
                    reference_role="anchor",
                    distance_m=DEFAULT_RADIUS_M,
                )

    if references and not any(sc.reference_role == "anchor" for sc in spatial_constraints):
        _ensure_spatial_constraint(
            spatial_constraints=spatial_constraints,
            relation="within_distance",
            target_role="primary",
            reference_role="anchor",
            distance_m=DEFAULT_RADIUS_M,
        )


def _repair_distance_defaults(
    spatial_constraints: list[SpatialConstraint],
    relations: list[RelationConstraint],
):
    repaired_spatial = []
    for sc in spatial_constraints:
        d = sc.distance_m
        if sc.relation in RELATION_DISTANCE_DEFAULTS and d is None:
            d = RELATION_DISTANCE_DEFAULTS[sc.relation]
        repaired_spatial.append(
            SpatialConstraint(
                relation=sc.relation,
                target_role=sc.target_role,
                reference_role=sc.reference_role,
                distance_m=d,
            )
        )

    repaired_relations = []
    for r in relations:
        d = r.distance_m
        if r.relation in RELATION_DISTANCE_DEFAULTS and d is None:
            d = RELATION_DISTANCE_DEFAULTS[r.relation]
        repaired_relations.append(
            RelationConstraint(
                relation=r.relation,
                source_role=r.source_role,
                target_role=r.target_role,
                distance_m=d,
            )
        )

    return repaired_spatial, repaired_relations


def _validate_repaired_semantics(
    supported: bool,
    targets: list[TargetSpec],
    attribute_constraints: list[AttributeConstraint],
    relations: list[RelationConstraint],
    ranking: Optional[RankingSpec],
) -> bool:
    role_map = _get_role_map(targets)
    valid = True

    if "primary" not in role_map:
        valid = False

    for t in targets:
        if t.entity not in SUPPORTED_TARGET_ENTITIES:
            valid = False
        if t.role not in {"primary", "support", "scope", "filter"}:
            valid = False
        if t.role == "scope" and t.entity not in SUPPORTED_SCOPE_ENTITIES:
            valid = False

    for ac in attribute_constraints:
        ent = role_map.get(ac.target_role)
        if ent is None:
            valid = False
            continue
        ds = DATASET_REGISTRY.get(ent)
        if ds is None:
            valid = False
            continue
        if ac.field not in ds.fields and ac.field not in ds.derived_fields:
            valid = False

    for r in relations:
        if r.relation not in SUPPORTED_RELATIONS:
            valid = False
        if r.source_role not in role_map or r.target_role not in role_map:
            valid = False
        if r.source_role == "scope" or r.target_role == "scope":
            valid = False

    if ranking is not None:
        rank_entity = role_map.get(ranking.target_role)
        if rank_entity is None:
            valid = False
        else:
            rank_ds = DATASET_REGISTRY.get(rank_entity)
            if rank_ds is None or get_dataset_key_field(rank_ds) is None:
                valid = False

    return bool(supported and valid)


def _resolve_harm_level_constraints(attribute_constraints: list[AttributeConstraint]):
    """
    For first_hrmf constraints with natural language values, resolve via match_harm_levels().
    1 match → use directly. Multiple matches → use "in" operator. 0 matches → keep raw.
    """
    for ac in attribute_constraints:
        if ac.field != FIRST_HARM_COL:
            continue
        if isinstance(ac.value, list):
            if all(v in FIRST_HARM_LEVELS for v in ac.value):
                if len(ac.value) > 1:
                    ac.operator = "in"
                continue
        raw_val = str(ac.value).strip() if ac.value else ""
        if not raw_val:
            continue
        if raw_val in FIRST_HARM_LEVELS:
            ac.value = raw_val
            ac.operator = "eq"
            continue
        matches = match_harm_levels(raw_val)
        if len(matches) == 1:
            ac.value = matches[0]
            ac.operator = "eq"
        elif len(matches) > 1:
            ac.value = matches
            ac.operator = "in"
        # else: 0 matches — keep raw value, SQL may return 0 rows


def validate_and_repair_semantic_frame(raw: dict, user_prompt: str) -> SemanticFrame:
    if not isinstance(raw, dict):
        raw = {}

    prompt_low = user_prompt.lower()

    targets = _repair_targets(raw.get("targets"), prompt_low, user_prompt)
    references = _repair_references(raw.get("references"))
    spatial_constraints = _repair_spatial_constraints(raw.get("spatial_constraints"))
    attribute_constraints = _repair_attribute_constraints(raw.get("attribute_constraints"))
    relations = _repair_relations(raw.get("relations"))
    ranking = _repair_ranking(raw.get("ranking"))
    outputs = normalize_output_modes(raw.get("outputs"))
    notes = raw.get("notes")
    supported = bool(raw.get("supported", False))

    _repair_ranking_dependencies(targets, spatial_constraints, ranking)
    _repair_filter_dependencies(targets, spatial_constraints, ranking, prompt_low)

    _repair_attribute_role_dependencies(targets, attribute_constraints, prompt_low)
    _repair_relation_dependencies(targets, relations)
    _repair_scope_dependencies(targets, prompt_low)

    role_map = _get_role_map(targets)
    attribute_constraints = _normalize_temporal_attribute_constraints(attribute_constraints, role_map, user_prompt)

    role_map = _get_role_map(targets)
    if any(ac.target_role == "primary" and ac.field == CRASH_SEVE_COL for ac in attribute_constraints):
        if role_map.get("primary") != "Crash":
            crash_role = _detect_crash_role(role_map, None)
            if crash_role is None:
                _ensure_target_entity(targets, "primary", "Crash")

    # Ensure Crash entity exists if first_hrmf constraint is present
    role_map = _get_role_map(targets)
    if any(ac.field == FIRST_HARM_COL for ac in attribute_constraints):
        crash_role = _detect_crash_role(role_map, None)
        if crash_role is None:
            _ensure_target_entity(targets, "primary", "Crash")

    # Resolve first_hrmf fuzzy values via match_harm_levels
    _resolve_harm_level_constraints(attribute_constraints)

    _repair_prompt_sidewalk_defaults(targets, attribute_constraints, prompt_low)

    if any(k in prompt_low for k in ["sidewalk presence", "sidewalk status", "sidewalk condition"]):
        notes = ((notes or "") + " sidewalk_visual").strip()

    role_map = _get_role_map(targets)
    _repair_prompt_anchor_defaults(references, spatial_constraints, role_map, user_prompt)

    # ── Bogus anchor cleanup (runs AFTER _repair_prompt_anchor_defaults) ──────
    # Catches two cases:
    #   1. LLM treats entity type names as geocodable places ("bus stops", "schools")
    #   2. LLM creates anchor for distance phrases ("500m of schools")
    # Runs after _repair_prompt_anchor_defaults so anchors added by that function
    # are also checked. Safe because legitimate place names (Amherst Center,
    # Palmer St @ Brockton Ave, Jacob Hiatt Magnet School) never match these patterns.
    _ENTITY_TYPE_EXACT = {
        "crash", "crashes", "road", "roads", "school", "schools",
        "busstop", "busstops", "bus stop", "bus stops", "bus_stop", "bus_stops",
        "crosswalk", "crosswalks", "town", "towns",
    }
    import re as _re_bogus
    bogus_refs = []
    for ref in references:
        if ref.role == "anchor" and ref.name:
            name_low = ref.name.strip().lower()
            # Exact entity type name match
            if name_low in _ENTITY_TYPE_EXACT:
                bogus_refs.append(ref)
                continue
            # Distance phrase match (e.g. "500m of schools", "1km around schools")
            if _re_bogus.search(r'\d+\s*(?:m|meter|meters|km|mile|miles)', name_low):
                bogus_refs.append(ref)
    for bogus in bogus_refs:
        references.remove(bogus)
        spatial_constraints[:] = [
            sc for sc in spatial_constraints
            if sc.reference_role != "anchor"
        ]

    spatial_constraints, relations = _repair_distance_defaults(spatial_constraints, relations)

    # ── Fix "show roads" queries where LLM wrongly used Crash primary ──
    # If the user said "show roads" / "roads with" / "roads without" and did NOT
    # mention crashes, the primary should be Road, not Crash.
    road_primary_kw = ["show roads", "show road", "show me roads", "show me road",
                       "roads with ", "roads without ", "roads in "]
    is_road_primary_intent = any(k in prompt_low for k in road_primary_kw)
    mentions_crash = "crash" in prompt_low or "fatal" in prompt_low
    if is_road_primary_intent and not mentions_crash:
        role_map = _get_role_map(targets)
        primary_t = _find_target(targets, "primary")
        if primary_t is not None and primary_t.entity == "Crash":
            primary_t.entity = "Road"
        elif primary_t is not None and primary_t.entity not in {"Road"}:
            # Don't override if it's School/Town/etc
            pass
        # Also re-target any attribute constraints from crash fields to Road
        role_map = _get_role_map(targets)
        if role_map.get("primary") == "Road":
            for ac in attribute_constraints:
                if ac.target_role == "primary" and ac.field in {DERIVED_SIDEWALK_STATUS, SPEED_LIM_COL, LT_SIDEWALK_COL, RT_SIDEWALK_COL}:
                    pass  # already on primary which is now Road — correct

    # ── FINAL CLEANUP: Crash filtering by merged road attributes ──
    # This runs LAST so no later repair can re-add Road or snap_match.
    # Rule: Crash data has speed_lim, lt_sidewlk, rt_sidewlk, rdwy_jnct_ columns
    # from merged road inventory. When the user is filtering crashes by any of these,
    # no Road entity or snap_match is needed.
    # We only keep Road if it's primary (user asked about roads themselves, e.g.,
    # "show roads without sidewalks", "show roads with speed limit above 30").

    road_attr_kw = [
        "no sidewalk", "no sidewalks", "without sidewalk", "without sidewalks",
        "no-sidewalk", "sidewalk_status", "left sidewalk", "right sidewalk",
        "speed limit", "speed above", "speed higher", "speed greater",
        "speed_lim", "junction", "intersection", "t-intersection",
        "four-way", "driveway", "rdwy_jnct",
    ]
    road_only_kw = ["show roads", "show road", "roads with", "roads without", "road with",
                    "road without", "sidewalk presence", "sidewalk condition",
                    "road segments", "road segment"]

    has_road_attr_intent = any(k in prompt_low for k in road_attr_kw)
    has_crash_intent = (
        "crash" in prompt_low or "fatal" in prompt_low
        or any(k in prompt_low for k in ["top ", "lowest ", "rank ", "most ", "fewest "])
    )
    is_road_only_query = any(k in prompt_low for k in road_only_kw) and not has_crash_intent

    if has_road_attr_intent and has_crash_intent and not is_road_only_query:
        # Strip ALL snap_match relations
        relations[:] = [r for r in relations if r.relation != "snap_match"]

        # Remove Road from support, filter roles (keep if primary)
        role_map = _get_role_map(targets)
        for remove_role in ["support", "filter"]:
            if role_map.get(remove_role) == "Road":
                targets[:] = [t for t in targets if not (t.role == remove_role and t.entity == "Road")]
                # Remove spatial constraints that reference the removed role
                spatial_constraints[:] = [
                    sc for sc in spatial_constraints
                    if not (sc.reference_role == remove_role)
                ]

        # Re-target any road-attribute constraints from removed roles to the crash role
        role_map = _get_role_map(targets)
        crash_role = _detect_crash_role(role_map, None)
        shared_fields = {DERIVED_SIDEWALK_STATUS, LT_SIDEWALK_COL, RT_SIDEWALK_COL, SPEED_LIM_COL, RDWY_JNCT_COL}
        if crash_role is not None:
            for ac in attribute_constraints:
                if ac.field in shared_fields:
                    if ac.target_role not in role_map:
                        ac.target_role = crash_role

            # Ensure sidewalk_status constraint if sidewalk keywords present
            sidewalk_kw = ["no sidewalk", "no sidewalks", "without sidewalk", "without sidewalks", "no-sidewalk"]
            if any(k in prompt_low for k in sidewalk_kw):
                if not any(ac.target_role == crash_role and ac.field == DERIVED_SIDEWALK_STATUS for ac in attribute_constraints):
                    attribute_constraints.append(
                        AttributeConstraint(
                            target_role=crash_role,
                            field=DERIVED_SIDEWALK_STATUS,
                            operator="eq",
                            value="none",
                        )
                    )

    supported = _validate_repaired_semantics(supported, targets, attribute_constraints, relations, ranking)

    # ── Deduplicate attribute constraints ───────────────────────────────────
    # Two passes:
    #
    # Pass A — cross-role sidewalk dedup:
    # When Road is primary and a sidewalk constraint exists on primary,
    # remove any sidewalk constraint on support (the LLM may have put it
    # on the wrong role and a repair moved it, leaving both).
    role_map_final = _get_role_map(targets)
    if role_map_final.get("primary") == "Road":
        has_primary_sidewalk = any(
            ac.target_role == "primary" and ac.field in {DERIVED_SIDEWALK_STATUS, LT_SIDEWALK_COL, RT_SIDEWALK_COL}
            for ac in attribute_constraints
        )
        if has_primary_sidewalk:
            attribute_constraints = [
                ac for ac in attribute_constraints
                if not (ac.target_role == "support" and ac.field in {DERIVED_SIDEWALK_STATUS, LT_SIDEWALK_COL, RT_SIDEWALK_COL})
            ]

    # Pass B — same role+field dedup:
    # Keep the last occurrence per (target_role, field) — repair passes always
    # append so the last entry reflects the most corrected value.
    seen_ac: dict[tuple, int] = {}
    for i, ac in enumerate(attribute_constraints):
        seen_ac[(ac.target_role, ac.field)] = i
    attribute_constraints = [attribute_constraints[i] for i in sorted(seen_ac.values())]

    return SemanticFrame(
        supported=supported,
        targets=targets,
        references=references,
        spatial_constraints=spatial_constraints,
        attribute_constraints=attribute_constraints,
        relations=relations,
        ranking=ranking,
        outputs=outputs,
        notes=notes,
    )


# =========================================================
# 10) COMPILER (DAG)
# =========================================================
# The compiler builds a typed directed acyclic graph rather than a flat list.
# Each node carries an explicit list of input node_ids representing data
# dependencies. The DAGBuilder helper tracks the "current head" for each role —
# i.e., the most recent node that wrote to that role's SQL state — and uses
# this to wire new nodes to the correct predecessors.
#
# The same call order from the legacy linear compiler is preserved (load →
# initialize → anchor spatial → scope → attribute → relation → match →
# aggregate → rank → materialize → outputs) because that order makes head
# tracking straightforward and exactly mirrors the dependency rules required
# by the executor's read/write contract over state.role_data.

class DAGBuilder:
    """
    Helper for accumulating typed nodes with auto-wired dependencies.

    Tracks the latest writer ("head") for each role so subsequent operations
    that read or write the same role get edges pointing to it. Also tracks
    auxiliary heads for shared resources like the dataset registry and
    reference objects.

    Each role that the executor mutates (state.role_data[role].sql_base) has
    its head advanced whenever a node modifies it. A node's `inputs` list is
    the union of (a) the prior head of every role it reads/writes, plus (b)
    auxiliary heads it depends on (registry, reference objects).
    """

    def __init__(self):
        self.nodes: dict[str, DAGNode] = {}
        self._role_heads: dict[str, str] = {}
        self._registry_head: Optional[str] = None
        self._role_loaded_head: dict[str, str] = {}  # LoadEntitySpec node per role
        self._reference_heads: dict[str, str] = {}    # role -> BuildReferenceObject node id
        self._match_heads: list[str] = []             # all match node ids in order
        self._aggregate_head: Optional[str] = None
        self._rank_head: Optional[str] = None
        self._materialize_heads: dict[str, str] = {}  # role -> MaterializeRole node id
        self._counter: dict[str, int] = {}            # for disambiguation when needed

    def _next_id(self, base: str) -> str:
        """Return a unique node id derived from `base`, suffixing on collision."""
        if base not in self.nodes:
            return base
        i = self._counter.get(base, 1)
        while f"{base}_{i}" in self.nodes:
            i += 1
        self._counter[base] = i + 1
        return f"{base}_{i}"

    def _add(self, node_id: str, op: str, params: dict, inputs: list[str]) -> str:
        nid = self._next_id(node_id)
        # Deduplicate inputs while preserving deterministic order
        seen: set[str] = set()
        deduped: list[str] = []
        for x in inputs:
            if x is not None and x not in seen:
                seen.add(x)
                deduped.append(x)
        self.nodes[nid] = DAGNode(node_id=nid, op=op, params=dict(params), inputs=deduped)
        return nid

    # --- registry / load ---
    def add_load_registry(self) -> str:
        nid = self._add("load_dataset_registry", "LoadDatasetRegistry", {}, [])
        self._registry_head = nid
        return nid

    def add_load_entity(self, role: str, entity: str, names: list[str]) -> str:
        assert self._registry_head is not None
        nid = self._add(
            f"load_entity_{role}",
            "LoadEntitySpec",
            {"role": role, "entity": entity, "names": list(names)},
            [self._registry_head],
        )
        self._role_loaded_head[role] = nid
        return nid

    def add_resolve_reference(self, role: str, entity: str, name: str, idx: int) -> str:
        # ResolveReference reads role_loaded_head for its role to know the entity
        deps = []
        if role in self._role_loaded_head:
            deps.append(self._role_loaded_head[role])
        elif self._registry_head is not None:
            deps.append(self._registry_head)
        nid = self._add(
            f"resolve_reference_{role}_{idx}",
            "ResolveReference",
            {"role": role, "entity": entity, "name": name},
            deps,
        )
        return nid

    def add_build_reference(self, role: str, resolve_node_id: str, idx: int) -> str:
        nid = self._add(
            f"build_reference_object_{role}_{idx}",
            "BuildReferenceObject",
            {"role": role},
            [resolve_node_id],
        )
        # Subsequent anchor spatial constraints look up the reference object by role
        self._reference_heads[role] = nid
        return nid

    # --- per-role pipeline ---
    def add_initialize_role(self, role: str) -> str:
        load_id = self._role_loaded_head.get(role)
        deps = [load_id] if load_id else []
        nid = self._add(f"init_role_{role}", "InitializeRoleQuery", {"role": role}, deps)
        self._role_heads[role] = nid
        return nid

    def add_name_filter(self, role: str, names: list[str]) -> str:
        prev = self._role_heads.get(role)
        deps = [prev] if prev else []
        nid = self._add(
            f"name_filter_{role}",
            "ApplyNameFilter",
            {"role": role, "names": list(names)},
            deps,
        )
        self._role_heads[role] = nid
        return nid

    def add_anchor_spatial(
        self,
        target_role: str,
        relation: str,
        reference_role: str,
        distance_m: Optional[float],
        anchor_idx: int,
    ) -> str:
        # Reads the target role's current SQL and the anchor reference object
        deps: list[str] = []
        if target_role in self._role_heads:
            deps.append(self._role_heads[target_role])
        # Anchor reference role — typically "anchor" — points to its build_reference node
        anchor_ref = self._reference_heads.get(reference_role)
        if anchor_ref is not None:
            deps.append(anchor_ref)
        nid = self._add(
            f"anchor_spatial_{target_role}_{anchor_idx}",
            "ApplySpatialConstraint",
            {
                "target_role": target_role,
                "relation": relation,
                "reference_role": reference_role,
                "distance_m": distance_m,
            },
            deps,
        )
        self._role_heads[target_role] = nid
        return nid

    def add_scope_constraint(self, target_role: str, scope_role: str, relation: str) -> str:
        # Reads the target role's current SQL and the scope role's current SQL
        deps: list[str] = []
        if target_role in self._role_heads:
            deps.append(self._role_heads[target_role])
        if scope_role in self._role_heads:
            deps.append(self._role_heads[scope_role])
        nid = self._add(
            f"scope_{target_role}",
            "ApplyScopeConstraint",
            {"target_role": target_role, "scope_role": scope_role, "relation": relation},
            deps,
        )
        self._role_heads[target_role] = nid
        return nid

    def add_attribute_constraints(self, target_role: str) -> str:
        prev = self._role_heads.get(target_role)
        deps = [prev] if prev else []
        nid = self._add(
            f"attribute_{target_role}",
            "ApplyAttributeConstraints",
            {"target_role": target_role},
            deps,
        )
        self._role_heads[target_role] = nid
        return nid

    def add_relation_constraint(
        self,
        relation: str,
        source_role: str,
        target_role: str,
        distance_m: Optional[float],
        rel_idx: int,
    ) -> str:
        # Relation constraint mutates BOTH source and target SQL
        deps: list[str] = []
        if source_role in self._role_heads:
            deps.append(self._role_heads[source_role])
        if target_role in self._role_heads:
            deps.append(self._role_heads[target_role])
        nid = self._add(
            f"relation_{rel_idx}_{source_role}_{target_role}",
            "ApplyRelationConstraint",
            {
                "relation": relation,
                "source_role": source_role,
                "target_role": target_role,
                "distance_m": distance_m,
            },
            deps,
        )
        # Relation writes both sides — advance both heads
        self._role_heads[source_role] = nid
        self._role_heads[target_role] = nid
        return nid

    def add_match_spatial_sets(
        self,
        left_role: str,
        right_role: str,
        relation: str,
        distance_m: Optional[float],
        match_idx: int,
        left_keep: bool = True,
        right_keep: bool = True,
    ) -> str:
        # MatchSpatialSets mutates both roles' SQL (when left_keep / right_keep)
        deps: list[str] = []
        if left_role in self._role_heads:
            deps.append(self._role_heads[left_role])
        if right_role in self._role_heads:
            deps.append(self._role_heads[right_role])
        nid = self._add(
            f"match_{match_idx}_{left_role}_{right_role}",
            "MatchSpatialSets",
            {
                "left_role": left_role,
                "right_role": right_role,
                "relation": relation,
                "distance_m": distance_m,
                "left_keep": left_keep,
                "right_keep": right_keep,
            },
            deps,
        )
        if left_keep:
            self._role_heads[left_role] = nid
        if right_keep:
            self._role_heads[right_role] = nid
        self._match_heads.append(nid)
        return nid

    def add_aggregate(self, params: dict) -> str:
        # Aggregate reads the latest SQL for both group_role and measure_role
        group_role = params.get("group_role")
        measure_role = params.get("measure_role")
        deps: list[str] = []
        if group_role and group_role in self._role_heads:
            deps.append(self._role_heads[group_role])
        if measure_role and measure_role in self._role_heads:
            deps.append(self._role_heads[measure_role])
        # Plus all match nodes already emitted, since aggregate semantics
        # rely on match-side narrowing happening first.
        for m in self._match_heads:
            if m not in deps:
                deps.append(m)
        nid = self._add("aggregate", "Aggregate", params, deps)
        self._aggregate_head = nid
        return nid

    def add_rank(self, params: dict) -> str:
        deps = [self._aggregate_head] if self._aggregate_head else []
        nid = self._add("rank", "Rank", params, deps)
        self._rank_head = nid
        return nid

    def add_materialize_role(self, role: str, ranking_role: Optional[str]) -> str:
        # Materialize depends on the role's latest head, plus rank if rank may
        # rewrite this role's SQL. Rank rewrites:
        #   - the ranking primary role (narrows candidates by rank result)
        #   - the filter role (narrows by ranked group polygons)
        # so materialize for those roles must wait for Rank to complete.
        deps: list[str] = []
        prev = self._role_heads.get(role)
        if prev:
            deps.append(prev)
        if self._rank_head is not None:
            if (ranking_role is not None and role == ranking_role) or role == "filter":
                deps.append(self._rank_head)
        nid = self._add(f"materialize_{role}", "MaterializeRole", {"role": role}, deps)
        self._materialize_heads[role] = nid
        return nid

    def add_output(self, op: str) -> str:
        # Output nodes depend on every materialized role
        deps = list(self._materialize_heads.values())
        # PrepareSummary additionally reads ranking outputs
        if op == "PrepareSummary" and self._rank_head is not None and self._rank_head not in deps:
            deps.append(self._rank_head)
        # PrepareMap reads aggregate output (for ranked-map consistency)
        if op == "PrepareMap":
            if self._aggregate_head is not None and self._aggregate_head not in deps:
                deps.append(self._aggregate_head)
            if self._rank_head is not None and self._rank_head not in deps:
                deps.append(self._rank_head)
        # Use lowercase op as id
        nid_base = {
            "PrepareTable": "prepare_table",
            "PrepareSummary": "prepare_summary",
            "PrepareMap": "prepare_map",
        }[op]
        return self._add(nid_base, op, {}, deps)

    def add_unsupported(self) -> str:
        return self._add("unsupported", "Unsupported", {}, [])


def topological_sort(nodes: dict[str, DAGNode]) -> list[str]:
    """
    Deterministic Kahn's-algorithm topological sort.

    Tiebreaker: lexicographic node_id order, so debug output is reproducible
    across runs even when multiple nodes are simultaneously eligible.

    Raises ValueError on cycle detection (any node remaining after processing).
    """
    # Build outgoing edges and in-degrees
    indeg: dict[str, int] = {nid: 0 for nid in nodes}
    out_edges: dict[str, list[str]] = {nid: [] for nid in nodes}
    for nid, node in nodes.items():
        for src in node.inputs:
            if src not in nodes:
                raise ValueError(
                    f"DAG node {nid!r} has unknown input {src!r}; this indicates a compiler bug."
                )
            indeg[nid] += 1
            out_edges[src].append(nid)

    # Sorted ready set for determinism
    ready = sorted([nid for nid, d in indeg.items() if d == 0])
    order: list[str] = []
    while ready:
        nid = ready.pop(0)
        order.append(nid)
        for succ in out_edges[nid]:
            indeg[succ] -= 1
            if indeg[succ] == 0:
                # Insert preserving sorted order
                lo, hi = 0, len(ready)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if ready[mid] < succ:
                        lo = mid + 1
                    else:
                        hi = mid
                ready.insert(lo, succ)

    if len(order) != len(nodes):
        remaining = [nid for nid in nodes if nid not in set(order)]
        raise ValueError(f"Cycle detected in execution DAG; nodes involved: {remaining}")

    return order


def _validate_dag(plan: DAGPlan) -> None:
    """
    Structural checks beyond acyclicity.

    - Every input id resolves to a known node
    - Every leaf is an output node (Prepare* family) or the unsupported sentinel —
      a leaf doing computation would mean we built a node nothing consumes.
    """
    output_ops = {"PrepareTable", "PrepareSummary", "PrepareMap", "Unsupported"}
    consumed: set[str] = set()
    for n in plan.nodes.values():
        for src in n.inputs:
            if src not in plan.nodes:
                raise ValueError(f"Node {n.node_id!r} references unknown input {src!r}")
            consumed.add(src)
    leaves = [nid for nid in plan.nodes if nid not in consumed]
    plan.leaves = sorted(leaves)
    plan.roots = sorted([nid for nid, n in plan.nodes.items() if not n.inputs])
    for lid in plan.leaves:
        op = plan.nodes[lid].op
        if op not in output_ops:
            # Materialize nodes can be leaves only if no output node was requested,
            # which happens only when sf.outputs is empty (defensive — should not
            # happen in normal flow because outputs default to ["map","summary"]).
            if op == "MaterializeRole":
                continue
            raise ValueError(
                f"Leaf node {lid!r} has op {op!r}; only output-family ops may be leaves."
            )


def render_dag_as_dot(plan: DAGPlan) -> str:
    """Render the DAG as a Graphviz DOT string for debugging or paper figures."""
    lines = ["digraph ExecutionDAG {", "  rankdir=TB;", "  node [shape=box, fontname=\"Helvetica\"];"]
    for nid in plan.order:
        n = plan.nodes[nid]
        label = f"{n.op}\\n{nid}"
        lines.append(f'  "{nid}" [label="{label}"];')
    for nid, n in plan.nodes.items():
        for src in n.inputs:
            lines.append(f'  "{src}" -> "{nid}";')
    lines.append("}")
    return "\n".join(lines)


def _infer_aggregate_spec_from_semantics(sf: SemanticFrame) -> Optional[AggregateSpec]:
    if sf.ranking is None:
        return None

    ranking_role = sf.ranking.target_role or "primary"
    role_map = _get_role_map(sf.targets)
    rank_entity = role_map.get(ranking_role)
    if rank_entity is None:
        return None

    if sf.ranking.metric == "crash_count":
        crash_role = _detect_crash_role(role_map, preferred_role="support")
        if crash_role is None or crash_role == ranking_role:
            return None

        agg_sc = next(
            (
                x for x in sf.spatial_constraints
                if x.target_role == crash_role
                and x.reference_role == ranking_role
                and x.relation in GENERIC_SET_MATCH_RELATIONS
            ),
            None,
        )

        agg_relation = agg_sc.relation if agg_sc is not None else "within_distance"
        # intersects/contains don't use distance; within_distance needs a default
        if agg_relation in {"intersects", "contains"}:
            agg_distance = None
        else:
            # Road ranking uses tighter snap distance than point-entity ranking
            if rank_entity == "Road":
                fallback_dist = ROAD_RANKING_SNAP_M
            else:
                fallback_dist = DEFAULT_RADIUS_M
            agg_distance = float(agg_sc.distance_m if agg_sc and agg_sc.distance_m is not None else fallback_dist)

        return AggregateSpec(
            metric=sf.ranking.metric,
            group_role=ranking_role,
            measure_role=crash_role,
            relation=agg_relation,
            distance_m=agg_distance,
            output_name="aggregate_table",
            value_column="crash_count",
        )

    return None


def compile_dag_plan(sf: SemanticFrame) -> DAGPlan:
    """
    Compile a validated SemanticFrame into a typed execution DAG.

    The orchestration order mirrors the original linear compiler so head
    tracking maps cleanly onto the executor's read/write contract over
    state.role_data. The resulting graph captures the same dependencies the
    linear plan implied by ordering, but makes them explicit on each node.
    """
    builder = DAGBuilder()

    if not sf.supported:
        builder.add_unsupported()
        plan = DAGPlan(nodes=builder.nodes)
        plan.order = topological_sort(plan.nodes)
        # Skip _validate_dag for the unsupported sentinel — it's a single-node
        # graph with op="Unsupported" and no edges.
        plan.roots = list(plan.nodes.keys())
        plan.leaves = list(plan.nodes.keys())
        return plan

    # --- 1. Load registry + entity specs ---
    builder.add_load_registry()
    for t in sf.targets:
        builder.add_load_entity(t.role, t.entity, t.names)

    # --- 2. Resolve references (anchors) and build reference objects ---
    for idx, r in enumerate(sf.references):
        rid = builder.add_resolve_reference(r.role, r.entity, r.name, idx)
        builder.add_build_reference(r.role, rid, idx)

    # --- 3. Initialize role queries + name filters ---
    for t in sf.targets:
        builder.add_initialize_role(t.role)
        if t.names:
            builder.add_name_filter(t.role, t.names)

    # --- 4. Anchor-based spatial constraints (run before scope, as in legacy) ---
    anchor_idx = 0
    for sc in sf.spatial_constraints:
        if sc.reference_role == "anchor":
            builder.add_anchor_spatial(
                target_role=sc.target_role,
                relation=sc.relation,
                reference_role=sc.reference_role,
                distance_m=sc.distance_m,
                anchor_idx=anchor_idx,
            )
            anchor_idx += 1

    # --- 5. Scope filters (one per non-scope role, only if scope target present) ---
    scope_target = _find_target(sf.targets, "scope")
    if scope_target is not None:
        for t in sf.targets:
            if t.role != "scope":
                builder.add_scope_constraint(
                    target_role=t.role,
                    scope_role="scope",
                    relation=SCOPE_FILTER_RELATION_DEFAULT,
                )

    # --- 6. Attribute constraints (one per role, even if no constraints —
    #        the handler is a no-op in that case but the linear plan emitted it,
    #        so we preserve identical behavior) ---
    seen_roles: list[str] = []
    for t in sf.targets:
        if t.role not in seen_roles:
            seen_roles.append(t.role)
    for role in seen_roles:
        builder.add_attribute_constraints(role)

    # --- 7. Relation constraints (snap_match, etc.) ---
    for rel_idx, rel in enumerate(sf.relations):
        builder.add_relation_constraint(
            relation=rel.relation,
            source_role=rel.source_role,
            target_role=rel.target_role,
            distance_m=rel.distance_m,
            rel_idx=rel_idx,
        )

    # --- 8. Spatial set matches (between roles, not anchor-based) ---
    match_idx = 0
    for sc in sf.spatial_constraints:
        if sc.reference_role in {"primary", "support"}:
            builder.add_match_spatial_sets(
                left_role=sc.target_role,
                right_role=sc.reference_role,
                relation=sc.relation,
                distance_m=sc.distance_m,
                match_idx=match_idx,
                left_keep=True,
                right_keep=True,
            )
            match_idx += 1
        elif sc.reference_role == "filter":
            # Filter entity is displayed on map → keep both sides
            builder.add_match_spatial_sets(
                left_role=sc.target_role,
                right_role=sc.reference_role,
                relation=sc.relation,
                distance_m=sc.distance_m,
                match_idx=match_idx,
                left_keep=True,
                right_keep=True,
            )
            match_idx += 1

    # --- 9. Aggregate (only if ranking_metric=crash_count and shape is right) ---
    agg = _infer_aggregate_spec_from_semantics(sf)
    if agg is not None:
        builder.add_aggregate({
            "metric": agg.metric,
            "group_role": agg.group_role,
            "measure_role": agg.measure_role,
            "relation": agg.relation,
            "distance_m": agg.distance_m,
            "output_name": agg.output_name,
            "value_column": agg.value_column,
        })

    # --- 10. Rank ---
    if sf.ranking is not None:
        builder.add_rank({
            "source_table": "aggregate_table",
            "target_role": sf.ranking.target_role,
            "order": sf.ranking.order,
            "top_n": sf.ranking.top_n,
        })

    # --- 11. Materialize each target role ---
    ranking_role = sf.ranking.target_role if sf.ranking else None
    for t in sf.targets:
        builder.add_materialize_role(t.role, ranking_role=ranking_role)

    # --- 12. Output preparation nodes ---
    if "table" in sf.outputs:
        builder.add_output("PrepareTable")
    if "summary" in sf.outputs:
        builder.add_output("PrepareSummary")
    if "map" in sf.outputs:
        builder.add_output("PrepareMap")

    plan = DAGPlan(nodes=builder.nodes)
    plan.order = topological_sort(plan.nodes)
    _validate_dag(plan)
    return plan


# Backward-compatible alias retained for any callers that imported the old
# entry-point name. Returns the same DAGPlan; DAGPlan.steps yields nodes in
# topological order so legacy iteration patterns continue to work.
def compile_linear_plan(sf: SemanticFrame) -> DAGPlan:
    return compile_dag_plan(sf)


# =========================================================
# 11) GENERIC SQL BUILDERS
# =========================================================
def _quote_col(col: str) -> str:
    return f'"{col}"'


def build_crash_date_expr(alias: str) -> str:
    col = f"{alias}.{_quote_col(CRASH_DATE_COL)}"
    return f"""
    CASE
      WHEN {col} IS NULL OR BTRIM({col}) = '' THEN NULL::date
      ELSE TO_DATE({col}, 'MM DD YYYY')
    END
    """


def build_crash_time_minutes_expr(alias: str) -> str:
    col = f"{alias}.{_quote_col(CRASH_TIME_COL)}"
    return f"""
    CASE
      WHEN {col} IS NULL OR BTRIM({col}) = '' THEN NULL::integer
      ELSE
        (
          EXTRACT(HOUR FROM TO_TIMESTAMP(BTRIM({col}), 'HH12:MI AM'))::int * 60
          +
          EXTRACT(MINUTE FROM TO_TIMESTAMP(BTRIM({col}), 'HH12:MI AM'))::int
        )
    END
    """


def build_town_name_match_expr(alias: str) -> str:
    col = f"{alias}.{_quote_col(TOWN_NAME_COL)}"

    base_expr = f"""
    TRIM(
      REGEXP_REPLACE(
        LOWER(
          REGEXP_REPLACE(COALESCE({col}, ''), '[^a-zA-Z0-9 ]', ' ', 'g')
        ),
        '\\s+',
        ' ',
        'g'
      )
    )
    """

    stripped_expr = f"""
    TRIM(
      REGEXP_REPLACE(
        {base_expr},
        '((\\s+)(town|city))+$',
        '',
        'g'
      )
    )
    """

    return stripped_expr


def build_derived_field_sql(field_name: str, alias: str, operator: str, value: Any, param_prefix: str) -> tuple[str, dict]:
    params = {}

    if field_name == DERIVED_SIDEWALK_STATUS:
        lf = f'{alias}.{_quote_col(LT_SIDEWALK_COL)}'
        rf = f'{alias}.{_quote_col(RT_SIDEWALK_COL)}'
        case_expr = f"""
        CASE
          WHEN COALESCE(NULLIF(BTRIM({lf}::text),'')::int, 0) > 0 AND COALESCE(NULLIF(BTRIM({rf}::text),'')::int, 0) > 0 THEN 'both'
          WHEN COALESCE(NULLIF(BTRIM({lf}::text),'')::int, 0) = 0 AND COALESCE(NULLIF(BTRIM({rf}::text),'')::int, 0) = 0 THEN 'none'
          WHEN COALESCE(NULLIF(BTRIM({lf}::text),'')::int, 0) > 0 AND COALESCE(NULLIF(BTRIM({rf}::text),'')::int, 0) = 0 THEN 'left_only'
          WHEN COALESCE(NULLIF(BTRIM({lf}::text),'')::int, 0) = 0 AND COALESCE(NULLIF(BTRIM({rf}::text),'')::int, 0) > 0 THEN 'right_only'
          ELSE 'partial'
        END
        """
        if operator == "eq":
            params[f"{param_prefix}_value"] = str(value)
            return f"({case_expr}) = %({param_prefix}_value)s", params
        if operator == "in":
            vals = value if isinstance(value, list) else [value]
            params[f"{param_prefix}_value"] = [str(v) for v in vals]
            return f"({case_expr}) = ANY(%({param_prefix}_value)s)", params
        if operator == "is_null":
            return f"({case_expr}) IS NULL", params
        if operator == "not_null":
            return f"({case_expr}) IS NOT NULL", params

    if field_name == DERIVED_CRASH_DATE_VALUE:
        date_expr = build_crash_date_expr(alias)
        if operator == "eq":
            params[f"{param_prefix}_value"] = str(value)
            return f"({date_expr}) = %({param_prefix}_value)s::date", params
        if operator == "in":
            vals = value if isinstance(value, list) else [value]
            params[f"{param_prefix}_value"] = [str(v) for v in vals]
            return f"({date_expr}) = ANY(%({param_prefix}_value)s::date[])", params
        if operator in {"gt", "gte", "lt", "lte"}:
            params[f"{param_prefix}_value"] = str(value)
            op_map = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}
            return f"({date_expr}) {op_map[operator]} %({param_prefix}_value)s::date", params
        if operator == "between":
            if isinstance(value, dict):
                v1 = str(value.get("min"))
                v2 = str(value.get("max"))
            elif isinstance(value, (list, tuple)) and len(value) == 2:
                v1, v2 = str(value[0]), str(value[1])
            else:
                raise ValueError(f"between requires 2 values for field {field_name}")
            d1 = parse_date_text(v1)
            d2 = parse_date_text(v2)
            if d1 > d2:
                d1, d2 = d2, d1
            params[f"{param_prefix}_min"] = d1.strftime("%Y-%m-%d")
            params[f"{param_prefix}_max"] = d2.strftime("%Y-%m-%d")
            return f"({date_expr}) BETWEEN %({param_prefix}_min)s::date AND %({param_prefix}_max)s::date", params
        if operator == "is_null":
            return f"({date_expr}) IS NULL", params
        if operator == "not_null":
            return f"({date_expr}) IS NOT NULL", params

    if field_name == DERIVED_CRASH_TIME_MINUTES:
        time_expr = build_crash_time_minutes_expr(alias)
        if operator == "eq":
            params[f"{param_prefix}_value"] = int(value)
            return f"({time_expr}) = %({param_prefix}_value)s", params
        if operator == "in":
            vals = value if isinstance(value, list) else [value]
            params[f"{param_prefix}_value"] = [int(v) for v in vals]
            return f"({time_expr}) = ANY(%({param_prefix}_value)s)", params
        if operator in {"gt", "gte", "lt", "lte"}:
            params[f"{param_prefix}_value"] = int(value)
            op_map = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}
            return f"({time_expr}) {op_map[operator]} %({param_prefix}_value)s", params
        if operator == "between":
            if isinstance(value, dict):
                v1 = int(value.get("min"))
                v2 = int(value.get("max"))
            elif isinstance(value, (list, tuple)) and len(value) == 2:
                v1, v2 = int(value[0]), int(value[1])
            else:
                raise ValueError(f"between requires 2 values for field {field_name}")
            if v1 > v2:
                v1, v2 = v2, v1
            params[f"{param_prefix}_min"] = int(v1)
            params[f"{param_prefix}_max"] = int(v2)
            return f"({time_expr}) BETWEEN %({param_prefix}_min)s AND %({param_prefix}_max)s", params
        if operator == "is_null":
            return f"({time_expr}) IS NULL", params
        if operator == "not_null":
            return f"({time_expr}) IS NOT NULL", params

    if field_name == DERIVED_NAME_MATCH_KEY:
        name_expr = build_town_name_match_expr(alias)
        if operator == "eq":
            params[f"{param_prefix}_value"] = normalize_town_base_name(str(value))
            return f"({name_expr}) = %({param_prefix}_value)s", params
        if operator == "in":
            vals = value if isinstance(value, list) else [value]
            params[f"{param_prefix}_value"] = [normalize_town_base_name(str(v)) for v in vals]
            return f"({name_expr}) = ANY(%({param_prefix}_value)s)", params
        if operator == "is_null":
            return f"({name_expr}) IS NULL", params
        if operator == "not_null":
            return f"({name_expr}) IS NOT NULL", params

    raise ValueError(f"Unsupported derived field adapter: {field_name}")


def build_raw_field_sql(alias: str, field_name: str, field_meta: dict, operator: str, value: Any, param_prefix: str) -> tuple[str, dict]:
    col = f"{alias}.{_quote_col(field_name)}"
    field_type = field_meta.get("type")
    params = {}

    # Fields from merged road inventory stored as character varying but contain numbers.
    # We need to safely cast them to numeric for comparison.
    TEXT_STORED_NUMERIC_FIELDS = {LT_SIDEWALK_COL, RT_SIDEWALK_COL, SPEED_LIM_COL}
    if field_name in TEXT_STORED_NUMERIC_FIELDS and field_type == "numeric":
        col = f"COALESCE(NULLIF(BTRIM({col}::text),'')::numeric, 0)"

    if operator == "is_null":
        return f"{col} IS NULL", params
    if operator == "not_null":
        return f"{col} IS NOT NULL", params

    if field_type == "numeric":
        if operator in {"gt", "gte", "lt", "lte", "eq"}:
            params[f"{param_prefix}_value"] = float(value)
            op_map = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<=", "eq": "="}
            return f"{col} {op_map[operator]} %({param_prefix}_value)s", params
        if operator == "between":
            if isinstance(value, dict):
                v1 = float(value.get("min"))
                v2 = float(value.get("max"))
            elif isinstance(value, (list, tuple)) and len(value) == 2:
                v1, v2 = float(value[0]), float(value[1])
            else:
                raise ValueError(f"between requires 2 values for field {field_name}")
            if v1 > v2:
                v1, v2 = v2, v1
            params[f"{param_prefix}_min"] = v1
            params[f"{param_prefix}_max"] = v2
            return f"{col} BETWEEN %({param_prefix}_min)s AND %({param_prefix}_max)s", params
        if operator == "in":
            vals = value if isinstance(value, list) else [value]
            params[f"{param_prefix}_value"] = [float(v) for v in vals]
            return f"{col} = ANY(%({param_prefix}_value)s)", params

    if field_type in {"categorical", "text", "id"}:
        if operator == "eq":
            params[f"{param_prefix}_value"] = value
            return f"{col} = %({param_prefix}_value)s", params
        if operator == "in":
            vals = value if isinstance(value, list) else [value]
            params[f"{param_prefix}_value"] = vals
            return f"{col} = ANY(%({param_prefix}_value)s)", params

    raise ValueError(f"Unsupported operator/type combination: field={field_name}, type={field_type}, operator={operator}")


def build_attribute_constraint_sql(dataset_spec: DatasetSpec, alias: str, constraint: AttributeConstraint, param_prefix: str) -> tuple[str, dict]:
    field_name = constraint.field
    operator = constraint.operator
    value = constraint.value

    if field_name in dataset_spec.fields:
        return build_raw_field_sql(alias, field_name, dataset_spec.fields[field_name], operator, value, param_prefix)

    if field_name in dataset_spec.derived_fields:
        return build_derived_field_sql(field_name, alias, operator, value, param_prefix)

    raise ValueError(f"Unknown field in attribute constraint: {field_name}")


def build_name_filter_sql(dataset_spec: DatasetSpec, alias: str, names: list[str], param_prefix: str) -> tuple[str, dict]:
    if not names:
        return "1=1", {}

    label_field = dataset_spec.name_match_field or dataset_spec.label_field or dataset_spec.primary_key
    if not label_field:
        raise ValueError(f"Dataset {dataset_spec.entity} does not support name filtering.")

    params = {}

    if dataset_spec.entity == "Town":
        full_norms, core_norms = normalize_name_match_inputs(names, dataset_spec.name_match_strip_suffixes)
        raw_col = f"{alias}.{_quote_col(label_field)}"
        core_expr = build_town_name_match_expr(alias)
        full_expr = f"TRIM(REGEXP_REPLACE(LOWER(REGEXP_REPLACE(COALESCE({raw_col}, ''), '[^a-zA-Z0-9 ]', ' ', 'g')), '\\s+', ' ', 'g'))"

        parts = []
        if full_norms:
            params[f"{param_prefix}_full"] = full_norms
            parts.append(f"({full_expr}) = ANY(%({param_prefix}_full)s)")
        if core_norms:
            params[f"{param_prefix}_core"] = core_norms
            parts.append(f"({core_expr}) = ANY(%({param_prefix}_core)s)")

        if not parts:
            return "1=0", {}

        return "(" + " OR ".join(parts) + ")", params

    params[f"{param_prefix}_names"] = names
    return f"{alias}.{_quote_col(label_field)} = ANY(%({param_prefix}_names)s)", params


def build_role_select_clause(dataset_spec: DatasetSpec, alias: str, include_geom_wkt: bool = True) -> str:
    cols = []
    pk = dataset_spec.primary_key
    if pk:
        cols.append(f"{alias}.{_quote_col(pk)} AS {pk}")
    label_field = dataset_spec.label_field
    if label_field and label_field != pk:
        cols.append(f"{alias}.{_quote_col(label_field)} AS {label_field}")
    for f in dataset_spec.display_fields:
        if f not in {pk, label_field}:
            cols.append(f"{alias}.{_quote_col(f)} AS {f}")
    if include_geom_wkt:
        cols.append(f"ST_AsText({alias}.geom) AS wkt")
    return ", ".join(cols)


def make_base_role_sql(dataset_spec: DatasetSpec, alias: str = "x") -> tuple[str, dict]:
    sql = f"""
    SELECT
      {build_role_select_clause(dataset_spec, alias=alias, include_geom_wkt=False)},
      {alias}.{dataset_spec.geometry_column} AS geom
    FROM {dataset_spec.table} {alias}
    WHERE {alias}.{dataset_spec.geometry_column} IS NOT NULL
    """
    return sql, {}


def wrap_sql_with_where(base_sql: str, where_sql: str) -> str:
    return f"""
    SELECT *
    FROM (
      {base_sql}
    ) q
    WHERE {where_sql}
    """


def build_spatial_join_condition(
    left_geom_sql: str,
    right_geom_sql: str,
    relation: str,
    distance_m: Optional[float] = None,
) -> str:
    if relation == "within_distance":
        if distance_m is None:
            raise ValueError("within_distance requires distance_m")
        return f"ST_DWithin({left_geom_sql}, {right_geom_sql}, {float(distance_m)})"

    if relation == "intersects":
        return f"ST_Intersects({left_geom_sql}, {right_geom_sql})"

    if relation == "contains":
        return f"ST_Contains({right_geom_sql}, {left_geom_sql})"

    raise ValueError(f"Unsupported spatial join relation: {relation}")


def wrap_sql_with_spatial_set_filter(
    target_sql: str,
    reference_sql: str,
    relation: str,
    distance_m: Optional[float],
    target_geom_col: str = "geom",
    reference_geom_col: str = "geom",
) -> str:
    join_condition = build_spatial_join_condition(
        left_geom_sql=f"q.{target_geom_col}",
        right_geom_sql=f"ref.{reference_geom_col}",
        relation=relation,
        distance_m=distance_m,
    )
    return f"""
    SELECT DISTINCT q.*
    FROM ({target_sql}) q
    JOIN ({reference_sql}) ref
      ON {join_condition}
    """


def wrap_sql_with_relation_join(
    source_sql: str,
    target_sql: str,
    source_geom_col: str,
    target_geom_col: str,
    relation: str,
    distance_m: float,
    keep_side: str,
) -> str:
    if relation != "snap_match":
        raise ValueError(f"Unsupported relation in generic join: {relation}")

    join_condition = f"ST_DWithin(s.{source_geom_col}, t.{target_geom_col}, {float(distance_m)})"

    if keep_side == "source":
        return f"""
        SELECT DISTINCT s.*
        FROM ({source_sql}) s
        JOIN ({target_sql}) t
          ON {join_condition}
        """
    if keep_side == "target":
        return f"""
        SELECT DISTINCT t.*
        FROM ({source_sql}) s
        JOIN ({target_sql}) t
          ON {join_condition}
        """
    raise ValueError("keep_side must be source or target")


def build_reference_sql_from_df(ref_df: pd.DataFrame) -> tuple[str, dict]:
    if ref_df.empty:
        return "SELECT NULL::geometry AS geom WHERE FALSE", {}

    ref_rows = []
    params = {}
    for i, row in ref_df.reset_index(drop=True).iterrows():
        key = f"ref_wkt_{i}"
        params[key] = str(row["wkt"])
        ref_rows.append(f"SELECT ST_GeomFromText(%({key})s, 26986) AS geom")

    return "\nUNION ALL\n".join(ref_rows), params


def merge_sql_params_with_prefix(base_params: dict, extra_params: dict, prefix: str, sql_text: str) -> tuple[dict, str]:
    merged = dict(base_params)
    new_sql = sql_text
    for k, v in extra_params.items():
        new_k = f"{prefix}{k}"
        merged[new_k] = v
        new_sql = new_sql.replace(f"%({k})s", f"%({new_k})s")
    return merged, new_sql


def build_generic_count_aggregation_sql(
    group_sql: str,
    measure_sql: str,
    group_ds: DatasetSpec,
    measure_ds: DatasetSpec,
    relation: str,
    distance_m: Optional[float],
    value_column: str,
    count_distinct_measure: bool = True,
    group_alias: str = "g",
    measure_alias: str = "m",
) -> str:
    group_pk = get_dataset_key_field(group_ds)
    if not group_pk:
        raise ValueError(f"Group dataset must have a primary key for aggregation: {group_ds.entity}")

    group_label = get_dataset_label_field(group_ds)
    group_select_parts = [f'{group_alias}.{_quote_col(group_pk)} AS group_key']
    if group_label:
        group_select_parts.append(f'{group_alias}.{_quote_col(group_label)} AS group_label')

    join_condition = build_spatial_join_condition(
        left_geom_sql=f"{measure_alias}.geom",
        right_geom_sql=f"{group_alias}.geom",
        relation=relation,
        distance_m=distance_m,
    )

    measure_pk = get_dataset_key_field(measure_ds)
    if count_distinct_measure and measure_pk:
        metric_expr = f'COUNT(DISTINCT {measure_alias}.{_quote_col(measure_pk)})'
    elif measure_pk:
        metric_expr = f'COUNT({measure_alias}.{_quote_col(measure_pk)})'
    else:
        metric_expr = "COUNT(*)"

    group_by_cols = [f"{group_alias}.{_quote_col(group_pk)}"]
    if group_label:
        group_by_cols.append(f"{group_alias}.{_quote_col(group_label)}")

    return f"""
    SELECT
      {", ".join(group_select_parts)},
      {metric_expr} AS {value_column}
    FROM ({group_sql}) {group_alias}
    LEFT JOIN ({measure_sql}) {measure_alias}
      ON {join_condition}
    GROUP BY {", ".join(group_by_cols)}
    """


def build_generic_match_detail_sql(
    left_sql: str,
    right_sql: str,
    left_ds: DatasetSpec,
    right_ds: DatasetSpec,
    relation: str,
    distance_m: Optional[float],
    left_tag_field: Optional[str] = None,
    right_tag_field: Optional[str] = None,
) -> str:
    join_condition = build_spatial_join_condition(
        left_geom_sql="l.geom",
        right_geom_sql="r.geom",
        relation=relation,
        distance_m=distance_m,
    )

    select_parts = []

    left_pk = get_dataset_key_field(left_ds)
    right_pk = get_dataset_key_field(right_ds)
    left_label = get_dataset_label_field(left_ds)
    right_label = get_dataset_label_field(right_ds)

    if left_tag_field and left_pk:
        select_parts.append(f'l.{_quote_col(left_pk)} AS {left_tag_field}')
    if right_tag_field and right_pk:
        select_parts.append(f'r.{_quote_col(right_pk)} AS {right_tag_field}')

    if left_pk:
        select_parts.append(f'l.{_quote_col(left_pk)} AS {left_pk}')
    if left_label and left_label != left_pk:
        select_parts.append(f'l.{_quote_col(left_label)} AS {left_label}')

    for f in left_ds.display_fields:
        if f not in {left_pk, left_label}:
            select_parts.append(f'l.{_quote_col(f)} AS {f}')

    select_parts.append("ST_AsText(l.geom) AS wkt")

    return f"""
    SELECT DISTINCT
      {", ".join(select_parts)}
    FROM ({left_sql}) l
    JOIN ({right_sql}) r
      ON {join_condition}
    """


def build_generic_pair_count_sql(
    left_sql: str,
    right_sql: str,
    relation: str,
    distance_m: Optional[float],
) -> str:
    join_condition = build_spatial_join_condition(
        left_geom_sql="l.geom",
        right_geom_sql="r.geom",
        relation=relation,
        distance_m=distance_m,
    )
    return f"""
    SELECT COUNT(*) AS pair_count
    FROM ({left_sql}) l
    JOIN ({right_sql}) r
      ON {join_condition}
    """


# =========================================================
# 12) GENERIC ANALYSIS HELPERS
# =========================================================
def materialize_role_to_gdf(conn, role_data: RoleData) -> tuple[gpd.GeoDataFrame, int]:
    ds = DATASET_REGISTRY[role_data.entity]
    render_limit = role_data.render_limit or get_default_render_limit(role_data.entity)
    params = dict(role_data.params)

    count_sql = f"SELECT COUNT(*) FROM ({role_data.sql_base}) z"
    selected_count = int(fetch_scalar(conn, count_sql, params) or 0)

    select_fields = build_role_select_clause(ds, alias="z", include_geom_wkt=False)
    sql = f"""
    SELECT {select_fields}, ST_AsText(z.geom) AS wkt
    FROM ({role_data.sql_base}) z
    LIMIT {int(render_limit)}
    """
    df = fetch_df(conn, sql, params)

    if role_data.entity == "Road":
        gdf = finalize_roads_gdf(df)
    elif ds.display_geometry_mode == "centroid_point":
        gdf = finalize_centroid_display_gdf(df)
    else:
        gdf = geodf_from_wkt_df(df)

    return gdf, selected_count


def materialize_sql_to_gdf_for_entity(
    conn,
    entity: str,
    sql: str,
    params: dict,
    render_limit: Optional[int] = None,
) -> tuple[gpd.GeoDataFrame, int]:
    ds = DATASET_REGISTRY[entity]
    limit_n = render_limit or get_default_render_limit(entity)

    count_sql = f"SELECT COUNT(*) FROM ({sql}) z"
    selected_count = int(fetch_scalar(conn, count_sql, params) or 0)

    select_fields = build_role_select_clause(ds, alias="z", include_geom_wkt=False)
    data_sql = f"""
    SELECT {select_fields}, ST_AsText(z.geom) AS wkt
    FROM ({sql}) z
    LIMIT {int(limit_n)}
    """
    df = fetch_df(conn, data_sql, params)

    if entity == "Road":
        gdf = finalize_roads_gdf(df)
    elif ds.display_geometry_mode == "centroid_point":
        gdf = finalize_centroid_display_gdf(df)
    else:
        gdf = geodf_from_wkt_df(df)

    return gdf, selected_count


def _get_role_constraints(sf: SemanticFrame, role: str) -> list[AttributeConstraint]:
    return [ac for ac in sf.attribute_constraints if ac.target_role == role]


def _get_first_spatial_constraint(
    sf: SemanticFrame,
    target_role: str,
    reference_role: str,
    relation: Optional[str] = None,
) -> Optional[SpatialConstraint]:
    for sc in sf.spatial_constraints:
        if sc.target_role == target_role and sc.reference_role == reference_role:
            if relation is None or sc.relation == relation:
                return sc
    return None


def _merge_role_and_reference_params(role_rd: RoleData, ref_sql: str, ref_params: dict, prefix: str) -> tuple[dict, str]:
    merged = dict(role_rd.params)
    new_ref_sql = ref_sql
    for k, v in ref_params.items():
        new_k = f"{prefix}{k}"
        merged[new_k] = v
        new_ref_sql = new_ref_sql.replace(f"%({k})s", f"%({new_k})s")
    return merged, new_ref_sql


def build_filtered_measure_sql_from_constraints(
    dataset_spec: DatasetSpec,
    constraints: list[AttributeConstraint],
    alias: str,
    base_prefix: str,
) -> tuple[str, dict]:
    base_sql, base_params = make_base_role_sql(dataset_spec, alias=alias)
    where_parts = ["1=1"]
    params = dict(base_params)

    for idx, ac in enumerate(constraints, start=1):
        w, p = build_attribute_constraint_sql(dataset_spec, alias, ac, f"{base_prefix}_{idx}")
        where_parts.append(w)
        params.update(p)

    final_sql = f"""
    SELECT *
    FROM (
      {base_sql}
    ) {alias}
    WHERE {' AND '.join(where_parts)}
    """
    return final_sql, params


def execute_generic_count_aggregation(
    conn,
    group_sql: str,
    group_params: dict,
    group_entity: str,
    measure_sql: str,
    measure_params: dict,
    measure_entity: str,
    relation: str,
    distance_m: Optional[float],
    value_column: str = "metric_value",
) -> tuple[pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, int]:
    group_ds = DATASET_REGISTRY[group_entity]
    measure_ds = DATASET_REGISTRY[measure_entity]

    agg_sql = build_generic_count_aggregation_sql(
        group_sql=group_sql,
        measure_sql=measure_sql,
        group_ds=group_ds,
        measure_ds=measure_ds,
        relation=relation,
        distance_m=distance_m,
        value_column=value_column,
        count_distinct_measure=True,
        group_alias="g",
        measure_alias="m",
    )

    agg_params = dict(group_params)
    agg_sql_local = agg_sql
    for k, v in measure_params.items():
        agg_params[f"m_{k}"] = v
        agg_sql_local = agg_sql_local.replace(f"%({k})s", f"%(m_{k})s")

    agg_df = fetch_df(conn, agg_sql_local, agg_params)

    group_gdf, _ = materialize_sql_to_gdf_for_entity(
        conn=conn,
        entity=group_entity,
        sql=group_sql,
        params=group_params,
        render_limit=get_default_render_limit(group_entity),
    )

    right_key = get_dataset_key_field(group_ds)
    right_tag_field = f"matched_{right_key}" if right_key else None

    detail_sql = build_generic_match_detail_sql(
        left_sql=measure_sql,
        right_sql=group_sql,
        left_ds=measure_ds,
        right_ds=group_ds,
        relation=relation,
        distance_m=distance_m,
        left_tag_field=None,
        right_tag_field=right_tag_field,
    )

    detail_params = dict(measure_params)
    detail_sql_local = detail_sql
    for k, v in group_params.items():
        detail_params[f"g_{k}"] = v
        detail_sql_local = detail_sql_local.replace(f"%({k})s", f"%(g_{k})s")

    detail_df = fetch_df(conn, detail_sql_local, detail_params)

    if not detail_df.empty:
        if measure_entity == "Road":
            detail_gdf = finalize_roads_gdf(detail_df)
        elif measure_ds.display_geometry_mode == "centroid_point":
            detail_gdf = finalize_centroid_display_gdf(detail_df)
        else:
            detail_gdf = geodf_from_wkt_df(detail_df)

        measure_pk = get_dataset_key_field(measure_ds)
        if measure_pk and measure_pk in detail_df.columns:
            matched_count = int(detail_df[measure_pk].nunique())
        else:
            matched_count = int(len(detail_df))
    else:
        empty_cols = []
        if right_tag_field:
            empty_cols.append(right_tag_field)
        measure_pk = get_dataset_key_field(measure_ds)
        measure_label = get_dataset_label_field(measure_ds)
        if measure_pk:
            empty_cols.append(measure_pk)
        if measure_label and measure_label not in empty_cols:
            empty_cols.append(measure_label)
        empty_cols += [c for c in measure_ds.display_fields if c not in empty_cols]
        empty_df = pd.DataFrame(columns=empty_cols)
        detail_gdf = gpd.GeoDataFrame(empty_df, geometry=gpd.GeoSeries([], crs="EPSG:4326"), crs="EPSG:4326")
        matched_count = 0

    return agg_df, group_gdf, detail_gdf, matched_count


def filter_gdf_by_label_values(gdf: Optional[gpd.GeoDataFrame], label_col: str, keep_values: list[Any]) -> Optional[gpd.GeoDataFrame]:
    if gdf is None or gdf.empty:
        return gdf
    if label_col not in gdf.columns:
        return gdf
    return gdf[gdf[label_col].isin(keep_values)].copy()


def deduplicate_gdf_by_entity_pk(gdf: Optional[gpd.GeoDataFrame], entity: str) -> Optional[gpd.GeoDataFrame]:
    if gdf is None or gdf.empty:
        return gdf
    ds = DATASET_REGISTRY[entity]
    pk = get_dataset_key_field(ds)
    if pk and pk in gdf.columns:
        return gdf.drop_duplicates(subset=[pk]).copy()
    return gdf.drop_duplicates().copy()


def get_match_tag_field_for_entity(entity: str) -> Optional[str]:
    ds = DATASET_REGISTRY[entity]
    pk = get_dataset_key_field(ds)
    if pk:
        return f"matched_{pk}"
    return None


def get_rank_keep_values_from_table(df: pd.DataFrame) -> list[Any]:
    if df is None or df.empty:
        return []
    if "group_key" in df.columns:
        return df["group_key"].tolist()
    if "name" in df.columns:
        return df["name"].tolist()
    return []


def get_rank_keep_field_for_roledata(rd: RoleData) -> Optional[str]:
    if rd is None:
        return None
    if rd.primary_key:
        return rd.primary_key
    return rd.label_field


def get_analysis_table(state: ExecutionState, table_name: str) -> Optional[pd.DataFrame]:
    if table_name == "aggregate_table":
        return state.analysis.get("aggregate_table")
    if table_name == "ranking_table":
        return state.analysis.get("ranking_table")
    return None


def set_analysis_table(state: ExecutionState, table_name: str, df: Optional[pd.DataFrame]) -> ExecutionState:
    if table_name == "aggregate_table":
        state.analysis["aggregate_table"] = df
    elif table_name == "ranking_table":
        state.analysis["ranking_table"] = df
    else:
        state.debug.setdefault("analysis_tables_extra", {})[table_name] = df
    return state


def get_primary_role_name(state: ExecutionState) -> Optional[str]:
    sf = state.semantic_frame
    if sf is None:
        return None
    for t in sf.targets:
        if t.role == "primary":
            return "primary"
    return None


def get_support_role_name(state: ExecutionState) -> Optional[str]:
    sf = state.semantic_frame
    if sf is None:
        return None
    for t in sf.targets:
        if t.role == "support":
            return "support"
    return None


def get_scope_role_name(state: ExecutionState) -> Optional[str]:
    sf = state.semantic_frame
    if sf is None:
        return None
    for t in sf.targets:
        if t.role == "scope":
            return "scope"
    return None


def get_display_role_gdf(state: ExecutionState, role: str) -> Optional[gpd.GeoDataFrame]:
    if role == "primary" and state.analysis.get("primary_display_gdf") is not None:
        return state.analysis.get("primary_display_gdf")
    if role == "support" and state.analysis.get("support_display_gdf") is not None:
        return state.analysis.get("support_display_gdf")
    rd = state.role_data.get(role)
    if rd is None:
        return None
    return rd.gdf


def get_display_table_for_output(state: ExecutionState) -> Optional[pd.DataFrame]:
    if state.analysis.get("ranking_table") is not None:
        return state.analysis.get("ranking_table")
    if state.analysis.get("aggregate_table") is not None:
        return state.analysis.get("aggregate_table")
    return None


# =========================================================
# 13) MAP CREATION
# =========================================================
def _legend_symbol_svg(shape: str, color: str) -> str:
    """
    Return inline SVG matching the actual map symbol.
    Shapes: circle, line, polygon, square (fallback).
    """
    size = 18
    if shape == "circle":
        return (
            f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">'
            f'<circle cx="{size//2}" cy="{size//2}" r="{size//2 - 2}" fill="{color}" stroke="{color}" stroke-width="1" opacity="0.9"/>'
            f'</svg>'
        )
    if shape == "line":
        return (
            f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">'
            f'<line x1="1" y1="{size//2}" x2="{size-1}" y2="{size//2}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>'
            f'</svg>'
        )
    if shape == "polygon":
        return (
            f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">'
            f'<rect x="1" y="1" width="{size-2}" height="{size-2}" fill="{color}" fill-opacity="0.12" '
            f'stroke="{color}" stroke-width="2" rx="2"/>'
            f'</svg>'
        )
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">'
        f'<rect x="0" y="0" width="{size}" height="{size}" fill="{color}" rx="2"/>'
        f'</svg>'
    )


def add_legend(m: folium.Map, title: str, items: list[tuple[str, str, str]]):
    """
    Add legend to map. items: list of (label, color, shape) tuples.
    shape: "circle", "line", "polygon", "square"
    """
    rows = "".join(
        f"""
        <div style="display:flex; align-items:center; margin:4px 0;">
          <div style="width:18px; height:18px; flex-shrink:0; margin-right:8px; display:flex; align-items:center; justify-content:center;">
            {_legend_symbol_svg(shape, color)}
          </div>
          <div style="font-size:12px; line-height:1.3;">{label}</div>
        </div>
        """
        for label, color, shape in items
    )

    template = Template(f"""
    {{% macro html(this, kwargs) %}}
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        z-index: 9999;
        background: white;
        padding: 10px 12px;
        border: 1px solid #777;
        border-radius: 6px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.2);
        ">
      <div style="font-weight:600; margin-bottom:6px; font-size:13px;">{title}</div>
      {rows}
    </div>
    {{% endmacro %}}
    """)

    macro = MacroElement()
    macro._template = template
    m.get_root().add_child(macro)
    return m


def _should_use_sidewalk_color_mode(state: ExecutionState) -> bool:
    sf = state.semantic_frame
    if sf is None:
        return False
    if sf.notes and "sidewalk_visual" in str(sf.notes):
        return True
    return any(ac.field == DERIVED_SIDEWALK_STATUS for ac in sf.attribute_constraints)


def _add_roads_geojson(roads_gdf: gpd.GeoDataFrame, fg: folium.FeatureGroup, color_mode: str = "uniform"):
    if roads_gdf is None or roads_gdf.empty:
        return

    def style_fn(feat):
        if color_mode == "uniform":
            return {"color": "black", "weight": 2, "opacity": 0.85}
        status = feat["properties"].get(DERIVED_SIDEWALK_STATUS, "partial")
        if status == "both":
            return {"color": "green", "weight": 3, "opacity": 0.9}
        if status == "none":
            return {"color": "red", "weight": 3, "opacity": 0.9}
        if status == "left_only":
            return {"color": "orange", "weight": 3, "opacity": 0.9}
        if status == "right_only":
            return {"color": "purple", "weight": 3, "opacity": 0.9}
        return {"color": "black", "weight": 2, "opacity": 0.7}

    keep_cols = [
        c for c in [
            SPEED_LIM_COL,
            OP_DIR_SPEED_LIM_COL,
            LT_SIDEWALK_COL,
            RT_SIDEWALK_COL,
            DERIVED_SIDEWALK_STATUS,
            "geometry",
        ]
        if c in roads_gdf.columns
    ]
    roads_small = roads_gdf[keep_cols].copy()

    tooltip_fields = [c for c in [SPEED_LIM_COL, OP_DIR_SPEED_LIM_COL] if c in roads_small.columns]
    if color_mode == "sidewalk_status":
        tooltip_fields += [c for c in [LT_SIDEWALK_COL, RT_SIDEWALK_COL, DERIVED_SIDEWALK_STATUS] if c in roads_small.columns]

    folium.GeoJson(
        data=json.loads(roads_small.to_json()),
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields),
        name="Road inventory",
    ).add_to(fg)


def _add_polygon_geojson(gdf: gpd.GeoDataFrame, fg: folium.FeatureGroup, label_fields: list[str], color: str = "#1f77b4"):
    if gdf is None or gdf.empty:
        return
    keep_cols = [c for c in label_fields if c in gdf.columns] + ["geometry"]
    small = gdf[keep_cols].copy()
    tooltip_fields = [c for c in label_fields if c in small.columns]
    folium.GeoJson(
        data=json.loads(small.to_json()),
        style_function=lambda feat: {
            "color": color,
            "weight": 2,
            "fillColor": color,
            "fillOpacity": 0.08,
            "opacity": 0.9,
        },
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields) if tooltip_fields else None,
        name="Polygon layer",
    ).add_to(fg)
def create_map_from_state(state: ExecutionState) -> folium.Map:
    primary_role = get_primary_role_name(state)
    support_role = get_support_role_name(state)
    scope_role = get_scope_role_name(state)

    role_gdfs = {}
    for role, rd in state.role_data.items():
        use_gdf = get_display_role_gdf(state, role)
        if use_gdf is not None and not use_gdf.empty:
            role_gdfs[role] = use_gdf

    ref_objects = state.references_by_role
    sidewalk_mode = "sidewalk_status" if _should_use_sidewalk_color_mode(state) else "uniform"

    center_lat, center_lon = 42.35, -71.06

    if ref_objects:
        first_ref = next(iter(ref_objects.values()))
        ref_gdf = gpd.GeoDataFrame(first_ref.df.copy(), geometry=first_ref.df["wkt"].apply(wkt.loads), crs="EPSG:26986").to_crs(4326)
        center_lat = float(ref_gdf.geometry.y.iloc[0])
        center_lon = float(ref_gdf.geometry.x.iloc[0])
        m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="CartoDB positron")
    elif role_gdfs:
        first_gdf = next(iter(role_gdfs.values()))
        if first_gdf.geom_type.iloc[0] in {"Point", "MultiPoint"}:
            center_lat = float(first_gdf.geometry.y.mean())
            center_lon = float(first_gdf.geometry.x.mean())
        else:
            cent = first_gdf.to_crs(26986).geometry.centroid
            cent = gpd.GeoSeries(cent, crs="EPSG:26986").to_crs(4326)
            center_lat = float(cent.y.mean())
            center_lon = float(cent.x.mean())
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")
    else:
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB positron")

    legend_items = []

    for ref_role, ref_obj in ref_objects.items():
        ref_gdf = gpd.GeoDataFrame(ref_obj.df.copy(), geometry=ref_obj.df["wkt"].apply(wkt.loads), crs="EPSG:26986").to_crs(4326)
        fg = folium.FeatureGroup(name=f"Reference: {ref_role}", show=True)
        for _, row in ref_gdf.iterrows():
            popup_parts = [
                f"Reference role: {ref_role}",
                f"Name: {row.get('name')}",
                f"Display: {row.get('display_name')}",
                f"Type: {row.get('reference_type')}",
            ]
            if "dist_m" in row and pd.notna(row["dist_m"]):
                popup_parts.append(f"Distance: {float(row['dist_m']):.1f} m")
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=8,
                color="blue",
                fill=True,
                fill_color="blue",
                popup="<br>".join(popup_parts),
            ).add_to(fg)
        fg.add_to(m)
        legend_items.append(("Reference", "blue", "circle"))

    for role, rd in state.role_data.items():
        gdf = role_gdfs.get(role)
        if gdf is None or gdf.empty:
            continue

        entity = rd.entity
        fg = folium.FeatureGroup(name=f"{entity} ({role})", show=True)

        if entity == "Road":
            _add_roads_geojson(gdf, fg, color_mode=sidewalk_mode)
            fg.add_to(m)
            if sidewalk_mode == "sidewalk_status" and DERIVED_SIDEWALK_STATUS in gdf.columns:
                present_statuses = set(gdf[DERIVED_SIDEWALK_STATUS].dropna().unique())
                sidewalk_legend_map = [
                    ("both", "Both sidewalks", "green"),
                    ("none", "No sidewalks", "red"),
                    ("left_only", "Left only", "orange"),
                    ("right_only", "Right only", "purple"),
                    ("partial", "Other / partial", "black"),
                ]
                for status_val, label, color in sidewalk_legend_map:
                    if status_val in present_statuses:
                        legend_items.append((label, color, "line"))
            elif sidewalk_mode == "sidewalk_status":
                legend_items.extend([
                    ("Both sidewalks", "green", "line"),
                    ("No sidewalks", "red", "line"),
                    ("Left only", "orange", "line"),
                    ("Right only", "purple", "line"),
                    ("Other / partial", "black", "line"),
                ])
            else:
                legend_items.append((f"Road ({role})", "black", "line"))

        elif entity == "Crash":
            for _, row in gdf.iterrows():
                sev = row.get(CRASH_SEVE_COL, None)
                sev_txt = f"<br>Severity: {sev}" if pd.notna(sev) else ""
                date_txt = f"<br>Date: {row.get(CRASH_DATE_COL)}" if CRASH_DATE_COL in gdf.columns and pd.notna(row.get(CRASH_DATE_COL)) else ""
                time_txt = f"<br>Time: {row.get(CRASH_TIME_COL)}" if CRASH_TIME_COL in gdf.columns and pd.notna(row.get(CRASH_TIME_COL)) else ""
                harm_txt = f"<br>First harmful: {row.get(FIRST_HARM_COL)}" if FIRST_HARM_COL in gdf.columns and pd.notna(row.get(FIRST_HARM_COL)) else ""

                extra_parts = []
                for c in gdf.columns:
                    if str(c).startswith("matched_") and pd.notna(row.get(c)):
                        extra_parts.append(f"<br>{c}: {row.get(c)}")
                extra = "".join(extra_parts)

                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=3,
                    color="red",
                    fill=True,
                    fill_color="red",
                    popup=f"Crash ID: {row.get(CRASH_ID_COL, '')}{sev_txt}{date_txt}{time_txt}{harm_txt}{extra}",
                ).add_to(fg)
            fg.add_to(m)

            if state.analysis.get("ranking_table") is not None and role == support_role:
                legend_items.append(("Rank-matched crashes", "red", "circle"))
            else:
                legend_items.append((f"Crash ({role})", "red", "circle"))

        elif entity == "School":
            school_label_col = rd.label_field if rd.label_field in gdf.columns else (SCHOOL_NAME_COL if SCHOOL_NAME_COL in gdf.columns else "name")
            for _, row in gdf.iterrows():
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=5,
                    color="cadetblue",
                    fill=True,
                    fill_color="cadetblue",
                    popup=f"School: {row.get(school_label_col, '')}",
                ).add_to(fg)
            fg.add_to(m)

            if state.analysis.get("ranking_table") is not None and role == primary_role:
                legend_items.append(("Ranked schools", "cadetblue", "circle"))
            else:
                legend_items.append((f"School ({role})", "cadetblue", "circle"))

        elif entity == "BusStop":
            label_col = rd.label_field if rd.label_field in gdf.columns else BUS_STOP_NAME_COL
            id_col = rd.primary_key if rd.primary_key in gdf.columns else BUS_STOP_ID_COL
            for _, row in gdf.iterrows():
                popup_parts = []
                if id_col in gdf.columns:
                    popup_parts.append(f"Stop ID: {row.get(id_col, '')}")
                if label_col in gdf.columns:
                    popup_parts.append(f"Stop Name: {row.get(label_col, '')}")
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=5,
                    color="darkgreen",
                    fill=True,
                    fill_color="darkgreen",
                    popup="<br>".join(popup_parts),
                ).add_to(fg)
            fg.add_to(m)

            if state.analysis.get("ranking_table") is not None and role == primary_role:
                legend_items.append(("Ranked bus stops", "darkgreen", "circle"))
            else:
                legend_items.append((f"BusStop ({role})", "darkgreen", "circle"))

        elif entity == "Crosswalk":
            id_col = rd.primary_key if rd.primary_key in gdf.columns else CROSSWALK_ID_COL
            for _, row in gdf.iterrows():
                popup_parts = []
                if id_col in gdf.columns:
                    popup_parts.append(f"Crosswalk ID: {row.get(id_col, '')}")
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=4,
                    color="orange",
                    fill=True,
                    fill_color="orange",
                    popup="<br>".join(popup_parts) if popup_parts else "Crosswalk",
                ).add_to(fg)
            fg.add_to(m)

            if state.analysis.get("ranking_table") is not None and role == primary_role:
                legend_items.append(("Ranked crosswalks", "orange", "circle"))
            else:
                legend_items.append((f"Crosswalk ({role})", "orange", "circle"))

        elif entity == "Town":
            _add_polygon_geojson(gdf, fg, label_fields=[TOWN_NAME_COL], color="#2b6cb0" if role == scope_role else "#6b46c1")
            fg.add_to(m)
            if role == scope_role:
                legend_items.append(("Scope boundary", "#2b6cb0", "polygon"))
            else:
                legend_items.append((f"Town ({role})", "#6b46c1", "polygon"))

    folium.LayerControl(collapsed=False).add_to(m)
    if legend_items:
        dedup = []
        seen = set()
        for item in legend_items:
            if item not in seen:
                dedup.append(item)
                seen.add(item)
        add_legend(m, "Legend", dedup)
    return m


# =========================================================
# 14) RESULT FORMATTING
# =========================================================
def summarize_result(state: ExecutionState) -> str:
    sf = state.semantic_frame
    lines = []

    primary = next((t for t in sf.targets if t.role == "primary"), None)
    if primary:
        lines.append(f"Primary target: {primary.entity}")

    scope = next((t for t in sf.targets if t.role == "scope"), None)
    if scope:
        scope_names_txt = ", ".join(scope.names) if scope.names else "all selected scope features"
        lines.append(f"Scope: {scope.entity} -> {scope_names_txt}")

    if sf.references:
        ref_txt = ", ".join([f"{r.entity}({r.role})={r.name}" for r in sf.references])
        lines.append(f"References: {ref_txt}")
    else:
        lines.append("References: none")

    if sf.spatial_constraints:
        sc_txt = []
        for sc in sf.spatial_constraints:
            if sc.relation == "within_distance":
                sc_txt.append(f"{sc.target_role} within {float(sc.distance_m):.0f} m of {sc.reference_role}")
            else:
                sc_txt.append(f"{sc.target_role} {sc.relation} {sc.reference_role}")
        lines.append("Spatial constraints: " + "; ".join(sc_txt))
    else:
        lines.append("Spatial constraints: none")

    if sf.attribute_constraints:
        parts = []
        for ac in sf.attribute_constraints:
            value_txt = ac.value
            if ac.field == DERIVED_CRASH_TIME_MINUTES and ac.operator == "between" and isinstance(ac.value, (list, tuple)) and len(ac.value) == 2:
                value_txt = [minutes_to_label(ac.value[0]), minutes_to_label(ac.value[1])]
            elif ac.field == DERIVED_CRASH_TIME_MINUTES and ac.operator in {"eq", "gt", "gte", "lt", "lte"}:
                try:
                    value_txt = minutes_to_label(int(ac.value))
                except Exception:
                    value_txt = ac.value
            parts.append(f"{ac.target_role}.{ac.field} {ac.operator} {value_txt}")
        lines.append("Attribute constraints: " + "; ".join(parts))
    else:
        lines.append("Attribute constraints: none")

    if sf.relations:
        rel_parts = []
        for r in sf.relations:
            if r.relation == "snap_match":
                rel_parts.append(f"{r.source_role} snap_match {r.target_role} within {float(r.distance_m):.1f} m")
            else:
                rel_parts.append(f"{r.source_role} {r.relation} {r.target_role}")
        lines.append("Relations: " + "; ".join(rel_parts))
    else:
        lines.append("Relations: none")

    if sf.ranking:
        lines.append(f"Ranking: {sf.ranking.order} {sf.ranking.top_n} by {sf.ranking.metric} on {sf.ranking.target_role}")

    for name, meta in state.match_metadata_by_name.items():
        lines.append(
            f"Match '{name}': "
            f"{meta.left_entity}({meta.left_role}) ↔ {meta.right_entity}({meta.right_role}) "
            f"using {meta.relation}"
            + (f" within {float(meta.distance_m):.1f} m" if meta.distance_m is not None else "")
        )
        if meta.matched_left_count is not None:
            lines.append(f"  matched {meta.left_role} count: {meta.matched_left_count}")
        if meta.matched_right_count is not None:
            lines.append(f"  matched {meta.right_role} count: {meta.matched_right_count}")
        if meta.pair_count is not None:
            lines.append(f"  matched pair count: {meta.pair_count}")

    for role, rd in state.role_data.items():
        lines.append(f"{rd.entity} ({role}) selected count: {rd.selected_count}")
        if rd.selected_names:
            lines.append(f"  selected names: {', '.join(rd.selected_names)}")

    if state.analysis.get("aggregate_table") is not None:
        lines.append(f"Analysis aggregate rows: {len(state.analysis['aggregate_table'])}")

    if state.analysis.get("ranking_table") is not None:
        lines.append(f"Analysis ranking rows returned: {len(state.analysis['ranking_table'])}")

    if state.analysis.get("primary_display_gdf") is not None:
        lines.append(f"Analysis primary display rows: {len(state.analysis['primary_display_gdf'])}")

    if state.analysis.get("support_display_gdf") is not None:
        lines.append(f"Analysis support display rows: {len(state.analysis['support_display_gdf'])}")

    lines.append(f"Crash-to-road snap distance assumption: {SNAP_CRASH_TO_ROAD_M:.1f} m")
    lines.append(
        "Map render limits: "
        f"roads={MAX_RENDER_ROADS}, crashes={MAX_RENDER_CRASHES}, schools={MAX_RENDER_SCHOOLS}, "
        f"bus_stops={MAX_RENDER_BUS_STOPS}, crosswalks={MAX_RENDER_CROSSWALKS}, towns={MAX_RENDER_TOWNS}"
    )
    return "\n".join(lines)


# =========================================================
# 14B) RESPONSE LAYER HELPERS
# =========================================================
def _safe_scalar_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if pd.isna(value):
        return None
    if isinstance(value, (datetime, date)):
        return str(value)
    return str(value)


def _df_preview_records(df: pd.DataFrame, limit: int = 5) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    preview = df.head(limit).copy()
    records = []
    for rec in preview.to_dict(orient="records"):
        clean = {str(k): _safe_scalar_json(v) for k, v in rec.items()}
        records.append(clean)
    return records


def _describe_spatial_constraint(sc: SpatialConstraint) -> str:
    if sc.relation == "within_distance":
        if sc.distance_m is not None:
            return f"{sc.target_role} within {float(sc.distance_m):.0f} m of {sc.reference_role}"
        return f"{sc.target_role} within distance of {sc.reference_role}"
    return f"{sc.target_role} {sc.relation} {sc.reference_role}"


def _describe_attribute_constraint(ac: AttributeConstraint) -> str:
    value_txt = ac.value
    if ac.field == DERIVED_CRASH_TIME_MINUTES:
        if ac.operator == "between" and isinstance(ac.value, (list, tuple)) and len(ac.value) == 2:
            value_txt = [minutes_to_label(ac.value[0]), minutes_to_label(ac.value[1])]
        elif ac.operator in {"eq", "gt", "gte", "lt", "lte"}:
            try:
                value_txt = minutes_to_label(int(ac.value))
            except Exception:
                value_txt = ac.value
    return f"{ac.target_role}.{ac.field} {ac.operator} {value_txt}"


def _describe_relation_constraint(rc: RelationConstraint) -> str:
    if rc.relation == "snap_match":
        if rc.distance_m is not None:
            return f"{rc.source_role} snap_match {rc.target_role} within {float(rc.distance_m):.1f} m"
        return f"{rc.source_role} snap_match {rc.target_role}"
    return f"{rc.source_role} {rc.relation} {rc.target_role}"


def _describe_ranking(ranking: Optional[RankingSpec]) -> Optional[str]:
    if ranking is None:
        return None
    return f"{ranking.order} {ranking.top_n} by {ranking.metric} on {ranking.target_role}"


def _detect_empty_result_from_state(state: ExecutionState) -> bool:
    total_selected = sum(int(rd.selected_count or 0) for rd in state.role_data.values())
    ranking_df = state.analysis.get("ranking_table")
    aggregate_df = state.analysis.get("aggregate_table")

    if ranking_df is not None:
        return len(ranking_df) == 0
    if aggregate_df is not None:
        return len(aggregate_df) == 0
    return total_selected == 0


def _collect_response_warnings(state: ExecutionState) -> list[str]:
    warnings = []

    for role, rd in state.role_data.items():
        if rd.gdf is not None and rd.render_limit is not None and rd.selected_count > len(rd.gdf):
            warnings.append(
                f"Rendered {len(rd.gdf)} of {rd.selected_count} selected rows for {rd.entity} ({role}) due to render limit."
            )

    if state.analysis.get("ranking_table") is not None and state.analysis.get("primary_display_gdf") is None:
        warnings.append("Ranking table exists but primary ranked display geometry was not materialized.")

    if state.analysis.get("aggregate_table") is not None and len(state.tables) == 0:
        warnings.append("Aggregate analysis exists but no tables were prepared for display.")

    return warnings


def _collect_response_limitations(state: ExecutionState) -> list[str]:
    limitations = [
        "Narrative answer is grounded only in structured execution outputs.",
        "Response layer does not perform planning, SQL generation, or reinterpretation beyond the structured bundle.",
    ]

    sf = state.semantic_frame
    if sf is not None and "map" in sf.outputs and state.map_object is None:
        limitations.append("Map output was requested but no map object was produced.")

    return limitations


def build_response_bundle(
    state: ExecutionState,
    plan: DAGPlan,
    success: bool,
    error_message: Optional[str] = None,
    failure_type: Optional[str] = None,
) -> StructuredResponseBundle:
    sf = state.semantic_frame
    if sf is None:
        raise ValueError("Cannot build response bundle without semantic frame.")

    primary = next((t for t in sf.targets if t.role == "primary"), None)
    support = next((t for t in sf.targets if t.role == "support"), None)
    scope = next((t for t in sf.targets if t.role == "scope"), None)

    role_summaries = []
    selected_counts_by_role = {}
    for role, rd in state.role_data.items():
        render_row_count = 0
        if rd.gdf is not None:
            try:
                render_row_count = len(rd.gdf)
            except Exception:
                render_row_count = 0

        role_summaries.append(
            ResponseBundleRoleSummary(
                role=role,
                entity=rd.entity,
                selected_count=int(rd.selected_count or 0),
                selected_names=list(rd.selected_names),
                has_materialized_gdf=rd.gdf is not None,
                render_row_count=render_row_count,
                label_field=rd.label_field,
                primary_key=rd.primary_key,
            )
        )
        selected_counts_by_role[role] = int(rd.selected_count or 0)

    match_summaries = []
    for name, meta in state.match_metadata_by_name.items():
        match_summaries.append(
            ResponseBundleMatchSummary(
                name=name,
                left_role=meta.left_role,
                right_role=meta.right_role,
                left_entity=meta.left_entity,
                right_entity=meta.right_entity,
                relation=meta.relation,
                distance_m=meta.distance_m,
                matched_left_count=meta.matched_left_count,
                matched_right_count=meta.matched_right_count,
                pair_count=meta.pair_count,
            )
        )

    table_summaries = []
    for t in state.tables:
        table_summaries.append(
            ResponseBundleTableSummary(
                name=t.name,
                row_count=int(len(t.df)),
                columns=[str(c) for c in t.df.columns.tolist()],
                preview_rows=_df_preview_records(t.df, limit=5),
            )
        )

    aggregate_df = state.analysis.get("aggregate_table")
    ranking_df = state.analysis.get("ranking_table")

    empty_result = _detect_empty_result_from_state(state)
    warnings = _collect_response_warnings(state)
    limitations = _collect_response_limitations(state)

    empty_note = None
    if empty_result and success:
        empty_note = "Execution completed but no rows matched the current filters and spatial conditions."

    downloads = [
        ResponseBundleDownloadItem(
            name="table_export",
            kind="table",
            available=False,
            filename=None,
            note="Placeholder only. No download file is generated yet.",
        ),
        ResponseBundleDownloadItem(
            name="map_export",
            kind="map",
            available=False,
            filename=None,
            note="Placeholder only. Map download metadata can be added later.",
        ),
    ]

    anchors = []
    for r in sf.references:
        if r.name:
            anchors.append(f"{r.entity}({r.role})={r.name}")

    partial_success = bool(success and warnings)

    bundle = StructuredResponseBundle(
        bundle_version=RESPONSE_LAYER_VERSION,
        original_user_prompt=state.user_prompt,
        supported=bool(sf.supported),
        success=bool(success),
        semantic_frame=asdict(sf),
        execution_plan=[asdict(step) for step in plan.steps],
        primary_entity=primary.entity if primary else None,
        primary_role=primary.role if primary else None,
        support_entity=support.entity if support else None,
        support_role=support.role if support else None,
        scope_entity=scope.entity if scope else None,
        scope_names=list(scope.names) if scope else [],
        anchor_descriptions=anchors,
        spatial_constraint_descriptions=[_describe_spatial_constraint(sc) for sc in sf.spatial_constraints],
        attribute_constraint_descriptions=[_describe_attribute_constraint(ac) for ac in sf.attribute_constraints],
        relation_descriptions=[_describe_relation_constraint(rc) for rc in sf.relations],
        ranking_description=_describe_ranking(sf.ranking),
        selected_counts_by_role=selected_counts_by_role,
        role_summaries=role_summaries,
        match_summaries=match_summaries,
        aggregate_exists=aggregate_df is not None,
        aggregate_row_count=int(len(aggregate_df)) if aggregate_df is not None else 0,
        ranking_exists=ranking_df is not None,
        ranking_row_count=int(len(ranking_df)) if ranking_df is not None else 0,
        map_exists=state.map_object is not None,
        table_count=len(state.tables),
        tables=table_summaries,
        empty_result=bool(empty_result),
        partial_success=partial_success,
        warnings=warnings,
        limitations=limitations,
        empty_result_note=empty_note,
        failure_type=failure_type,
        error_message=error_message,
        deterministic_summary=state.summary_text,
        downloads=downloads,
        conversation_context=ResponseBundleConversationContext(
            prior_context_available=False,
            compact_context=None,
        ),
    )
    return bundle


def build_failure_response_bundle(
    user_prompt: str,
    raw_semantic_frame: Optional[dict],
    validated_semantic_frame: Optional[dict],
    plan: Optional[DAGPlan],
    error_message: str,
    failure_type: str,
) -> StructuredResponseBundle:
    semantic_frame_dict = validated_semantic_frame or raw_semantic_frame or {}
    plan_list = [asdict(step) for step in plan.steps] if plan is not None else []

    limitations = [
        "Narrative answer is grounded only in structured execution outputs.",
        "Because execution failed, no result interpretation beyond the failure details is available.",
    ]

    return StructuredResponseBundle(
        bundle_version=RESPONSE_LAYER_VERSION,
        original_user_prompt=user_prompt,
        supported=bool(semantic_frame_dict.get("supported", False)) if isinstance(semantic_frame_dict, dict) else False,
        success=False,
        semantic_frame=semantic_frame_dict if isinstance(semantic_frame_dict, dict) else {},
        execution_plan=plan_list,
        primary_entity=None,
        primary_role=None,
        support_entity=None,
        support_role=None,
        scope_entity=None,
        scope_names=[],
        anchor_descriptions=[],
        spatial_constraint_descriptions=[],
        attribute_constraint_descriptions=[],
        relation_descriptions=[],
        ranking_description=None,
        selected_counts_by_role={},
        role_summaries=[],
        match_summaries=[],
        aggregate_exists=False,
        aggregate_row_count=0,
        ranking_exists=False,
        ranking_row_count=0,
        map_exists=False,
        table_count=0,
        tables=[],
        empty_result=False,
        partial_success=False,
        warnings=[],
        limitations=limitations,
        empty_result_note=None,
        failure_type=failure_type,
        error_message=error_message,
        deterministic_summary=None,
        downloads=[
            ResponseBundleDownloadItem(
                name="table_export",
                kind="table",
                available=False,
                filename=None,
                note="Placeholder only. No download file is generated yet.",
            ),
            ResponseBundleDownloadItem(
                name="map_export",
                kind="map",
                available=False,
                filename=None,
                note="Placeholder only. Map download metadata can be added later.",
            ),
        ],
        conversation_context=ResponseBundleConversationContext(
            prior_context_available=False,
            compact_context=None,
        ),
    )


def build_response_agent_prompt(bundle: StructuredResponseBundle) -> str:
    payload = asdict(bundle)
    payload_json = json.dumps(payload, indent=2, ensure_ascii=False)

    return f"""
You are a downstream response agent for a Road Safety Assistant.

You must produce a short grounded user-facing answer using ONLY the structured bundle below.

Hard rules:
- Do not invent counts, findings, entities, locations, or interpretations.
- Do not generate SQL.
- Do not describe hidden execution logic beyond what is explicitly present.
- Do not claim success if success=false.
- If empty_result=true, say clearly that no matching rows were found.
- If failure_type or error_message is present, explain the failure plainly.
- Mention important filters only when they appear in the bundle.
- Mention scope names if present.
- Mention ranking only if ranking_exists=true or ranking_description is present.
- Mention warnings or limitations briefly when relevant.
- Keep the answer concise, clear, and grounded.
- Use plain paragraphs, not JSON.

Structured bundle:
{payload_json}
""".strip()


def generate_narrative_answer(
    bundle: StructuredResponseBundle,
    response_llm: OptionalGeminiClient,
) -> NarrativeResponse:
    if not response_llm.enabled:
        reason = response_llm.init_error or "Response-agent API key is missing."
        return NarrativeResponse(
            available=False,
            used_response_agent=False,
            text=None,
            fallback_reason=reason,
            raw_text=None,
        )

    prompt = build_response_agent_prompt(bundle)
    text = response_llm.generate_text(prompt, temperature=0.0).strip()

    if not text:
        return NarrativeResponse(
            available=False,
            used_response_agent=False,
            text=None,
            fallback_reason="Response agent returned empty text.",
            raw_text=text,
        )

    if len(text) > RESPONSE_AGENT_MAX_CHARS:
        text = text[:RESPONSE_AGENT_MAX_CHARS].rstrip() + "..."

    return NarrativeResponse(
        available=True,
        used_response_agent=True,
        text=text,
        fallback_reason=None,
        raw_text=text,
    )


# =========================================================
# 15) EXECUTION ENGINE
# =========================================================
class RoadSafetyExecutor:
    def __init__(self, conn, road_geom_col: str):
        self.conn = conn
        self.road_geom_col = road_geom_col

    def execute(self, state: ExecutionState, plan: DAGPlan) -> ExecutionState:
        state.road_geom_col = self.road_geom_col

        # Iterate nodes in topological order. Sequential dispatch — the DAG
        # structure encodes dependencies explicitly, but parallel execution
        # is intentionally out of scope for this refactor (psycopg2 connection
        # sharing has its own concerns and the architectural extension can be
        # made later without changing this contract).
        for idx, nid in enumerate(plan.order, start=1):
            node = plan.nodes[nid]
            state.debug[f"step_{idx}_{node.op}"] = {
                "node_id": node.node_id,
                "params": node.params,
                "inputs": list(node.inputs),
            }
            handler = getattr(self, f"_handle_{node.op}", None)
            if handler is None:
                if node.op == "Unsupported":
                    raise ValueError("Unsupported query.")
                raise ValueError(f"No handler implemented for node op: {node.op}")
            state = handler(state, node.params)

        return state

    def _get_dataset_spec_by_role(self, state: ExecutionState, role: str) -> DatasetSpec:
        if role not in state.dataset_specs_by_role:
            raise ValueError(f"No dataset spec loaded for role: {role}")
        return state.dataset_specs_by_role[role]

    def _get_reference_df_by_role(self, state: ExecutionState, role: str) -> pd.DataFrame:
        if role not in state.references_by_role:
            raise ValueError(f"No reference object loaded for role: {role}")
        return state.references_by_role[role].df

    def _get_materialized_role(self, state: ExecutionState, role: str) -> RoleData:
        if role not in state.role_data:
            raise ValueError(f"Role not materialized: {role}")
        return state.role_data[role]

    def _replace_role_sql(self, state: ExecutionState, role: str, new_sql: str, new_params: dict):
        rd = self._get_materialized_role(state, role)
        rd.sql_base = new_sql
        rd.params = new_params
        state.role_data[role] = rd

    def _materialize_current_role(self, state: ExecutionState, role: str) -> ExecutionState:
        rd = self._get_materialized_role(state, role)
        gdf, count = materialize_role_to_gdf(self.conn, rd)
        rd.gdf = gdf
        rd.selected_count = count
        state.role_data[role] = rd
        return state

    def _get_role_sql_and_params(self, state: ExecutionState, role: str) -> tuple[str, dict]:
        rd = self._get_materialized_role(state, role)
        return rd.sql_base, dict(rd.params)

    def _get_reference_sql_and_params(self, state: ExecutionState, role: str) -> tuple[str, dict]:
        ref_df = self._get_reference_df_by_role(state, role)
        return build_reference_sql_from_df(ref_df)

    def _apply_generic_set_filter_to_role(
        self,
        state: ExecutionState,
        target_role: str,
        reference_role: str,
        relation: str,
        distance_m: Optional[float],
    ) -> ExecutionState:
        rd = self._get_materialized_role(state, target_role)
        target_sql = rd.sql_base
        target_params = dict(rd.params)

        if reference_role == "anchor":
            ref_sql, ref_params = self._get_reference_sql_and_params(state, "anchor")
        elif reference_role in state.role_data:
            ref_sql, ref_params = self._get_role_sql_and_params(state, reference_role)
        else:
            raise ValueError(f"Unsupported reference role for set filter: {reference_role}")

        merged_params, new_ref_sql = merge_sql_params_with_prefix(target_params, ref_params, f"{reference_role}_", ref_sql)
        new_sql = wrap_sql_with_spatial_set_filter(
            target_sql=target_sql,
            reference_sql=new_ref_sql,
            relation=relation,
            distance_m=distance_m,
            target_geom_col="geom",
            reference_geom_col="geom",
        )

        rd.sql_base = new_sql
        rd.params = merged_params
        rd.applied_spatial_constraints.append(
            {
                "target_role": target_role,
                "reference_role": reference_role,
                "relation": relation,
                "distance_m": distance_m,
            }
        )
        state.role_data[target_role] = rd
        return state

    def _record_match_metadata(
        self,
        state: ExecutionState,
        left_role: str,
        right_role: str,
        relation: str,
        distance_m: Optional[float],
        left_tag_field: Optional[str],
        right_tag_field: Optional[str],
        matched_left_count: Optional[int],
        matched_right_count: Optional[int],
        pair_count: Optional[int],
    ) -> ExecutionState:
        left_ds = self._get_dataset_spec_by_role(state, left_role)
        right_ds = self._get_dataset_spec_by_role(state, right_role)
        match_name = f"{left_role}_to_{right_role}_{relation}"
        state.match_metadata_by_name[match_name] = MatchMetadata(
            left_role=left_role,
            right_role=right_role,
            left_entity=left_ds.entity,
            right_entity=right_ds.entity,
            relation=relation,
            distance_m=distance_m,
            left_key_field=get_dataset_key_field(left_ds),
            right_key_field=get_dataset_key_field(right_ds),
            left_tag_field=left_tag_field,
            right_tag_field=right_tag_field,
            matched_left_count=matched_left_count,
            matched_right_count=matched_right_count,
            pair_count=pair_count,
        )
        return state

    def _debug_scope_name_resolution(self, state: ExecutionState, role: str, names: list[str]) -> ExecutionState:
        ds = self._get_dataset_spec_by_role(state, role)
        if ds.entity != "Town":
            return state

        debug_payload = {
            "input_names": list(names),
            "normalized_inputs": normalize_town_names_debug_payload(names, ds.name_match_strip_suffixes),
        }

        try:
            where_sql, where_params = build_name_filter_sql(
                dataset_spec=ds,
                alias="q",
                names=names,
                param_prefix=f"{role}_debug_name",
            )
            preview_sql = f"""
            SELECT
              q.{_quote_col(TOWN_NAME_COL)} AS raw_name,
              {build_town_name_match_expr('q')} AS core_name
            FROM ({make_base_role_sql(ds, alias='x')[0]}) q
            WHERE {where_sql}
            ORDER BY q.{_quote_col(TOWN_NAME_COL)}
            """
            preview_df = fetch_df(self.conn, preview_sql, where_params)
            debug_payload["matched_rows"] = preview_df.to_dict(orient="records")
            debug_payload["matched_count"] = int(len(preview_df))
        except Exception as e:
            debug_payload["resolution_error"] = str(e)

        state.debug.setdefault("town_name_resolution", {})[role] = debug_payload
        return state

    def _handle_LoadDatasetRegistry(self, state: ExecutionState, params: dict) -> ExecutionState:
        state.debug["dataset_registry_entities"] = list(DATASET_REGISTRY.keys())
        return state

    def _handle_LoadEntitySpec(self, state: ExecutionState, params: dict) -> ExecutionState:
        role = params["role"]
        entity = params["entity"]
        names = normalize_names_list(params.get("names"))

        if entity not in DATASET_REGISTRY:
            raise ValueError(f"Entity not found in registry: {entity}")

        ds = DATASET_REGISTRY[entity]
        if entity == "Road":
            ds = DatasetSpec(
                entity=ds.entity,
                table=ds.table,
                geometry_column=self.road_geom_col,
                geometry_family=ds.geometry_family,
                primary_key=ds.primary_key,
                label_field=ds.label_field,
                display_fields=list(ds.display_fields),
                fields=dict(ds.fields),
                derived_fields=dict(ds.derived_fields),
                relation_capabilities=list(ds.relation_capabilities),
                scope_capable=ds.scope_capable,
                name_match_field=ds.name_match_field,
                name_match_strip_suffixes=list(ds.name_match_strip_suffixes),
                display_geometry_mode=ds.display_geometry_mode,
            )

        state.role_bindings[role] = RoleBinding(role=role, entity=entity)
        state.dataset_specs_by_role[role] = ds
        state.debug.setdefault("role_target_names", {})[role] = names
        return state

    def _handle_ResolveReference(self, state: ExecutionState, params: dict) -> ExecutionState:
        ref = ReferenceSpec(entity=params["entity"], role=params["role"], name=params["name"])
        geocode_sel = state.debug.get("_geocode_selection")
        rr = resolve_reference(self.conn, ref, geocode_selection=geocode_sel)
        state.resolver_results[ref.role] = rr
        return state

    def _handle_BuildReferenceObject(self, state: ExecutionState, params: dict) -> ExecutionState:
        role = params["role"]
        rr = state.resolver_results[role]

        if rr.entity == "School":
            df = pd.DataFrame(
                {
                    "role": [rr.role],
                    "name": [rr.name],
                    "display_name": [rr.display_name],
                    "reference_type": ["school"],
                    "dist_m": [rr.dist_m],
                    "wkt": [rr.wkt_26986],
                }
            )
        else:
            df = make_reference_row_from_point(role, rr.name, rr.lat, rr.lon, rr.display_name)

        state.references_by_role[role] = ReferenceObject(role=role, entity=rr.entity, df=df)
        return state

    def _handle_InitializeRoleQuery(self, state: ExecutionState, params: dict) -> ExecutionState:
        role = params["role"]
        ds = self._get_dataset_spec_by_role(state, role)
        sql, sql_params = make_base_role_sql(ds, alias="x")

        selected_names = normalize_names_list(state.debug.get("role_target_names", {}).get(role, []))

        state.role_data[role] = RoleData(
            role=role,
            entity=ds.entity,
            table=ds.table,
            geometry_column="geom",
            geometry_family=ds.geometry_family,
            primary_key=ds.primary_key,
            label_field=ds.label_field,
            sql_base=sql,
            params=dict(sql_params),
            render_limit=get_default_render_limit(ds.entity),
            display_fields=list(ds.display_fields),
            selected_names=selected_names,
        )
        return state

    def _handle_ApplyNameFilter(self, state: ExecutionState, params: dict) -> ExecutionState:
        role = params["role"]
        names = normalize_names_list(params.get("names"))
        if not names:
            return state

        ds = self._get_dataset_spec_by_role(state, role)
        rd = self._get_materialized_role(state, role)

        where_sql, where_params = build_name_filter_sql(
            dataset_spec=ds,
            alias="q",
            names=names,
            param_prefix=f"{role}_name",
        )
        rd.sql_base = wrap_sql_with_where(rd.sql_base, where_sql)
        rd.params = {**rd.params, **where_params}
        rd.selected_names = names
        rd.applied_name_filters.append({"names": names})

        if ds.entity == "Town":
            state = self._debug_scope_name_resolution(state, role, names)

        state.role_data[role] = rd
        return state

    def _handle_ApplySpatialConstraint(self, state: ExecutionState, params: dict) -> ExecutionState:
        target_role = params["target_role"]
        relation = params["relation"]
        reference_role = params["reference_role"]
        distance_m = params.get("distance_m")

        if relation in GENERIC_SET_MATCH_RELATIONS:
            return self._apply_generic_set_filter_to_role(
                state=state,
                target_role=target_role,
                reference_role=reference_role,
                relation=relation,
                distance_m=distance_m,
            )

        if relation == "nearest_to":
            raise ValueError("nearest_to is not implemented in execution yet.")

        raise ValueError(f"Unsupported spatial relation currently: {relation}")

    def _handle_ApplyScopeConstraint(self, state: ExecutionState, params: dict) -> ExecutionState:
        target_role = params["target_role"]
        scope_role = params["scope_role"]
        relation = params.get("relation", SCOPE_FILTER_RELATION_DEFAULT)

        scope_rd = self._get_materialized_role(state, scope_role)
        scope_ds = self._get_dataset_spec_by_role(state, scope_role)

        if scope_ds.geometry_family != "polygon":
            raise ValueError(f"Scope role must be polygon-based. Got {scope_ds.entity} / {scope_ds.geometry_family}")

        if relation not in {"intersects", "contains"}:
            raise ValueError(f"Unsupported scope relation: {relation}")

        if scope_ds.entity == "Town":
            state.debug.setdefault("scope_debug", {})[scope_role] = {
                "selected_scope_count_sql": scope_rd.sql_base,
                "selected_scope_names": list(scope_rd.selected_names),
            }

        return self._apply_generic_set_filter_to_role(
            state=state,
            target_role=target_role,
            reference_role=scope_role,
            relation=relation,
            distance_m=None,
        )

    def _handle_ApplyAttributeConstraints(self, state: ExecutionState, params: dict) -> ExecutionState:
        target_role = params["target_role"]
        ds = self._get_dataset_spec_by_role(state, target_role)
        rd = self._get_materialized_role(state, target_role)

        role_constraints = [ac for ac in state.semantic_frame.attribute_constraints if ac.target_role == target_role]
        if not role_constraints:
            return state

        where_parts = ["1=1"]
        new_params = dict(rd.params)

        for idx, ac in enumerate(role_constraints, start=1):
            w, p = build_attribute_constraint_sql(ds, "q", ac, f"{target_role}_attr_{idx}")
            where_parts.append(w)
            new_params.update(p)
            rd.applied_attribute_constraints.append(asdict(ac))

        rd.sql_base = wrap_sql_with_where(rd.sql_base, " AND ".join(where_parts))
        rd.params = new_params
        state.role_data[target_role] = rd
        return state

    def _handle_ApplyRelationConstraint(self, state: ExecutionState, params: dict) -> ExecutionState:
        relation = params["relation"]
        source_role = params["source_role"]
        target_role = params["target_role"]
        distance_m = float(params["distance_m"])

        source_rd = self._get_materialized_role(state, source_role)
        target_rd = self._get_materialized_role(state, target_role)

        filtered_source_sql = wrap_sql_with_relation_join(
            source_sql=source_rd.sql_base,
            target_sql=target_rd.sql_base,
            source_geom_col="geom",
            target_geom_col="geom",
            relation=relation,
            distance_m=distance_m,
            keep_side="source",
        )
        filtered_target_sql = wrap_sql_with_relation_join(
            source_sql=source_rd.sql_base,
            target_sql=target_rd.sql_base,
            source_geom_col="geom",
            target_geom_col="geom",
            relation=relation,
            distance_m=distance_m,
            keep_side="target",
        )

        merged_params_source = dict(source_rd.params)
        for k, v in target_rd.params.items():
            merged_params_source[f"t_{k}"] = v
            filtered_source_sql = filtered_source_sql.replace(f"%({k})s", f"%(t_{k})s")

        merged_params_target = dict(target_rd.params)
        for k, v in source_rd.params.items():
            merged_params_target[f"s_{k}"] = v
            filtered_target_sql = filtered_target_sql.replace(f"%({k})s", f"%(s_{k})s")

        source_rd.sql_base = filtered_source_sql
        source_rd.params = merged_params_source
        source_rd.applied_relations.append(params)

        target_rd.sql_base = filtered_target_sql
        target_rd.params = merged_params_target
        target_rd.applied_relations.append(params)

        state.role_data[source_role] = source_rd
        state.role_data[target_role] = target_rd
        return state

    def _handle_MatchSpatialSets(self, state: ExecutionState, params: dict) -> ExecutionState:
        left_role = params["left_role"]
        right_role = params["right_role"]
        relation = params["relation"]
        distance_m = params.get("distance_m")
        left_keep = bool(params.get("left_keep", True))
        right_keep = bool(params.get("right_keep", True))

        left_rd = self._get_materialized_role(state, left_role)
        right_rd = self._get_materialized_role(state, right_role)
        left_ds = self._get_dataset_spec_by_role(state, left_role)
        right_ds = self._get_dataset_spec_by_role(state, right_role)

        left_sql = left_rd.sql_base
        right_sql = right_rd.sql_base
        left_params = dict(left_rd.params)
        right_params = dict(right_rd.params)

        left_key = get_dataset_key_field(left_ds)
        right_key = get_dataset_key_field(right_ds)
        left_tag_field = f"matched_{left_key}" if left_key else None
        right_tag_field = f"matched_{right_key}" if right_key else None

        if left_keep:
            new_left_sql = wrap_sql_with_spatial_set_filter(
                target_sql=left_sql,
                reference_sql=right_sql,
                relation=relation,
                distance_m=distance_m,
                target_geom_col="geom",
                reference_geom_col="geom",
            )
            merged_left_params = dict(left_params)
            for k, v in right_params.items():
                merged_left_params[f"r_{k}"] = v
                new_left_sql = new_left_sql.replace(f"%({k})s", f"%(r_{k})s")
            left_rd.sql_base = new_left_sql
            left_rd.params = merged_left_params
            left_rd.applied_spatial_constraints.append(
                {
                    "target_role": left_role,
                    "reference_role": right_role,
                    "relation": relation,
                    "distance_m": distance_m,
                    "match_step": True,
                }
            )

        if right_keep:
            new_right_sql = wrap_sql_with_spatial_set_filter(
                target_sql=right_sql,
                reference_sql=left_sql,
                relation=relation,
                distance_m=distance_m,
                target_geom_col="geom",
                reference_geom_col="geom",
            )
            merged_right_params = dict(right_params)
            for k, v in left_params.items():
                merged_right_params[f"l_{k}"] = v
                new_right_sql = new_right_sql.replace(f"%({k})s", f"%(l_{k})s")
            right_rd.sql_base = new_right_sql
            right_rd.params = merged_right_params
            right_rd.applied_spatial_constraints.append(
                {
                    "target_role": right_role,
                    "reference_role": left_role,
                    "relation": relation,
                    "distance_m": distance_m,
                    "match_step": True,
                }
            )

        pair_sql = build_generic_pair_count_sql(
            left_sql=left_sql,
            right_sql=right_sql,
            relation=relation,
            distance_m=distance_m,
        )
        pair_params = dict(left_params)
        for k, v in right_params.items():
            pair_params[f"r_{k}"] = v
            pair_sql = pair_sql.replace(f"%({k})s", f"%(r_{k})s")
        pair_count = int(fetch_scalar(self.conn, pair_sql, pair_params) or 0)

        state.role_data[left_role] = left_rd
        state.role_data[right_role] = right_rd

        matched_left_count = int(fetch_scalar(self.conn, f"SELECT COUNT(*) FROM ({left_rd.sql_base}) q", left_rd.params) or 0) if left_keep else None
        matched_right_count = int(fetch_scalar(self.conn, f"SELECT COUNT(*) FROM ({right_rd.sql_base}) q", right_rd.params) or 0) if right_keep else None

        state = self._record_match_metadata(
            state=state,
            left_role=left_role,
            right_role=right_role,
            relation=relation,
            distance_m=distance_m,
            left_tag_field=left_tag_field,
            right_tag_field=right_tag_field,
            matched_left_count=matched_left_count,
            matched_right_count=matched_right_count,
            pair_count=pair_count,
        )
        return state

    def _handle_Aggregate(self, state: ExecutionState, params: dict) -> ExecutionState:
        metric = params["metric"]
        group_role = params["group_role"]
        measure_role = params["measure_role"]
        relation = params["relation"]
        distance_m = params.get("distance_m")
        output_name = params.get("output_name", "aggregate_table")
        value_column = params.get("value_column", "metric_value")

        group_ds = self._get_dataset_spec_by_role(state, group_role)
        measure_ds = self._get_dataset_spec_by_role(state, measure_role)

        if metric != "crash_count":
            raise ValueError(f"Unsupported aggregate request: metric={metric}")

        group_rd = self._get_materialized_role(state, group_role)
        measure_rd = self._get_materialized_role(state, measure_role)

        agg_df, group_gdf, measure_gdf, matched_measure_count = execute_generic_count_aggregation(
            conn=self.conn,
            group_sql=group_rd.sql_base,
            group_params=group_rd.params,
            group_entity=group_ds.entity,
            measure_sql=measure_rd.sql_base,
            measure_params=measure_rd.params,
            measure_entity=measure_ds.entity,
            relation=relation,
            distance_m=distance_m,
            value_column=value_column,
        )

        state = set_analysis_table(state, output_name, agg_df)
        state.analysis["primary_display_gdf"] = None
        state.analysis["support_display_gdf"]
        state.debug["aggregate_metric"] = metric
        state.debug["aggregate_group_role"] = group_role
        state.debug["aggregate_measure_role"] = measure_role
        state.debug["aggregation_distance_m"] = distance_m
        state.debug["aggregate_output_name"] = output_name
        state.debug["aggregate_group_display_count"] = len(group_gdf)
        state.debug["aggregate_measure_display_count"] = matched_measure_count

        state.debug["aggregate_preview_group_entity"] = group_ds.entity
        state.debug["aggregate_preview_measure_entity"] = measure_ds.entity

        state.debug["_aggregate_group_gdf"] = group_gdf
        state.debug["_aggregate_measure_gdf"] = measure_gdf
        return state

    def _handle_Rank(self, state: ExecutionState, params: dict) -> ExecutionState:
        source_table = params.get("source_table", "aggregate_table")
        target_role = params["target_role"]
        order = params["order"]
        top_n = int(params["top_n"])
        ascending = order == "lowest"

        source_df = get_analysis_table(state, source_table)
        if source_df is None:
            raise ValueError(f"No aggregate table available for ranking: {source_table}")

        ranked = source_df.copy()
        if "group_key" not in ranked.columns:
            raise ValueError("Ranking source table must contain group_key.")
        if "group_label" not in ranked.columns:
            ranked["group_label"] = ranked["group_key"]

        metric_cols = [c for c in ranked.columns if c not in {"group_key", "group_label"}]
        if not metric_cols:
            raise ValueError("No metric column found for ranking.")
        value_column = metric_cols[0]

        ranked = ranked.sort_values(["group_key"]).copy()
        ranked = ranked.sort_values([value_column, "group_label"], ascending=[ascending, True]).head(top_n).reset_index(drop=True)

        state = set_analysis_table(state, "ranking_table", ranked)

        keep_values = get_rank_keep_values_from_table(ranked)

        primary_role = get_primary_role_name(state)
        support_role = get_support_role_name(state)

        primary_gdf = state.debug.get("_aggregate_group_gdf")
        support_gdf = state.debug.get("_aggregate_measure_gdf")

        if primary_gdf is None and target_role in state.role_data:
            primary_gdf = state.role_data[target_role].gdf

        if primary_gdf is not None and not primary_gdf.empty:
            target_rd = state.role_data.get(target_role)
            keep_field = get_rank_keep_field_for_roledata(target_rd) if target_rd is not None else None
            if keep_field and keep_field in primary_gdf.columns:
                primary_gdf = primary_gdf[primary_gdf[keep_field].isin(keep_values)].copy()
            primary_gdf = deduplicate_gdf_by_entity_pk(
                primary_gdf,
                target_rd.entity if target_rd is not None else state.dataset_specs_by_role[target_role].entity
            )

        if support_gdf is None and support_role and support_role in state.role_data:
            support_gdf = state.role_data[support_role].gdf

        if support_gdf is not None and not support_gdf.empty and target_role in state.role_data:
            target_entity = state.role_data[target_role].entity
            match_label = get_match_tag_field_for_entity(target_entity)
            if match_label and match_label in support_gdf.columns:
                support_gdf = support_gdf[support_gdf[match_label].isin(keep_values)].copy()
            support_entity = state.role_data[support_role].entity if support_role in state.role_data else None
            if support_entity:
                support_gdf = deduplicate_gdf_by_entity_pk(support_gdf, support_entity)

        if target_role == primary_role:
            state.analysis["primary_display_gdf"] = primary_gdf
            if support_role is not None:
                state.analysis["support_display_gdf"] = support_gdf
        elif target_role == support_role:
            state.analysis["support_display_gdf"] = primary_gdf
            if primary_role is not None:
                state.analysis["primary_display_gdf"] = support_gdf
        else:
            state.analysis["primary_display_gdf"] = primary_gdf
            state.analysis["support_display_gdf"] = support_gdf

        state.debug["ranking_value_column"] = value_column
        state.debug["ranking_keep_values"] = keep_values

        # Narrow filter role GDF to only features inside the ranked group polygons
        filter_role = None
        for r in state.role_data:
            if r == "filter":
                filter_role = r
                break
        if filter_role and filter_role in state.role_data and primary_gdf is not None and not primary_gdf.empty:
            filter_rd = state.role_data[filter_role]
            filter_ds = self._get_dataset_spec_by_role(state, filter_role)

            # Build SQL to narrow filter entity to within ranked town polygons
            ranked_town_names = list(keep_values)
            if ranked_town_names and filter_rd.sql_base:
                town_ds = DATASET_REGISTRY.get("Town")
                if town_ds is not None:
                    # Build a subquery for just the ranked towns
                    name_params = {}
                    name_placeholders = []
                    for i, tn in enumerate(ranked_town_names):
                        pk = f"_rtn_{i}"
                        name_params[pk] = str(tn)
                        name_placeholders.append(f"%({pk})s")

                    ranked_towns_sql = f"""
                    SELECT * FROM {town_ds.table}
                    WHERE {_quote_col(TOWN_NAME_COL)} IN ({", ".join(name_placeholders)})
                    """

                    # Spatially filter the filter role to inside ranked towns
                    narrowed_filter_sql = wrap_sql_with_spatial_set_filter(
                        target_sql=filter_rd.sql_base,
                        reference_sql=ranked_towns_sql,
                        relation="intersects",
                        distance_m=None,
                        target_geom_col="geom",
                        reference_geom_col="geom",
                    )

                    merged_params = dict(filter_rd.params)
                    for k, v in name_params.items():
                        merged_params[k] = v

                    filter_rd.sql_base = narrowed_filter_sql
                    filter_rd.params = merged_params
                    state.role_data[filter_role] = filter_rd

        return state

    def _handle_MaterializeRole(self, state: ExecutionState, params: dict) -> ExecutionState:
        role = params["role"]
        return self._materialize_current_role(state, role)

    def _handle_PrepareTable(self, state: ExecutionState, params: dict) -> ExecutionState:
        ranking_df = state.analysis.get("ranking_table")
        aggregate_df = state.analysis.get("aggregate_table")

        if ranking_df is not None:
            state.tables.append(TableObject(name="ranking_table", df=ranking_df))
        elif aggregate_df is not None:
            state.tables.append(TableObject(name="aggregate_table", df=aggregate_df))

        if state.match_metadata_by_name:
            match_df = pd.DataFrame([asdict(v) for v in state.match_metadata_by_name.values()])
            state.tables.append(TableObject(name="match_metadata", df=match_df))

        for role, ref_obj in state.references_by_role.items():
            state.tables.append(
                TableObject(
                    name=f"reference_{role}",
                    df=ref_obj.df.drop(columns=["wkt"], errors="ignore"),
                )
            )

        if ranking_df is None:
            for role, rd in state.role_data.items():
                if rd.gdf is None or rd.gdf.empty:
                    continue

                if rd.entity in {"School", "Crash", "Road", "BusStop", "Crosswalk", "Town"}:
                    df = rd.gdf.drop(columns=["geometry"], errors="ignore").copy()
                    state.tables.append(TableObject(name=f"{role}_{rd.entity.lower()}", df=df))

        return state

    def _handle_PrepareSummary(self, state: ExecutionState, params: dict) -> ExecutionState:
        state.summary_text = summarize_result(state)
        return state

    def _handle_PrepareMap(self, state: ExecutionState, params: dict) -> ExecutionState:
        state.map_object = create_map_from_state(state)
        return state


# =========================================================
# 16) ASSISTANT INTERFACE
# =========================================================


class RoadSafetyAssistant:
    """
    Road Safety Assistant.

    Pluggable LLM provider (Gemini or OpenAI) for the interpretation layer,
    with an optional response-layer LLM for narrative summaries.
    """
    def __init__(
        self,
        conn,
        # Preferred unified kwargs
        llm_provider: str = "gemini",
        llm_api_key: str = "",
        llm_model: str = "",
        response_provider: str = "gemini",
        response_api_key: Optional[str] = None,
        response_model: str = "",
        # Legacy Gemini-only kwargs (backwards compat)
        gemini_api_key: Optional[str] = None,
        gemini_model: Optional[str] = None,
        response_gemini_api_key: Optional[str] = None,
        response_gemini_model: Optional[str] = None,
    ):
        # Resolve legacy kwargs
        if gemini_api_key and not llm_api_key:
            llm_api_key = gemini_api_key
            llm_provider = "gemini"
        if gemini_model and not llm_model:
            llm_model = gemini_model
        if response_gemini_api_key and not response_api_key:
            response_api_key = response_gemini_api_key
        if response_gemini_model and not response_model:
            response_model = response_gemini_model

        # Apply model defaults
        if not llm_model:
            llm_model = OPENAI_MODEL if llm_provider.lower() == "openai" else GEMINI_MODEL
        if not response_model:
            response_model = OPENAI_MODEL if response_provider.lower() == "openai" else RESPONSE_GEMINI_MODEL

        self.conn = conn
        self.llm_provider = llm_provider.lower()
        self.llm_model = llm_model
        self.road_geom_col = detect_road_geom_col(conn)

        base_road = DATASET_REGISTRY["Road"]
        DATASET_REGISTRY["Road"] = DatasetSpec(
            entity=base_road.entity,
            table=base_road.table,
            geometry_column=self.road_geom_col,
            geometry_family=base_road.geometry_family,
            primary_key=base_road.primary_key,
            label_field=base_road.label_field,
            display_fields=list(base_road.display_fields),
            fields=dict(base_road.fields),
            derived_fields=dict(base_road.derived_fields),
            relation_capabilities=list(base_road.relation_capabilities),
            scope_capable=base_road.scope_capable,
            name_match_field=base_road.name_match_field,
            name_match_strip_suffixes=list(base_road.name_match_strip_suffixes),
            display_geometry_mode=base_road.display_geometry_mode,
        )

        # Primary LLM for query interpretation.
        self.llm = make_llm_client(llm_provider, llm_api_key, llm_model)

        # Optional response-layer LLM for narrative summaries.
        self.response_llm = OptionalGeminiClient(
            api_key=response_api_key or RESPONSE_GEMINI_API_KEY or "",
            model=response_model,
            provider=response_provider,
        )

        self.executor = RoadSafetyExecutor(conn, self.road_geom_col)

    def help_message(self) -> str:
        return (
            "Supported examples:\n"
            '  - "show crashes"\n'
            '  - "show roads"\n'
            '  - "show schools"\n'
            '  - "show bus stops"\n'
            '  - "show crosswalks"\n'
            '  - "show towns"\n'
            '  - "show Quincy"\n'
            '  - "show Lenox town"\n'
            '  - "show crashes in Quincy"\n'
            '  - "show roads in Lenox"\n'
            '  - "show schools in Quincy city"\n'
            '  - "show bus stops in Lenox town"\n'
            '  - "show crosswalks in Quincy"\n'
            '  - "show crashes in Quincy and Lenox"\n'
            '  - "show roads in Amherst, Hadley, and Northampton"\n'
            '  - "show roads with no sidewalk statewide"\n'
            '  - "show crashes on roads with speed limits higher than 30"\n'
            '  - "find crashes within 200m around Amherst CVS"\n'
            '  - "show roads within 500m around Boston Common"\n'
            '  - "show crosswalks around Amherst Center within 500m"\n'
            '  - "show roads with speed limits above 30 around Amherst Regional High School"\n'
            '  - "show sidewalk presence on roads around Amherst CVS"\n'
            '  - "top 20 schools by crashes within 200m"\n'
            '  - "top 10 schools by crashes within 500m in Quincy"\n'
            '  - "lowest 10 schools by crashes within 300m"\n'
            '  - "top 10 bus stops by crashes within 500m"\n'
            '  - "top 20 bus stops by crashes within 500m in Quincy city"\n'
            '  - "show fatal crashes between January 6 2025 and February 5 2025"\n'
            '  - "show crashes between 01 06 2025 and 02 05 2025"\n'
            '  - "show crashes between 12 and 14"\n'
            '  - "show fatal crashes between 7:30 AM and 9:00 AM"\n'
            '  - "show crashes around Amherst Center within 1km between January 6 2025 and February 5 2025"\n'
            '  - "top 10 schools by crashes within 500m between 7am and 10am"\n'
            '  - "show crashes within 500m of all schools"\n'
            '  - "show crashes within 500m of all bus stops"\n'
            '  - "show crashes within 500m of all crosswalks"\n'
            '  - "show roads intersecting crosswalks"\n'
            '  - "show schools near crosswalks in Quincy"\n'
            '  - "show crashes near crosswalks"\n'
            '  - "show roads without sidewalks within 500m of all schools"\n'
            '  - "show crashes involving pedestrian"\n'
            '  - "show crashes involving animal"\n'
            '  - "show fatal crashes involving motor vehicle"\n'
            '  - "top 20 towns by crashes"\n'
            '  - "top 10 towns by fatal crashes"\n'
            '  - "top 20 towns by crashes involving pedestrian"\n'
        )

    def plan(self, user_prompt: str) -> tuple[SemanticFrame, DAGPlan, dict]:
        raw_sf = extract_semantic_frame(self.llm, user_prompt, self.road_geom_col)
        sf = validate_and_repair_semantic_frame(raw_sf, user_prompt)
        plan = compile_linear_plan(sf)
        return sf, plan, raw_sf

    def _generate_temporal_plots(self, state: ExecutionState) -> list:
        """
        Generate time distribution plots from the actual filtered crash result set.

        Uses the Crash role's fully-filtered SQL (all spatial, attribute, scope,
        and temporal filters applied) — so the plots show the distribution of
        exactly the crashes the user sees on the map.

        Produces up to two plots:
        1. Hour-of-day histogram (always, when crash data has time info)
        2. Daily crash count line/bar plot (always, when crash data has date info)

        If temporal constraints exist, the queried window is highlighted.
        """
        sf = state.semantic_frame
        if sf is None:
            return []

        role_map = _get_role_map(sf.targets)
        crash_role = _detect_crash_role(role_map, None)
        if crash_role is None:
            return []

        crash_rd = state.role_data.get(crash_role)
        if crash_rd is None:
            return []

        # Use the crash role's fully-filtered SQL — same crashes as on the map
        base_sql = crash_rd.sql_base
        base_params = dict(crash_rd.params)

        # Detect temporal constraints for highlighting
        time_constraint = None
        date_constraint = None
        for ac in sf.attribute_constraints:
            if ac.field == DERIVED_CRASH_TIME_MINUTES and ac.operator == "between":
                time_constraint = ac
            if ac.field == DERIVED_CRASH_DATE_VALUE and ac.operator == "between":
                date_constraint = ac

        # Only generate plots if there are temporal constraints
        if time_constraint is None and date_constraint is None:
            return []

        plots = []

        # --- Hour-of-day histogram (only when time-of-day filter exists) ---
        if time_constraint is not None:
            try:
                time_minutes_expr = build_crash_time_minutes_expr("c")
                hour_sql = f"""
                SELECT FLOOR(({time_minutes_expr}) / 60.0)::int AS hour_of_day, COUNT(*) AS cnt
                FROM ({base_sql}) c
                WHERE ({time_minutes_expr}) IS NOT NULL
                GROUP BY hour_of_day
                ORDER BY hour_of_day
                """
                hour_df = fetch_df(self.conn, hour_sql, base_params)

                if not hour_df.empty:
                    all_hours = pd.DataFrame({"hour_of_day": range(24)})
                    hour_df = all_hours.merge(hour_df, on="hour_of_day", how="left").fillna(0)
                    hour_df["cnt"] = hour_df["cnt"].astype(int)

                    filter_vals = time_constraint.value
                    h_start, h_end = None, None
                    if isinstance(filter_vals, list) and len(filter_vals) == 2:
                        h_start = int(filter_vals[0]) / 60.0
                        h_end = int(filter_vals[1]) / 60.0

                    fig, ax = plt.subplots(figsize=(8, 3))
                    colors = []
                    for h in hour_df["hour_of_day"]:
                        if h_start is not None and h_start <= h < h_end:
                            colors.append("#e74c3c")
                        else:
                            colors.append("#3498db")
                    ax.bar(hour_df["hour_of_day"], hour_df["cnt"], color=colors, edgecolor="white", width=0.85)

                    if h_start is not None:
                        h_s = int(h_start)
                        h_e = int(h_end)
                        label_start = f"{h_s % 12 or 12}{'am' if h_s < 12 else 'pm'}"
                        label_end = f"{h_e % 12 or 12}{'am' if h_e < 12 else 'pm'}"
                        ax.axvline(x=h_start - 0.5, color="#c0392b", linestyle="--", linewidth=1.2, alpha=0.7)
                        ax.axvline(x=h_end - 0.5, color="#c0392b", linestyle="--", linewidth=1.2, alpha=0.7)
                        ax.set_title(f"Selected Crashes by Hour of Day (filter: {label_start}–{label_end})", fontsize=11)
                    else:
                        ax.set_title("Selected Crashes by Hour of Day", fontsize=11)

                    total = int(hour_df["cnt"].sum())
                    ax.text(0.98, 0.95, f"Total: {total:,} crashes", transform=ax.transAxes,
                            ha="right", va="top", fontsize=9, color="#555")

                    ax.set_xlabel("Hour of Day")
                    ax.set_ylabel("Crash Count")
                    ax.set_xticks(range(0, 24, 2))
                    ax.set_xticklabels([f"{h % 12 or 12}{'a' if h < 12 else 'p'}" for h in range(0, 24, 2)], fontsize=9)
                    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    fig.tight_layout()
                    plots.append(fig)
            except Exception:
                pass

        # --- Daily crash count plot (only when date-range filter exists) ---
        if date_constraint is not None:
            try:
                date_expr = f"""
                CASE
                  WHEN c.{_quote_col(CRASH_DATE_COL)} IS NULL OR BTRIM(c.{_quote_col(CRASH_DATE_COL)}) = '' THEN NULL::date
                  ELSE TO_DATE(BTRIM(c.{_quote_col(CRASH_DATE_COL)}), 'MM/DD/YYYY')
                END
                """
                daily_sql = f"""
                SELECT ({date_expr}) AS crash_day, COUNT(*) AS cnt
                FROM ({base_sql}) c
                WHERE ({date_expr}) IS NOT NULL
                GROUP BY crash_day
                ORDER BY crash_day
                """
                daily_df = fetch_df(self.conn, daily_sql, base_params)

                if not daily_df.empty:
                    daily_df["crash_day"] = pd.to_datetime(daily_df["crash_day"])
                    daily_df = daily_df.sort_values("crash_day")

                    filter_vals = date_constraint.value
                    d_start, d_end = None, None
                    if isinstance(filter_vals, list) and len(filter_vals) == 2:
                        d_start = pd.to_datetime(filter_vals[0])
                        d_end = pd.to_datetime(filter_vals[1])

                    fig, ax = plt.subplots(figsize=(8, 3))

                    if len(daily_df) <= 60:
                        bar_colors = []
                        for _, row in daily_df.iterrows():
                            if d_start is not None and d_start <= row["crash_day"] <= d_end:
                                bar_colors.append("#e74c3c")
                            else:
                                bar_colors.append("#3498db")
                        ax.bar(daily_df["crash_day"], daily_df["cnt"], color=bar_colors, width=0.8)
                    else:
                        ax.plot(daily_df["crash_day"], daily_df["cnt"], color="#3498db", linewidth=0.8, alpha=0.9)
                        if d_start is not None:
                            mask = (daily_df["crash_day"] >= d_start) & (daily_df["crash_day"] <= d_end)
                            highlighted = daily_df[mask]
                            if not highlighted.empty:
                                ax.fill_between(highlighted["crash_day"], 0, highlighted["cnt"],
                                                color="#e74c3c", alpha=0.35)
                                ax.plot(highlighted["crash_day"], highlighted["cnt"],
                                        color="#e74c3c", linewidth=1.2)

                    if d_start is not None:
                        ax.axvline(x=d_start, color="#c0392b", linestyle="--", linewidth=1, alpha=0.7)
                        ax.axvline(x=d_end, color="#c0392b", linestyle="--", linewidth=1, alpha=0.7)
                        ax.set_title(f"Selected Crashes by Day (filter: {d_start.strftime('%b %d, %Y')}–{d_end.strftime('%b %d, %Y')})", fontsize=11)
                    else:
                        ax.set_title("Selected Crashes by Day", fontsize=11)

                    total = int(daily_df["cnt"].sum())
                    ax.text(0.98, 0.95, f"Total: {total:,} crashes", transform=ax.transAxes,
                            ha="right", va="top", fontsize=9, color="#555")

                    ax.set_xlabel("Date")
                    ax.set_ylabel("Crashes per Day")
                    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    fig.autofmt_xdate()
                    fig.tight_layout()
                    plots.append(fig)
            except Exception:
                pass

        return plots

    def _build_success_public_result(
        self,
        state: ExecutionState,
        plan: DAGPlan,
        sf: SemanticFrame,
        warnings_extra: Optional[list[str]] = None,
    ) -> PublicResult:
        bundle = build_response_bundle(
            state=state,
            plan=plan,
            success=True,
            error_message=None,
            failure_type=None,
        )

        narrative = generate_narrative_answer(bundle, self.response_llm)

        warnings = list(bundle.warnings)
        if warnings_extra:
            warnings.extend(list(warnings_extra))
        if not narrative.available and narrative.fallback_reason:
            warnings.append(f"Response-agent fallback: {narrative.fallback_reason}")

        tables = {t.name: t.df for t in state.tables}

        # Collect spatial layers for download (roles with a materialized GDF).
        gdfs = {
            role: rd.gdf
            for role, rd in state.role_data.items()
            if rd.gdf is not None and not rd.gdf.empty
        }

        # Generate temporal distribution plots if applicable
        temporal_plots = self._generate_temporal_plots(state)

        return PublicResult(
            semantic_frame=asdict(sf),
            execution_plan=[asdict(s) for s in plan.steps],
            summary=state.summary_text,
            map_object=state.map_object,
            tables=tables,
            debug=state.debug,
            response_bundle=asdict(bundle),
            narrative_answer=narrative.text,
            warnings=warnings,
            downloads=[asdict(x) for x in bundle.downloads],
            temporal_plots=temporal_plots,
            gdfs=gdfs,
        )

    def _build_failure_public_result(
        self,
        user_prompt: str,
        raw_sf: Optional[dict],
        sf: Optional[SemanticFrame],
        plan: Optional[DAGPlan],
        error: Exception,
    ) -> PublicResult:
        error_message = str(error)
        failure_type = type(error).__name__

        validated_dict = asdict(sf) if sf is not None else None
        bundle = build_failure_response_bundle(
            user_prompt=user_prompt,
            raw_semantic_frame=raw_sf,
            validated_semantic_frame=validated_dict,
            plan=plan,
            error_message=error_message,
            failure_type=failure_type,
        )

        narrative = generate_narrative_answer(bundle, self.response_llm)

        warnings = list(bundle.warnings)
        if not narrative.available and narrative.fallback_reason:
            warnings.append(f"Response-agent fallback: {narrative.fallback_reason}")

        debug = {
            "raw_semantic_frame": raw_sf,
            "validated_semantic_frame": validated_dict,
            "failure_type": failure_type,
            "error_message": error_message,
        }

        return PublicResult(
            semantic_frame=validated_dict or {},
            execution_plan=[asdict(s) for s in plan.steps] if plan is not None else [],
            summary=None,
            map_object=None,
            tables={},
            debug=debug,
            response_bundle=asdict(bundle),
            narrative_answer=narrative.text,
            warnings=warnings,
            downloads=[asdict(x) for x in bundle.downloads],
        )

    def run(self, user_prompt: str, geocode_selection: Optional[int] = None) -> PublicResult:
        raw_sf = None
        sf = None
        plan = None
        try:
            sf, plan, raw_sf = self.plan(user_prompt)

            if not sf.supported:
                raise ValueError("Unsupported query.\n\n" + self.help_message())

            state = ExecutionState(user_prompt=user_prompt, semantic_frame=sf)
            state.debug["raw_semantic_frame"] = raw_sf
            state.debug["validated_semantic_frame"] = asdict(sf)
            if geocode_selection is not None:
                state.debug["_geocode_selection"] = geocode_selection
            state.debug["dataset_registry_snapshot"] = {
                k: {
                    "table": v.table,
                    "geometry_column": v.geometry_column,
                    "geometry_family": v.geometry_family,
                    "primary_key": v.primary_key,
                    "label_field": v.label_field,
                    "display_fields": v.display_fields,
                    "fields": v.fields,
                    "derived_fields": v.derived_fields,
                    "relation_capabilities": v.relation_capabilities,
                    "scope_capable": v.scope_capable,
                    "name_match_field": v.name_match_field,
                    "name_match_strip_suffixes": v.name_match_strip_suffixes,
                    "display_geometry_mode": v.display_geometry_mode,
                }
                for k, v in DATASET_REGISTRY.items()
            }

            state = self.executor.execute(state, plan)
            return self._build_success_public_result(state=state, plan=plan, sf=sf)

        except AmbiguousLocationError:
            raise  # Propagate to UI for user selection

        except Exception as e:
            return self._build_failure_public_result(
                user_prompt=user_prompt,
                raw_sf=raw_sf,
                sf=sf,
                plan=plan,
                error=e,
            )

    def run_and_display(self, user_prompt: str, save_html_path: str = "roadsafety_map.html") -> PublicResult:
        result = self.run(user_prompt)

        print("\n[Semantic Frame]")
        print(json.dumps(result.semantic_frame, indent=2))

        print("\n[Execution Plan]")
        print(json.dumps(result.execution_plan, indent=2))

        if result.response_bundle is not None:
            print("\n[Response Bundle]")
            print(json.dumps(result.response_bundle, indent=2, ensure_ascii=False))

        if result.summary:
            print("\n[Summary]")
            print(result.summary)

        if result.narrative_answer:
            print("\n[Narrative Answer]")
            print(result.narrative_answer)

        if result.warnings:
            print("\n[Warnings]")
            for w in result.warnings:
                print(f"- {w}")

        for name, df in result.tables.items():
            print(f"\n[Table: {name}]")
            if ipy_display is not None:
                ipy_display(df)
            else:
                print(df)

        if result.map_object is not None:
            if ipy_display is not None:
                try:
                    ipy_display(result.map_object)
                except Exception:
                    result.map_object.save(save_html_path)
                    print(f"Saved map to: {save_html_path}")
            else:
                result.map_object.save(save_html_path)
                print(f"Saved map to: {save_html_path}")

        return result


# =========================================================
# 17) CHAT / NOTEBOOK USAGE
# =========================================================
def chat(assistant: RoadSafetyAssistant, save_html_path: str = "roadsafety_map.html"):
    print("\nTry prompts like:")
    print(assistant.help_message())
    print("Type 'exit' to quit.")

    while True:
        user_prompt = input("\nAsk: ").strip()
        if user_prompt.lower() in {"exit", "quit"}:
            break
        result = assistant.run_and_display(user_prompt, save_html_path=save_html_path)
        if result.response_bundle is not None and result.response_bundle.get("success") is False and not result.narrative_answer:
            err = result.response_bundle.get("error_message") or "Unknown error."
            print("\nERROR:", err)
