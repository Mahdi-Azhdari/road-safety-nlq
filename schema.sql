-- Road Safety Assistant — database schema
--
-- PostgreSQL + PostGIS CREATE TABLE statements for the six entity types
-- used by the system.  Table names, column names, and geometry column names
-- must match these definitions exactly; they are referenced by the constants
-- in core.py and are not configurable at runtime.
--
-- All geometry columns use EPSG:26986 (Massachusetts State Plane, metres).
-- Spatial indexes are included for each geometry column.
--
-- Load order: PostGIS extension must be enabled before running this file.
--   CREATE EXTENSION IF NOT EXISTS postgis;
--
-- Crash records are from MassDOT and are not distributed with this repository.
-- Contact MassDOT or visit https://www.mass.gov/info-details/massachusetts-crash-data.


-- ── Crash ─────────────────────────────────────────────────────────────────────
-- Point layer. One row per crash event.
-- Roadway attributes (speed limit, sidewalk status, junction type) are merged
-- directly from the road inventory so that attribute filtering on crashes does
-- not require a separate join at query time.

CREATE TABLE IF NOT EXISTS public."Crash" (
    id              SERIAL PRIMARY KEY,
    geom            geometry(Point, 26986),

    -- Crash attributes
    crash_seve      TEXT,           -- severity: 'Fatal injury' | 'Non-fatal injury' |
                                    --           'Property damage only (none injured)' | 'Unknown'
    crash_date      TEXT,           -- stored as text 'MM DD YYYY'; parsed at query time
    crash_time      TEXT,           -- stored as text 'HH:MI AM'; parsed at query time
    first_hrmf      TEXT,           -- first harmful event (30 canonical categories)

    -- Road inventory attributes merged from Road_Inventory_2025
    speed_lim       INTEGER,        -- posted speed limit (mph)
    lt_sidewlk      INTEGER,        -- left-side sidewalk presence (0/1 or coded)
    rt_sidewlk      INTEGER,        -- right-side sidewalk presence (0/1 or coded)
    rdwy_jnct_      TEXT            -- roadway junction type
);

CREATE INDEX IF NOT EXISTS idx_crash_geom
    ON public."Crash" USING GIST (geom);

CREATE INDEX IF NOT EXISTS idx_crash_seve
    ON public."Crash" (crash_seve);

CREATE INDEX IF NOT EXISTS idx_crash_first_hrmf
    ON public."Crash" (first_hrmf);


-- ── Road Inventory ────────────────────────────────────────────────────────────
-- Line layer. Massachusetts road network with roadway attributes.
-- The geometry column name is detected at runtime via geometry_columns;
-- 'geom' is the fallback if detection fails.

CREATE TABLE IF NOT EXISTS public."Road_Inventory_2025" (
    objectid        SERIAL PRIMARY KEY,
    geom            geometry(MultiLineString, 26986),

    -- Roadway attributes
    speed_lim       INTEGER,        -- posted speed limit (mph)
    op_dir_sl       INTEGER,        -- opposing-direction speed limit (mph)
    lt_sidewlk      INTEGER,        -- left-side sidewalk presence
    rt_sidewlk      INTEGER        -- right-side sidewalk presence
);

CREATE INDEX IF NOT EXISTS idx_road_geom
    ON public."Road_Inventory_2025" USING GIST (geom);


-- ── Schools ───────────────────────────────────────────────────────────────────
-- Point layer. Public and private school locations.

CREATE TABLE IF NOT EXISTS public."SCHOOLS_PT" (
    name            TEXT PRIMARY KEY,
    geom            geometry(Point, 26986)
);

CREATE INDEX IF NOT EXISTS idx_school_geom
    ON public."SCHOOLS_PT" USING GIST (geom);

CREATE INDEX IF NOT EXISTS idx_school_name
    ON public."SCHOOLS_PT" (name);


-- ── Bus Stops ─────────────────────────────────────────────────────────────────
-- Point layer. Transit stop locations.

CREATE TABLE IF NOT EXISTS public."Bus_stops" (
    stop_id         TEXT PRIMARY KEY,
    stop_name       TEXT,
    geom            geometry(Point, 26986)
);

CREATE INDEX IF NOT EXISTS idx_busstop_geom
    ON public."Bus_stops" USING GIST (geom);


-- ── Crosswalks ────────────────────────────────────────────────────────────────
-- Polygon layer. Marked crosswalk footprints.

CREATE TABLE IF NOT EXISTS public."Crosswalks.shp" (
    id              SERIAL PRIMARY KEY,
    geom            geometry(Polygon, 26986)
);

CREATE INDEX IF NOT EXISTS idx_crosswalk_geom
    ON public."Crosswalks.shp" USING GIST (geom);


-- ── Towns ─────────────────────────────────────────────────────────────────────
-- Polygon layer. Massachusetts municipal boundaries.
-- namelsad20 is the full legal name used for geographic scoping
-- (e.g. 'Quincy city', 'Amherst town').

CREATE TABLE IF NOT EXISTS public."Towns.Mass" (
    namelsad20      TEXT PRIMARY KEY,
    geom            geometry(MultiPolygon, 26986)
);

CREATE INDEX IF NOT EXISTS idx_town_geom
    ON public."Towns.Mass" USING GIST (geom);

CREATE INDEX IF NOT EXISTS idx_town_name
    ON public."Towns.Mass" (namelsad20);
