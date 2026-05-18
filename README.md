# Road Safety Assistant

A schema-grounded natural language interface for transportation safety analysis.

The system allows users to query an authoritative spatial database using plain English. A large language model interprets each query into a structured semantic frame, a rule-based validation and repair layer enforces schema conformance, and a typed directed acyclic graph (DAG) of spatial operations is compiled and executed against a PostGIS database. Language interpretation is fully separated from execution, keeping results reproducible and grounded in the underlying data.

This repository accompanies the paper:

> **Broadening Access to Transportation Safety Evidence with Generative AI: A Schema-Grounded Framework for Natural Language Querying**
> Mahdi Azhdari, Eric J. Gonzales
> Department of Civil and Environmental Engineering, University of Massachusetts Amherst
> *Transportation Research Part A: Policy and Practice*

```bash
git clone https://github.com/Mahdi-Azhdari/road-safety-nlq.git
```

---

## Repository Structure

```
core.py                            # Core pipeline: LLM interpretation, validation/repair, DAG compiler, PostGIS executor
app.py                             # Streamlit web interface
benchmark_dag_strict.py            # Evaluation script (80-query benchmark)
benchmark_ground_truth_strict.py   # Ground truth definitions for the benchmark
schema.sql                         # PostgreSQL CREATE TABLE statements for the six entity types
requirements.txt                   # Python dependencies
.env.example                       # Environment variable template
```

---

## System Architecture

```
User query (natural language)
        │
        ▼
┌─────────────────────┐
│   LLM Interpreter   │  Gemini or OpenAI — produces a structured semantic frame (JSON)
└─────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Validation & Repair Layer  │  Rule-based: schema validation, value normalization,
│                             │  anchor resolution, structural correction
└─────────────────────────────┘
        │
        ▼
┌─────────────────────┐
│    DAG Compiler     │  Translates the validated frame into a typed directed acyclic
│                     │  graph of spatial operations
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   PostGIS Executor  │  Executes the DAG against the spatial database
└─────────────────────┘
        │
        ▼
  Maps · Tables · Summaries
```

---

## Supported Entity Types

| Entity     | Geometry | Description                              |
|------------|----------|------------------------------------------|
| Crash      | Point    | Crash records with severity, date, time, first harmful event, roadway attributes |
| Road       | Line     | Road inventory with speed limits and sidewalk status |
| School     | Point    | Public and private school locations      |
| BusStop    | Point    | Transit stop locations                   |
| Crosswalk  | Polygon  | Marked crosswalk footprints              |
| Town       | Polygon  | Municipal boundaries (used for geographic scoping) |

---

## Data

The system is implemented on a statewide Massachusetts transportation safety database. Crash records are sourced from the **Massachusetts Department of Transportation (MassDOT)** and are **not included** in this repository due to data licensing restrictions.

The database schema is provided in [`schema.sql`](schema.sql). To obtain crash data, contact MassDOT or visit the [MassDOT crash portal](https://www.mass.gov/info-details/massachusetts-crash-data).

All six entity types must be present in a PostGIS-enabled PostgreSQL database before running the system. Table and column names must match the constants defined in `core.py` exactly.

---

## Requirements

- Python 3.11+
- PostgreSQL 14+ with the PostGIS extension
- A Gemini or OpenAI API key

### Python dependencies

```bash
pip install -r requirements.txt
```

For DAG visualization in the Streamlit interface, the Graphviz system package is also required:

```bash
# macOS
brew install graphviz

# Ubuntu / Debian
sudo apt-get install graphviz
```

---

## Environment Variables

A `.env.example` file is included showing the variable names the project uses. It is a template — copy it, fill in your values, and keep the resulting `.env` file out of version control (it is already in `.gitignore`).

```bash
cp .env.example .env
```

The variables and where each one is used:

| Variable | Used by | Description |
|---|---|---|
| `GEMINI_API_KEY` | `core.py` | Gemini API key (primary LLM, if using Gemini) |
| `RESPONSE_GEMINI_API_KEY` | `core.py` | Response-layer Gemini key (optional; can be the same key) |
| `RESPONSE_GEMINI_MODEL` | `core.py` | Override the response-layer model name (optional) |
| `DB_PASSWORD` | `core.py`, benchmark | PostgreSQL password |
| `DB_HOST` / `DB_PORT` / `DB_NAME` / `DB_USER` | benchmark only | Database connection (the Streamlit app takes these from the sidebar at runtime) |

The Streamlit app takes all credentials from the sidebar at runtime, so none of the environment variables are required when using the app directly. They are mainly useful for running the benchmark script without hardcoding credentials.

---

## Database Setup

1. Create a PostgreSQL database with PostGIS enabled:

```sql
CREATE DATABASE roadsafety;
\c roadsafety
CREATE EXTENSION IF NOT EXISTS postgis;
```

2. Run the schema file to create the required tables:

```sql
\i schema.sql
```

3. Load your data into the tables. Column names and types must match the schema exactly.

---

## Running the App

```bash
streamlit run app.py
```

Open the Streamlit interface in your browser, enter your LLM API key and database credentials in the sidebar, and click **Connect & Start**.

Example queries:
- `show crashes in Quincy`
- `show fatal pedestrian crashes in Amherst between 7am and 10am`
- `top 10 schools by crashes within 500m in Boston`
- `top 20 towns by crashes without sidewalks`
- `show roads with speed limit above 30 near bus stops in Springfield`

---

## Running the Benchmark

The benchmark evaluates the full pipeline against 80 natural language queries spanning nine analytical groups. Results are written to `benchmark_out_<provider>/`.

1. Edit the configuration block at the top of `benchmark_dag_strict.py`:

```python
LLM_PROVIDER = "gemini"           # "gemini" | "openai"
LLM_API_KEY  = "your-key-here"
LLM_MODEL    = "gemini-2.5-flash"
```

2. Set the database password via environment variable or in the config block:

```bash
export DB_PASSWORD=your_password
```

3. Run:

```bash
python benchmark_dag_strict.py
```

Outputs: `results.xlsx`, `debug.json`, `execution_log.json`, `summary.txt`.

---

## Evaluation Results (Paper)

Results reported in the paper using GPT-4o on the 80-query benchmark:

| Metric | Result |
|--------|--------|
| Execution success | 80/80 (100%) |
| Intent completeness | 80/80 (100%) |
| Frames requiring repair | 23/80 (29%) |
| Mean runtime | 18.6 s |
| Max runtime | 178.8 s |

---

## License

This code is released under the MIT License. See [LICENSE](LICENSE) for the full text.
