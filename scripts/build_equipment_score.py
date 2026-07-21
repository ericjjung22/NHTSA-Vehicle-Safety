"""Combine rule-based and LLM classifications into a per-vehicle equipment score.

Reads the `equipment` long table, classifies each (feature, raw_value) pair
using the rule matcher first and falling back to the LLM cache
(data/equipment_llm_cache.json), then writes two tables back to DuckDB:

- equipment_levels: vehicle_id, feature, level, source (rule|llm|unresolved)
- vehicle_equipment_score: vehicle_id, equipment_score (mean level across
  CORE_FEATURES present for that vehicle, on a 0-2 scale), n_features_scored
"""
import json
import os

import duckdb
import pandas as pd

from equipment_rules import CORE_FEATURES, classify_rule_based

DB_PATH = "data/nhtsa.duckdb"
CACHE_PATH = "data/equipment_llm_cache.json"


def load_llm_cache(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def classify_row(feature, raw_value, llm_cache):
    rule_result = classify_rule_based(raw_value)
    if rule_result is not None:
        level, confidence = rule_result
        return level, "rule"
    cached = llm_cache.get(f"{feature}||{raw_value}")
    if cached is not None:
        return cached["level"], "llm"
    return None, "unresolved"


def main():
    con = duckdb.connect(DB_PATH)
    llm_cache = load_llm_cache(CACHE_PATH)

    placeholders = ", ".join("?" for _ in CORE_FEATURES)
    rows = con.execute(
        f"SELECT vehicle_id, feature, raw_value FROM equipment WHERE feature IN ({placeholders})",
        CORE_FEATURES,
    ).fetchall()

    scored = []
    source_counts = {"rule": 0, "llm": 0, "unresolved": 0}
    for vehicle_id, feature, raw_value in rows:
        level, source = classify_row(feature, raw_value, llm_cache)
        source_counts[source] += 1
        scored.append((vehicle_id, feature, level, source))

    scored_df = pd.DataFrame(scored, columns=["vehicle_id", "feature", "level", "source"])
    con.execute("CREATE OR REPLACE TABLE equipment_levels AS SELECT * FROM scored_df")
    print(f"equipment_levels: {len(scored)} rows")
    print(f"  by source: {source_counts}")

    con.execute("""
        CREATE OR REPLACE TABLE vehicle_equipment_score AS
        SELECT
            vehicle_id,
            AVG(level) AS equipment_score,
            COUNT(*) AS n_features_scored
        FROM equipment_levels
        WHERE level IS NOT NULL
        GROUP BY vehicle_id
    """)
    n_vehicles = con.execute("SELECT COUNT(*) FROM vehicle_equipment_score").fetchone()[0]
    print(f"vehicle_equipment_score: {n_vehicles} vehicles scored")


if __name__ == "__main__":
    main()
