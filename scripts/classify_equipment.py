"""Classify free-text equipment values the keyword matcher (equipment_rules.py)
can't handle, using the Claude API. Results are cached to a JSON file so the
same string is never billed twice, and batched (default 25/call) to keep the
number of API calls small.

Usage:
    python scripts/classify_equipment.py                 # classify unresolved values
    python scripts/classify_equipment.py --validate 40    # agreement check vs. the rule matcher
"""
import argparse
import json
import os
import time

import anthropic
import duckdb

from equipment_rules import CORE_FEATURES, classify_rule_based

DB_PATH = "data/nhtsa.duckdb"
CACHE_PATH = "data/equipment_llm_cache.json"
MODEL = "claude-opus-4-8"
BATCH_SIZE = 25
MAX_RETRIES = 3

SYSTEM_PROMPT = """\
You classify vehicle safety-equipment descriptions pulled from raw NHTSA data. \
Each description states whether a piece of equipment was standard, optional, or \
not present on a vehicle. Classify each one:

- level 2 = standard equipment (came with every vehicle)
- level 1 = available/optional equipment (could be added, not standard)
- level 0 = not present / not available

The descriptions are messy: abbreviations ("S", "O", "Sd", "Ab"), fragments of a \
front/rear or driver/passenger split ("F= Standard; R= Optional" -> treat as \
optional overall, since it's not standard on every seating position), percentages \
("Avail (57%)" -> optional), and free text with stray punctuation or OCR noise \
("Optional?46??" -> optional). Use your judgment for ambiguous fragments, and set \
confidence to "low" when you're guessing rather than reading a clear signal.

Return ONLY a JSON object matching the given schema. No prose, no markdown fences.\
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "classifications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "level": {"type": "integer", "enum": [0, 1, 2]},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["index", "level", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["classifications"],
    "additionalProperties": False,
}


def load_cache(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_cache(cache, path):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def cache_key(feature, raw_value):
    return f"{feature}||{raw_value}"


def fetch_unique_values(con, features):
    placeholders = ", ".join("?" for _ in features)
    query = f"SELECT DISTINCT feature, raw_value FROM equipment WHERE feature IN ({placeholders})"
    rows = con.execute(query, features).fetchall()
    return [(feat, val) for feat, val in rows if val and val.strip()]


def classify_batch(client, items):
    """items: list of (feature, raw_value). Returns {index: (level, confidence)}."""
    numbered = "\n".join(
        f"{i}. feature={feat!r} value={val!r}" for i, (feat, val) in enumerate(items)
    )
    user_message = f"Classify each of these {len(items)} values:\n\n{numbered}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                output_config={"format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}},
                messages=[{"role": "user", "content": user_message}],
            )
            text = next(b.text for b in response.content if b.type == "text")
            parsed = json.loads(text)
            result = {}
            for entry in parsed["classifications"]:
                result[entry["index"]] = (entry["level"], entry["confidence"])
            missing = set(range(len(items))) - set(result.keys())
            if missing:
                raise ValueError(f"response missing indices: {sorted(missing)}")
            return result
        except (anthropic.APIStatusError, anthropic.APIConnectionError, ValueError, KeyError) as e:
            print(f"    batch attempt {attempt} failed: {e}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2 ** attempt)


def classify_values(client, cache, values, batch_size=BATCH_SIZE):
    """Classify (feature, raw_value) pairs not already in cache. Mutates cache in place."""
    to_classify = [(feat, val) for feat, val in values if cache_key(feat, val) not in cache]
    print(f"  {len(values)} values requested, {len(to_classify)} not yet cached")

    for start in range(0, len(to_classify), batch_size):
        batch = to_classify[start : start + batch_size]
        print(f"  classifying batch {start // batch_size + 1} ({len(batch)} values)...")
        results = classify_batch(client, batch)
        for i, (feat, val) in enumerate(batch):
            level, confidence = results[i]
            cache[cache_key(feat, val)] = {
                "feature": feat,
                "value": val,
                "level": level,
                "confidence": confidence,
                "source": "llm",
            }
        save_cache(cache, CACHE_PATH)


def run_validation(client, cache, con, sample_size, features):
    """Run the LLM on values the rule matcher already handles, measure agreement."""
    all_values = fetch_unique_values(con, features)
    rule_covered = [(feat, val) for feat, val in all_values if classify_rule_based(val) is not None]
    sample = rule_covered[:sample_size]
    print(f"Validating against {len(sample)} rule-covered values...")

    classify_values(client, cache, sample)

    agree = 0
    for feat, val in sample:
        rule_level, _ = classify_rule_based(val)
        llm_entry = cache[cache_key(feat, val)]
        match = rule_level == llm_entry["level"]
        agree += match
        if not match:
            print(f"  DISAGREE  feature={feat!r} value={val!r} rule={rule_level} llm={llm_entry['level']}")

    pct = 100 * agree / len(sample) if sample else 0
    print(f"\nAgreement: {agree}/{len(sample)} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        help="Comma-separated feature names (default: CORE_FEATURES from equipment_rules.py)",
    )
    parser.add_argument(
        "--validate",
        type=int,
        metavar="N",
        help="Validate agreement against N rule-classified values instead of classifying the residual",
    )
    args = parser.parse_args()

    features = args.features.split(",") if args.features else CORE_FEATURES
    con = duckdb.connect(DB_PATH, read_only=True)
    client = anthropic.Anthropic()
    cache = load_cache(CACHE_PATH)

    if args.validate:
        run_validation(client, cache, con, args.validate, features)
        return

    all_values = fetch_unique_values(con, features)
    unresolved = [(feat, val) for feat, val in all_values if classify_rule_based(val) is None]
    print(f"{len(all_values)} distinct values, {len(unresolved)} not handled by the rule matcher")
    classify_values(client, cache, unresolved)
    print(f"Done. Cache has {len(cache)} entries at {CACHE_PATH}")


if __name__ == "__main__":
    main()
