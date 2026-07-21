"""Clean raw NHTSA Safercar data and load it into a local DuckDB database.

Pipeline stages: dedupe -> sentinel-zero handling -> type/category normalization
-> split into a wide `vehicles` table and a long/tidy `equipment` table -> load
into DuckDB (data/nhtsa.duckdb).
"""
import re

import duckdb
import pandas as pd

RAW_PATH = "data/Safercar_data.csv"
DB_PATH = "data/nhtsa.duckdb"

# Columns describing standard/optional/not-present safety equipment. Kept in a
# long table (vehicle_id, feature, raw_value) so downstream code (SQL queries,
# the LLM classifier in Phase 2) can work with "the distinct values of one
# feature column" without needing to know the full wide schema.
EQUIPMENT_COLS = [
    "UPPER_BELT_ANCHORAGE", "UPPER_BELT_ANCHORAGE_LOC",
    "SEAT_BELT_PRETENSIONER", "SEAT_BELT_PRETENSIONER_LOC",
    "LOAD_LIMITERS", "LOAD_LIMITERS_LOC",
    "FRNT_BELT_INDICATOR", "FRNT_BELT_LOC",
    "REAR_BELT_INDICATOR", "LATCH_REAR_POSITION",
    "HEAD_SAB", "HEAD_SAB_TYPE", "HEAD_SAB_LOC", "HEAD_SAB_MOUNT_LOC",
    "HEAD_SAB_MEET_REQUIREMENTS", "HEAD_SAB_DEPLOY_IN_ROLLOVER",
    "TORSO_SAB", "TORSO_SAB_TYPE", "TORSO_SAB_LOC", "TORSO_SAB_MOUNT_LOC",
    "KNEE_BOLSTERS", "KNEE_BOLSTERS_LOC",
    "ADL", "HEAD_RESTRAINT_IND", "DYNAMIC_HEAD_RESTRAINT_IND",
    "BETI", "BLIND_SPOT_DETECTION", "DAY_RUN_LIGHTS",
    "ADAPTIVE_CRUISE_CONTROL", "ABS", "ARS", "ARS_LOC",
    "AUTO_CRASH_NOTIFICATION", "CRASH_DATA_RECORDER",
    "ANTI_THEFT_DEVICE", "ANTI_THEFT_DEVICE_TYPE",
    "FRNT_COLLISION_WARNING", "NHTSA_FRNT_COLLISION_WARNING", "NHTSA_FCW_EVALUATION",
    "LANE_DEPARTURE_WARNING", "NHTSA_LANE_DEPARTURE_WARNING", "NHTSA_LDW_EVALUATION",
    "CRASH_IMMINENT_BRAKE", "NHTSA_CRASH_IMMINENT_BRAKE", "NHTSA_CIB_EVALUATION",
    "DYNAMIC_BRAKE_SUPPORT", "NHTSA_DYNAMIC_BRAKE_SUPPORT", "NHTSA_DBS_EVALUATION",
    "NHTSA_ESC",
    "PELVIS_SAB", "PELVIS_SAB_TYPE", "PELVIS_SAB_LOC", "PELVIS_SAB_MOUNT_LOC",
    "NHTSA_BACKUP_CAMERA", "BACKUP_CAMERA",
]

# Known typos / abbreviations seen in the raw DRIVE_TRAIN column, normalized
# to the dominant spelling for that drivetrain. Combo values (e.g. "RWD/AWD")
# are left as-is since collapsing them would lose real information.
DRIVE_TRAIN_FIXES = {
    "ADW": "AWD",
    "FWD": "FWD",
    "4X4": "4WD",
    "4X2": "2WD",
}

BODY_FRAME_MAP = {
    "uni-body": "Uni-Body",
    "uni body": "Uni-Body",
    "body-on-frame": "Body-on-Frame",
    "body on frame": "Body-on-Frame",
    "frame-based": "Body-on-Frame",
}


def load_raw(path):
    """Read the raw CSV."""
    return pd.read_csv(path, low_memory=False)


def drop_duplicates(df):
    """Remove exact duplicate rows."""
    n_dupes = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"  dropped {n_dupes} exact duplicate rows")
    return df


def fix_sentinel_zeros(df):
    """Treat known sentinel zeros as missing data.

    CURB_WEIGHT uses 0 to mean "not recorded" (no vehicle weighs 0 lbs);
    the star-rating and other weight columns already use NaN natively.
    """
    n_zero = (df["CURB_WEIGHT"] == 0).sum()
    df.loc[df["CURB_WEIGHT"] == 0, "CURB_WEIGHT"] = pd.NA
    print(f"  converted {n_zero} CURB_WEIGHT sentinel zeros to NaN")
    return df


def normalize_drive_train(df):
    """Normalize typos/case in DRIVE_TRAIN; blank strings become NaN."""
    def fix(val):
        if pd.isna(val):
            return pd.NA
        val = val.strip().upper()
        if not val:
            return pd.NA
        return DRIVE_TRAIN_FIXES.get(val, val)

    df["DRIVE_TRAIN"] = df["DRIVE_TRAIN"].apply(fix)
    return df


def normalize_body_frame(df):
    """Collapse casing/whitespace variants of BODY_FRAME into one spelling."""
    def fix(val):
        if pd.isna(val):
            return pd.NA
        key = val.strip().lower()
        if not key:
            return pd.NA
        return BODY_FRAME_MAP.get(key, val.strip())

    df["BODY_FRAME"] = df["BODY_FRAME"].apply(fix)
    return df


def normalize_num_seating(df):
    """Extract the smallest seat count mentioned (e.g. "5 or 6" -> 5).

    Keeps the original free-text value in NUM_OF_SEATING for reference.
    """
    def first_number(val):
        if pd.isna(val):
            return pd.NA
        match = re.search(r"\d+", str(val))
        return int(match.group()) if match else pd.NA

    df["NUM_OF_SEATING_MIN"] = df["NUM_OF_SEATING"].apply(first_number).astype("Int64")
    return df


def split_vehicles_equipment(df):
    """Split the cleaned wide frame into a `vehicles` table and a long
    `equipment` table keyed by a surrogate vehicle_id.
    """
    df = df.reset_index(drop=True)
    df.insert(0, "vehicle_id", df.index)

    equipment_cols = [c for c in EQUIPMENT_COLS if c in df.columns]
    equipment = df[["vehicle_id"] + equipment_cols].melt(
        id_vars="vehicle_id", var_name="feature", value_name="raw_value"
    )
    equipment = equipment.dropna(subset=["raw_value"])
    equipment = equipment[equipment["raw_value"].str.strip() != ""]

    vehicles = df.drop(columns=equipment_cols)
    return vehicles, equipment


def load_to_duckdb(vehicles, equipment, db_path):
    con = duckdb.connect(db_path)
    con.execute("CREATE OR REPLACE TABLE vehicles AS SELECT * FROM vehicles")
    con.execute("CREATE OR REPLACE TABLE equipment AS SELECT * FROM equipment")
    con.close()


def main():
    print("Loading raw data...")
    df = load_raw(RAW_PATH)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    print("Cleaning...")
    df = drop_duplicates(df)
    df = fix_sentinel_zeros(df)
    df = normalize_drive_train(df)
    df = normalize_body_frame(df)
    df = normalize_num_seating(df)

    print("Splitting into vehicles/equipment tables...")
    vehicles, equipment = split_vehicles_equipment(df)
    print(f"  vehicles: {len(vehicles)} rows, {len(vehicles.columns)} columns")
    print(f"  equipment: {len(equipment)} rows (long format)")

    print(f"Loading into DuckDB at {DB_PATH}...")
    load_to_duckdb(vehicles, equipment, DB_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
