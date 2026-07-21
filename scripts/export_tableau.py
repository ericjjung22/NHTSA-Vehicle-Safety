"""Export a flat CSV for the Tableau Public dashboard.

Tableau Public (the free tier) only connects to flat files, not directly to
DuckDB, so this denormalizes vehicles + vehicle_equipment_score into one
tidy, wide CSV that Tableau can ingest as-is.
"""
import duckdb

DB_PATH = "data/nhtsa.duckdb"
OUT_PATH = "data/tableau_export.csv"


def main():
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute("""
        SELECT
            v.vehicle_id,
            v.MAKE,
            v.MODEL,
            v.MODEL_YR,
            v.VEHICLE_TYPE,
            v.VEHICLE_CLASS,
            v.DRIVE_TRAIN,
            v.OVERALL_STARS,
            s.equipment_score,
            s.n_features_scored,
            CASE WHEN v.MODEL_YR >= 2011 THEN 'Post-2011 protocol' ELSE 'Pre-2011 protocol' END AS protocol_era
        FROM vehicles v
        JOIN vehicle_equipment_score s USING (vehicle_id)
        WHERE v.OVERALL_STARS IS NOT NULL
          AND v.VEHICLE_TYPE IN ('PC', 'MPV', 'TRUCK')
        ORDER BY v.MODEL_YR
    """).df()
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
