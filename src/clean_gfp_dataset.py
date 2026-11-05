from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from extract_gfp_pdfs import (
    CORE_FIELDS,
    DEFAULT_OUTPUT_DIR,
    PERCENT_FIELDS,
    RANK_FIELDS,
    TARGET_FIELDS,
    VALUE_FIELDS,
    parse_numeric_value,
)


STRING_FIELDS = {"country", "source_file", "source_url", "last_reviewed_date"}
FLOAT_FIELDS = {"pwrindx"} | PERCENT_FIELDS
NUMERIC_FIELDS = [field for field in TARGET_FIELDS if field not in STRING_FIELDS]
INTEGERISH_FIELDS = [field for field in NUMERIC_FIELDS if field not in FLOAT_FIELDS]


def clean_dataset(output_dir: Path) -> pd.DataFrame:
    table_dir = output_dir / "tables"
    raw_path = table_dir / "gfp_2026_raw_extracted.csv"
    if not raw_path.exists():
        raise SystemExit(f"Missing raw extraction file: {raw_path}. Run extract_gfp_pdfs.py first.")

    raw_df = pd.read_csv(raw_path, dtype=str, keep_default_na=False, na_values=[])
    clean_columns = {}

    for field in TARGET_FIELDS:
        if field in STRING_FIELDS:
            clean_columns[field] = raw_df.get(field, pd.Series(index=raw_df.index, dtype=object)).replace({"": pd.NA})
        elif field in FLOAT_FIELDS:
            clean_columns[field] = raw_df.get(field, pd.Series(index=raw_df.index, dtype=object)).map(
                lambda value: parse_numeric_value(value, field=field)
            )
        else:
            clean_columns[field] = raw_df.get(field, pd.Series(index=raw_df.index, dtype=object)).map(
                lambda value: parse_numeric_value(value, field=field)
            )

    clean_df = pd.DataFrame(clean_columns, index=raw_df.index)

    for field in INTEGERISH_FIELDS:
        clean_df[field] = pd.to_numeric(clean_df[field], errors="coerce").round().astype("Int64")

    for field in FLOAT_FIELDS:
        clean_df[field] = pd.to_numeric(clean_df[field], errors="coerce")

    if "year" in clean_df:
        clean_df["year"] = pd.to_numeric(clean_df["year"], errors="coerce").astype("Int64")

    reviewed = pd.to_datetime(clean_df["last_reviewed_date"], errors="coerce", format="%m/%d/%Y")
    clean_df["last_reviewed_date"] = reviewed.dt.date.astype("string").where(reviewed.notna(), pd.NA)

    for extra in ["extraction_engine", "text_line_count"]:
        if extra in raw_df.columns:
            clean_df[extra] = raw_df[extra].replace({"": pd.NA})
    if "text_line_count" in clean_df.columns:
        clean_df["text_line_count"] = pd.to_numeric(clean_df["text_line_count"], errors="coerce").astype("Int64")

    clean_csv = table_dir / "gfp_2026_clean.csv"
    clean_xlsx = table_dir / "gfp_2026_clean.xlsx"
    clean_df.to_csv(clean_csv, index=False)

    with pd.ExcelWriter(clean_xlsx, engine="openpyxl") as writer:
        clean_df.to_excel(writer, index=False, sheet_name="gfp_2026_clean")
        worksheet = writer.sheets["gfp_2026_clean"]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for column_cells in worksheet.columns:
            header = str(column_cells[0].value or "")
            max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells[:200])
            worksheet.column_dimensions[column_cells[0].column_letter].width = min(max(max_len + 2, len(header) + 2), 32)

    print(f"Wrote {clean_csv}")
    print(f"Wrote {clean_xlsx}")
    return clean_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean the raw Global Firepower 2026 extraction.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Folder containing output tables.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_dataset(args.output_dir)


if __name__ == "__main__":
    main()
