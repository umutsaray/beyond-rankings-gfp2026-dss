from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from extract_gfp_pdfs import DEFAULT_OUTPUT_DIR, RANK_FIELDS, TARGET_FIELDS, VALUE_FIELDS


DEFICIT_FIELDS = {"oil_deficit", "natural_gas_deficit", "coal_deficit"}
STOCK_READINESS_PAIRS = [
    ("aircraft_total", "aircraft_readiness"),
    ("fighters", "fighters_readiness"),
    ("attack_aircraft", "attack_aircraft_readiness"),
    ("transports", "transports_readiness"),
    ("trainers", "trainers_readiness"),
    ("special_mission", "special_mission_readiness"),
    ("tanker_fleet", "tanker_fleet_readiness"),
    ("helicopters", "helicopters_readiness"),
    ("attack_helicopters", "attack_helicopters_readiness"),
    ("tanks", "tanks_readiness"),
    ("vehicles", "vehicles_readiness"),
    ("self_propelled_artillery", "self_propelled_artillery_readiness"),
    ("towed_artillery", "towed_artillery_readiness"),
    ("mlrs", "mlrs_readiness"),
]


def numeric(df: pd.DataFrame, field: str) -> pd.Series:
    return pd.to_numeric(df[field], errors="coerce") if field in df.columns else pd.Series(dtype="float64")


def add_issue(issues: list[dict[str, object]], row: pd.Series, field: str, issue: str, value: object, severity: str = "warning") -> None:
    issues.append(
        {
            "country": row.get("country"),
            "source_file": row.get("source_file"),
            "field": field,
            "value": value,
            "issue": issue,
            "severity": severity,
        }
    )


def validate_dataset(output_dir: Path) -> None:
    table_dir = output_dir / "tables"
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    clean_path = table_dir / "gfp_2026_clean.csv"
    raw_path = table_dir / "gfp_2026_raw_extracted.csv"
    if not clean_path.exists():
        raise SystemExit(f"Missing clean dataset: {clean_path}. Run clean_gfp_dataset.py first.")

    clean_df = pd.read_csv(clean_path)
    raw_df = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()

    present_fields = [field for field in TARGET_FIELDS if field in clean_df.columns]
    missing_counts = clean_df[present_fields].isna().sum().sort_values(ascending=False)
    missing_report = pd.DataFrame(
        {
            "field": missing_counts.index,
            "missing_count": missing_counts.values,
            "missing_pct": (missing_counts.values / max(len(clean_df), 1) * 100).round(2),
        }
    )
    missing_report.to_csv(report_dir / "missing_values_report.csv", index=False)

    extraction_report = pd.DataFrame(
        {
            "country": clean_df.get("country"),
            "source_file": clean_df.get("source_file"),
            "rank": clean_df.get("rank"),
            "pwrindx": clean_df.get("pwrindx"),
            "missing_target_fields": clean_df[present_fields].isna().sum(axis=1),
            "present_target_fields": clean_df[present_fields].notna().sum(axis=1),
        }
    )
    if not raw_df.empty:
        for extra in ["extraction_engine", "text_line_count"]:
            if extra in raw_df.columns:
                extraction_report[extra] = raw_df[extra]
    extraction_report.to_csv(report_dir / "extraction_report.csv", index=False)

    issues: list[dict[str, object]] = []
    total_countries = numeric(clean_df, "total_countries")
    for idx, row in clean_df.iterrows():
        rank = pd.to_numeric(row.get("rank"), errors="coerce")
        total = pd.to_numeric(row.get("total_countries"), errors="coerce")
        if pd.notna(rank) and pd.notna(total) and (rank < 1 or rank > total):
            add_issue(issues, row, "rank", "Country rank is outside 1..total_countries", rank, "error")

        pwrindx = pd.to_numeric(row.get("pwrindx"), errors="coerce")
        if pd.notna(pwrindx) and (pwrindx <= 0 or pwrindx > 10):
            add_issue(issues, row, "pwrindx", "PwrIndx is outside an expected positive range", pwrindx, "warning")

        for field in RANK_FIELDS:
            value = pd.to_numeric(row.get(field), errors="coerce")
            if pd.notna(value) and pd.notna(total) and (value < 1 or value > total):
                add_issue(issues, row, field, "Overview rank is outside 1..total_countries", value, "warning")

        for field in VALUE_FIELDS:
            value = pd.to_numeric(row.get(field), errors="coerce")
            if field not in DEFICIT_FIELDS and pd.notna(value) and value < 0:
                add_issue(issues, row, field, "Negative value in a non-deficit field", value, "warning")

        active = pd.to_numeric(row.get("active_personnel"), errors="coerce")
        reserve = pd.to_numeric(row.get("reserve_personnel"), errors="coerce")
        paramilitary = pd.to_numeric(row.get("paramilitary_personnel"), errors="coerce")
        total_military = pd.to_numeric(row.get("total_military_personnel"), errors="coerce")
        subtotal = active + reserve + paramilitary
        if pd.notna(total_military) and pd.notna(subtotal) and subtotal > total_military * 1.05:
            add_issue(
                issues,
                row,
                "total_military_personnel",
                "Active + reserve + paramilitary exceeds total military personnel by more than 5%",
                total_military,
                "warning",
            )

        for stock_field, readiness_field in STOCK_READINESS_PAIRS:
            stock = pd.to_numeric(row.get(stock_field), errors="coerce")
            readiness = pd.to_numeric(row.get(readiness_field), errors="coerce")
            if pd.notna(stock) and pd.notna(readiness) and readiness > stock:
                add_issue(
                    issues,
                    row,
                    readiness_field,
                    f"Readiness exceeds stock field {stock_field}",
                    readiness,
                    "warning",
                )

    suspicious_report = pd.DataFrame(
        issues,
        columns=["country", "source_file", "field", "value", "issue", "severity"],
    )
    suspicious_report.to_csv(report_dir / "suspicious_values_report.csv", index=False)

    high_missing = missing_report[missing_report["missing_pct"] >= 25].head(15)
    summary_lines = [
        "# Global Firepower 2026 Extraction Summary",
        "",
        f"- Rows extracted: {len(clean_df)}",
        f"- Target schema fields checked: {len(present_fields)}",
        f"- Average missing target fields per country: {extraction_report['missing_target_fields'].mean():.1f}",
        f"- Suspicious values flagged: {len(suspicious_report)}",
        f"- Countries with missing rank: {clean_df['rank'].isna().sum() if 'rank' in clean_df else 'n/a'}",
        "",
        "## Highest-Missing Fields",
    ]
    if high_missing.empty:
        summary_lines.append("No fields have 25% or higher missingness.")
    else:
        for _, row in high_missing.iterrows():
            summary_lines.append(f"- {row['field']}: {int(row['missing_count'])} missing ({row['missing_pct']}%)")
    summary_lines.extend(
        [
            "",
            "## Output Files",
            "- reports/extraction_report.csv",
            "- reports/missing_values_report.csv",
            "- reports/suspicious_values_report.csv",
        ]
    )
    (report_dir / "summary_report.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Wrote {report_dir / 'extraction_report.csv'}")
    print(f"Wrote {report_dir / 'missing_values_report.csv'}")
    print(f"Wrote {report_dir / 'suspicious_values_report.csv'}")
    print(f"Wrote {report_dir / 'summary_report.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the cleaned Global Firepower 2026 dataset.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Folder containing output tables.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_dataset(args.output_dir)


if __name__ == "__main__":
    main()
