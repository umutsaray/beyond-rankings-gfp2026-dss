from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COMPOSITION = ROOT / "outputs/capability_profiles/tables/s8_corrected_domain_composition.csv"
OUTPUT_DIR = ROOT / "outputs/capability_profiles/major_revision"

PC1_RULE = (
    "Within each multi-variable connected component, fit PCA to StandardScaler-transformed "
    "features; multiply PC1 by -1 only when the un-oriented PC1 loading sum is negative, "
    "so the reported PC1 has sum(loadings) >= 0."
)


READINESS_FIELDS = {
    "aircraft_readiness": "Aircraft Total: / Readiness:",
    "fighters_readiness": "Fighters: / Readiness:",
    "attack_aircraft_readiness": "Attack Types: / Readiness:",
    "transports_readiness": "Transports (Fixed-Wing): / Readiness:",
    "trainers_readiness": "Trainers: / Readiness:",
    "special_mission_readiness": "Special-Mission: / Readiness:",
    "tanker_fleet_readiness": "Tanker Fleet: / Readiness:",
    "helicopters_readiness": "Helicopters: / Readiness:",
    "attack_helicopters_readiness": "Attack Helicopters: / Readiness:",
    "tanks_readiness": "Tanks: / Readiness:",
    "vehicles_readiness": "Vehicles: / Readiness:",
    "self_propelled_artillery_readiness": "Self-Propelled Artillery: / Readiness:",
    "towed_artillery_readiness": "Towed Artillery: / Readiness:",
    "mlrs_readiness": "MLRS (Rocket Artillery): / Readiness:",
}


ENERGY_FIELDS = [
    ("oil_production", "Oil Production:", "Flow; barrels; reporting period not stated in the source PDF", "Source-reported crude-oil production. The extracted GFP PDF supplies the unit bbl but does not identify a daily or annual period."),
    ("oil_consumption", "Oil Consumption:", "Flow; barrels; reporting period not stated in the source PDF", "Source-reported crude-oil consumption. The extracted GFP PDF supplies the unit bbl but does not identify a daily or annual period."),
    ("oil_deficit", "Oil Deficit:", "Flow balance; barrels; same unstated period as production and consumption", "In every complete 2026 record, this source field equals production minus consumption. Negative means deficit; positive means surplus. The pipeline retains the source sign."),
    ("oil_proven_reserves", "Oil Proven Reserves:", "Stock; barrels; source-PDF review date", "Source-reported proved crude-oil reserve stock. No flow period applies."),
    ("natural_gas_production", "Natural Gas Production:", "Flow; cubic metres; reporting period not stated in the source PDF", "Source-reported natural-gas production. The extracted GFP PDF supplies cu.m but no daily or annual period."),
    ("natural_gas_consumption", "Natural Gas Consumption:", "Flow; cubic metres; reporting period not stated in the source PDF", "Source-reported natural-gas consumption. The extracted GFP PDF supplies cu.m but no daily or annual period."),
    ("natural_gas_deficit", "Natural Gas Deficit:", "Flow balance; cubic metres; same unstated period as production and consumption", "In every complete 2026 record, this source field equals production minus consumption. Negative means deficit; positive means surplus. The pipeline retains the source sign."),
    ("natural_gas_proven_reserves", "Nat.Gas Proven Rez:", "Stock; cubic metres; source-PDF review date", "Source-reported proved natural-gas reserve stock. No flow period applies."),
    ("coal_production", "Coal Production:", "Flow; metric tonnes; reporting period not stated in the source PDF", "Source-reported coal production. The extracted GFP PDF supplies mt but no daily or annual period."),
    ("coal_consumption", "Coal Consumption:", "Flow; metric tonnes; reporting period not stated in the source PDF", "Source-reported coal consumption. The extracted GFP PDF supplies mt but no daily or annual period."),
    ("coal_deficit", "Coal Deficit:", "Flow balance; metric tonnes; same unstated period as production and consumption", "In every complete 2026 record, this source field equals production minus consumption. Negative means deficit; positive means surplus. The pipeline retains the source sign."),
    ("coal_proven_reserves", "Coal Proven Reserves:", "Stock; metric tonnes; source-PDF review date", "Source-reported proved coal reserve stock. No flow period applies."),
]


def build_mapping() -> pd.DataFrame:
    composition = pd.read_csv(COMPOSITION)
    rows: list[dict[str, object]] = []
    for _, row in composition.iterrows():
        members = composition.loc[
            composition["redundancy_group"].eq(row["redundancy_group"]), "original_feature"
        ].tolist()
        is_pc1 = len(members) > 1
        rows.append(
            {
                "original_variable": row["original_feature"],
                "domain": row["domain"],
                "correlation_group_id": row["redundancy_group"],
                "group_members": "; ".join(member for member in members if member != row["original_feature"]),
                "represented_by_PC1": is_pc1,
                "pc1_orientation_rule": PC1_RULE if is_pc1 else "Not applicable: singleton standardized feature retained directly.",
            }
        )
    result = pd.DataFrame(rows)
    assert len(result) == 76
    assert result["correlation_group_id"].nunique() == 56
    return result


def build_definitions() -> pd.DataFrame:
    rows = [
        {
            "variable": variable,
            "gfp_source_field_name": source_name,
            "time_period_or_stock_flow": period,
            "definition_note": note,
        }
        for variable, source_name, period, note in ENERGY_FIELDS
    ]
    for variable, source_name in READINESS_FIELDS.items():
        platform = "aircraft" if variable in {
            "aircraft_readiness", "fighters_readiness", "attack_aircraft_readiness",
            "transports_readiness", "trainers_readiness", "special_mission_readiness",
            "tanker_fleet_readiness", "helicopters_readiness", "attack_helicopters_readiness",
        } else "land-system"
        basis = "U.S. Air Force average" if platform == "aircraft" else "U.S. Army average"
        rows.append(
            {
                "variable": variable,
                "gfp_source_field_name": source_name,
                "time_period_or_stock_flow": "Estimated current ready inventory count; source-PDF review date; not a time rate",
                "definition_note": (
                    f"Estimated number of ready {platform} assets, not an operational-readiness percentage. "
                    f"GFP describes the estimate as based on a {basis} across categories and discounts assets "
                    "unavailable because of maintenance, modernization, refurbishment, and similar causes."
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mapping = build_mapping()
    definitions = build_definitions()
    mapping.to_csv(OUTPUT_DIR / "task_variable_to_component_mapping.csv", index=False)
    definitions.to_csv(OUTPUT_DIR / "task_variable_definitions.csv", index=False)
    print(f"mapping rows={len(mapping)}, groups={mapping.correlation_group_id.nunique()}")
    print(f"definition rows={len(definitions)}")


if __name__ == "__main__":
    main()
