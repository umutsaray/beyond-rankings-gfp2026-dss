from __future__ import annotations

import argparse
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover - gives a clear runtime failure
    raise SystemExit("PyMuPDF is required. Run: python -m pip install -r requirements.txt") from exc

try:
    import pdfplumber
except ImportError as exc:  # pragma: no cover - gives a clear runtime failure
    raise SystemExit("pdfplumber is required. Run: python -m pip install -r requirements.txt") from exc


DEFAULT_INPUT_DIR = Path("gfp_pdfs")
DEFAULT_OUTPUT_DIR = Path("outputs")

CORE_FIELDS = [
    "country",
    "year",
    "source_file",
    "source_url",
    "rank",
    "total_countries",
    "pwrindx",
    "last_reviewed_date",
]

FINANCIAL_FIELDS = [
    "purchasing_power",
    "foreign_reserves",
    "defense_budget",
    "external_debt",
]

GEOGRAPHY_FIELDS = [
    "square_land_area",
    "coastline",
    "shared_borders",
    "waterways",
]

MANPOWER_FIELDS = [
    "total_population",
    "available_manpower",
    "fit_for_service",
    "reaching_military_age_annually",
    "total_military_personnel",
    "active_personnel",
    "reserve_personnel",
    "paramilitary_personnel",
    "air_force_personnel",
    "army_personnel",
    "navy_personnel",
    "yearly_mobilization_potential",
]

AIRPOWER_FIELDS = [
    "aircraft_total",
    "aircraft_readiness",
    "fighters",
    "fighters_readiness",
    "attack_aircraft",
    "attack_aircraft_readiness",
    "transports",
    "transports_readiness",
    "trainers",
    "trainers_readiness",
    "special_mission",
    "special_mission_readiness",
    "tanker_fleet",
    "tanker_fleet_readiness",
    "helicopters",
    "helicopters_readiness",
    "attack_helicopters",
    "attack_helicopters_readiness",
]

LAND_FIELDS = [
    "tanks",
    "tanks_readiness",
    "vehicles",
    "vehicles_readiness",
    "self_propelled_artillery",
    "self_propelled_artillery_readiness",
    "towed_artillery",
    "towed_artillery_readiness",
    "mlrs",
    "mlrs_readiness",
]

NAVAL_FIELDS = [
    "naval_total_assets",
    "naval_total_tonnage",
    "aircraft_carriers",
    "helicopter_carriers",
    "destroyers",
    "frigates",
    "corvettes",
    "submarines",
    "patrol_vessels",
    "mine_warfare",
]

NATURAL_RESOURCE_FIELDS = [
    "oil_production",
    "oil_consumption",
    "oil_deficit",
    "oil_proven_reserves",
    "natural_gas_production",
    "natural_gas_consumption",
    "natural_gas_deficit",
    "natural_gas_proven_reserves",
    "coal_production",
    "coal_consumption",
    "coal_deficit",
    "coal_proven_reserves",
]

LOGISTICS_FIELDS = [
    "internet_coverage",
    "labor_force",
    "merchant_marine_fleet",
    "ports_trade_terminals",
    "airports",
    "roadway_coverage",
    "railway_coverage",
]

VALUE_FIELDS = (
    FINANCIAL_FIELDS
    + GEOGRAPHY_FIELDS
    + MANPOWER_FIELDS
    + AIRPOWER_FIELDS
    + LAND_FIELDS
    + NAVAL_FIELDS
    + NATURAL_RESOURCE_FIELDS
    + LOGISTICS_FIELDS
)

RANK_FIELDS = [
    "population_rank",
    "available_manpower_rank",
    "fit_for_service_rank",
    "military_age_rank",
    "active_personnel_rank",
    "reserve_personnel_rank",
    "paramilitary_rank",
    "aircraft_rank",
    "fighters_rank",
    "attack_aircraft_rank",
    "transports_rank",
    "trainers_rank",
    "special_mission_rank",
    "tankers_rank",
    "helicopters_rank",
    "attack_helicopters_rank",
    "tanks_rank",
    "vehicles_rank",
    "artillery_sp_rank",
    "artillery_towed_rank",
    "mlrs_rank",
    "ships_rank",
    "aircraft_carriers_rank",
    "helicopter_carriers_rank",
    "destroyers_rank",
    "frigates_rank",
    "corvettes_rank",
    "submarines_rank",
    "patrol_vessels_rank",
    "mine_warfare_rank",
    "oil_production_rank",
    "oil_consumption_rank",
    "oil_reserves_rank",
    "natural_gas_production_rank",
    "natural_gas_consumption_rank",
    "natural_gas_reserves_rank",
    "coal_production_rank",
    "coal_consumption_rank",
    "coal_reserves_rank",
    "labor_force_rank",
    "merchant_marine_rank",
    "ports_rank",
    "roadways_rank",
    "railways_rank",
    "airports_rank",
    "defense_budget_rank",
    "external_debt_rank",
    "foreign_reserves_rank",
    "purchasing_power_rank",
    "square_land_area_rank",
    "coastline_rank",
    "shared_border_rank",
    "waterways_rank",
]

TARGET_FIELDS = CORE_FIELDS + VALUE_FIELDS + RANK_FIELDS
PERCENT_FIELDS = {"internet_coverage"}


@dataclass(frozen=True)
class PdfLine:
    page: int
    text: str


@dataclass
class Extracted:
    raw_value: str | None = None
    raw_snippet: str | None = None
    page: int | None = None


SIMPLE_FIELD_ALIASES: dict[str, tuple[tuple[str, ...], ...]] = {
    "purchasing_power": (("Purchasing Power", "Parity:"),),
    "foreign_reserves": (("Foreign", "Exchange/Gold:"),),
    "defense_budget": (("Defense Budget:",),),
    "external_debt": (("External Debt:",),),
    "square_land_area": (("Square Land Area:",),),
    "coastline": (("Coastline Coverage:",),),
    "shared_borders": (("Shared Borders:",),),
    "waterways": (("Waterways (usable):",),),
    "total_population": (("Total Population:",),),
    "available_manpower": (("Available Manpower",),),
    "fit_for_service": (("Fit-for-Service",),),
    "reaching_military_age_annually": (("Reaching Mil Age", "Annually"),),
    "total_military_personnel": (("Tot Mil. Personnel", "(est.)"), ("Tot Mil. Personnel",)),
    "active_personnel": (("Active Personnel",),),
    "reserve_personnel": (("Reserve Personnel",),),
    "paramilitary_personnel": (("Paramilitary",),),
    "air_force_personnel": (("Air Force Personnel*",),),
    "army_personnel": (("Army Personnel*",),),
    "navy_personnel": (("Navy Personnel*",),),
    "yearly_mobilization_potential": (("Yearly Mobilization", "Potential"),),
    "naval_total_assets": (("Total Assets:",),),
    "naval_total_tonnage": (("Total Tonnage:",),),
    "aircraft_carriers": (("Aircraft Carriers:",),),
    "helicopter_carriers": (("Helicopter Carriers:",),),
    "destroyers": (("Destroyers:",),),
    "frigates": (("Frigates:",),),
    "corvettes": (("Corvettes:",),),
    "submarines": (("Submarines:",),),
    "patrol_vessels": (("Patrol Vessels:",),),
    "mine_warfare": (("Mine Warfare:",),),
    "oil_production": (("Oil Production:",),),
    "oil_consumption": (("Oil Consumption:",),),
    "oil_deficit": (("Oil Deficit:",),),
    "oil_proven_reserves": (("Oil Proven Reserves:",),),
    "natural_gas_production": (("Natural Gas", "Production:"),),
    "natural_gas_consumption": (("Natural Gas", "Consumption:"),),
    "natural_gas_deficit": (("Natural Gas Deficit:",),),
    "natural_gas_proven_reserves": (("Nat.Gas Proven Rez:",),),
    "coal_production": (("Coal Production:",),),
    "coal_consumption": (("Coal Consumption:",),),
    "coal_deficit": (("Coal Deficit:",),),
    "coal_proven_reserves": (("Coal Proven", "Reserves:"), ("Coal Proven Reserves:",)),
    "internet_coverage": (("Internet Coverage:",),),
    "labor_force": (("Labor Force:",),),
    "merchant_marine_fleet": (("Merchant Marine", "Fleet:"), ("Fleet:",)),
    "ports_trade_terminals": (("Ports / Trade", "Terminals:"), ("Terminals:",)),
    "airports": (("Airports:",),),
    "roadway_coverage": (("Roadway Coverage:",),),
    "railway_coverage": (("Railway Coverage:",),),
}

STOCK_FIELD_SPECS = [
    (("Aircraft Total:",), "aircraft_total", "aircraft_readiness"),
    (("Fighters:",), "fighters", "fighters_readiness"),
    (("Attack Types:",), "attack_aircraft", "attack_aircraft_readiness"),
    (("Transports (Fixed-", "Wing):"), "transports", "transports_readiness"),
    (("Trainers:",), "trainers", "trainers_readiness"),
    (("Special-Mission:",), "special_mission", "special_mission_readiness"),
    (("Tanker Fleet:",), "tanker_fleet", "tanker_fleet_readiness"),
    (("Helicopters:",), "helicopters", "helicopters_readiness"),
    (("Attack Helicopters:",), "attack_helicopters", "attack_helicopters_readiness"),
    (("Tanks:",), "tanks", "tanks_readiness"),
    (("Vehicles:",), "vehicles", "vehicles_readiness"),
    (("Self-Propelled", "Artillery:"), "self_propelled_artillery", "self_propelled_artillery_readiness"),
    (("Towed Artillery:",), "towed_artillery", "towed_artillery_readiness"),
    (("MLRS (Rocket", "Artillery):"), "mlrs", "mlrs_readiness"),
]

OVERVIEW_RANK_LABELS: dict[str, tuple[str, ...]] = {
    "population_rank": ("Population",),
    "available_manpower_rank": ("Avail.Manpwr",),
    "fit_for_service_rank": ("Fit-for-Serv",),
    "military_age_rank": ("Mil.Age.Annu",),
    "active_personnel_rank": ("Active Persn",),
    "reserve_personnel_rank": ("Rez.Persn",),
    "paramilitary_rank": ("Paramil.Persn",),
    "aircraft_rank": ("Aircraft",),
    "fighters_rank": ("Fighters",),
    "attack_aircraft_rank": ("Attack Types",),
    "transports_rank": ("Transports",),
    "trainers_rank": ("Trainers",),
    "special_mission_rank": ("Spl.Mission",),
    "tankers_rank": ("Tankers",),
    "helicopters_rank": ("Helicopters",),
    "attack_helicopters_rank": ("Attack Helos",),
    "tanks_rank": ("Tanks",),
    "vehicles_rank": ("Vehicles",),
    "artillery_sp_rank": ("Artillery (SP)",),
    "artillery_towed_rank": ("Art.(Towed)",),
    "mlrs_rank": ("MLRS",),
    "ships_rank": ("Ships",),
    "aircraft_carriers_rank": ("AC Carriers",),
    "helicopter_carriers_rank": ("Helo.Carriers",),
    "destroyers_rank": ("Destroyers",),
    "frigates_rank": ("Frigates",),
    "corvettes_rank": ("Corvettes",),
    "submarines_rank": ("Submarines",),
    "patrol_vessels_rank": ("Offsh.Patrol",),
    "mine_warfare_rank": ("Mine Warfare",),
    "oil_production_rank": ("Oil Prod",),
    "oil_consumption_rank": ("Oil Consump",),
    "oil_reserves_rank": ("Oil Rez.Proven",),
    "natural_gas_production_rank": ("Nat.Gas Prod",),
    "natural_gas_consumption_rank": ("NatGas Cons",),
    "natural_gas_reserves_rank": ("NatGas Rez",),
    "coal_production_rank": ("Coal Prod",),
    "coal_consumption_rank": ("Coal Consump",),
    "coal_reserves_rank": ("Coal Rez.Prvn",),
    "labor_force_rank": ("Labor Force",),
    "merchant_marine_rank": ("Merch.Marines",),
    "ports_rank": ("Ports & Harbors",),
    "roadways_rank": ("Roadways",),
    "railways_rank": ("Railways",),
    "airports_rank": ("Airports",),
    "defense_budget_rank": ("Def.Budget",),
    "external_debt_rank": ("Ext.Debt",),
    "foreign_reserves_rank": ("Foreign Rez",),
    "purchasing_power_rank": ("Purch Power", "Purch.Power"),
    "square_land_area_rank": ("Sq Land Area", "Sq.Land Area"),
    "coastline_rank": ("Coastline",),
    "shared_border_rank": ("Shared Border",),
    "waterways_rank": ("Waterways",),
}

FIN_GEO_SECTION_RANK_FIELDS = [
    "purchasing_power_rank",
    "foreign_reserves_rank",
    "defense_budget_rank",
    "external_debt_rank",
    "square_land_area_rank",
    "coastline_rank",
    "shared_border_rank",
    "waterways_rank",
]

def normalize_line(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold().strip())


KNOWN_FIELD_LABEL_TOKENS = set()
for aliases in SIMPLE_FIELD_ALIASES.values():
    for alias in aliases:
        KNOWN_FIELD_LABEL_TOKENS.update(alias)
for alias, _, _ in STOCK_FIELD_SPECS:
    KNOWN_FIELD_LABEL_TOKENS.update(alias)
KNOWN_FIELD_LABEL_TOKENS.update({"Stock:", "Readiness:"})

OVERVIEW_LABEL_LOOKUP = {
    normalize_line(label)
    for labels in OVERVIEW_RANK_LABELS.values()
    for label in labels
}


def label_matches(line: str, token: str) -> bool:
    lhs = normalize_line(line).rstrip(":")
    rhs = normalize_line(token).rstrip(":")
    return lhs == rhs


def line_is_known_label(line: str) -> bool:
    return any(label_matches(line, token) for token in KNOWN_FIELD_LABEL_TOKENS)


def is_footer_or_noise(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if re.match(r"^\d{1,2}\.\d{1,2}\.\d{4}\s+\d{1,2}:\d{2}", stripped):
        return True
    if re.match(r"^\d{4}\s+.+\s+Military Strength$", stripped):
        return True
    if stripped.startswith("https://www.globalfirepower.com/"):
        return True
    if re.match(r"^\d+\s*/\s*\d+$", stripped):
        return True
    return False


def is_rank_token(line: str) -> bool:
    text = line.strip()
    if re.fullmatch(r"\d+\s*/\s*\d+(?:\s+\d+)?", text):
        return True
    if re.fullmatch(r"\d+\s*/", text):
        return True
    return False


def is_unit_only(line: str) -> bool:
    return normalize_line(line) in {"usd", "cu.m", "bbl", "mt", "tonnes", "km", "k"}


def looks_like_value(line: str) -> bool:
    text = line.strip()
    if is_footer_or_noise(text) or is_rank_token(text) or is_unit_only(text):
        return False
    if normalize_line(text) in {"landlocked", "none", "n/a", "na"}:
        return True
    return bool(re.search(r"[$+\-]?\d", text))


def match_label_at(lines: list[PdfLine], start: int, tokens: tuple[str, ...], max_gap: int = 3) -> int | None:
    if start >= len(lines) or not label_matches(lines[start].text, tokens[0]):
        return None
    pos = start + 1
    while pos < len(lines) and label_matches(lines[pos].text, tokens[0]):
        pos += 1
    for token in tokens[1:]:
        found = None
        for probe in range(pos, min(len(lines), pos + max_gap + 1)):
            if label_matches(lines[probe].text, token):
                found = probe
                break
        if found is None:
            return None
        pos = found + 1
        while pos < len(lines) and label_matches(lines[pos].text, token):
            pos += 1
    return pos


def find_label_positions(
    lines: list[PdfLine],
    aliases: tuple[tuple[str, ...], ...],
    start_at: int = 0,
) -> Iterable[tuple[int, int, tuple[str, ...]]]:
    for idx in range(start_at, len(lines)):
        for alias in aliases:
            after = match_label_at(lines, idx, alias)
            if after is not None:
                yield idx, after, alias


def find_first_label_index(lines: list[PdfLine], aliases: tuple[tuple[str, ...], ...], start_at: int = 0) -> int | None:
    for start, _, _ in find_label_positions(lines, aliases, start_at=start_at):
        return start
    return None


def context_start_for_field(lines: list[PdfLine], field: str) -> int:
    if field in LOGISTICS_FIELDS:
        return find_first_label_index(lines, (("Internet Coverage:",),)) or 0
    if field in GEOGRAPHY_FIELDS:
        return find_first_label_index(lines, (("Square Land Area:",),)) or 0
    if field in MANPOWER_FIELDS:
        return find_first_label_index(lines, (("Total Population:",),)) or 0
    if field in NATURAL_RESOURCE_FIELDS:
        return find_first_label_index(lines, (("Oil Production:",),)) or 0
    return 0


def capture_value_after_label(
    lines: list[PdfLine],
    aliases: tuple[tuple[str, ...], ...],
    start_at: int = 0,
    max_lookahead: int = 35,
) -> Extracted:
    for label_start, after_label, _ in find_label_positions(lines, aliases, start_at=start_at):
        for idx in range(after_label, min(len(lines), after_label + max_lookahead)):
            text = lines[idx].text.strip()
            if looks_like_value(text):
                raw_lines = [text]
                j = idx + 1
                while j < len(lines) and len(raw_lines) < 4:
                    next_text = lines[j].text.strip()
                    if next_text == text or is_unit_only(next_text):
                        raw_lines.append(next_text)
                        j += 1
                        continue
                    break
                snippet_end = min(len(lines), j + 4)
                snippet = "\n".join(line.text for line in lines[label_start:snippet_end])
                return Extracted("\n".join(raw_lines), snippet, lines[idx].page)
            if line_is_known_label(text):
                break
            if is_footer_or_noise(text):
                continue
        # Try later occurrences; page splits often leave a dangling first label.
    return Extracted()


def capture_stock_readiness(
    lines: list[PdfLine],
    label_alias: tuple[str, ...],
    start_at: int = 0,
    max_lookahead: int = 45,
) -> tuple[Extracted, Extracted]:
    aliases = (label_alias,)
    for label_start, after_label, _ in find_label_positions(lines, aliases, start_at=start_at):
        stock: Extracted | None = None
        readiness: Extracted | None = None
        for idx in range(after_label, min(len(lines), after_label + max_lookahead)):
            text = lines[idx].text.strip()
            if normalize_line(text).startswith("stock:") and stock is None:
                stock = Extracted(text, "\n".join(line.text for line in lines[label_start : min(len(lines), idx + 5)]), lines[idx].page)
            elif normalize_line(text).startswith("readiness:") and readiness is None:
                readiness = Extracted(text, "\n".join(line.text for line in lines[label_start : min(len(lines), idx + 5)]), lines[idx].page)
            elif stock is None and line_is_known_label(text):
                break
            if stock is not None and readiness is not None:
                return stock, readiness
    return Extracted(), Extracted()


def collect_unique_value_lines(
    lines: list[PdfLine],
    start_aliases: tuple[tuple[str, ...], ...],
    end_aliases: tuple[tuple[str, ...], ...],
    max_lines: int = 100,
) -> list[Extracted]:
    start_idx = find_first_label_index(lines, start_aliases)
    if start_idx is None:
        return []
    end_idx = find_first_label_index(lines, end_aliases, start_at=start_idx + 1)
    if end_idx is None:
        end_idx = min(len(lines), start_idx + max_lines)
    else:
        end_idx = min(end_idx, start_idx + max_lines)

    values: list[Extracted] = []
    previous_number = None
    for idx in range(start_idx + 1, end_idx):
        text = lines[idx].text.strip()
        if not looks_like_value(text) or line_is_known_label(text):
            continue
        number = parse_numeric_value(text)
        if pd.isna(number):
            continue
        if previous_number is not None and number == previous_number:
            continue
        previous_number = number
        values.append(Extracted(text, "\n".join(line.text for line in lines[max(start_idx, idx - 4) : min(len(lines), idx + 5)]), lines[idx].page))
    return values


def fill_ordered_manpower_fallbacks(
    lines: list[PdfLine],
    values: dict[str, str | None],
    debug_rows: list[dict[str, object]],
) -> None:
    ordered_fields = [
        "total_population",
        "available_manpower",
        "fit_for_service",
        "reaching_military_age_annually",
    ]
    if all(values.get(field) for field in ordered_fields):
        return
    candidates = collect_unique_value_lines(lines, (("Total Population:",),), (("Tot Mil. Personnel", "(est.)"),))
    if len(candidates) < 4:
        return
    for field, extracted in zip(ordered_fields, candidates[:4]):
        if values.get(field):
            continue
        values[field] = extracted.raw_value
        debug_rows.append(
            {
                "field": field,
                "page": extracted.page,
                "raw_value": extracted.raw_value,
                "raw_snippet": extracted.raw_snippet,
            }
        )


def parse_filename(pdf_path: Path) -> tuple[int | None, str]:
    match = re.match(r"^(?P<year>\d{4})\s+(?P<country>.+?)\s+Military Strength\.pdf$", pdf_path.name)
    if not match:
        return None, pdf_path.stem
    return int(match.group("year")), match.group("country").strip()


def extract_lines_pymupdf(pdf_path: Path) -> tuple[list[PdfLine], str]:
    doc = fitz.open(pdf_path)
    lines: list[PdfLine] = []
    page_texts = []
    for page_index in range(doc.page_count):
        text = doc.load_page(page_index).get_text("text") or ""
        page_texts.append(text)
        for line in text.splitlines():
            if line.strip():
                lines.append(PdfLine(page_index + 1, line.strip()))
    return lines, "\n".join(page_texts)


def extract_lines_pdfplumber(pdf_path: Path) -> tuple[list[PdfLine], str]:
    lines: list[PdfLine] = []
    page_texts = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pdfplumber.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf.pages):
                text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
                page_texts.append(text)
                for line in text.splitlines():
                    if line.strip():
                        lines.append(PdfLine(page_index + 1, line.strip()))
    return lines, "\n".join(page_texts)


def extract_pdf_text(pdf_path: Path) -> tuple[list[PdfLine], str, str]:
    lines, text = extract_lines_pymupdf(pdf_path)
    if len(text.strip()) >= 1000:
        return lines, text, "pymupdf"
    fallback_lines, fallback_text = extract_lines_pdfplumber(pdf_path)
    if len(fallback_text.strip()) > len(text.strip()):
        return fallback_lines, fallback_text, "pdfplumber"
    return lines, text, "pymupdf"


def extract_core_fields(pdf_path: Path, full_text: str) -> dict[str, str | int | None]:
    year, country = parse_filename(pdf_path)
    row: dict[str, str | int | None] = {
        "country": country,
        "year": year,
        "source_file": pdf_path.name,
        "source_url": None,
        "rank": None,
        "total_countries": None,
        "pwrindx": None,
        "last_reviewed_date": None,
    }

    url_match = re.search(r"https?://www\.globalfirepower\.com/[^\s]+", full_text)
    if url_match:
        row["source_url"] = url_match.group(0).strip()

    rank_match = re.search(
        r"For\s+(\d{4}),\s+.+?\s+is\s+ranked\s+([\d,]+)\s+of\s+([\d,]+)",
        full_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if rank_match:
        row["rank"] = rank_match.group(2)
        row["total_countries"] = rank_match.group(3)

    pwr_match = re.search(r"PwrIndx\*\s+score\s+of\s+([0-9.]+)", full_text, flags=re.IGNORECASE)
    if pwr_match:
        row["pwrindx"] = pwr_match.group(1)

    reviewed_match = re.search(r"last\s+reviewed\s+on\s+([0-9]{1,2}/[0-9]{1,2}/[0-9]{4})", full_text, flags=re.IGNORECASE)
    if reviewed_match:
        row["last_reviewed_date"] = reviewed_match.group(1)

    return row


def get_overview_window(lines: list[PdfLine]) -> list[PdfLine]:
    start = next((idx for idx, line in enumerate(lines) if normalize_line(line.text) == "overview [+]"), 0)
    return lines[start : min(len(lines), start + 260)]


def parse_rank_from_line(line: str) -> str | None:
    slash_match = re.search(r"(\d+)\s*/\s*\d+", line)
    if slash_match:
        return slash_match.group(1)
    bare_match = re.fullmatch(r"\d{1,3}", line.strip())
    if bare_match:
        return bare_match.group(0)
    return None


def extract_overview_ranks(lines: list[PdfLine]) -> dict[str, str | None]:
    window = get_overview_window(lines)
    norm_to_indexes: dict[str, list[int]] = {}
    for idx, line in enumerate(window):
        norm_to_indexes.setdefault(normalize_line(line.text), []).append(idx)

    ranks: dict[str, str | None] = {field: None for field in RANK_FIELDS}
    for field, labels in OVERVIEW_RANK_LABELS.items():
        label_indexes: list[int] = []
        for label in labels:
            label_indexes.extend(norm_to_indexes.get(normalize_line(label), []))
        if not label_indexes:
            continue
        label_idx = min(label_indexes)
        for probe in range(label_idx + 1, min(len(window), label_idx + 8)):
            probe_norm = normalize_line(window[probe].text)
            if probe_norm in OVERVIEW_LABEL_LOOKUP:
                break
            rank = parse_rank_from_line(window[probe].text)
            if rank is not None:
                ranks[field] = rank
                break
    return ranks


def collect_rank_tokens_between(
    lines: list[PdfLine],
    start_aliases: tuple[tuple[str, ...], ...],
    end_aliases: tuple[tuple[str, ...], ...] | None = None,
    max_lines: int = 220,
) -> list[str]:
    start_idx = find_first_label_index(lines, start_aliases)
    if start_idx is None:
        return []
    end_idx = min(len(lines), start_idx + max_lines)
    if end_aliases:
        candidate = find_first_label_index(lines, end_aliases, start_at=start_idx + 1)
        if candidate is not None:
            end_idx = min(end_idx, candidate)
    ranks: list[str] = []
    for line in lines[start_idx:end_idx]:
        for match in re.finditer(r"(\d+)\s*/\s*\d+", line.text):
            ranks.append(match.group(1))
    return ranks


def fill_rank_fallbacks(lines: list[PdfLine], ranks: dict[str, str | None]) -> None:
    fin_geo = collect_rank_tokens_between(lines, (("Purchasing Power", "Parity:"),), (("Aircraft Total:",),), max_lines=260)
    if len(fin_geo) >= 8:
        for field, value in zip(FIN_GEO_SECTION_RANK_FIELDS, fin_geo[:8]):
            ranks[field] = ranks.get(field) or value

    manpower = collect_rank_tokens_between(lines, (("Total Population:",),), (("Aircraft Total:",),), max_lines=180)
    for field, value in zip(
        [
            "population_rank",
            "available_manpower_rank",
            "fit_for_service_rank",
            "military_age_rank",
            "active_personnel_rank",
            "reserve_personnel_rank",
            "paramilitary_rank",
        ],
        manpower[:7],
    ):
        ranks[field] = ranks.get(field) or value

    air = collect_rank_tokens_between(lines, (("Aircraft Total:",),), (("Tanks:",),), max_lines=180)
    for field, value in zip(
        [
            "aircraft_rank",
            "fighters_rank",
            "attack_aircraft_rank",
            "transports_rank",
            "trainers_rank",
            "special_mission_rank",
            "tankers_rank",
            "helicopters_rank",
            "attack_helicopters_rank",
        ],
        air[:9],
    ):
        ranks[field] = ranks.get(field) or value

    land = collect_rank_tokens_between(lines, (("Tanks:",),), (("Total Assets:",),), max_lines=160)
    for field, value in zip(
        ["tanks_rank", "vehicles_rank", "artillery_sp_rank", "artillery_towed_rank", "mlrs_rank"],
        land[:5],
    ):
        ranks[field] = ranks.get(field) or value

    natural = collect_rank_tokens_between(lines, (("Oil Production:",),), (("Internet Coverage:",),), max_lines=260)
    for field, value in zip(
        [
            "oil_production_rank",
            "oil_consumption_rank",
            "oil_reserves_rank",
            "natural_gas_production_rank",
            "natural_gas_consumption_rank",
            "natural_gas_reserves_rank",
            "coal_production_rank",
            "coal_consumption_rank",
            "coal_reserves_rank",
        ],
        natural[:9],
    ):
        ranks[field] = ranks.get(field) or value


def extract_values(lines: list[PdfLine]) -> tuple[dict[str, str | None], list[dict[str, object]]]:
    values: dict[str, str | None] = {field: None for field in VALUE_FIELDS}
    debug_rows: list[dict[str, object]] = []

    for field, aliases in SIMPLE_FIELD_ALIASES.items():
        start_at = context_start_for_field(lines, field)
        extracted = capture_value_after_label(lines, aliases, start_at=start_at)
        values[field] = extracted.raw_value
        debug_rows.append(
            {
                "field": field,
                "page": extracted.page,
                "raw_value": extracted.raw_value,
                "raw_snippet": extracted.raw_snippet,
            }
        )

    fill_ordered_manpower_fallbacks(lines, values, debug_rows)

    for label_alias, value_field, readiness_field in STOCK_FIELD_SPECS:
        start_at = 0
        if value_field in LAND_FIELDS:
            start_at = find_first_label_index(lines, (("Tanks:",),)) or 0
        stock, readiness = capture_stock_readiness(lines, label_alias, start_at=start_at)
        values[value_field] = stock.raw_value
        values[readiness_field] = readiness.raw_value
        debug_rows.append(
            {
                "field": value_field,
                "page": stock.page,
                "raw_value": stock.raw_value,
                "raw_snippet": stock.raw_snippet,
            }
        )
        debug_rows.append(
            {
                "field": readiness_field,
                "page": readiness.page,
                "raw_value": readiness.raw_value,
                "raw_snippet": readiness.raw_snippet,
            }
        )

    return values, debug_rows


def extract_one_pdf(pdf_path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    lines, full_text, engine = extract_pdf_text(pdf_path)
    row: dict[str, object] = {field: None for field in TARGET_FIELDS}
    row.update(extract_core_fields(pdf_path, full_text))
    values, debug_rows = extract_values(lines)
    row.update(values)

    ranks = extract_overview_ranks(lines)
    fill_rank_fallbacks(lines, ranks)
    row.update(ranks)

    row["extraction_engine"] = engine
    row["text_line_count"] = len(lines)

    for debug_row in debug_rows:
        debug_row.update(
            {
                "country": row["country"],
                "year": row["year"],
                "source_file": row["source_file"],
                "parsed_preview": parse_numeric_value(
                    debug_row.get("raw_value"),
                    field=str(debug_row.get("field") or ""),
                ),
            }
        )
    return row, debug_rows


def collapse_pdfplumber_doubled_chars(text: str) -> str:
    # pdfplumber sometimes emits strings like "2200,,995533,,000000".
    if len(text) < 8:
        return text
    pairs = [text[i : i + 2] for i in range(0, len(text) - 1, 2)]
    repeated_pairs = sum(1 for pair in pairs if len(pair) == 2 and pair[0] == pair[1])
    if pairs and repeated_pairs / len(pairs) >= 0.65:
        return text[::2]
    return text


def choose_numeric_fragment(raw: object) -> str | None:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    text = str(raw).replace("\r", "\n")
    for original_line in text.splitlines():
        line = collapse_pdfplumber_doubled_chars(original_line.strip())
        if not line or is_rank_token(line) or is_footer_or_noise(line) or is_unit_only(line):
            continue
        if normalize_line(line) in {"landlocked", "none", "n/a", "na"}:
            return "0"
        if ":" in line and re.search(r"[A-Za-z]", line.split(":", 1)[0]):
            line = line.split(":", 1)[1].strip()
        if re.search(r"[$+\-]?\d", line):
            return line
    compact = collapse_pdfplumber_doubled_chars(text.strip())
    return compact or None


def collapse_repeated_number_groups(groups: list[str]) -> list[str]:
    if len(groups) >= 4 and len(groups) % 2 == 0:
        pairs_are_duplicated = all(groups[index] == groups[index + 1] for index in range(0, len(groups), 2))
        if pairs_are_duplicated:
            return [groups[index] for index in range(0, len(groups), 2)]
    return groups


def parse_numeric_value(raw: object, field: str = "") -> float:
    fragment = choose_numeric_fragment(raw)
    if fragment is None:
        return np.nan

    fragment = fragment.strip().replace("*", "")
    fragment = re.sub(r"\([^)]*\)", "", fragment)
    fragment = fragment.replace("$", "")
    fragment = re.sub(r"\b(USD|km|bbl|cu\.m|mt|tonnes|recruits|incl\.Tot\.Persnnl|cumulative)\b", "", fragment, flags=re.IGNORECASE)
    fragment = fragment.strip()

    if field in PERCENT_FIELDS or "%" in fragment:
        percent_match = re.search(r"([+\-]?\d+(?:[.,]\d+)?|[+\-]?\d+\s+\d)\s*%", fragment)
        if percent_match:
            percent_text = percent_match.group(1).replace(",", ".")
            if " " in percent_text and "." not in percent_text:
                left, right = percent_text.split(None, 1)
                percent_text = f"{left}.{right}"
            try:
                return float(percent_text)
            except ValueError:
                return np.nan

    decimal_match = re.search(r"[+\-]?\d+\.\d+", fragment)
    if decimal_match and (field == "pwrindx" or "," not in decimal_match.group(0)):
        try:
            return float(decimal_match.group(0))
        except ValueError:
            return np.nan

    sign = -1 if re.search(r"-\s*\d", fragment) else 1
    groups = re.findall(r"\d+", fragment)
    if not groups:
        return np.nan
    groups = collapse_repeated_number_groups(groups)
    digits = "".join(groups)
    if not digits:
        return np.nan
    try:
        return float(sign * int(digits))
    except ValueError:
        return np.nan


def build_outputs(input_dir: Path, output_dir: Path) -> None:
    table_dir = output_dir / "tables"
    report_dir = output_dir / "reports"
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(input_dir.glob("2026 * Military Strength.pdf"))
    if not pdfs:
        raise SystemExit(f"No matching PDF files found in {input_dir}")

    rows: list[dict[str, object]] = []
    debug_rows: list[dict[str, object]] = []
    for pdf_path in pdfs:
        row, pdf_debug = extract_one_pdf(pdf_path)
        rows.append(row)
        debug_rows.extend(pdf_debug)

    output_columns = TARGET_FIELDS + ["extraction_engine", "text_line_count"]
    raw_df = pd.DataFrame(rows)
    raw_df = raw_df.reindex(columns=output_columns)
    raw_df.to_csv(table_dir / "gfp_2026_raw_extracted.csv", index=False)

    debug_df = pd.DataFrame(debug_rows)
    debug_df = debug_df.reindex(
        columns=["country", "year", "source_file", "field", "page", "raw_value", "parsed_preview", "raw_snippet"]
    )
    debug_df.to_csv(report_dir / "gfp_2026_debug_snippets.csv", index=False)

    print(f"Extracted {len(raw_df)} PDF rows")
    print(f"Wrote {table_dir / 'gfp_2026_raw_extracted.csv'}")
    print(f"Wrote {report_dir / 'gfp_2026_debug_snippets.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract raw Global Firepower 2026 PDF data.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Folder containing GFP PDF files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Folder for output tables and reports.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_outputs(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
