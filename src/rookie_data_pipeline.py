"""Reusable data collection and dataset building for rookie projections."""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FINAL_DIR = PROJECT_ROOT / "data" / "final"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
REPORT_DIR = PROJECT_ROOT / "reports"

POSITIONS = ["QB", "RB", "WR", "TE"]
SPORTS_REFERENCE_BASE = "https://www.sports-reference.com"
PFR_BASE = "https://www.pro-football-reference.com"


@dataclass(frozen=True)
class PositionSpec:
    college_candidates: tuple[Path, ...]
    nfl_candidates: tuple[Path, ...]
    college_stat_map: dict[str, str]
    target_columns: tuple[str, ...]


POSITION_SPECS: dict[str, PositionSpec] = {
    "QB": PositionSpec(
        college_candidates=(RAW_DIR / "college_qb_data_2.csv", RAW_DIR / "college_qb_data.csv"),
        nfl_candidates=(RAW_DIR / "nfl_qb_data_2.csv", RAW_DIR / "nfl_qb_data.csv"),
        college_stat_map={
            "games": "games",
            "pass_cmp": "pass_cmp",
            "pass_att": "pass_att",
            "cmp_pct": "cmp%",
            "pass_yds": "yds",
            "pass_td": "TDs",
            "pass_td_pct": "TD%",
            "pass_int": "Int",
            "pass_int_pct": "Int%",
            "pass_rating": "Rating",
            "rush_att": "Rush Attempts",
            "rush_yds": "Rush Yards",
            "rush_td": "Rush TDs",
        },
        target_columns=(
            "nfl_passingYDS",
            "nfl_passingTDS",
            "nfl_passingINT",
            "nfl_rushingYDS",
            "nfl_rushingTDS",
            "fantasy_points",
        ),
    ),
    "RB": PositionSpec(
        college_candidates=(RAW_DIR / "college_rb_data_15-24.csv", RAW_DIR / "college_rb_data_2.csv"),
        nfl_candidates=(RAW_DIR / "nfl_rb_data_2.csv", RAW_DIR / "nfl_rb_data.csv"),
        college_stat_map={
            "games": "games",
            "rush_att": "rushATT",
            "rush_yds": "rushYDS",
            "rush_td": "rushTDS",
            "rec": "recs",
            "rec_yds": "recYDS",
            "rec_td": "recTDS",
        },
        target_columns=(
            "nfl_rushingYDS",
            "nfl_rushingTDS",
            "nfl_recs",
            "nfl_recYDS",
            "nfl_recTDS",
            "fantasy_points",
        ),
    ),
    "WR": PositionSpec(
        college_candidates=(RAW_DIR / "college_wr_data_2.csv", PROJECT_ROOT / "models" / "college_wr_data_5.csv"),
        nfl_candidates=(RAW_DIR / "nfl_wr_data_2.csv", PROJECT_ROOT / "models" / "nfl_wr_data_5.csv"),
        college_stat_map={
            "games": "games",
            "rec": "recs",
            "rec_yds": "recYDS",
            "rec_td": "recTDS",
            "rush_att": "rushAttempts",
            "rush_yds": "rushYards",
            "rush_td": "rushTDS",
        },
        target_columns=("nfl_recs", "nfl_recYDS", "nfl_recTDS", "fantasy_points"),
    ),
    "TE": PositionSpec(
        college_candidates=(RAW_DIR / "college_te_data_15-24.csv", RAW_DIR / "college_te_data_2.csv"),
        nfl_candidates=(RAW_DIR / "nfl_te_data_2.csv", RAW_DIR / "nfl_te_data.csv"),
        college_stat_map={
            "games": "games",
            "rec": "recs",
            "rec_yds": "recYDS",
            "rec_td": "recTDS",
            "rush_att": "rushAttempts",
            "rush_yds": "rushYards",
            "rush_td": "rushTDS",
        },
        target_columns=("nfl_recs", "nfl_recYDS", "nfl_recTDS", "fantasy_points"),
    ),
}


COMMON_FEATURES = [
    "age",
    "draft_pick",
    "college_games",
    "total_yards",
    "total_tds",
    "scrimmage_yds_pg",
    "tds_pg",
    "volume_pg",
    "yards_per_touch",
    "latest_vs_prev_yards_pg",
    "latest_vs_prev_tds_pg",
    "draft_pick_log",
    "draft_pick_inverse",
    "conference_ACC",
    "conference_Big 12",
    "conference_Big Ten",
    "conference_Other",
    "conference_SEC",
    "position_QB",
    "position_RB",
    "position_WR",
    "position_TE",
]


POSITION_FEATURES = {
    "QB": [
        "pass_yds_pg",
        "pass_tds_pg",
        "pass_int_pg",
        "rush_yds_pg",
        "rush_tds_pg",
        "completion_pct",
        "pass_rating",
        "yards_per_attempt",
    ],
    "RB": [
        "rush_att_pg",
        "rush_yds_pg",
        "rush_tds_pg",
        "rec_pg",
        "rec_yds_pg",
        "rec_tds_pg",
        "rush_yards_per_att",
    ],
    "WR": [
        "rec_pg",
        "rec_yds_pg",
        "rec_tds_pg",
        "rush_yds_pg",
        "rush_tds_pg",
        "yards_per_rec",
    ],
    "TE": [
        "rec_pg",
        "rec_yds_pg",
        "rec_tds_pg",
        "rush_yds_pg",
        "rush_tds_pg",
        "yards_per_rec",
    ],
}


CONFERENCE_MAP = {
    "Alabama": "SEC",
    "Arkansas": "SEC",
    "Auburn": "SEC",
    "Florida": "SEC",
    "Georgia": "SEC",
    "Kentucky": "SEC",
    "LSU": "SEC",
    "Mississippi": "SEC",
    "Mississippi St.": "SEC",
    "Missouri": "SEC",
    "Oklahoma": "SEC",
    "South Carolina": "SEC",
    "Tennessee": "SEC",
    "Texas": "SEC",
    "Texas A&M": "SEC",
    "Vanderbilt": "SEC",
    "Clemson": "ACC",
    "Duke": "ACC",
    "Florida St.": "ACC",
    "Georgia Tech": "ACC",
    "Louisville": "ACC",
    "Miami (FL)": "ACC",
    "North Carolina": "ACC",
    "North Carolina St.": "ACC",
    "Pittsburgh": "ACC",
    "SMU": "ACC",
    "Syracuse": "ACC",
    "Virginia": "ACC",
    "Virginia Tech": "ACC",
    "Wake Forest": "ACC",
    "Arizona": "Big 12",
    "Arizona St.": "Big 12",
    "Baylor": "Big 12",
    "BYU": "Big 12",
    "Central Florida": "Big 12",
    "Cincinnati": "Big 12",
    "Colorado": "Big 12",
    "Houston": "Big 12",
    "Iowa St.": "Big 12",
    "Kansas": "Big 12",
    "Kansas St.": "Big 12",
    "Oklahoma St.": "Big 12",
    "TCU": "Big 12",
    "Texas Tech": "Big 12",
    "Utah": "Big 12",
    "West Virginia": "Big 12",
    "Illinois": "Big Ten",
    "Indiana": "Big Ten",
    "Iowa": "Big Ten",
    "Maryland": "Big Ten",
    "Michigan": "Big Ten",
    "Michigan St.": "Big Ten",
    "Minnesota": "Big Ten",
    "Nebraska": "Big Ten",
    "Northwestern": "Big Ten",
    "Ohio St.": "Big Ten",
    "Oregon": "Big Ten",
    "Penn St.": "Big Ten",
    "Purdue": "Big Ten",
    "Rutgers": "Big Ten",
    "UCLA": "Big Ten",
    "USC": "Big Ten",
    "Washington": "Big Ten",
    "Wisconsin": "Big Ten",
}


def first_existing(paths: tuple[Path, ...]) -> Path:
    for path in paths:
        if path.exists() and path.stat().st_size > 2:
            return path
    raise FileNotFoundError(f"No usable CSV found in: {[str(path) for path in paths]}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]
    return df


def extract_year(value: Any) -> float:
    match = re.search(r"(\d{4})", str(value))
    return float(match.group(1)) if match else np.nan


def safe_divide(numerator: Any, denominator: Any) -> Any:
    denominator = pd.Series(denominator).replace(0, np.nan)
    return pd.Series(numerator) / denominator


def add_conference_dummies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["conference"] = df.get("school", "").map(CONFERENCE_MAP).fillna("Other")
    for conf in ["ACC", "Big 12", "Big Ten", "Other", "SEC"]:
        df[f"conference_{conf}"] = (df["conference"] == conf).astype(int)
    return df


def add_position_dummies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for position in POSITIONS:
        df[f"position_{position}"] = (df["position"] == position).astype(int)
    return df


def fantasy_points(position: str, df: pd.DataFrame) -> pd.Series:
    def stat(column: str) -> pd.Series:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce").fillna(0)
        return pd.Series(0.0, index=df.index)

    if position == "QB":
        return (
            stat("nfl_passingYDS") * 0.04
            + stat("nfl_passingTDS") * 4
            - stat("nfl_passingINT") * 2
            + stat("nfl_rushingYDS") * 0.1
            + stat("nfl_rushingTDS") * 6
        )
    return (
        stat("nfl_rushingYDS") * 0.1
        + stat("nfl_rushingTDS") * 6
        + stat("nfl_recs")
        + stat("nfl_recYDS") * 0.1
        + stat("nfl_recTDS") * 6
    )


def build_position_dataset(position: str, include_targets: bool = True) -> pd.DataFrame:
    spec = POSITION_SPECS[position]
    college = normalize_columns(pd.read_csv(first_existing(spec.college_candidates)))
    college["position"] = position
    college["season"] = college["season"].apply(extract_year)
    for column in college.columns:
        if column not in {"name", "position", "school", "collegeURL"}:
            college[column] = pd.to_numeric(college[column], errors="coerce")

    college = college.dropna(subset=["name", "season"]).copy()
    college = add_conference_dummies(college)

    rows: list[dict[str, Any]] = []
    for name, group in college.groupby("name"):
        group = group.sort_values("season")
        latest = group.iloc[-1]
        previous = group.iloc[-2] if len(group) >= 2 else None
        row = base_feature_row(position, latest, previous)
        rows.append(row)

    features = pd.DataFrame(rows)
    if include_targets:
        nfl = normalize_columns(pd.read_csv(first_existing(spec.nfl_candidates)))
        nfl["position"] = position
        for column in nfl.columns:
            if column not in {"name", "position"}:
                nfl[column] = pd.to_numeric(nfl[column], errors="coerce")
        nfl["fantasy_points"] = fantasy_points(position, nfl)
        keep = ["name", "position", "nfl_games", *spec.target_columns]
        keep = [column for column in keep if column in nfl.columns]
        features = features.merge(nfl[keep], on=["name", "position"], how="inner")
    features = add_position_dummies(features)
    for column in COMMON_FEATURES + POSITION_FEATURES[position]:
        if column not in features.columns:
            features[column] = 0.0
    return features.replace([np.inf, -np.inf], np.nan)


def base_feature_row(position: str, latest: pd.Series, previous: pd.Series | None) -> dict[str, Any]:
    games = latest.get("games", np.nan)
    prev_games = previous.get("games", np.nan) if previous is not None else np.nan

    def latest_pg(column: str) -> float:
        return float(latest.get(column, 0) or 0) / games if games and np.isfinite(games) else np.nan

    def prev_pg(column: str) -> float:
        if previous is None or not prev_games or not np.isfinite(prev_games):
            return np.nan
        return float(previous.get(column, 0) or 0) / prev_games

    row: dict[str, Any] = {
        "name": latest.get("name"),
        "position": position,
        "school": latest.get("school"),
        "age": latest.get("age"),
        "draft_pick": latest.get("draft_pick"),
        "season": latest.get("season"),
        "college_games": games,
        "conference": latest.get("conference", "Other"),
        "conference_ACC": latest.get("conference_ACC", 0),
        "conference_Big 12": latest.get("conference_Big 12", 0),
        "conference_Big Ten": latest.get("conference_Big Ten", 0),
        "conference_Other": latest.get("conference_Other", 1),
        "conference_SEC": latest.get("conference_SEC", 0),
    }

    if position == "QB":
        total_yards = latest.get("yds", 0) + latest.get("Rush Yards", 0)
        total_tds = latest.get("TDs", 0) + latest.get("Rush TDs", 0)
        prev_yards = 0 if previous is None else previous.get("yds", 0) + previous.get("Rush Yards", 0)
        prev_tds = 0 if previous is None else previous.get("TDs", 0) + previous.get("Rush TDs", 0)
        row.update(
            {
                "pass_yds_pg": latest_pg("yds"),
                "pass_tds_pg": latest_pg("TDs"),
                "pass_int_pg": latest_pg("Int"),
                "rush_yds_pg": latest_pg("Rush Yards"),
                "rush_tds_pg": latest_pg("Rush TDs"),
                "completion_pct": latest.get("cmp%", np.nan),
                "pass_rating": latest.get("Rating", np.nan),
                "yards_per_attempt": latest.get("yds", np.nan) / latest.get("pass_att", np.nan),
                "volume_pg": latest.get("pass_att", 0) / games if games else np.nan,
            }
        )
    elif position == "RB":
        total_yards = latest.get("rushYDS", 0) + latest.get("recYDS", 0)
        total_tds = latest.get("rushTDS", 0) + latest.get("recTDS", 0)
        prev_yards = 0 if previous is None else previous.get("rushYDS", 0) + previous.get("recYDS", 0)
        prev_tds = 0 if previous is None else previous.get("rushTDS", 0) + previous.get("recTDS", 0)
        touches = latest.get("rushATT", 0) + latest.get("recs", 0)
        row.update(
            {
                "rush_att_pg": latest_pg("rushATT"),
                "rush_yds_pg": latest_pg("rushYDS"),
                "rush_tds_pg": latest_pg("rushTDS"),
                "rec_pg": latest_pg("recs"),
                "rec_yds_pg": latest_pg("recYDS"),
                "rec_tds_pg": latest_pg("recTDS"),
                "rush_yards_per_att": latest.get("rushYDS", np.nan) / latest.get("rushATT", np.nan),
                "volume_pg": touches / games if games else np.nan,
            }
        )
    else:
        total_yards = latest.get("recYDS", 0) + latest.get("rushYards", 0)
        total_tds = latest.get("recTDS", 0) + latest.get("rushTDS", 0)
        prev_yards = 0 if previous is None else previous.get("recYDS", 0) + previous.get("rushYards", 0)
        prev_tds = 0 if previous is None else previous.get("recTDS", 0) + previous.get("rushTDS", 0)
        touches = latest.get("recs", 0) + latest.get("rushAttempts", 0)
        row.update(
            {
                "rec_pg": latest_pg("recs"),
                "rec_yds_pg": latest_pg("recYDS"),
                "rec_tds_pg": latest_pg("recTDS"),
                "rush_yds_pg": latest_pg("rushYards"),
                "rush_tds_pg": latest_pg("rushTDS"),
                "yards_per_rec": latest.get("recYDS", np.nan) / latest.get("recs", np.nan),
                "volume_pg": touches / games if games else np.nan,
            }
        )

    row.update(
        {
            "total_yards": total_yards,
            "total_tds": total_tds,
            "scrimmage_yds_pg": total_yards / games if games else np.nan,
            "tds_pg": total_tds / games if games else np.nan,
            "yards_per_touch": total_yards / row.get("volume_pg", np.nan) / games
            if row.get("volume_pg", 0) and games
            else 0.0,
            "latest_vs_prev_yards_pg": (total_yards / games) - (prev_yards / prev_games)
            if previous is not None and games and prev_games
            else 0.0,
            "latest_vs_prev_tds_pg": (total_tds / games) - (prev_tds / prev_games)
            if previous is not None and games and prev_games
            else 0.0,
            "draft_pick_log": np.log1p(latest.get("draft_pick", np.nan)),
            "draft_pick_inverse": 1.0 / np.sqrt(max(float(latest.get("draft_pick", np.nan) or np.nan), 1.0)),
        }
    )
    return row


def build_all_processed() -> dict[str, Path]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    frames = []
    for position in POSITIONS:
        df = build_position_dataset(position, include_targets=True)
        path = PROCESSED_DIR / f"{position.lower()}_training_dataset.csv"
        df.to_csv(path, index=False)
        outputs[position] = path
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(PROCESSED_DIR / "all_positions_training_dataset.csv", index=False)
    outputs["ALL"] = PROCESSED_DIR / "all_positions_training_dataset.csv"
    return outputs


def cached_get(url: str, cache_name: str, delay: float = 3.0, refresh: bool = False) -> str:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / cache_name
    if cache_path.exists() and not refresh:
        return cache_path.read_text(encoding="utf-8")
    time.sleep(delay)
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 rookie-projection-research"})
    response.raise_for_status()
    cache_path.write_text(response.text, encoding="utf-8")
    return response.text


def parse_draft_page(html: str, year: int) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"id": "drafts"})
    if table is None:
        raise ValueError("Could not find PFR draft table.")
    rows = []
    for tr in table.select("tbody tr"):
        if tr.get("class") and "thead" in tr.get("class", []):
            continue
        player_cell = tr.find("td", {"data-stat": "player"})
        pos_cell = tr.find("td", {"data-stat": "pos"})
        pick_cell = tr.find("td", {"data-stat": "draft_pick"})
        if player_cell is None or pos_cell is None or pick_cell is None:
            continue
        position = pos_cell.get_text(strip=True)
        if position not in POSITIONS:
            continue
        college_cell = tr.find("td", {"data-stat": "college_link"})
        college_link = None
        if college_cell is not None and college_cell.find("a"):
            college_link = urljoin(SPORTS_REFERENCE_BASE, college_cell.find("a")["href"])
        rows.append(
            {
                "name": player_cell.get_text(strip=True),
                "position": position,
                "age": text_to_float(tr.find("td", {"data-stat": "age"})),
                "school": text_or_none(tr.find("td", {"data-stat": "college_id"})),
                "draft_pick": text_to_float(pick_cell),
                "rookie_year": year,
                "college_profile_url": college_link,
            }
        )
    return pd.DataFrame(rows)


def text_or_none(node: Any) -> str | None:
    if node is None:
        return None
    value = node.get_text(strip=True)
    return value or None


def text_to_float(node: Any) -> float:
    value = text_or_none(node)
    if value is None:
        return np.nan
    value = value.replace(",", "")
    try:
        return float(value)
    except ValueError:
        return np.nan


def scrape_rookie_class(year: int = 2026, refresh: bool = False, max_players: int | None = None) -> dict[str, Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    draft_url = f"{PFR_BASE}/years/{year}/draft.htm"
    try:
        draft_html = cached_get(draft_url, f"pfr_draft_{year}.html", refresh=refresh)
        draft_df = parse_draft_page(draft_html, year)
    except Exception as exc:
        print(f"PFR draft crawl failed ({type(exc).__name__}: {exc}); falling back to Wikipedia.")
        wiki_url = f"https://en.wikipedia.org/wiki/{year}_NFL_draft"
        draft_html = cached_get(wiki_url, f"wikipedia_draft_{year}.html", delay=0.5, refresh=refresh)
        draft_df = parse_wikipedia_draft_page(draft_html, year)
    if max_players:
        draft_df = draft_df.head(max_players)
    draft_path = RAW_DIR / f"rookies_{year}_draft.csv"
    draft_df.to_csv(draft_path, index=False)

    outputs = {"draft": draft_path}
    sports_reference_blocked = False
    for position in POSITIONS:
        position_rows = draft_df[draft_df["position"] == position].copy()
        college_rows = []
        for _, row in position_rows.iterrows():
            if sports_reference_blocked:
                college_rows.append({**row.to_dict(), "scrape_error": "skipped after Sports Reference 403"})
                continue
            if not row.get("college_profile_url"):
                continue
            try:
                html = cached_get(
                    row["college_profile_url"],
                    f"college_profile_{year}_{slug(row['name'])}.html",
                    refresh=refresh,
                )
                college_rows.extend(parse_college_profile(row, html))
            except Exception as exc:
                if "403" in str(exc):
                    sports_reference_blocked = True
                college_rows.append({**row.to_dict(), "scrape_error": f"{type(exc).__name__}: {exc}"})
        raw_college = pd.DataFrame(college_rows)
        raw_path = RAW_DIR / f"{position.lower()}_rookies_{year}_college.csv"
        raw_college.to_csv(raw_path, index=False)
        outputs[position] = raw_path
        if not raw_college.empty and "season" in raw_college.columns and raw_college["season"].notna().any():
            inference = build_inference_from_college(position, raw_college)
        else:
            inference = build_inference_from_draft(position, position_rows)
        if not inference.empty:
            final_path = FINAL_DIR / f"{position.lower()}_rookies_{year}_features.csv"
            inference.to_csv(final_path, index=False)
            outputs[f"{position}_features"] = final_path
    write_collection_report(year, draft_df, outputs)
    return outputs


def parse_wikipedia_draft_page(html: str, year: int) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    draft_table = None
    for table in soup.find_all("table"):
        headers = [cell.get_text(" ", strip=True) for cell in table.find_all("th")]
        if {"Rnd.", "Pick", "Player", "Pos.", "College"}.issubset(set(headers)):
            draft_table = table
            break
    if draft_table is None:
        raise ValueError("Could not find Wikipedia draft table.")

    header_cells = [cell.get_text(" ", strip=True) for cell in draft_table.find("tr").find_all(["th", "td"])]
    rows = []
    for tr in draft_table.find_all("tr")[1:]:
        cells = [cell.get_text(" ", strip=True) for cell in tr.find_all(["th", "td"])]
        if len(cells) < len(header_cells):
            continue
        row = dict(zip(header_cells, cells))
        position = row.get("Pos.")
        if position not in POSITIONS:
            continue
        player_name = clean_wiki_text(row.get("Player"))
        college = clean_wiki_text(row.get("College"))
        rows.append(
            {
                "name": player_name,
                "position": position,
                "age": np.nan,
                "school": college,
                "draft_pick": text_to_number(row.get("Pick")),
                "rookie_year": year,
                "college_profile_url": infer_college_profile_url(player_name),
            }
        )
    return pd.DataFrame(rows)


def clean_wiki_text(value: Any) -> str:
    value = str(value or "").strip()
    value = re.sub(r"\[[^\]]+\]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value


def text_to_number(value: Any) -> float:
    match = re.search(r"\d+", str(value or "").replace(",", ""))
    return float(match.group(0)) if match else np.nan


def infer_college_profile_url(player_name: str) -> str:
    suffixes = {"jr", "sr", "ii", "iii", "iv", "v"}
    cleaned = re.sub(r"[^A-Za-z0-9 .'-]", "", player_name).strip()
    parts = [part.strip(".").lower() for part in cleaned.split() if part.strip(".")]
    if parts and parts[-1] in suffixes:
        parts = parts[:-1]
    if len(parts) < 2:
        slug_name = "-".join(parts)
    else:
        slug_name = f"{parts[0]}-{parts[-1]}"
    return f"{SPORTS_REFERENCE_BASE}/cfb/players/{slug_name}-1.html"


def parse_college_profile(draft_row: pd.Series, html: str) -> list[dict[str, Any]]:
    position = draft_row["position"]
    spec = POSITION_SPECS[position]
    soup = BeautifulSoup(html, "html.parser")
    table_id = "passing_standard" if position == "QB" else ("rushing_standard" if position == "RB" else "receiving_standard")
    table = soup.find("table", {"id": table_id})
    if table is None:
        return [{**draft_row.to_dict(), "scrape_error": f"missing table {table_id}"}]
    rows = []
    for tr in table.select("tbody tr"):
        if tr.get("class"):
            continue
        year = extract_year(tr.find("th").get_text(strip=True) if tr.find("th") else "")
        if not np.isfinite(year):
            continue
        parsed = {
            "name": draft_row["name"],
            "age": draft_row["age"],
            "position": position,
            "school": draft_row["school"],
            "draft_pick": draft_row["draft_pick"],
            "season": year,
            "college_profile_url": draft_row["college_profile_url"],
        }
        for source_stat, output_column in spec.college_stat_map.items():
            parsed[output_column] = text_to_float(tr.find("td", {"data-stat": source_stat}))
        rows.append(parsed)
    return rows


def build_inference_from_college(position: str, college_df: pd.DataFrame) -> pd.DataFrame:
    college_df = normalize_columns(college_df)
    college_df["season"] = college_df["season"].apply(extract_year)
    path = RAW_DIR / f"_tmp_{position.lower()}_inference_college.csv"
    college_df.to_csv(path, index=False)
    original = POSITION_SPECS[position]
    patched = PositionSpec(
        college_candidates=(path,),
        nfl_candidates=original.nfl_candidates,
        college_stat_map=original.college_stat_map,
        target_columns=original.target_columns,
    )
    POSITION_SPECS[position] = patched
    try:
        return build_position_dataset(position, include_targets=False)
    finally:
        POSITION_SPECS[position] = original
        path.unlink(missing_ok=True)


def build_inference_from_draft(position: str, draft_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in draft_df.iterrows():
        latest = pd.Series(
            {
                "name": row.get("name"),
                "position": position,
                "school": row.get("school"),
                "age": row.get("age"),
                "draft_pick": row.get("draft_pick"),
                "season": row.get("rookie_year", np.nan) - 1 if pd.notna(row.get("rookie_year", np.nan)) else np.nan,
                "games": 0,
            }
        )
        feature_row = base_feature_row(position, latest, None)
        rows.append(feature_row)
    df = pd.DataFrame(rows)
    df = add_conference_dummies(df)
    df = add_position_dummies(df)
    for column in COMMON_FEATURES + POSITION_FEATURES[position]:
        if column not in df.columns:
            df[column] = 0.0
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def write_collection_report(year: int, draft_df: pd.DataFrame, outputs: dict[str, Path]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rookie_year": year,
        "drafted_skill_players": int(len(draft_df)),
        "by_position": draft_df["position"].value_counts().to_dict() if not draft_df.empty else {},
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    (REPORT_DIR / f"data_collection_{year}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and refresh rookie projection datasets.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build-processed")
    crawl = sub.add_parser("crawl-rookies")
    crawl.add_argument("--year", type=int, default=2026)
    crawl.add_argument("--refresh", action="store_true")
    crawl.add_argument("--max-players", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "build-processed":
        outputs = build_all_processed()
        for position, path in outputs.items():
            print(f"{position}: {path}")
    elif args.command == "crawl-rookies":
        outputs = scrape_rookie_class(args.year, refresh=args.refresh, max_players=args.max_players)
        for key, path in outputs.items():
            print(f"{key}: {path}")


if __name__ == "__main__":
    main()
