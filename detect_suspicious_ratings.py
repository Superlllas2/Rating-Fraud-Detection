"""Command-line tool to detect suspicious player ratings."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class CLIArgs:
    """Dataclass describing command-line arguments."""

    input_path: Path
    sport: str
    output_dir: Path
    suspicion_threshold: float


def parse_args(argv: Optional[Iterable[str]] = None) -> CLIArgs:
    """Parse command-line arguments.

    Args:
        argv: Optional iterable of argument strings.

    Returns:
        Parsed CLIArgs object.
    """

    parser = argparse.ArgumentParser(
        description="Detect suspicious player ratings from a JSON export."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="Path to the JSON file containing player data.",
    )
    parser.add_argument(
        "-s",
        "--sport",
        default="Football",
        help="Sport to filter on (default: Football).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=Path("./output"),
        type=Path,
        help="Directory where CSV outputs will be written.",
    )
    parser.add_argument(
        "--suspicion-threshold",
        default=4.0,
        type=float,
        help="Suspicion score threshold for flagging ratings (default: 4.0).",
    )

    args = parser.parse_args(argv)
    input_path = args.input
    if not input_path.exists():
        parser.error(f"Input file does not exist: {input_path}")

    output_dir = args.output_dir
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    return CLIArgs(
        input_path=input_path,
        sport=args.sport,
        output_dir=output_dir,
        suspicion_threshold=args.suspicion_threshold,
    )


def load_player_data(path: Path) -> List[Dict[str, Any]]:
    """Load player data JSON.

    Args:
        path: Path to JSON file.

    Returns:
        List of player dictionaries.
    """

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON to be a list of players.")

    return data


def parse_decimal(value: Any) -> Optional[float]:
    """Convert Mongo decimal to float."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict) and "$numberDecimal" in value:
        try:
            return float(value["$numberDecimal"])
        except (TypeError, ValueError):
            return None
    return None


def safe_get(dictionary: Optional[Dict[str, Any]], key: str) -> Any:
    """Safely get a key from a possibly None dict."""

    if dictionary is None:
        return None
    return dictionary.get(key)


def flatten_ratings(players: List[Dict[str, Any]]) -> pd.DataFrame:
    """Flatten gradingsAsVoter entries into a DataFrame."""

    rows: List[Dict[str, Any]] = []
    for player in players:
        gradings = player.get("gradingsAsVoter") or []
        for grading in gradings:
            voter_info = grading.get("voterId") or {}
            target_info = grading.get("targetId") or {}
            match_info = grading.get("archivedMatchId") or {}

            rows.append(
                {
                    "rating_id": grading.get("_id"),
                    "sport": grading.get("sport"),
                    "rating_value": parse_decimal(grading.get("grade")),
                    "rater_id": safe_get(voter_info, "_id"),
                    "rater_name": safe_get(voter_info, "name"),
                    "rater_email": safe_get(voter_info, "email"),
                    "ratee_id": safe_get(target_info, "_id"),
                    "ratee_name": safe_get(target_info, "name"),
                    "ratee_email": safe_get(target_info, "email"),
                    "match_id": safe_get(match_info, "_id"),
                    "match_name": safe_get(match_info, "name"),
                    "match_sport": safe_get(match_info, "sport"),
                    "match_game_date": safe_get(match_info, "game_date"),
                    "createdAt": grading.get("createdAt"),
                    "updatedAt": grading.get("updatedAt"),
                }
            )

    df = pd.DataFrame(rows)
    return df


def compute_mad(series: pd.Series) -> float:
    """Compute Median Absolute Deviation for a series."""

    if series.empty:
        return 0.0
    median = np.median(series)
    deviations = np.abs(series - median)
    return float(np.median(deviations))


def ensure_output_dir(path: Path) -> None:
    """Create output directory if needed."""

    path.mkdir(parents=True, exist_ok=True)


def add_ratee_statistics(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add per-ratee statistics to DataFrame.

    Returns:
        Tuple containing the augmented ratings DataFrame and a per-ratee
        statistics DataFrame.
    """

    ratee_groups = df.groupby("ratee_id")["rating_value"]
    ratee_stats = ratee_groups.agg(
        median_rating="median",
        mean_rating_received="mean",
        total_ratings_received="count",
    )
    ratee_stats["mad"] = ratee_groups.apply(compute_mad)
    ratee_stats = ratee_stats.reset_index()

    df = df.merge(
        ratee_stats[["ratee_id", "median_rating", "mad"]], on="ratee_id", how="left"
    )
    return df, ratee_stats


def add_rater_statistics(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add per-rater statistics.

    Returns:
        Tuple containing the augmented ratings DataFrame and a per-rater
        statistics DataFrame.
    """

    rater_stats = (
        df.groupby("rater_id")["rating_value"]
        .agg(mean_given="mean", std_given="std", count_given="count")
        .reset_index()
    )
    df = df.merge(rater_stats, on="rater_id", how="left")
    return df, rater_stats


def compute_outlier_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute outlier and suspicion scores for ratings."""

    mad_safe = df["mad"].fillna(0.0)
    mad_safe = mad_safe.where(mad_safe > 0, 0.1)
    df["diff_from_median"] = df["rating_value"] - df["median_rating"]
    df["player_outlier_score"] = df["diff_from_median"].abs() / mad_safe

    std_safe = df["std_given"].fillna(0.0)
    std_safe = std_safe.where(std_safe > 0, 0.1)
    df["rater_outlier_score"] = (df["rating_value"] - df["mean_given"]).abs() / std_safe

    df["suspicion_score"] = df["player_outlier_score"] + df["rater_outlier_score"]
    df["suspicion_direction"] = np.select(
        [df["diff_from_median"] > 0, df["diff_from_median"] < 0],
        ["high", "low"],
        default="neutral",
    )
    return df


def build_top_raters_strings(df: pd.DataFrame, direction: str) -> pd.Series:
    """Build summary strings for top suspicious raters per player."""

    subset = df[df["suspicion_direction"] == direction]
    if subset.empty:
        return pd.Series(dtype=object)

    counts = (
        subset.groupby(["ratee_id", "rater_name"])
        .size()
        .reset_index(name="count")
    )
    counts["rater_name"] = counts["rater_name"].fillna("Unknown")
    counts = counts.sort_values(["ratee_id", "count"], ascending=[True, False])

    def join_top(group: pd.DataFrame) -> str:
        top = group.head(3)
        return ", ".join(f"{row['rater_name']} ({int(row['count'])})" for _, row in top.iterrows())

    return counts.groupby("ratee_id").apply(join_top)


def build_player_summary(
    df: pd.DataFrame, ratee_stats: pd.DataFrame, suspicion_threshold: float
) -> pd.DataFrame:
    """Build per-ratee summary DataFrame."""

    suspicious = df[df["suspicion_score"] >= suspicion_threshold]

    identity = (
        df.groupby("ratee_id")
        .agg(ratee_name=("ratee_name", "first"), ratee_email=("ratee_email", "first"))
        .reset_index()
    )
    summary = identity.merge(ratee_stats, on="ratee_id", how="left")

    high_counts = (
        suspicious[suspicious["suspicion_direction"] == "high"]
        .groupby("ratee_id")
        .size()
        .rename("count_suspicious_high")
    )
    low_counts = (
        suspicious[suspicious["suspicion_direction"] == "low"]
        .groupby("ratee_id")
        .size()
        .rename("count_suspicious_low")
    )

    summary = summary.merge(high_counts, on="ratee_id", how="left")
    summary = summary.merge(low_counts, on="ratee_id", how="left")
    summary["count_suspicious_high"] = summary["count_suspicious_high"].fillna(0).astype(int)
    summary["count_suspicious_low"] = summary["count_suspicious_low"].fillna(0).astype(int)

    top_high = build_top_raters_strings(suspicious, "high")
    top_low = build_top_raters_strings(suspicious, "low")
    summary = summary.merge(
        top_high.rename("top_raters_suspicious_high"), on="ratee_id", how="left"
    )
    summary = summary.merge(
        top_low.rename("top_raters_suspicious_low"), on="ratee_id", how="left"
    )
    summary["top_raters_suspicious_high"] = summary[
        "top_raters_suspicious_high"
    ].fillna("")
    summary["top_raters_suspicious_low"] = summary[
        "top_raters_suspicious_low"
    ].fillna("")

    return summary


def build_rater_summary(
    df: pd.DataFrame, rater_stats: pd.DataFrame, suspicion_threshold: float
) -> pd.DataFrame:
    """Build per-rater summary DataFrame."""

    suspicious = df[df["suspicion_score"] >= suspicion_threshold]

    identity = (
        df.groupby("rater_id")
        .agg(rater_name=("rater_name", "first"), rater_email=("rater_email", "first"))
        .reset_index()
    )
    summary = identity.merge(rater_stats, on="rater_id", how="left")
    summary["std_given"] = summary["std_given"].fillna(0.0)

    df = df.copy()
    df["is_extreme"] = df["rating_value"].isin([1.0, 6.0])
    extreme_counts = (
        df.groupby("rater_id")
        .agg(extreme_count=("is_extreme", "sum"), total_count=("rating_value", "count"))
        .reset_index()
    )
    extreme_counts["extreme_rating_fraction"] = np.where(
        extreme_counts["total_count"] > 0,
        extreme_counts["extreme_count"] / extreme_counts["total_count"],
        0.0,
    )

    summary = summary.merge(
        extreme_counts[["rater_id", "extreme_rating_fraction"]],
        on="rater_id",
        how="left",
    )
    summary["extreme_rating_fraction"] = summary["extreme_rating_fraction"].fillna(0.0)

    high_counts = (
        suspicious[suspicious["suspicion_direction"] == "high"]
        .groupby("rater_id")
        .size()
        .rename("count_suspicious_high")
    )
    low_counts = (
        suspicious[suspicious["suspicion_direction"] == "low"]
        .groupby("rater_id")
        .size()
        .rename("count_suspicious_low")
    )

    summary = summary.merge(high_counts, on="rater_id", how="left")
    summary = summary.merge(low_counts, on="rater_id", how="left")
    summary["count_suspicious_high"] = summary["count_suspicious_high"].fillna(0).astype(int)
    summary["count_suspicious_low"] = summary["count_suspicious_low"].fillna(0).astype(int)

    return summary


def main() -> None:
    """Entry point for the suspicious rating detector."""

    args = parse_args()
    print(f"Loading data from {args.input_path}...")
    players = load_player_data(args.input_path)
    print(f"Loaded {len(players)} players.")

    df = flatten_ratings(players)
    if df.empty:
        print("No rating events found in input data.")
        return
    print(f"Flattened {len(df)} rating events.")

    sport_filtered = df[df["sport"] == args.sport].copy()
    print(f"Filtering for sport: {args.sport}")
    if sport_filtered.empty:
        print(f"No rating events found for sport '{args.sport}'.")
        return

    ensure_output_dir(args.output_dir)
    all_ratings_path = args.output_dir / "all_ratings.csv"
    sport_filtered.to_csv(all_ratings_path, index=False)
    print(f"Saved flattened ratings to {all_ratings_path}.")

    sport_filtered, ratee_stats = add_ratee_statistics(sport_filtered)
    sport_filtered, rater_stats = add_rater_statistics(sport_filtered)
    sport_filtered = compute_outlier_scores(sport_filtered)

    all_scores_path = args.output_dir / "all_ratings_with_scores.csv"
    sport_filtered.to_csv(all_scores_path, index=False)

    suspicious = sport_filtered[sport_filtered["suspicion_score"] >= args.suspicion_threshold]
    suspicious_path = args.output_dir / "suspicious_ratings.csv"
    suspicious.sort_values("suspicion_score", ascending=False).to_csv(
        suspicious_path, index=False
    )
    print(
        f"Flagged {len(suspicious)} suspicious ratings (threshold={args.suspicion_threshold})."
    )

    player_summary = build_player_summary(sport_filtered, ratee_stats, args.suspicion_threshold)
    player_summary_path = args.output_dir / "player_summary.csv"
    player_summary.to_csv(player_summary_path, index=False)

    rater_summary = build_rater_summary(sport_filtered, rater_stats, args.suspicion_threshold)
    rater_summary_path = args.output_dir / "rater_summary.csv"
    rater_summary.to_csv(rater_summary_path, index=False)

    print(f"Saved detailed ratings with scores to {all_scores_path}.")
    print(f"Saved suspicious ratings to {suspicious_path}.")
    print(f"Saved player summary to {player_summary_path}.")
    print(f"Saved rater summary to {rater_summary_path}.")


if __name__ == "__main__":
    main()
