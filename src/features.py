"""
    This module is responsible for data ingestion and preprocessing for the AIFFPredictor project.
"""
import nflreadpy as nfl
import polars as pl
import pandas as pd
import os
from src.data_ingestion import load_or_fetch_weekly_data


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Orchestrator: takes cleaned weekly data, returns it with all
    engineered feature columns added, sorted and ready for modeling.
    Calls the helper functions below in order.
    """
    # 1. sort by player, season, week — critical, rolling calcs depend on order
    df = df.sort(by=['player_name', 'season', 'week'])
    # 2. call add_rolling_points()
    rolling = add_rolling_points(df)
    # 3. call add_rolling_volume()
    volume = add_rolling_volume(rolling)
    # 4. call add_trend_features()
    trended = add_trend_features(volume)
    # 5. call add_context_features()
    contexted = add_context_features(trended)
    # 6. call drop_insufficient_history()
    final_df = drop_insufficient_history(contexted)
    # 7. return final df
    return final_df


def add_rolling_points(df: pl.DataFrame, windows: list[int] = [3, 5]) -> pl.DataFrame:
    """
    For each player, adds columns like 'avg_points_last_3', 'avg_points_last_5' —
    the mean of 'my_fantasy_points' over the trailing N games, NOT including
    the current row's own points (this is the leakage check point — the
    current week's actual score must never feed into its own feature).
    Grouped per player, ordered by season/week.
    """
    for i in range(len(windows)):
        window = windows[i]
        col_name = f"avg_points_last_{window}"
        df = df.with_columns(
            pl.col("my_fantasy_points")
            .shift(1)  # shift by 1 to exclude current row's points
            .rolling_mean(window)
            .over("player_name")
            .alias(col_name)
        )
    return df


def add_rolling_volume(df: pl.DataFrame, windows: list[int] = [3, 5]) -> pl.DataFrame:
    """
    Same trailing-window idea, applied to targets, carries, attempts
    instead of points. E.g. 'avg_targets_last_3'. This is often more
    stable/predictive than points themselves — a role change shows up
    here before it shows up in scoring.
    """
    
    for i in range(len(windows)):
        window = windows[i]
        for col in ['targets', 'carries', 'attempts']:
            col_name = f"avg_{col}_in_last_{window}"
            df = df.with_columns(
                pl.col(col)
                .shift(1)  # shift by 1 to exclude current row's values
                .rolling_mean(window)
                .over("player_name")
                .alias(col_name)
            )
    return df


def add_trend_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compares a short window average to a longer window average
    (e.g. avg_points_last_3 vs avg_points_last_5, or vs season-to-date average)
    to capture direction: is usage/production trending up or down recently,
    not just what the recent level is.
    """
    df = df.with_columns(
        (pl.col("avg_points_last_3") - pl.col("avg_points_last_5")).alias("points_trend_3_vs_5")
    )
    return df


def add_context_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Adds situational columns: home/away flag, maybe a rest-days or
    bye-week-return flag if derivable from the week sequence.
    Lower priority than the rolling stats — nice-to-have, not core.
    """
    # add the snap count context features
    snap = nfl.load_snap_counts()
    # gets only the values from snap counts that I want to join into the main df
    snap = snap.select(['player_name', 'season', 'week', 'team', 'game_id', 'offense_snaps', 'defense_snaps', 'st_snaps', 'offens'])
    df.jo
    df = df.with_columns(
        


def drop_insufficient_history(df: pl.DataFrame, min_games: int = 3) -> pl.DataFrame:
    """
    Removes rows where a player doesn't yet have enough prior games for
    the rolling windows to be meaningful (e.g. their first 1-2 games of
    a season, or of the whole dataset). Prevents feeding the model
    rows full of nulls/defaults from insufficient history.
    """