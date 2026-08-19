"""
    This module is responsible for data ingestion and preprocessing for the AIFFPredictor project.
"""
import nflreadpy as nfl
import polars as pl
import pandas as pd
import os
from src.data_ingestion import load_or_fetch_weekly_data, save_cleaned_data


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
    
    print("before:", contexted.shape[0])
    print("after:", final_df.shape[0])
    print(final_df.null_count())
    save_cleaned_data(final_df, "features.parquet")
    
    return final_df


def add_rolling_points(df: pl.DataFrame, windows: list[int] = [1, 3, 5]) -> pl.DataFrame:
    """
    For each player, adds columns like 'avg_points_last_1', 'avg_points_last_3', 'avg_points_last_5' —
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
            .over("player_id")
            .alias(col_name)
        )
    return df


def add_rolling_volume(df: pl.DataFrame, windows: list[int] = [1, 3, 5]) -> pl.DataFrame:
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
                .over("player_id")
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
    snap = nfl.load_snap_counts([2023, 2024, 2025])
    
    players = nfl.load_players()
    players = players.select(['gsis_id', 'pfr_id'])

    snap = snap.join(players, left_on='pfr_player_id', right_on='pfr_id', how='left')
    # gets only the values from snap counts that I want to join into the main df
    snap = snap.select(['gsis_id', 'game_id', 'offense_snaps', 'defense_snaps', 'st_snaps', 'offense_pct', 'defense_pct', 'st_pct'])
    df = df.join(
        snap,
        left_on=['player_id', 'game_id'],
        right_on=['gsis_id', 'game_id'],
        how='left',
    )
    snap_cols = ['offense_snaps', 'defense_snaps', 'st_snaps', 'offense_pct', 'defense_pct', 'st_pct']
    print(f"shape df: {df.shape[0]}")
    print(f"non null {df.filter(pl.col('offense_snaps').is_not_null()).shape[0]}")
    df = df.with_columns([pl.col(c).fill_null(0) for c in snap_cols])  
    
    # add columns for last 1, 3 and last 5 games' snap counts and percentages
    for window in [1, 3, 5]:
        for col in snap_cols:
            col_name = f"{col}_last_{window}"
            df = df.with_columns(
                pl.col(col)
                .shift(1)  # shift by 1 to exclude current row's values
                .rolling_mean(window)
                .over("player_id")
                .alias(col_name)
            )

    return df    


def drop_insufficient_history(df: pl.DataFrame, min_games: int = 3) -> pl.DataFrame:
    """
    Removes rows where a player doesn't yet have enough prior games for
    the rolling windows to be meaningful (e.g. their first 1-2 games of
    a season, or of the whole dataset). Prevents feeding the model
    rows full of nulls/defaults from insufficient history.
    """
    rolling_cols = (
        [f"avg_points_last_{window}" for window in [3, 5]]
        + [f"avg_{col}_in_last_{window}" for col in ['targets', 'carries', 'attempts'] for window in [3, 5]]
           + [f"{col}_last_{window}" for col in ['offense_snaps', 'defense_snaps', 'st_snaps', 'offense_pct', 'defense_pct', 'st_pct'] for window in [3, 5]]
           
    )
    df = df.drop_nulls(subset=rolling_cols)
    
    return df