"""
    This module is responsible for data ingestion and preprocessing for the AIFFPredictor project.
"""
import nflreadpy as nfl
import polars as pl
import pandas as pd
import os
import numpy as np

from pathlib import Path

# At the top of data_ingestion.py, after imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # src/ -> project root
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

def load_or_fetch_weekly_data(years: list[int], filename: str = "weekly_data.parquet", data_dir: Path = DEFAULT_DATA_DIR) -> pl.DataFrame:
    """
        Fetches weekly data for the specified years using the nflreadpy library.
    """
    path = data_dir / filename
    if os.path.exists(path):
        print(f"Loading weekly data from {path}")
        return pl.read_parquet(path)
    print(f"No cache found at {path}. Fetching weekly data for years: {years}")
    cleaned_data = clean_weekly_data(nfl.load_player_stats(years))
    save_cleaned_data(cleaned_data, filename, data_dir)
    return cleaned_data

def clean_weekly_data(df: pl.DataFrame) -> pl.DataFrame:
    """
        Cleans the weekly data by removing unnecessary columns and handling missing values, as well as adding my fantasy points column.
    """
    # Remove unnecessary columns by selecting only the columns we need to keep
    columns_to_keep = [
        'player_id', 'player_name', 'position', 'week', 'team', 'season', 'game_id', 'opponent_team', 
        'completions', 'attempts', 'sacks_suffered', 'passing_first_downs', 'passing_yards', 
        'carries', 'rushing_yards', 'rushing_first_downs', 'receptions', 'targets', 
        'receiving_yards', 'passing_tds', 'rushing_tds', 'receiving_tds', 'receiving_first_downs',
        'passing_interceptions', 'fumbles_lost_total', 'rushing_2pt_conversions',
        'receiving_2pt_conversions', 'passing_2pt_conversions', 'pt_return_yards',
        'pt_return_tds', 'kickoff_return_yards', 'special_teams_tds',
        'def_safeties', 'def_interceptions', 'def_fumbles', 'def_sacks', 'def_tds', 
        'fg_made_0_19', 'fg_made_20_29', 'fg_made_30_39', 'fg_made_40_49', 'fg_made_50_59', 'fg_made_60_',
        'fg_missed_0_19', 'fg_missed_20_29', 'fg_missed_30_39', 'fg_missed_40_49', 'fg_missed_50_59', 'fg_missed_60_',
        'pat_made', 'pat_missed', 'pt_net_yards', 'pt_inside_20', 'pt_att', 'pt_blocked', 'punt_net_yards_per_punt', 'my_fantasy_points'
    ]

    
    
    df = df.with_columns(
        pl.when(pl.col('position').is_in(['P']))
        .then(
            pl.col('pt_att') - pl.col('pt_blocked')
        ).alias('punts_taken')
    )
    df = df.with_columns(
        pl.when(pl.col('position').is_in(['P']))
        .then(
            pl.col('pt_net_yards') / pl.col('punts_taken')
        ).alias('punt_net_yards_per_punt')
    )
    
    stat_cols = [
            'passing_yards', 'rushing_yards', 'receiving_yards', 'passing_tds', 'rushing_tds', 'receiving_tds',
            'passing_interceptions', 'fumbles_lost_total', 'rushing_2pt_conversions', 'receiving_2pt_conversions',
            'passing_2pt_conversions', 'pt_return_yards', 'pt_return_tds', 'kickoff_return_yards', 'special_teams_tds',
            'def_safties', 'def_interceptions', 'def_fumbles', 'def_sacks', 'def_tds',
            'fg_made_0_19', 'fg_made_20_29', 'fg_made_30_39',
            'fg_made_40_49', 'fg_made_50_59', 'fg_made_60_', 'fg_missed_0_19', 'fg_missed_20_29', 'fg_missed_30_39',
            'fg_missed_40_49', 'fg_missed_50_59', 'fg_missed_60_', 'pat_made', 'pat_missed', 'pt_net_yards', 'pt_inside_20','pt_att', 'pt_blocked', 'punt_net_yards_per_punt'
        ]
    df = df.with_columns([pl.col(c).fill_null(0) for c in stat_cols if c in df.columns])
    
    punt_net_yards_per_punt_points = (
        pl.when((pl.col('punt_net_yards_per_punt') >= 40) & (pl.col('punt_net_yards_per_punt') < 42)).then(1)
        .when((pl.col('punt_net_yards_per_punt') >= 42) & (pl.col('punt_net_yards_per_punt') < 44)).then(2)
        .when(pl.col('punt_net_yards_per_punt') >= 44).then(3)
        .otherwise(0)
    )
    
    # method to make my fantasy score for a player
    df = df.with_columns(
        pl.when(pl.col('position').is_in(['QB', 'RB', 'WR', 'TE', 'K', 'P']))
        .then(
            (pl.col('passing_yards') // 25) * 2 + 
            pl.col('passing_tds') * 6 +
            (pl.col('rushing_yards') // 10)* 1 +
            pl.col('rushing_tds') * 6 +
            (pl.col('receiving_yards') // 10) * 1.5 +
            pl.col('receiving_tds') * 6 +
            pl.col('fumbles_lost_total') * -2 +
            pl.col('passing_interceptions') * -2 + 
            pl.col('passing_2pt_conversions') * 2 + 
            pl.col('rushing_2pt_conversions') * 2 +
            pl.col('receiving_2pt_conversions') * 2 +
            pl.col('pt_return_tds') * 6 +
            pl.col('special_teams_tds') * 6 +
            pl.col('pt_return_yards') // 10 * 2 +
            pl.col('kickoff_return_yards') // 25 * 3 +
            pl.col('pat_made') * 1.5 +
            pl.col('fg_made_0_19') * 3 +
            pl.col('fg_made_20_29') * 3 +
            pl.col('fg_made_30_39') * 3 +
            pl.col('fg_made_40_49') * 5 +
            pl.col('fg_made_50_59') * 6 +
            pl.col('fg_made_60_') * 6 +
            pl.col('fg_missed_0_19') * -2 +
            pl.col('fg_missed_20_29') * -2 +
            pl.col('fg_missed_30_39') * -2 +
            pl.col('fg_missed_40_49') * -1 +
            pl.col('pat_missed') * -1 +
            pl.col('pt_inside_20') * 1 +
            punt_net_yards_per_punt_points
        )
        .otherwise(pl.col('fantasy_points'))
        .alias('my_fantasy_points')
    )
    
    df = df.select(columns_to_keep)
    df = df.drop_nulls(subset=['player_id'])
        
    return df

def save_cleaned_data(df: pl.DataFrame, filename: str, data_dir: Path = DEFAULT_DATA_DIR) -> None:
    """
        Saves the cleaned data to a CSV file in the specified directory.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / filename
    df.write_parquet(path)
    print(f"Cleaned data of {df.shape[0]} rows saved to {path}")