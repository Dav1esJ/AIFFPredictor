"""
    This module is responsible for data ingestion and preprocessing for the AIFFPredictor project.
"""
import polars as pl
import pandas as pd
import numpy as np
import sklearn as sk
from src.data_ingestion import load_or_fetch_weekly_data, save_cleaned_data
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
import joblib


def encode_categoricals(df: pl.DataFrame) -> pl.DataFrame:
    """
    One-hot encodes categorical columns. Done on the FULL dataset before
    any train/test split, so both sides always end up with identical
    columns — encoding train and test separately risks producing
    mismatched columns if a category appears in one but not the other.
    """
    return df.to_dummies(columns=['position', 'team', 'opponent_team'])

def chronological_split(df: pl.DataFrame, test_start_season: int, test_start_week: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Splits data into train/test by time, NOT randomly — everything before
    the given season/week goes to train, everything from that point
    onward goes to test. This simulates genuinely predicting the future
    from the past, matching how the model will actually be used.
    """
    train_df = df.filter((pl.col("season") < test_start_season) |
                         (pl.col('season') == test_start_season) & (pl.col('week') < test_start_week)
    )
    test_df = df.filter((pl.col("season") > test_start_season) |
                        (pl.col('season') == test_start_season) & (pl.col('week') >= test_start_week)
    )
    
    save_cleaned_data(train_df, "train.parquet")
    save_cleaned_data(test_df, "test.parquet")
    
    return train_df, test_df


def get_feature_columns(df: pl.DataFrame) -> list[str]:
    """
    Returns the explicit list of column names the model should train on
    (your rolling averages, trend, snap features, etc.) — kept as one
    source of truth so train/predict always use the identical feature
    set, and so you're not accidentally including columns like
    'player_name' or the target itself as an input.
    """
    static_features = [
        'week', 'season',
        'avg_points_last_1', 'avg_points_last_3', 'avg_points_last_5',
        'avg_targets_in_last_1', 'avg_targets_in_last_3', 'avg_targets_in_last_5',
        'avg_carries_in_last_1', 'avg_carries_in_last_3', 'avg_carries_in_last_5',
        'avg_attempts_in_last_1', 'avg_attempts_in_last_3', 'avg_attempts_in_last_5',
        'points_trend_3_vs_5',
        'offense_snaps_last_1', 'offense_snaps_last_3', 'offense_snaps_last_5',
        'defense_snaps_last_1', 'defense_snaps_last_3', 'defense_snaps_last_5',
        'st_snaps_last_1', 'st_snaps_last_3', 'st_snaps_last_5',
        'offense_pct_last_1', 'offense_pct_last_3', 'offense_pct_last_5',
        'defense_pct_last_1', 'defense_pct_last_3', 'defense_pct_last_5',
        'st_pct_last_1', 'st_pct_last_3', 'st_pct_last_5',
    ]
    dummy_features = [c for c in df.columns if c.startswith(('position_', 'team_', 'opponent_team_'))]
    return static_features + dummy_features


def train_baseline(train_df: pl.DataFrame) -> pl.DataFrame:
    """
    Not a real model — just returns/uses 'avg_points_last_3' directly as
    the prediction. This exists purely to give you a number to beat;
    if your real model can't outperform this, it's not adding value.
    """
    return train_df.select(['player_id', 'season', 'week', 'avg_points_last_3']).rename({'avg_points_last_3': 'predicted_points'})



def evaluate(predictions, actuals) -> dict:
    """
    Computes error metrics (e.g. MAE, RMSE) comparing predicted vs actual
    fantasy points. Used identically for the baseline and the real model,
    so the comparison is apples-to-apples.
    """
    return {
        'RMSE': root_mean_squared_error(actuals, predictions),
        'MAE': mean_absolute_error(actuals, predictions)
    }


def train_Linear_model(train_df: pl.DataFrame, feature_cols: list[str]) -> BaseEstimator:
    """
    Fits a real model (start with something simple/interpretable like
    linear regression or a random forest — scikit-learn) using
    feature_cols as input and 'my_fantasy_points' as the target.
    Returns the fitted model object.
    """
    
    train_y = train_df.select('my_fantasy_points').to_numpy().ravel()
    train_X = train_df.select(feature_cols).to_numpy()
    
    
    model = LinearRegression()  # or RandomForestRegressor(), etc.
    model.fit(train_X, train_y)
    
    return model
    
def train_tree_model(train_df: pl.DataFrame, feature_cols: list[str]) -> BaseEstimator:
    """
    Fits a real model (start with something simple/interpretable like
    linear regression or a random forest — scikit-learn) using
    feature_cols as input and 'my_fantasy_points' as the target.
    Returns the fitted model object.
    """
    
    train_y = train_df.select('my_fantasy_points').to_numpy().ravel()
    train_X = train_df.select(feature_cols).to_numpy()
    
    
    model = RandomForestRegressor(n_estimators=100, random_state=17)  # or RandomForestRegressor(), etc.
    model.fit(train_X, train_y)
    
    return model
    
def predict(model, df: pl.DataFrame, feature_cols: list[str]) -> pl.DataFrame:
    """
    Runs the trained model on new rows, returns predictions attached
    alongside player identifying info (so you can tell whose prediction
    is whose, not just a bare array of numbers).
    """
    train_x = df.select(feature_cols).to_numpy()
    predictions = model.predict(train_x)
    
    return df.select(['player_id', 'player_name', 'season', 'week']).with_columns(pl.Series('predicted_points', predictions))
    


def save_model(model, path: str) -> None:
    """
    Persists the trained model to disk (e.g. via joblib or pickle) so you
    don't need to retrain every time you want to use it — same caching
    instinct as your data pipeline.
    """
    joblib.dump(model, path)
    print(f"Model saved to {path}")
    


def load_model(path: str) -> BaseEstimator:
    """
    Loads a previously trained model from disk.
    """
    return joblib.load(path)