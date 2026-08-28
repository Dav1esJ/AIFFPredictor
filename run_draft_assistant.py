from src.data_ingestion import load_or_fetch_weekly_data, save_cleaned_data
from src.features import build_features
from src.models import *
from src.draft_assistant import *
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
import os

def find_player(name: str, pool: pl.DataFrame) -> str | None:
    """
    Looks up a player by partial/case-insensitive name match.
    Returns the player_id if exactly one match found.
    Prints options and returns None if zero or multiple matches,
    so the caller can ask the user to be more specific.
    """
    
    # first lookup all players in the pool that have name in their player name
    players = pool.filter(pl.col('player_name').str.contains(name))
    
    if len(players) == 0:
        print(f"No player matching found with name {name}")
        return None
    elif len(players) == 1:
        return players["player_id"].item()
    elif len(players) <= 5:
        names = players["player_name"].to_list()
        print(f"is it {', '.join(names[:-1])}, or {names[-1]}?")
        return None
    else:
        # take the top 5 players that score the most and reduce it to just those players
        reduced = players.top_k(5, by="predicted_points")
        names = reduced['player_name'].to_list()
        print(f"too many options, is it one of these top options: {', '.join(names[:-1])}, or {names[-1]}?")
        return None

    
def print_instructions():
    
    print("Command drafted followed by a player name adds that player to the list of drafted players")
    print("Command mine followed by a player name adds that player to my list of drafted players")
    print("Command Reccomend will list 5 players that it recommends you draft, if the command is followed by a position then it will list players in that position")
    print("Command quit exits the program")
    
def run():
    # one-time setup
    df = pl.read_parquet("data/features.parquet") #load features.parquet
    model = load_model("data/model.joblib")
    
    encoded_full = encode_categoricals(df)
    feature_cols = get_feature_columns(encoded_full)
    pool = build_draft_pool(df, model, feature_cols)

    # print instructions (what commands are available)
    print_instructions()

    while True:
        command = input("> ")
        
        if command.startswith("drafted "):
            name = command[len("drafted "):]
            player_id = find_player(name, pool)
            if player_id: mark_player_drafted(player_id, my_draft=False)
        
        elif command.startswith("mine "):
            name = command[len("mine "):]
            player_id = find_player(name, pool)
            if player_id: mark_player_drafted(player_id, my_draft=True)
        
        elif command == "recommend":
            names = recommend_next_pick(pool)
            print(names)
        
        elif command.startswith("recommend "):
            position = command[len("recommend "):]
            names = recommend_next_pick(pool, position=position)
            print(names)
        
        elif command == "quit":
            break
        
        else:
            print("unrecognized command")