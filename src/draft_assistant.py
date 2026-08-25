"""   
    It will use the draft assistant to generate a draft response and then use the final assistant to generate a final response
"""
import polars as pl
import json
import os
import sys
from src.models import *
import nflreadpy as nfl


def build_draft_pool(df: pl.DataFrame, model, feature_cols) -> pl.DataFrame:
    """
    One-time setup: pulls current-season roster, attaches each player's
    most recent feature snapshot, runs the model, returns one table of
    every candidate player with their predicted points.
    """
    # load the current roster of players
    roster = nfl.load_rosters([2026]).select(['gsis_id'])
    
    # get only the last row for each player
    last_row_df = last_row(df)
    
    # filter the last row of each player based on who is currently on a roster
    pool = last_row_df.join(roster, left_on='player_id', right_on='gsis_id', how='semi')
        
    prediction = predict(model, pool, feature_cols)
    prediction = prediction.join(pool.select(['player_id', 'position']), on='player_id')
    return prediction

    
    
    
def last_row(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(['player_id', 'season', 'week'])  # guarantee chronological order, don't assume it
    last_rows = df.group_by('player_id', maintain_order=True).last()
    return last_rows

def load_draft_state(path: str = "../data/draft_state.json") -> dict:
    """
    Simple persistence for the drafted-players list and position counts,
    so you're not starting over if you close/reopen mid-draft.
    Could be as simple as a small JSON file.
    """
    if not os.path.exists(path):
        return {"drafted_players:": [], "my_team:": []}
    with open(path, "r") as f:
        return json.load(f)
        
    

def mark_player_drafted(player_id, my_draft = False, path: str = "../data/draft_state.json"):
    """
    Updates draft state: adds player to drafted list, increments
    position count.
    """
    state = load_draft_state(path)
    if my_draft:
        state['my_team'].append(player_id)
    state['drafted_players'].append(player_id)
    with open(path, 'w') as f:
        json.dump(state, f, indent=2)
    

def recommend_next_pick(pool, position: str | None = None, n: int = 5, path: str = "../data/draft_state.json"):
    """
    Filters the pool to exclude drafted players, optionally filters by
    position, sorts by predicted_points, returns top n.
    """
    # remove all drafted players from the current state
    state = load_draft_state(path)
    undrafted = pool.filter(~pl.col('player_id').is_in(state['drafted_players']))
        
    # filter by position and then sort by predicted point and take top 5
    if position is not None:
        undrafted = undrafted.filter(pl.col("position") == position)
    top_picks = undrafted.sort('predicted_points', descending=True).head(n)
    
    # return the top 5 player names
    return top_picks['player_name'].to_list()
