# main.py, at project root (same level as src/, data/)
from src.data_ingestion import load_or_fetch_weekly_data, save_cleaned_data
from src.features import build_features
from src.models import *

def run_pipeline():
    saved_model = load_model('data/model.joblib')
    
    if not saved_model:
    
        print("Loading weekly data...")
        df = load_or_fetch_weekly_data(years=[2023, 2024, 2025])

        print("Building features...")
        final_df = build_features(df)

        print("Saving features...")
        save_cleaned_data(final_df, "features.parquet")

        print(f"Done. Final shape: {final_df.shape}")
        print(final_df.head(5))
        print(final_df.null_count())
    
        encoded = encode_categoricals(final_df)
        
        train_df, test_df = chronological_split(encoded, 2025, 4)
        
        feature_column = get_feature_columns(train_df)
        baseline_model = train_baseline(train_df)
        
        model = train_tree_model(train_df, feature_column)

        predictions = predict(model, test_df, feature_column)
        print(evaluate(predictions['predicted_points'], test_df['my_fantasy_points']))
        
        save_model(model, 'data/model.joblib')
    
    

if __name__ == "__main__":
    run_pipeline()