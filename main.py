# main.py, at project root (same level as src/, data/)
from src.data_ingestion import load_or_fetch_weekly_data, save_cleaned_data
from src.features import build_features

def run_pipeline():
    print("Loading weekly data...")
    df = load_or_fetch_weekly_data(years=[2023, 2024, 2025])

    print("Building features...")
    final_df = build_features(df)

    print("Saving features...")
    save_cleaned_data(final_df, "features.parquet")

    print(f"Done. Final shape: {final_df.shape}")
    print(final_df.head(5))
    print(final_df.null_count())

if __name__ == "__main__":
    run_pipeline()