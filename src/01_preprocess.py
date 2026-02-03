# src/preprocess.py

import pandas as pd
import numpy as np

MOVIES_DATASET = "data/raw/tmdb_5000_movies.csv"
CREDITS_DATASET = "data/raw/tmdb_5000_credits.csv"
MERGED_DATASET = "data/cleaned/merged.csv"
OUTPUT_PATH = "data/cleaned/cleaned.csv"

def merge_datasets():
    df1 = pd.read_csv(MOVIES_DATASET)
    df2 = pd.read_csv(CREDITS_DATASET)

    # Merge on 'id' and 'movie_id'
    merged_df = pd.merge(df1, df2, left_on="id", right_on="movie_id", how="inner")

    # Drop redundant 'movie_id' column
    merged_df = merged_df.drop(columns=["movie_id"])

    merged_df.to_csv(MERGED_DATASET, index=False)
    print("✅ Datasets merged and saved -> merged.csv")

def clean_data():
    df = pd.read_csv(MERGED_DATASET)

    # Drop long/unnecessary columns
    df = df.drop(['homepage', 'keywords','original_language' ,'original_title','overview',
              'spoken_languages','status','tagline','title_y'], axis=1)
    df = df.rename(columns={"title_x": "title"})

    #Get release year and release month to identify any effects like holidays
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = df['release_date'].dt.year.astype('Int64')
    df['release_month'] = df['release_date'].dt.month.astype('Int64')

    #Calculate Profit and ROI to calculate whether the movie is hit or flop
    df['profit'] = df['revenue'] - df['budget']
    df['roi'] = np.where(df['budget'] > 0, df['profit'] / df['budget'], 0)

    #Calculate the target column Hitflop
    df['hitflop'] = np.where(df['profit'] > 0, 1, 0)

    #Drop the null columns
    df = df.dropna(subset=['runtime', 'release_year', 'release_month','release_date'])

    # Remove zero budget/revenue
    df = df[(df["budget"] > 0) & (df["revenue"] > 0)]

    # Drop null rows
    df = df.dropna()

    df.to_csv(OUTPUT_PATH, index=False)
    print("✅ Cleaned data saved -> cleaned.csv")


if __name__ == "__main__":
    merge_datasets()
    clean_data()