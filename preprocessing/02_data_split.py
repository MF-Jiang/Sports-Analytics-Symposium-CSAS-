import pandas as pd
import os

file_path = '../data/feature_cleaned.csv'
output_dir1 = '../data/pitcher_data'
output_dir2 = '../data/batter_data'
df = pd.read_csv(file_path)

os.makedirs(output_dir1, exist_ok=True)

unique_pitchers = df['pitcher'].unique()

for pitcher in unique_pitchers:
    pitcher_df = df[df['pitcher'] == pitcher]
    output_file = os.path.join(output_dir1, f'{pitcher}.csv')
    pitcher_df.to_csv(output_file, index=False)

print(f"Split into {len(unique_pitchers)} CSV files in directory: {output_dir1}")

os.makedirs(output_dir2, exist_ok=True)

unique_batters = df['batter'].unique()

for batter in unique_batters:
    batter_df = df[df['batter'] == batter]
    output_file = os.path.join(output_dir2, f'{batter}.csv')
    batter_df.to_csv(output_file, index=False)

print(f"Split into {len(unique_batters)} CSV files in directory: {output_dir2}")