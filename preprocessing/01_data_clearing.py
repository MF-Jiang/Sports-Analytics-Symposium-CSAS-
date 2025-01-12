import polars as pl
import pandas as pd

## Read the arrow file into a Table
# ad = '../data/statcast_pitch_swing_data_20240402_20240630.arrow'
# df = pl.read_ipc(ad, use_pyarrow = True)

df = pd.read_csv('../data/statcast_pitch_swing_data_20240402_20241030_with_arm_angle2.csv')
print(df.columns.tolist())

# Discard features or invalid (only one value) features
columns_to_drop = ['spin_dir', 'spin_rate_deprecated','break_angle_deprecated', 'break_length_deprecated',
                  'tfs_deprecated', 'tfs_zulu_deprecated', 'umpire','game_year', 'fielder_2',
                   'fielder_3','fielder_4','fielder_5','fielder_6','fielder_7','fielder_8',
                   'fielder_9','home_team','away_team','sv_id','player_name','des']

df = df.drop(columns=columns_to_drop)

df = df[df['game_type'] == 'R']
df = df.drop(columns=['game_type'])

# pdf = df.to_pandas()

for col in df.columns:
    print(f"Feature: {col}")

    duplicates = df[col].value_counts()
    duplicates = duplicates[duplicates > 1]
    if not duplicates.empty:
        print("Duplicate values:")
        print(duplicates)
    else:
        print("No duplicates")

    null_count = df[col].isnull().sum()
    print(f"Missing values: {null_count}")

    print("-" * 30)

# are_columns_equal = pdf['fielder_2'].equals(pdf['fielder_2_1'])
# print(f"Are 'fielder_2' and 'fielder_2_1' the same? {are_columns_equal}")
# are_columns_equal = pdf['pitcher'].equals(pdf['pitcher_1'])
# print(f"Are 'pitcher' and 'pitcher_1' the same? {are_columns_equal}")

df.to_csv('../data/feature_cleaned.csv', index=False)