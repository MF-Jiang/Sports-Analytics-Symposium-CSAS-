import polars as pl

## Read the arrow file into a Table
ad = '../data/statcast_pitch_swing_data_20240402_20240630.arrow'
df = pl.read_ipc(ad, use_pyarrow = True)

# Discard features or invalid (only one value) features
columns_to_drop = ['spin_dir', 'spin_rate_deprecated','break_angle_deprecated', 'break_length_deprecated',
                  'tfs_deprecated', 'tfs_zulu_deprecated', 'umpire','game_type','game_year', 'fielder_2_1',
                   'pitcher_1','pitch_name']

df = df.drop(columns_to_drop)
pdf = df.to_pandas()

for col in pdf.columns:
    print(f"Feature: {col}")

    duplicates = pdf[col].value_counts()
    duplicates = duplicates[duplicates > 1]
    if not duplicates.empty:
        print("Duplicate values:")
        print(duplicates)
    else:
        print("No duplicates")

    null_count = pdf[col].isnull().sum()
    print(f"Missing values: {null_count}")

    print("-" * 30)

# are_columns_equal = pdf['fielder_2'].equals(pdf['fielder_2_1'])
# print(f"Are 'fielder_2' and 'fielder_2_1' the same? {are_columns_equal}")
# are_columns_equal = pdf['pitcher'].equals(pdf['pitcher_1'])
# print(f"Are 'pitcher' and 'pitcher_1' the same? {are_columns_equal}")

pdf.to_csv('../data/feature_cleaned.csv', index=False)