import pandas as pd
import numpy as np
import time
from multiprocessing import Pool

file_path = "../data/pitcher_prediction_dataset/pitcher_prediction_dataset.csv"
# data = pd.read_csv(file_path)
data = pd.read_csv(file_path, low_memory=False)

def encode_column(column):
    column = column.fillna("0")

    value_counts = column.value_counts()
    non_zero_values = value_counts[value_counts.index != "0"]

    encoding = {value: idx + 1 for idx, (value, _) in enumerate(non_zero_values.items())}

    encoding["0"] = 0

    encoded_column = column.map(encoding)

    return encoded_column, encoding


data['pitch_type'], pitch_type_mapping = encode_column(data['pitch_type'])
data['bb_type'], bb_type_mapping = encode_column(data['bb_type'])
data['stand'], stand_mapping = encode_column(data['stand'])
data['inning_topbot'], inning_topbot_mapping = encode_column(data['inning_topbot'])

print("Pitch Type Mapping:", pitch_type_mapping)
print("BB Type Mapping:", bb_type_mapping)
print("Stand Mapping:", stand_mapping)
print("Inning Topbot Mapping:", inning_topbot_mapping)

required_columns = ['on_1b', 'on_2b', 'on_3b']
for column in required_columns:
    data[column] = data[column].apply(lambda x: 1 if pd.notna(x) else 0)
    # print(data[column])

data.drop(columns=['batter'], inplace=True)
data.drop(columns=['batter.1'], inplace=True)

data.sort_values(by=['pitcher', 'game_pk', 'at_bat_number', 'pitch_number'], inplace=True)

data.to_csv("../data/pitcher_prediction_dataset/pitcher_prediction_dataset_V2.csv", index=False)
print("save v2")

missing_values = data.isnull().sum()

print("Columns with missing values:")
print(missing_values[missing_values > 0])



for i in range(1, 6):
    data[f'prev_pitch_type_{i}'] = 0
    data[f'prev_delta_run_exp_{i}'] = 0.0
data.to_csv("../data/pitcher_prediction_dataset/pitcher_prediction_dataset_V3.csv", index=False)

# def add_previous_pitches(group):
#     group_data = group.copy().reset_index(drop=True)
#     # group_data = group.drop(columns=['pitcher', 'game_pk']).copy()
#     # print(group_data)
#     for idx in range(len(group_data)):
#         prev_rows = group_data.iloc[max(0, idx - 5):idx]
#         # print(prev_rows)
#
#         for i in range(1, 6):
#             if len(prev_rows) >= i:
#                 # print("here")
#                 # print(prev_rows.iloc[-i]['pitch_type'])
#                 group_data.loc[idx, f'prev_pitch_type_{i}'] = prev_rows.iloc[-i]['pitch_type']
#                 # print(group_data.loc[idx, f'prev_pitch_type_{i}'])
#                 group_data.loc[idx, f'prev_delta_run_exp_{i}'] = prev_rows.iloc[-i]['delta_run_exp']
#                 # group_data.at[idx, f'prev_pitch_type_{i}'] = prev_rows.iloc[-i]['pitch_type']
#                 # group_data.at[idx, f'prev_delta_run_exp_{i}'] = prev_rows.iloc[-i]['delta_run_exp']
#             else:
#                 # print("there")
#                 group_data.loc[idx, f'prev_pitch_type_{i}'] = 0
#                 group_data.loc[idx, f'prev_delta_run_exp_{i}'] = 0.0
#         # print(group_data)
#     group.update(group_data)
#     return group
#
# start_time = time.time()
# data = data.groupby(['pitcher', 'game_pk'], group_keys=False).apply(add_previous_pitches)
# end_time = time.time()
# execution_time = end_time - start_time
# print(f"Time taken to Combine Memory dataset: {execution_time:.2f} seconds")






