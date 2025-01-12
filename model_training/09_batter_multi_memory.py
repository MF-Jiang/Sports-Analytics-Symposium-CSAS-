import pandas as pd
import numpy as np
import time
from multiprocessing import Pool

def add_previous_battes(group):
    group_data = group.copy().reset_index(drop=True)
    for idx in range(len(group_data)):
        prev_rows = group_data.iloc[max(0, idx - 5):idx]
        for i in range(1, 6):
            if len(prev_rows) >= i:
                group_data.loc[idx, f'prev_batter_type_{i}'] = prev_rows.iloc[-i]['bb_type']
                group_data.loc[idx, f'prev_bat_win_exp_{i}'] = prev_rows.iloc[-i]['bat_win_exp']
            else:
                group_data.loc[idx, f'prev_batter_type_{i}'] = 0
                group_data.loc[idx, f'prev_bat_win_exp_{i}'] = 0.0
    print(group_data)
    return group_data

def process_group(group):
    return add_previous_battes(group)

def parallelize_dataframe(df, func, num_workers=4):
    groups = [group for _, group in df.groupby(['batter', 'game_pk'])]
    with Pool(num_workers) as pool:
        result = pool.map(func, groups)
    return pd.concat(result)

if __name__ == '__main__':
    data=pd.read_csv('../data/batter_prediction_dataset/batter_prediction_dataset_V3.csv', low_memory=False)
    start_time = time.time()
    data_parallelized = parallelize_dataframe(data, process_group, num_workers=8)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time taken to process the data with parallelization: {execution_time:.2f} seconds")


    data_parallelized.to_csv("../data/batter_prediction_dataset/batter_prediction_dataset_V4.csv", index=False)
    print("save v3")


    missing_values = data.isnull().sum()

    print("Columns with missing values:")
    print(missing_values[missing_values > 0])