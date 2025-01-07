import os
import pandas as pd
import time

from agent.batter_agent import BatterAgent


def generate_agents_from_csv(folder_path):
    """
    Read all CSV files from a folder and generate pitcher_agent instances.

    Parameters:
        folder_path (str): Path to the folder containing CSV files.

    Returns:
        dict: A dictionary where keys are CSV file names (without extension),
              and values are pitcher_agent instances.
    """
    # Initialize an empty dictionary to store agents
    agents = {}
    i=1
    # List all files in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file is a CSV
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)

            # Create a pitcher_agent instance
            # print(file_path)
            if file_name == '543518.csv' or file_name == '592325.csv' or file_name == '622491.csv'\
                    or file_name == '676679.csv' or file_name=='676896.csv':
                continue
            # if i==3:
            #     break
            print("Creating the "+ str(i) +" Batter Agent instance")
            print(file_name)
            agent_instance = BatterAgent(file_path)

            # Use the file name (without extension) as the key
            base_name = os.path.splitext(file_name)[0]
            agents[base_name] = agent_instance
        i = i+1
    return agents

folder_path = "../data/batter_data"
# folder_path = "../data/pitcher_test"

start_time = time.time()
agents_dict = generate_agents_from_csv(folder_path)
end_time = time.time()
execution_time = end_time - start_time
print(f"Time taken to generate agents: {execution_time:.2f} seconds")

# print(agents_dict.items())

batter_prediction_df = pd.read_csv("../data/feature_cleaned.csv")

start_time = time.time()
combined_data = []

for index, row in batter_prediction_df.iterrows():
    pitcher_id = row['pitcher']
    batter_id = row['batter']
    # print(1)
    # print(batter_id)
    agent = next((agent for name, agent in agents_dict.items() if agent.id == batter_id), None)
    # print(2)
    # print(agent.id)
    if agent is not None:
        agent_basic_df = agent.basic_df.reset_index(drop=True)
        row_df = row.to_frame().T.reset_index(drop=True)

        pitcher_row = agent.against_pitcher_df.loc[
            agent.against_pitcher_df['pitcher'] == pitcher_id
            ]

        if not pitcher_row.empty:
            pitcher_row = pitcher_row.reset_index(drop=True)
            combined_df = pd.concat([row_df, agent_basic_df, pitcher_row], axis=1)
            combined_data.append(combined_df)
end_time = time.time()
execution_time = end_time - start_time
print(f"Time taken to Combine dataset: {execution_time:.2f} seconds")
# print(combined_data)

final_combined_df = pd.concat(combined_data, ignore_index=True)

# final_combined_df = pd.read_csv("../data/pitcher_prediction_dataset/pitcher_prediction_dataset.csv")

columns_to_keep = [
    "pitch_type", "pitcher", "release_speed", "release_pos_x", "release_pos_y",
    "release_pos_z", "pfx_x", "pfx_z", "vx0", "vy0",
    "vz0", "ax", "ay", "az", "effective_speed",
    "release_spin_rate", "release_extension", "plate_x",
    "plate_z", "p_throws",
    "balls", "strikes", "on_3b", "on_2b", "on_1b", "outs_when_up",
    "inning", "inning_topbot", "home_score", "away_score", "at_bat_number",
    "pitch_number", "game_pk", "delta_run_exp",
    'zone_level', 'hc_x_level', 'hc_y_level', 'hit_distance_sc_level', 'estimated_ba_using_speedangle_level',
    'babip_value_level', 'iso_value_level','bat_speed_level', 'swing_length_level','stand',
    'P_bb_type_fly_ball','P_bb_type_ground_ball',
    'P_bb_type_line_drive','P_bb_type_popup',
    'P_bb_type_other','P_launch_speed_angle_0','P_launch_speed_angle_1','P_launch_speed_angle_2','P_launch_speed_angle_3',
    'P_launch_speed_angle_4', 'P_launch_speed_angle_5', 'P_launch_speed_angle_6','bb_type','batter'
]

filtered_combined_df = final_combined_df[columns_to_keep]

filtered_combined_df.to_csv("../data/batter_prediction_dataset/batter_prediction_dataset.csv", index=False)







