import os
import pandas as pd
import time
from agent.pitcher_agent import PitcherAgent


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
            # if i==3:
            #     break
            print("Creating the "+ str(i) +" Pitcher Agent instance")
            agent_instance = PitcherAgent(file_path)

            # Use the file name (without extension) as the key
            base_name = os.path.splitext(file_name)[0]
            agents[base_name] = agent_instance
        i = i+1
    return agents

folder_path = "../data/pitcher_data"
# folder_path = "../data/pitcher_test"

start_time = time.time()
agents_dict = generate_agents_from_csv(folder_path)
end_time = time.time()
execution_time = end_time - start_time
print(f"Time taken to generate agents: {execution_time:.2f} seconds")



pitcher_prediction_df = pd.read_csv("../data/feature_cleaned.csv")

start_time = time.time()
combined_data = []

for index, row in pitcher_prediction_df.iterrows():
    pitcher_id = row['pitcher']
    batter_id = row['batter']

    agent = next((agent for name, agent in agents_dict.items() if agent.id == pitcher_id), None)

    if agent is not None:
        agent_basic_df = agent.basic_df.reset_index(drop=True)
        row_df = row.to_frame().T.reset_index(drop=True)

        batter_row = agent.against_batter_df.loc[
            agent.against_batter_df['batter'] == batter_id
            ]

        if not batter_row.empty:
            batter_row = batter_row.reset_index(drop=True)
            combined_df = pd.concat([row_df, agent_basic_df, batter_row], axis=1)
            combined_data.append(combined_df)
end_time = time.time()
execution_time = end_time - start_time
print(f"Time taken to Combine dataset: {execution_time:.2f} seconds")
# print(combined_data)

final_combined_df = pd.concat(combined_data, ignore_index=True)

# final_combined_df = pd.read_csv("../data/pitcher_prediction_dataset/pitcher_prediction_dataset.csv")

columns_to_keep = [
    "pitch_type", "release_speed_level", "release_pos_x_level", "release_pos_y_level",
    "release_pos_z_level", "pfx_x_level", "pfx_z_level", "vx0_level", "vy0_level",
    "vz0_level", "ax_level", "ay_level", "az_level", "effective_speed_level",
    "release_spin_rate_level", "release_extension_level", "plate_x_level",
    "plate_z_level", "p_throws_L", "p_throws_R", "type_B", "type_S", "type_X",
    "pitch_type_CH", "pitch_type_CS", "pitch_type_CU", "pitch_type_EP", "pitch_type_FA",
    "pitch_type_FC", "pitch_type_FF", "pitch_type_FO", "pitch_type_FS", "pitch_type_KC",
    "pitch_type_KN", "pitch_type_PO", "pitch_type_SI", "pitch_type_SL", "pitch_type_ST",
    "pitch_type_SV", "batter", "B_release_speed_level", "B_release_pos_x_level",
    "B_release_pos_y_level", "B_release_pos_z_level", "B_pfx_x_level", "B_pfx_z_level",
    "B_vx0_level", "B_vy0_level", "B_vz0_level", "B_ax_level", "B_ay_level", "B_az_level",
    "B_effective_speed_level", "B_release_spin_rate_level", "B_release_extension_level",
    "B_plate_x_level", "B_plate_z_level", "B_p_throws_L", "B_p_throws_R", "B_type_B",
    "B_type_S", "B_type_X", "B_pitch_type_CH", "B_pitch_type_CS", "B_pitch_type_CU",
    "B_pitch_type_EP", "B_pitch_type_FA", "B_pitch_type_FC", "B_pitch_type_FF",
    "B_pitch_type_FO", "B_pitch_type_FS", "B_pitch_type_KC", "B_pitch_type_KN",
    "B_pitch_type_PO", "B_pitch_type_SI", "B_pitch_type_SL", "B_pitch_type_ST",
    "B_pitch_type_SV", "balls", "strikes", "on_3b", "on_2b", "on_1b", "outs_when_up",
    "inning", "inning_topbot", "home_score", "away_score", "at_bat_number",
    "pitch_number", "game_pk", "delta_run_exp", "stand", "pitcher"
]

filtered_combined_df = final_combined_df[columns_to_keep]

filtered_combined_df.to_csv("../data/pitcher_prediction_dataset/pitcher_prediction_dataset.csv", index=False)







