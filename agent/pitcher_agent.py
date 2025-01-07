import pandas as pd



class PitcherAgent:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(self.file_path)
        # self.data = pd.read_excel(self.file_path)
        self.id = self.data['pitcher'].iloc[0]
        self.basic_df = self.set_basic_performance()
        # self.display_basic_info()
        # print(self.basic_df)
        self.against_batter_df = self.set_performance_by_opponent()
        # print(self.against_batter_df)
        print("Pitcher "+str(self.id) +" Created")
        # print(self.get_basic_with_batter_df(665833))

    def set_basic_performance(self):
        # Initialize a dictionary to store all statistics
        stats_dict = {}

        # Helper function to add stats for a given variable
        def add_stats(variable_name):
            std = self.data[variable_name].std()
            std = std if pd.notna(std) else 0
            # stats_dict[f"{variable_name}_min"] = self.data[variable_name].min()
            # stats_dict[f"{variable_name}_max"] = self.data[variable_name].max()
            # stats_dict[f"{variable_name}_mean"] = self.data[variable_name].mean()
            # # stats_dict[f"{variable_name}_mode"] = self.data[variable_name].mode().iloc[0] if not self.data[
            # #     variable_name].mode().empty else None
            # stats_dict[f"{variable_name}_mode"] = calculate_expanded_mode(self.data[variable_name])
            # stats_dict[f"{variable_name}_std"] = self.data[variable_name].std()
            stats_dict[f"{variable_name}_level"] = calculate_normalized_score(self.data[variable_name].mean(),
                                                                 calculate_expanded_mode(self.data[variable_name]),
                                                                 self.data[variable_name].min(), self.data[variable_name].max(),
                                                                 std)

        # Add stats for each variable
        for var in ['release_speed', 'release_pos_x', 'release_pos_y', 'release_pos_z',
                    'pfx_x', 'pfx_z', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
                    'effective_speed', 'release_spin_rate', 'release_extension', 'plate_x',
                    'plate_z']:
            add_stats(var)

        # Categorical stats
        stats_dict['p_throws_L'] = (self.data['p_throws'] == 'L').mean()
        stats_dict['p_throws_R'] = (self.data['p_throws'] == 'R').mean()
        stats_dict['type_B'] = (self.data['type'] == 'B').mean()
        stats_dict['type_S'] = (self.data['type'] == 'S').mean()
        stats_dict['type_X'] = (self.data['type'] == 'X').mean()

        # Pitch type stats
        pitch_types = ['CH', 'CS', 'CU', 'EP', 'FA', 'FC', 'FF', 'FO', 'FS', 'KC', 'KN', 'PO', 'SI', 'SL', 'ST', 'SV']
        for pitch in pitch_types:
            stats_dict[f"pitch_type_{pitch}"] = (self.data['pitch_type'] == pitch).mean()

        # Convert dictionary to DataFrame
        stats_df = pd.DataFrame([stats_dict])

        return stats_df

    def display_basic_info(self):
        print(self.basic_df)

    def get_basic_df(self):
        return self.basic_df

    def set_performance_by_opponent(self):
        # Helper function to calculate stats for a given variable within a group
        def add_stats(group, variable_name):
            # print(variable_name)
            try:
                std = group[variable_name].std()
                std = std if pd.notna(std) else 0
                # print(std)
                return {
                    # f"B_{variable_name}_min": group[variable_name].min(),
                    # f"B_{variable_name}_max": group[variable_name].max(),
                    # f"B_{variable_name}_mean": group[variable_name].mean(),
                    # # f"B_{variable_name}_mode": group[variable_name].mode().iloc[0] if not group[
                    # #     variable_name].mode().empty else None,
                    # f"{variable_name}_mode" : calculate_expanded_mode(group[variable_name]),
                    # f"B_{variable_name}_std": group[variable_name].std(),
                    f"B_{variable_name}_level" : calculate_normalized_score(group[variable_name].mean(),calculate_expanded_mode(group[variable_name]),
                                                                          group[variable_name].min(),group[variable_name].max(),std),
                }
            except:
                return {f"B_{variable_name}_level" : self.basic_df[f"{variable_name}_level"]}

        # Initialize a list to store stats for each batter
        performance_list = []

        # Group data by 'batter'
        grouped_data = self.data.groupby('batter')

        # Loop through each group (batter-specific data)
        for batter, group in grouped_data:
            stats_dict = {'batter': batter}

            # Add stats for numeric variables
            for var in ['release_speed', 'release_pos_x', 'release_pos_y', 'release_pos_z',
                    'pfx_x', 'pfx_z', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
                    'effective_speed', 'release_spin_rate', 'release_extension', 'plate_x', 'plate_z']:
                stats_dict.update(add_stats(group, var))

            stats_dict['B_p_throws_L'] = (group['p_throws'] == 'L').mean()
            stats_dict['B_p_throws_R'] = (group['p_throws'] == 'R').mean()
            stats_dict['B_type_B'] = (group['type'] == 'B').mean()
            stats_dict['B_type_S'] = (group['type'] == 'S').mean()
            stats_dict['B_type_X'] = (group['type'] == 'X').mean()

            # Add pitch type percentages
            pitch_types = ['CH', 'CS', 'CU', 'EP', 'FA', 'FC', 'FF', 'FO', 'FS', 'KC', 'KN', 'PO', 'SI', 'SL', 'ST',
                           'SV']
            for pitch in pitch_types:
                stats_dict[f"B_pitch_type_{pitch}"] = (group['pitch_type'] == pitch).mean()

            # Append batter-specific stats to the performance list
            performance_list.append(stats_dict)

        # Convert performance list to DataFrame
        performance_df = pd.DataFrame(performance_list)

        return performance_df

    def get_Against_Batter_df(self):
        return self.against_batter_df

    def get_basic_with_batter_df(self, batter):
        filtered_batter_df = self.against_batter_df[self.against_batter_df['batter'] == batter]

        if filtered_batter_df.empty:
            filtered_batter_df = self.basic_df.copy()
            filtered_batter_df.columns = [f"B_{col}" for col in filtered_batter_df.columns]
        else:
            filtered_batter_df = filtered_batter_df.drop(columns=['batter'])

        # print(filtered_batter_df)

        # Concatenate basic_df and the filtered row from batter_df
        combined_df = pd.concat([self.basic_df, filtered_batter_df], axis=1)
        return combined_df

def calculate_expanded_mode(series, window_size=1):
    """
    Calculate the mode considering a sliding window around each value.
    The function expands the window size until it finds a dominant mode.

    Parameters:
        series (pd.Series): The input data series.
        window_size (int): Initial window size to consider around each value.

    Returns:
        float: The calculated expanded mode.
    """
    # print(series)
    if series.empty:
        return None

    # Sort the series to facilitate window-based calculations
    sorted_series = series.sort_values().reset_index(drop=True)

    max_count = 0
    expanded_mode = None

    # Iterate through each value in the sorted series
    for center_index in range(len(sorted_series)):
        center_value = sorted_series[center_index]

        # Define the window around the current value
        lower_bound = center_value - window_size
        upper_bound = center_value + window_size

        # Count values within the window
        count_in_window = sorted_series[(sorted_series >= lower_bound) & (sorted_series <= upper_bound)].count()

        # Update the mode if this window has more values
        if count_in_window > max_count:
            max_count = count_in_window
            expanded_mode = center_value

    return expanded_mode

def calculate_normalized_score(mean_val, mode_val, min_val, max_val, std, alpha=0.3, beta=0.3, gamma=0.2, delta=0.2):
    range_val = max_val - min_val
    if range_val == 0:
        range_val = 0.001 # consider as R^2
    return alpha * (mean_val - min_val) / range_val + beta * (mode_val - min_val) / range_val + gamma * range_val - delta * std

# test
# pitcher_agent = PitcherAgent("../data/pitcher_data/621016.csv")
# pitcher_agent = PitcherAgent("../data/test.xlsx")

# print(calculate_normalized_score(91,91,91,91,0))