import pandas as pd

class BatterAgent:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(self.file_path)
        self.id = self.data['batter'].iloc[0]
        self.data['bb_type'] = self.data['bb_type'].fillna('other')
        self.data['launch_speed_angle'] = self.data['launch_speed_angle'].fillna(0)
        self.data = self.data.dropna(subset=['bat_speed', 'swing_length'])
        self.id = self.data['batter'].iloc[0]
        self.basic_df = self.set_basic_performance()
        self.against_pitcher_df = self.set_performance_by_opponent()

        print("Batter " + str(self.id) + " Created")

    def set_basic_performance(self):
        stats_dict = {}
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
        for var in ['zone', 'hc_x', 'hc_y', 'hit_distance_sc', 'estimated_ba_using_speedangle',
                    'babip_value', 'iso_value','bat_speed', 'swing_length']:
            add_stats(var)

        # stats_dict['stand_L'] = (self.data['stand'] == 'L').mean()
        # stats_dict['stand_R'] = (self.data['stand'] == 'R').mean()

        bb_types = ['fly_ball', 'ground_ball', 'line_drive', 'popup', 'other']
        for bb in bb_types:
            stats_dict[f"bb_type_{bb}"] = (self.data['bb_type'] == bb).mean()
        launch_speed_angle = [0,1,2,3,4,5,6]
        for lsa in launch_speed_angle:
            stats_dict[f"launch_speed_angle_{lsa}"] = (self.data['launch_speed_angle'] == lsa).mean()

        stats_df = pd.DataFrame([stats_dict])

        return stats_df

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
                    f"P_{variable_name}_level" : calculate_normalized_score(group[variable_name].mean(),calculate_expanded_mode(group[variable_name]),
                                                                          group[variable_name].min(),group[variable_name].max(),std),
                }
            except:
                return {f"P_{variable_name}_level" : self.basic_df[f"{variable_name}_level"]}

        # Initialize a list to store stats for each pitcher
        performance_list = []

        # Group data by 'pitcher'
        grouped_data = self.data.groupby('pitcher')

        # Loop through each group (pitcher-specific data)
        for pitcher, group in grouped_data:
            stats_dict = {'pitcher': pitcher}

            # Add stats for numeric variables
            for var in ['zone', 'hc_x', 'hc_y', 'hit_distance_sc', 'estimated_ba_using_speedangle',
                    'babip_value', 'iso_value','bat_speed', 'swing_length']:
                stats_dict.update(add_stats(group, var))

            # stats_dict['P_stand_L'] = (self.data['stand'] == 'L').mean()
            # stats_dict['P_stand_R'] = (self.data['stand'] == 'R').mean()

            bb_types = ['fly_ball', 'ground_ball', 'line_drive', 'popup', 'other']
            for bb in bb_types:
                stats_dict[f"P_bb_type_{bb}"] = (self.data['bb_type'] == bb).mean()
            launch_speed_angle = [0, 1, 2, 3, 4, 5, 6]
            for lsa in launch_speed_angle:
                stats_dict[f"P_launch_speed_angle_{lsa}"] = (self.data['launch_speed_angle'] == lsa).mean()

            # Append pitcher-specific stats to the performance list
            performance_list.append(stats_dict)

        # Convert performance list to DataFrame
        performance_df = pd.DataFrame(performance_list)

        return performance_df

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
# batter_agent = BatterAgent("../data/batter_data/444482.csv")
# print(batter_agent.id)