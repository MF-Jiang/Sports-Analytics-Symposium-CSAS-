import pandas as pd

class pitcher_agent:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(self.file_path)
        self.id = self.data['pitcher'].iloc[0]
        self.basic_df = self.set_basic_performance()
        # self.display_basic_info()

        self.against_batter_df = self.set_performance_by_opponent()
        # print(self.against_batter_df)
        print("Create Pitcher "+str(self.id))

    def set_basic_performance(self):
        # Initialize a dictionary to store all statistics
        stats_dict = {}

        # Helper function to add stats for a given variable
        def add_stats(variable_name):
            stats_dict[f"{variable_name}_min"] = self.data[variable_name].min()
            stats_dict[f"{variable_name}_max"] = self.data[variable_name].max()
            stats_dict[f"{variable_name}_mean"] = self.data[variable_name].mean()
            stats_dict[f"{variable_name}_mode"] = self.data[variable_name].mode().iloc[0] if not self.data[
                variable_name].mode().empty else None
            stats_dict[f"{variable_name}_std"] = self.data[variable_name].std()

        # Add stats for each variable
        for var in ['release_speed', 'release_pos_x', 'release_pos_y', 'release_pos_z',
                    'pfx_x', 'pfx_z', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
                    'effective_speed', 'release_spin_rate', 'release_extension', 'plate_x', 'plate_z']:
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
            return {
                f"B_{variable_name}_min": group[variable_name].min(),
                f"B_{variable_name}_max": group[variable_name].max(),
                f"B_{variable_name}_mean": group[variable_name].mean(),
                f"B_{variable_name}_mode": group[variable_name].mode().iloc[0] if not group[
                    variable_name].mode().empty else None,
                f"B_{variable_name}_std": group[variable_name].std(),
            }

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
# test
pitcher_agent = pitcher_agent("../data/pitcher_data/434378.csv")
