import pandas as pd

class pitcher_agent:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(self.file_path)
        self.id = self.data['pitcher'].iloc[0]
        self.basic_df = self.set_basic_performance()
        self.display_basic_info()

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

# test
pitcher_agent = pitcher_agent("../data/pitcher_data/434378.csv")
