import pandas as pd

class pitcher_agent:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(self.file_path)
        self.id = self.data['pitcher'].iloc[0]
        self.basic_df = self.set_basic_performance()
        self.display_basic_info()

        self.against_batter_df = self.set_performance_by_opponent()
        print(self.against_batter_df)

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
        # Create a list to store performance data for each opponent
        performance_list = []

        # Group data by 'batter'
        grouped_data = self.data.groupby('batter')

        for batter, group in grouped_data:
            performance = {
                'batter': batter,
                'release_speed_min': group['release_speed'].min(),
                'release_speed_max': group['release_speed'].max(),
                'release_speed_mean': group['release_speed'].mean(),
                'release_speed_mode': group['release_speed'].mode().iloc[0] if not group[
                    'release_speed'].mode().empty else None,
                'release_speed_std': group['release_speed'].std(),

                'release_pos_x_min': group['release_pos_x'].min(),
                'release_pos_x_max': group['release_pos_x'].max(),
                'release_pos_x_mean': group['release_pos_x'].mean(),
                'release_pos_x_mode': group['release_pos_x'].mode().iloc[0] if not group[
                    'release_pos_x'].mode().empty else None,
                'release_pos_x_std': group['release_pos_x'].std(),

                # Add other variables in the same format as above
                'release_pos_y_min': group['release_pos_y'].min(),
                'release_pos_y_max': group['release_pos_y'].max(),
                'release_pos_y_mean': group['release_pos_y'].mean(),
                'release_pos_y_mode': group['release_pos_y'].mode().iloc[0] if not group[
                    'release_pos_y'].mode().empty else None,
                'release_pos_y_std': group['release_pos_y'].std(),

                'release_pos_z_min': group['release_pos_z'].min(),
                'release_pos_z_max': group['release_pos_z'].max(),
                'release_pos_z_mean': group['release_pos_z'].mean(),
                'release_pos_z_mode': group['release_pos_z'].mode().iloc[0] if not group[
                    'release_pos_z'].mode().empty else None,
                'release_pos_z_std': group['release_pos_z'].std(),
            }

            # Append pitch type percentages
            for pitch in [
                'CH', 'CS', 'CU', 'EP', 'FA',
                'FC', 'FF', 'FO', 'FS', 'KC',
                'KN', 'PO', 'SI', 'SL', 'ST', 'SV'
            ]:
                performance[f'pitch_type_{pitch}_percentage'] = (group['pitch_type'] == pitch).mean()

            performance_list.append(performance)

        # Convert performance list to a DataFrame
        performance_df = pd.DataFrame(performance_list)
        return performance_df

# test
pitcher_agent = pitcher_agent("../data/pitcher_data/434378.csv")
