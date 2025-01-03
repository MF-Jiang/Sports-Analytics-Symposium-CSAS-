import pandas as pd

class pitcher_agent:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(self.file_path)
        self.id = self.data['pitcher'].iloc[0]

        # Belief:
        # Self ability performance
        # release speed
        self.release_speed_min = self.data['release_speed'].min()
        self.release_speed_max = self.data['release_speed'].max()
        self.release_speed_mean = self.data['release_speed'].mean()
        self.release_speed_mode = self.data['release_speed'].mode().iloc[0] if not self.data[
            'release_speed'].mode().empty else None
        self.release_speed_std = self.data['release_speed'].std()

        # release pos
        self.release_pos_x_min = self.data['release_pos_x'].min()
        self.release_pos_x_max = self.data['release_pos_x'].max()
        self.release_pos_x_mean = self.data['release_pos_x'].mean()
        self.release_pos_x_mode = self.data['release_pos_x'].mode().iloc[0] if not self.data[
            'release_pos_x'].mode().empty else None
        self.release_pos_x_std = self.data['release_pos_x'].std()

        self.release_pos_y_min = self.data['release_pos_y'].min()
        self.release_pos_y_max = self.data['release_pos_y'].max()
        self.release_pos_y_mean = self.data['release_pos_y'].mean()
        self.release_pos_y_mode = self.data['release_pos_y'].mode().iloc[0] if not self.data[
            'release_pos_y'].mode().empty else None
        self.release_pos_y_std = self.data['release_pos_y'].std()


        self.release_pos_z_min = self.data['release_pos_z'].min()
        self.release_pos_z_max = self.data['release_pos_z'].max()
        self.release_pos_z_mean = self.data['release_pos_z'].mean()
        self.release_pos_z_mode = self.data['release_pos_z'].mode().iloc[0] if not self.data[
            'release_pos_z'].mode().empty else None
        self.release_pos_z_std = self.data['release_pos_z'].std()

        # Throw hand
        self.p_throws_L = (self.data['p_throws'] == 'L').mean()
        self.p_throws_R = (self.data['p_throws'] == 'R').mean()

        # Type
        self.type_B = (self.data['type'] == 'B').mean()
        self.type_S = (self.data['type'] == 'S').mean()
        self.type_X = (self.data['type'] == 'X').mean()



    def display_basic_info(self):
        print("Pitcher Agent Information")
        print("Pitcher ID: " + str(self.id))
        print("=" * 30)
        print(f"File Path: {self.file_path}")
        print("\nRelease Speed Statistics:")
        print(f"  Min: {self.release_speed_min}")
        print(f"  Max: {self.release_speed_max}")
        print(f"  Mean: {self.release_speed_mean}")
        print(f"  Mode: {self.release_speed_mode}")
        print(f"  Std Dev: {self.release_speed_std}")

        print("\nRelease Position X Statistics:")
        print(f"  Min: {self.release_pos_x_min}")
        print(f"  Max: {self.release_pos_x_max}")
        print(f"  Mean: {self.release_pos_x_mean}")
        print(f"  Mode: {self.release_pos_x_mode}")
        print(f"  Std Dev: {self.release_pos_x_std}")

        print("\nRelease Position Y Statistics:")
        print(f"  Min: {self.release_pos_y_min}")
        print(f"  Max: {self.release_pos_y_max}")
        print(f"  Mean: {self.release_pos_y_mean}")
        print(f"  Mode: {self.release_pos_y_mode}")
        print(f"  Std Dev: {self.release_pos_y_std}")

        print("\nRelease Position Z Statistics:")
        print(f"  Min: {self.release_pos_z_min}")
        print(f"  Max: {self.release_pos_z_max}")
        print(f"  Mean: {self.release_pos_z_mean}")
        print(f"  Mode: {self.release_pos_z_mode}")
        print(f"  Std Dev: {self.release_pos_z_std}")

        print("\nThrow Hand Proportions:")
        print(f"  Left-handed (L): {self.p_throws_L:.2%}")
        print(f"  Right-handed (R): {self.p_throws_R:.2%}")

        print("\nThrow Type Proportions:")
        print(f"  Type B: {self.type_B:.2%}")
        print(f"  Type S: {self.type_S:.2%}")
        print(f"  Type X: {self.type_X:.2%}")

# test
pitcher_agent = pitcher_agent("../data/pitcher_data/434378.csv")
pitcher_agent.display_basic_info()