import pandas as pd

class BatterAgent:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(self.file_path)
        self.id = self.data['batter'].iloc[0]
        self.basic_df = self.set_basic_performance()

    def set_basic_performance(self):
        stats_dict = {}

        return stats_dict