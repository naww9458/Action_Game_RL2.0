import json
import os
import datetime

class Recorder:

    def __init__(self, task: str = "game_history_record"):
        """
        tasks:
        1. game_history_record
        2. temp_memory
        """
        # CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        CURRENT_DIR = ""
        if task == "game_history_record":
            collection_name = self.get_newest_record_name()
            self.json_file_path = CURRENT_DIR + "./game_history/" + collection_name + ".json"

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.json_file_path), exist_ok=True)

        if os.path.exists(self.json_file_path):
            print("Loading the json memory file")
            self.memory = self.load(self.json_file_path)
        else:
            print("The json memory file does not exist. Creating new file.")
            self.memory = {"game_records": []}  # Direct dictionary instead of json.loads
            with open(self.json_file_path, "w") as f:
                json.dump(self.memory, f)

    def get(self):
        print("Getting the json memory")
        return self.memory

    def add_no_limit(self, data: float, ):
        """
        Add a records.

        Args:
            role: The role of the sender (e.g., 'user', 'assistant')
            message: The message content
        """
        self.memory["game_records"].append({
            "game_total_duration": data,
            "timestamp": str(datetime.datetime.now())
        })

        self.save(self.json_file_path)

    def save(self, file_path):
        try:
            with open(file_path, 'w') as f:
                json.dump(self.memory, f)
        except Exception as e:
            print(f"Error saving memory to {file_path}: {e}")

    def load(self, file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading memory from {file_path}: {e}")
            return {"game_records": []}

    def get_newest_record_name(self) -> str:
        """
        傳回最新的對話歷史資料和集的名稱 (game_YYYY_MM)
            - 例如: "game_2022-01"
        """

        this_month = datetime.datetime.now().strftime("%Y-%m")
        return "record_" + this_month