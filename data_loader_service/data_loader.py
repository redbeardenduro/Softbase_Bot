import logging

class DataLoader:
    def load_parts_data(self):
        # Simulated data loading
        parts_data = [
            {"part_id": 1, "part_name": "Part A", "part_description": "Description of Part A"},
            {"part_id": 2, "part_name": "Part B", "part_description": "Description of Part B"}
            # Add more parts as needed
        ]
        logging.info(f"Loaded parts data: {parts_data}")
        return parts_data
