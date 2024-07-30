import requests
import logging
from config.settings import Config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_bot():
    try:
        response = requests.get("http://localhost:8003/load_parts_data")
        if response.status_code == 200:
            parts_data = response.json()["parts_data"]
            logging.info(f"Loaded parts data batch: {parts_data}")
        else:
            logging.error(f"Failed to load parts data: {response.status_code} - {response.text}")
            return

        # Ensure parts_data is a list of dictionaries
        if not isinstance(parts_data, list) or not all(isinstance(item, dict) for item in parts_data):
            logging.error("Invalid parts_data structure")
            return

        # Wrap parts_data in a dictionary with the key "parts_df"
        report_request_data = {"parts_df": parts_data}

        response = requests.post("http://localhost:8001/generate_report", json=report_request_data)
        report = response.json().get("report")
        if report:
            logging.info("Generated report")
            print(report)
        else:
            logging.error("Failed to generate report: %s", response.json())

        response = requests.post("http://localhost:8001/find_duplicates", json=report_request_data)
        duplicates = response.json().get("duplicates")
        if duplicates:
            logging.info("Identified duplicates")
            print(duplicates)
        else:
            logging.error("Failed to find duplicates: %s", response.json())

        merge_data = {"duplicates": duplicates, "engine": "database_engine"}
        create_data = {"parts_df": parts_data, "engine": "database_engine"}
        update_data = {"parts_df": parts_data, "engine": "database_engine"}

        response = requests.post("http://localhost:8002/merge_duplicates", json=merge_data)
        if response.status_code != 200:
            logging.error(f"Failed to merge duplicates: {response.json()}")

        response = requests.post("http://localhost:8002/create_new_part_numbers", json=create_data)
        if response.status_code != 200:
            logging.error(f"Failed to create new part numbers: {response.json()}")

        response = requests.post("http://localhost:8002/update_database", json=update_data)
        if response.status_code != 200:
            logging.error(f"Failed to update database: {response.json()}")

        email_data = {"to_email": "recipient@example.com", "subject": "Parts Database Update", "body": "The parts database has been updated successfully."}
        response = requests.post("http://localhost:8004/send_email", json=email_data)
        if response.status_code == 200:
            logging.info("Email sent successfully")
        else:
            logging.error("Failed to send email: %s", response.json())

        response = requests.post("http://localhost:8004/generate_report", json=report_request_data)
        if response.status_code == 200:
            logging.info("Generated report via email notification service")
        else:
            logging.error("Failed to generate report via email notification service: %s", response.json())

        # Shut down the services
        logging.info("Shutting down services...")
        requests.get("http://localhost:8001/shutdown")
        requests.get("http://localhost:8002/shutdown")
        requests.get("http://localhost:8003/shutdown")
        requests.get("http://localhost:8004/shutdown")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    run_bot()
