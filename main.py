import subprocess
import time
import os
import sys

# Add the project root directory to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(project_root)
os.environ['PYTHONPATH'] = project_root

services = [
    ("AI Service", "ai_service/main.py", 8001),
    ("Data Cleaner Service", "data_cleaner_service/main.py", 8002),
    ("Data Loader Service", "data_loader_service/main.py", 8003),
    ("Notification Service", "notification_service/main.py", 8004),
    ("Orchestrator", "orchestrator/main.py", None)
]

processes = []

try:
    for service_name, script_path, port in services:
        print(f"Starting {service_name}...")
        process = subprocess.Popen(["python", script_path], env=os.environ.copy())
        processes.append((service_name, process))
        if port:
            time.sleep(5)  # Wait for the service to start up
        print(f"{service_name} started.")

    # Wait for all processes to complete
    for service_name, process in processes:
        process.wait()

except KeyboardInterrupt:
    print("Shutting down all services...")
    for service_name, process in processes:
        print(f"Terminating {service_name}...")
        process.terminate()
        process.wait()
        print(f"{service_name} terminated.")
