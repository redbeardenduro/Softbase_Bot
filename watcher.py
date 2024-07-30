import os
import logging
import subprocess
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logging.basicConfig(level=logging.INFO)

class CodeChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith('.py'):
            logging.info(f"Modified file: {event.src_path}")
            try:
                subprocess.run(['python3', 'get_code_suggestions.py', event.src_path], check=True)
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to run get_code_suggestions.py: {e}")
            except Exception as e:
                logging.error(f"Unexpected error: {e}")

def main():
    path = '.'  # Specify the directory to monitor
    event_handler = CodeChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
