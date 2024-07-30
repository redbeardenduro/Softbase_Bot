import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

class Config:
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = int(os.getenv('DB_PORT', '5432'))  # Default to 5432 if not set
    DB_USER = os.getenv('DB_USER', 'test_user')
    DB_PASSWORD = os.getenv('DB_PASSWORD', 'test_password')
    DB_NAME = os.getenv('DB_NAME', 'test_db')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your_openai_api_key')
    FROM_EMAIL = os.getenv('FROM_EMAIL', 'your_email@example.com')
    FROM_PASSWORD = os.getenv('FROM_PASSWORD', 'your_email_password')