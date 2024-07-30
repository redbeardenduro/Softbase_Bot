import os
import ast
import shutil
import subprocess
import time
import logging
import re
import autopep8
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import json
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from black import format_file_contents, Mode
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Configure logging
logging.basicConfig(filename='code_suggestions.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load API key from .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize context for maintaining conversation history
context = []

# Paths for machine learning models and vectorizers
MODEL_PATH = "suggestion_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

# Load or initialize machine learning model
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(VECTORIZER_PATH, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
else:
    model = DecisionTreeClassifier()
    vectorizer = TfidfVectorizer()
    # Initial non-empty training data
    initial_code = ["def foo():\n    return 'bar'"]
    X_train = vectorizer.fit_transform(initial_code)
    y_train = [1]  # Assume the initial suggestion is valid
    model.fit(X_train, y_train)

# Load the trained syntax correction model and tokenizer
syntax_model = T5ForConditionalGeneration.from_pretrained('syntax_correction_model')
syntax_tokenizer = T5Tokenizer.from_pretrained('syntax_correction_tokenizer')

def log_info(message):
    """Log informational messages."""
    logging.info(message)

def log_error(message):
    """Log error messages."""
    logging.error(message)

def get_suggestions(prompt, max_retries=3):
    """Get suggestions from OpenAI based on the given prompt."""
    context.append({"role": "user", "content": prompt})
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=context
            )
            suggestions = response.choices[0].message.content.strip()
            context.append({"role": "assistant", "content": suggestions})
            if validate_suggestions(suggestions):
                return suggestions
        except Exception as e:
            log_error(f"Error getting suggestions (attempt {attempt + 1}/{max_retries}): {e}")
    return ""

def validate_suggestions(suggestions):
    """Validate the code suggestions for syntax errors using a machine learning model."""
    try:
        # Remove invalid characters and check for unterminated string literals
        suggestions = suggestions.replace('’', "'").replace('“', '"').replace('”', '"')
        compile(suggestions, "<string>", "exec")
        formatted_code = format_file_contents(suggestions, fast=False, mode=Mode())
        return True
    except Exception as e:
        log_error(f"SyntaxError in suggestions: {e}")
        return False

def correct_syntax_errors_with_model(suggestions, max_retries=3):
    """Request corrections for syntax errors using a pre-trained model."""
    for attempt in range(max_retries):
        try:
            inputs = syntax_tokenizer.encode(suggestions, return_tensors='pt', max_length=512, truncation=True)
            outputs = syntax_model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
            corrected_suggestions = syntax_tokenizer.decode(outputs[0], skip_special_tokens=True)
            if validate_suggestions(corrected_suggestions):
                return corrected_suggestions
        except Exception as e:
            log_error(f"Error correcting syntax with model (attempt {attempt + 1}/{max_retries}): {e}")
    return suggestions

def read_code_file(file_path):
    """Read code from a file."""
    try:
        with open(file_path, 'r') as file:
            code = file.read()
        return code
    except Exception as e:
        log_error(f"Error reading file {file_path}: {e}")
        return ""

def write_code_file(file_path, code):
    """Write code to a file."""
    try:
        with open(file_path, 'w') as file:
            file.write(code)
    except Exception as e:
        log_error(f"Error writing file {file_path}: {e}")

def backup_file(file_path):
    """Create a backup of the original file."""
    try:
        backup_path = file_path + ".bak"
        shutil.copy(file_path, backup_path)
        return backup_path
    except Exception as e:
        log_error(f"Error creating backup for file {file_path}: {e}")
        return ""

def apply_suggestions(code, suggestions):
    """Apply code suggestions to the existing code."""
    updated_code = code
    if suggestions:
        lines = suggestions.split('\n')
        for line in lines:
            if line.startswith('#'):
                continue  # Skip comments
            updated_code += '\n' + line
    try:
        formatted_code = format_file_contents(updated_code, fast=False, mode=Mode())
        return formatted_code
    except Exception as e:
        log_error(f"Error formatting code with black: {e}")
        return updated_code

def extract_functions(code):
    """Extract function names from code."""
    try:
        tree = ast.parse(code)
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        return functions
    except SyntaxError as e:
        log_error(f"SyntaxError while parsing code: {e}")
        return []

def generate_tests(file_path, functions):
    """Generate a basic test file for each code file."""
    test_file_path = file_path.replace('.py', '_test.py')
    try:
        with open(test_file_path, 'w') as test_file:
            test_file.write('import unittest\n\n')
            for func in functions:
                test_file.write(f'class Test{func.capitalize()}(unittest.TestCase):\n')
                test_file.write(f'    def test_{func}(self):\n')
                test_file.write(f'        # TODO: Add tests for {func}\n')
                test_file.write('        pass\n\n')
    except Exception as e:
        log_error(f"Error generating tests for file {file_path}: {e}")

def run_script(file_path):
    """Run a Python script and log its output or errors."""
    try:
        result = subprocess.run(['python3', file_path], capture_output=True, text=True)
        if result.returncode == 0:
            log_info(f"Script {file_path} ran successfully.")
            return ""
        else:
            log_error(f"Script {file_path} failed with error:\n{result.stderr}")
            return result.stderr
    except Exception as e:
        log_error(f"Error running {file_path}: {e}")
        return str(e)

def run_tests():
    """Run all test files and log results."""
    try:
        result = subprocess.run(['pytest', '--maxfail=1', '--disable-warnings', '-q'], capture_output=True, text=True)
        if result.returncode == 0:
            log_info("All tests passed.")
        else:
            log_error(f"Tests failed with output:\n{result.stdout}\n{result.stderr}")
    except Exception as e:
        log_error(f"Error running tests: {e}")

def run_main():
    """Run the main.py script."""
    try:
        result = subprocess.run(['python3', 'main.py'], capture_output=True, text=True)
        if result.returncode == 0:
            log_info("main.py ran successfully.")
        else:
            log_error(f"main.py failed with error:\n{result.stderr}")
    except Exception as e:
        log_error(f"Error running main.py: {e}")

def check_processed_files(processed_files_path):
    """Load the list of processed files."""
    if not os.path.exists(processed_files_path):
        return set()

    with open(processed_files_path, 'r') as file:
        processed_files = set(line.strip() for line in file)
    return processed_files

def mark_file_as_processed(processed_files_path, file_path):
    """Mark a file as processed."""
    with open(processed_files_path, 'a') as file:
        file.write(file_path + '\n')

def update_model(suggestions, is_valid):
    """Update the machine learning model with new data."""
    global model, vectorizer
    X_train = vectorizer.transform([suggestions])
    y_train = [1 if is_valid else 0]
    model.fit(X_train, y_train)
    # Save updated model and vectorizer
    with open(MODEL_PATH, 'wb') as model_file:
        pickle.dump(model, model_file)
    with open(VECTORIZER_PATH, 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

def process_file(code_file_path, processed_files):
    """Process an individual file."""
    log_info(f"Processing {code_file_path}...")
    
    code = read_code_file(code_file_path)
    
    if not code:
        log_info(f"Skipping {code_file_path} due to empty content.")
        return
    
    # Create a backup of the original file
    backup_path = backup_file(code_file_path)
    
    # Get code suggestions
    suggestions = get_suggestions(f"Here is some Python code:\n{code}\nWhat changes or improvements can be made?")
    log_info(f"Suggestions for {code_file_path}:\n{suggestions}\n")
    
    # Validate suggestions
    if not validate_suggestions(suggestions):
        suggestions = correct_syntax_errors_with_model(suggestions)
        if not validate_suggestions(suggestions):
            log_error(f"Invalid suggestions for {code_file_path}, skipping.")
            return
    
    # Apply suggestions
    updated_code = apply_suggestions(code, suggestions)
    
    # Write the updated code back to the file
    write_code_file(code_file_path, updated_code)
    
    # Generate tests for the script
    functions = extract_functions(updated_code)
    if functions:
        generate_tests(code_file_path, functions)
    
    # Run the updated script to ensure it works
    errors = run_script(code_file_path)
    if errors:
        log_error(f"Errors encountered while running {code_file_path}: {errors}")
        # Revert changes if errors are detected
        log_info(f"Reverting changes for {code_file_path} due to errors.")
        shutil.copy(backup_path, code_file_path)
        update_model(suggestions, False)
    else:
        update_model(suggestions, True)
    
    # Run all tests to ensure everything works
    run_tests()
    
    # Run main.py to ensure the main application is functional
    run_main()
    
    # Mark file as processed
    mark_file_as_processed('processed_files.txt', code_file_path)

def scan_and_process_directory(root_dir, exclude_files=None, sleep_interval=60):
    """Scan through all Python files in the directory and apply suggestions."""
    if exclude_files is None:
        exclude_files = set()

    processed_files_path = 'processed_files.txt'
    processed_files = check_processed_files(processed_files_path)

    while True:
        for subdir, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.py'):
                    code_file_path = os.path.join(subdir, file)

                    # Skip excluded files and already processed files
                    if code_file_path in exclude_files or code_file_path in processed_files:
                        continue

                    process_file(code_file_path, processed_files)
        
        # Sleep for a while before re-checking
        time.sleep(sleep_interval)

if __name__ == "__main__":
    root_directory = os.path.dirname(os.path.abspath(__file__))
    # Exclude specific files from processing
    exclude_files = {os.path.join(root_directory, 'watcher.py')}
    scan_and_process_directory(root_directory, exclude_files=exclude_files)
