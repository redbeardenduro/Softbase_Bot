from openai import OpenAI
from config.settings import Config
import logging

client = OpenAI(api_key=Config.OPENAI_API_KEY)

class ChatGPTIntegration:
    def __init__(self):
        pass

    def query_openai(self, prompt):
        try:
            response = client.chat.completions.create(
                model="gpt-4",  # Use the gpt-4 model
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"OpenAI API call failed: {e}")
            return None

    def find_duplicates(self, parts_df):
        logging.info(f"parts_df type in find_duplicates: {type(parts_df)}")
        logging.info(f"parts_df contents in find_duplicates: {parts_df}")
        prompt = "Identify duplicates in the following parts data:\n"
        for row in parts_df:
            prompt += f"Part ID: {row['part_id']}, Name: {row['part_name']}, Description: {row['part_description']}\n"
        prompt += "\nProvide a list of duplicate parts based on their names and descriptions."
        return self.query_openai(prompt)

    def generate_report(self, parts_df):
        logging.info(f"parts_df type in generate_report: {type(parts_df)}")
        logging.info(f"parts_df contents in generate_report: {parts_df}")
        prompt = "Generate a detailed report based on the following parts data:\n"
        for row in parts_df:
            prompt += f"Part ID: {row['part_id']}, Name: {row['part_name']}, Description: {row['part_description']}\n"
        prompt += "\nInclude findings, identified duplicates, and suggestions for improvements."
        return self.query_openai(prompt)
