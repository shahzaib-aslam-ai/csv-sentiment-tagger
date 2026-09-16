import os
import logging
import time
import csv
import json
from datetime import datetime

from dotenv import load_dotenv
from groq import Groq


# Load environment variables
load_dotenv()


# Logging configuration
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


# API key validation
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("Error: GROQ_API_KEY is missing.")
    logging.error("GROQ_API_KEY is missing.")
    raise SystemExit()


# System prompt
system_prompt = """
You are an AI assistant specialized in customer feedback sentiment analysis.

Your task is to:
1. Classify the customer feedback as Positive, Negative, or Neutral.
2. Provide a short reason for the selected sentiment.

Output format:
{
    "sentiment": "Positive",
    "reason": "Short reason here"
}

Important rules:
1. The sentiment must be exactly one of: Positive, Negative, Neutral.
2. The reason must be short and directly related to the feedback.
3. Return only a valid JSON object.
4. Do not include markdown, explanations, or any extra text.
"""


# Input file validation
file_name = "feedback.csv"
required_column = "feedback_text"

if not os.path.isfile(file_name):
    print(f"Error: {file_name} does not exist.")
    logging.error(f"{file_name} does not exist.")
    raise SystemExit()


# Create Groq client once
client = Groq(api_key=api_key)


# Start timer
start_time = time.time()


# Allowed sentiment values
allowed_sentiments = [
    "Positive",
    "Negative",
    "Neutral"
]


results = []

row_count = 0
processed_count = 0
failed_count = 0
skipped_count = 0


# Read CSV
with open(
    file_name,
    "r",
    encoding="utf-8",
    newline=""
) as file:

    reader = csv.DictReader(file)

    # Check empty CSV
    if not reader.fieldnames:
        print(f"Error: {file_name} is empty.")
        logging.error(f"{file_name} is empty.")
        raise SystemExit()

    # Check required column
    if required_column not in reader.fieldnames:
        print(
            f"Error: Required column '{required_column}' is missing."
        )
        logging.error(
            f"Required column '{required_column}' is missing."
        )
        raise SystemExit()


    # Process rows
    for row in reader:

        row_count += 1

        feedback = row[required_column].strip()

        # Skip empty feedback
        if not feedback:
            skipped_count += 1

            logging.warning(
                f"Empty feedback row skipped: {row_count}"
            )

            continue


        try:

            # API request
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": feedback
                    }
                ],
                model="openai/gpt-oss-20b",
                response_format={"type": "json_object"}
            )


            # Get AI response
            response = chat_completion.choices[0].message.content


            # Parse JSON
            result = json.loads(response)


            # Validate required keys
            sentiment = result["sentiment"]
            reason = result["reason"]


            # Validate sentiment
            if sentiment not in allowed_sentiments:
                raise ValueError(
                    f"Invalid sentiment: {sentiment}"
                )
            required_keys = ["sentiment", "reason"]
            for key in required_keys:
                if key not in result:
                    raise ValueError(f"Missing required key: {key}")


            # Merge original feedback with AI result
            final_result = {
                "feedback": feedback,
                "sentiment": sentiment,
                "reason": reason
            }


            results.append(final_result)

            processed_count += 1


        except Exception as e:

            failed_count += 1

            logging.error(
                f"Row {row_count} failed: {e}"
            )

            continue


# Check if there are no data rows
if row_count == 0:
    logging.warning("CSV contains no data rows.")


# Processing summary
logging.info(
    f"{processed_count}/{row_count} rows processed successfully, "
    f"{failed_count} failed, "
    f"{skipped_count} skipped"
)


# Create output directory
os.makedirs("output", exist_ok=True)


# Create timestamped output filename
timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

output_file = (
    f"output/tagged_feedback_{timestamp}.csv"
)


# Save results
with open(
    output_file,
    "w",
    encoding="utf-8",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "feedback",
            "sentiment",
            "reason"
        ]
    )

    writer.writeheader()
    writer.writerows(results)


# Calculate total processing time
end_time = time.time()

total_time = end_time - start_time


logging.info(
    f"Total processing time: {total_time:.2f} seconds"
)

print(
    f"{processed_count}/{row_count} rows processed successfully, "
    f"{failed_count} failed, "
    f"{skipped_count} skipped"
)

print(f"Output saved to: {output_file}")
print(f"Total processing time: {total_time:.2f} seconds")

