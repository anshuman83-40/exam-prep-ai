"""Level 0: Send one question to Gemini and print the answer.

Run:  python levels/level0_hello_ai.py
"""

import os

from dotenv import load_dotenv
from google import genai

# 1. Load the secret API key from the .env file (never write keys in code!)
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key or api_key == "your-key-here":
    raise SystemExit("No API key found. Copy .env.example to .env and paste your Gemini key.")

# 2. Connect to Gemini
client = genai.Client(api_key=api_key)
model = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")

# 3. Ask a question
question = input("Ask the AI anything: ")
response = client.models.generate_content(model=model, contents=question)

# 4. Print the answer
print("\nAI says:\n")
print(response.text)
