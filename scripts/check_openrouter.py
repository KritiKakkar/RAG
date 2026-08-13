#!/usr/bin/env python3

import os
import sys
import json
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    print("Error: OPENROUTER_API_KEY not found in .env file")
    sys.exit(1)

print(f"Found API key (first 5 chars: {api_key[:5]}...)")

# Make the API request
try:
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost:8000",  # Replace with your site URL
            "X-OpenRouter-Title": "RAG System Test",  # Replace with your site name
        },
        data=json.dumps({
            "model": "nvidia/nemotron-3.5-lightning:free",  # Using a specific model for testing
            "messages": [
                {
                    "role": "user",
                    "content": "What is the meaning of life? Answer in one short sentence."
                }
            ]
        })
    )

    # Check if the request was successful
    if response.status_code == 200:
        result = response.json()
        print("\nAPI Test Successful! ✅")
        print(f"Model used: {result.get('model', 'unknown')}")
        print(f"Response: {result['choices'][0]['message']['content']}")
    else:
        print(f"\nAPI Test Failed! ❌ Status code: {response.status_code}")
        print(f"Error: {response.text}")

except Exception as e:
    print(f"\nAPI Test Failed! ❌")
    print(f"Exception: {str(e)}")
