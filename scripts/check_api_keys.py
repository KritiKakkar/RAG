#!/usr/bin/env python3

import os
import sys
import json
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment
openai_key = os.getenv("OPENAI_API_KEY")
google_key = os.getenv("GOOGLE_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")

print("API Key Check Tool\n")

# Check Pinecone API key
if pinecone_key:
    print(f"✅ Pinecone API key found (first 5 chars: {pinecone_key[:5]}...)")
else:
    print("❌ Pinecone API key not found in .env file")

# Check OpenAI API key
if openai_key:
    print(f"✅ OpenAI API key found (first 5 chars: {openai_key[:5]}...)")
    print("Testing OpenAI connection...")

    try:
        response = requests.post(
            url="https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openai_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "user",
                        "content": "Say hello in one word."
                    }
                ],
                "max_tokens": 10
            }
        )

        if response.status_code == 200:
            result = response.json()
            print("✅ OpenAI API test successful!")
            print(f"Response: {result['choices'][0]['message']['content'].strip()}")
        else:
            print(f"❌ OpenAI API test failed! Status code: {response.status_code}")
            print(f"Error: {response.text}")

    except Exception as e:
        print(f"❌ OpenAI API test failed!")
        print(f"Exception: {str(e)}")
else:
    print("⚠️ OpenAI API key not found in .env file (only needed if using OpenAI models)")

# Check Google API key
if google_key:
    print(f"✅ Google API key found (first 5 chars: {google_key[:5]}...)")
    print("Note: Google Gemini API can't be easily tested via direct requests. Please try running the embedding test script.")
else:
    print("⚠️ Google API key not found in .env file (only needed if using Google Gemini models)")

print("\nReminder: Never commit your .env file containing API keys to GitHub!")