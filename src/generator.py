import os
import json
import requests
from typing import Optional, List, Dict, Any

from src.config import CHAT_MODEL

class OpenRouterLLM:
    """Chat model using OpenRouter API as a unified endpoint for multiple LLMs."""

    def __init__(self,
                 model_name: Optional[str] = None,
                 temperature: float = 0.0,
                 api_key: Optional[str] = None):
        """Initialize the OpenRouter chat model.

        Args:
            model_name: Model identifier on OpenRouter (default from config)
            temperature: Temperature for generation (0.0 = deterministic)
            api_key: OpenRouter API key (default from env)
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key is required. Set OPENROUTER_API_KEY in .env file."
            )

        self.model_name = model_name or CHAT_MODEL
        self.temperature = temperature
        self.api_base = "https://openrouter.ai/api/v1"
        self.site_name = "RAG System"
        self.site_url = "http://localhost:8000"

        print(f"Using OpenRouter LLM with model: {self.model_name}")

    def invoke(self, messages):
        """Generate a response from the model using the provided messages.

        Args:
            messages: A list of messages in LangChain format or a single message string

        Returns:
            A response object with content attribute
        """
        # Convert LangChain message format to OpenRouter format if needed
        if isinstance(messages, str):
            formatted_messages = [{"role": "user", "content": messages}]
        else:
            try:
                # Handle LangChain message objects
                formatted_messages = []
                for msg in messages.messages:
                    role = "system" if msg.type == "system" else "user" if msg.type == "human" else "assistant"
                    formatted_messages.append({"role": role, "content": msg.content})
            except (AttributeError, TypeError):
                # Assume it's already in the right format
                formatted_messages = messages

        # Make the API call
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.site_url,
            "X-OpenRouter-Title": self.site_name,
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            "temperature": self.temperature
        }

        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                data=json.dumps(payload)
            )
            response.raise_for_status()  # Raise exception for HTTP errors

            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # Create a response object similar to what LangChain expects
            return MessageResponse(content)

        except Exception as e:
            print(f"Error calling OpenRouter API: {str(e)}")
            if 'response' in locals() and response:
                print(f"Response: {response.text}")
            raise

class MessageResponse:
    """Simple response class that mimics LangChain's response structure."""

    def __init__(self, content):
        self.content = content


def get_llm(model_name=None, temperature=0.0):
    """Get an LLM instance using OpenRouter."""
    return OpenRouterLLM(
        model_name=model_name or CHAT_MODEL,
        temperature=temperature
    )