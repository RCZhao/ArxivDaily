"""
Utility functions for interacting with Large Language Models (LLMs).
"""
import os
import json
import google.generativeai as genai
from openai import OpenAI
from config import LLM_PROVIDER, LLM_API_KEY

def query_llm(prompt, model_name, temperature=0.2, max_tokens=150, is_json=False):
    """
    Queries a configured LLM provider and returns the response.

    Args:
        prompt (str): The prompt to send to the LLM.
        model_name (str): The specific model to use for this query.
        temperature (float): The sampling temperature.
        max_tokens (int): The maximum number of tokens to generate.
        is_json (bool): Whether to expect a JSON response.

    Returns:
        str or dict: The LLM's response, or None if an error occurs.
    """
    provider = LLM_PROVIDER.lower()
    api_key = LLM_API_KEY

    if not api_key or 'YOUR' in api_key:
        print(f"Warning: LLM API key for '{provider}' not configured in config.ini.")
        return None

    if provider == 'openai':
        try:
            client = OpenAI(api_key=api_key)
            messages = [{"role": "user", "content": prompt}]
            response_format = {"type": "json_object"} if is_json else {"type": "text"}

            completion = client.chat.completions.create(
                model=model_name, messages=messages, temperature=temperature,
                max_tokens=max_tokens, response_format=response_format
            )
            content = completion.choices[0].message.content
            return json.loads(content) if is_json else content.strip()
        except Exception as e:
            print(f"Error querying OpenAI: {e}")
            return None
    elif provider == 'gemini':
        try:
            genai.configure(api_key=api_key)
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            if is_json:
                generation_config["response_mime_type"] = "application/json"
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt, generation_config=generation_config)
            content = response.text
            return json.loads(content) if is_json else content.strip()
        except Exception as e:
            print(f"Error querying Gemini: {e}")
            return None
    else:
        print(f"Unsupported LLM provider: {provider}")
        return None