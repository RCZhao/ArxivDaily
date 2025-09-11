"""
Utility functions for interacting with Large Language Models (LLMs).
"""
import os
import json
import configparser
import google.generativeai as genai
from openai import OpenAI
from config import CONFIG_FILE

def get_llm_config():
    """Loads LLM configuration from config.ini."""
    config = configparser.ConfigParser()
    if not os.path.exists(CONFIG_FILE):
        return None
    config.read(CONFIG_FILE)
    if 'llm' not in config:
        return None
    return config['llm']

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
    llm_config = get_llm_config()
    if not llm_config:
        print("Warning: [llm] section not found in config.ini. Cannot query LLM.")
        return None

    provider = llm_config.get('provider', 'openai').lower()
    api_key = llm_config.get('api_key')

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