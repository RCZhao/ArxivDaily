"""
Utility functions for interacting with Large Language Models (LLMs).
"""
import os
import json
import time
from functools import wraps
import google.generativeai as genai
from openai import OpenAI
from llama_cpp import Llama
from config import (
    LLM_PROVIDER, LLM_API_KEY, LLM_MODEL, LLM_MODEL_PATH, LLM_SYSTEM_PROMPT
)

# Global variable to cache the loaded llama_cpp model
_llama_cpp_model = None

# Rate limiting for Gemini API (free tier: 10 requests per minute)
_last_gemini_call_time = 0
_GEMINI_MIN_INTERVAL = 6.5  # Seconds between calls (slightly more than 60/10 to be safe)

def _rate_limit_gemini():
    """Ensures minimum interval between Gemini API calls to avoid rate limits."""
    global _last_gemini_call_time
    current_time = time.time()
    time_since_last_call = current_time - _last_gemini_call_time
    
    if time_since_last_call < _GEMINI_MIN_INTERVAL:
        sleep_time = _GEMINI_MIN_INTERVAL - time_since_last_call
        print(f"Rate limiting: Sleeping for {sleep_time:.1f}s to avoid Gemini quota...")
        time.sleep(sleep_time)
    
    _last_gemini_call_time = time.time()

def _get_llama_cpp_model():
    """Loads and caches the llama_cpp model."""
    global _llama_cpp_model
    if _llama_cpp_model is None:
        if not LLM_MODEL_PATH or not os.path.exists(LLM_MODEL_PATH):
            raise ValueError(f"llama_cpp model path not configured or not found: {LLM_MODEL_PATH}")
        
        print(f"Loading local LLM from: {LLM_MODEL_PATH}...")
        _llama_cpp_model = Llama(
            model_path=LLM_MODEL_PATH,
            n_ctx=4096,
            n_gpu_layers=-1, # Offload all layers to GPU if possible. Set to 0 for CPU only.
            verbose=False,
            chat_format="chatml" # Qwen2 models use ChatML format.
        )
        print("Local LLM loaded successfully.")
    return _llama_cpp_model

def query_llm(prompt, model_name, temperature=0.2, max_tokens=8192, is_json=False):
    """
    Queries a configured LLM provider and returns the response.

    Args:
        prompt (str): The prompt to send to the LLM.
        model_name (str): The specific model to use for this query (for cloud providers).
        temperature (float): The sampling temperature.
        max_tokens (int): The maximum number of tokens to generate.
        is_json (bool): Whether to expect a JSON response.

    Returns:
        str or dict: The LLM's response, or None if an error occurs.
    """
    provider = LLM_PROVIDER.lower()

    if provider == 'llama_cpp':
        try:
            llm = _get_llama_cpp_model()
            
            messages = []
            if LLM_SYSTEM_PROMPT:
                messages.append({"role": "system", "content": LLM_SYSTEM_PROMPT})
            messages.append({"role": "user", "content": prompt})

            response_format = {"type": "json_object"} if is_json else {"type": "text"}
            
            completion = llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
            content = completion['choices'][0]['message']['content']
            return json.loads(content) if is_json else content.strip()
        except Exception as e:
            print(f"Error querying llama_cpp: {e}")
            return None

    api_key = LLM_API_KEY
    if not api_key or 'YOUR' in api_key:
        print(f"Warning: LLM API key for '{provider}' not configured in config.ini.")
        return None

    if provider == 'openai':
        try:
            client = OpenAI(api_key=api_key)
            messages = []
            if LLM_SYSTEM_PROMPT:
                messages.append({"role": "system", "content": LLM_SYSTEM_PROMPT})
            messages.append({"role": "user", "content": prompt})
            response_format = {"type": "json_object"} if is_json else {"type": "text"}

            completion = client.chat.completions.create(
                model=model_name, messages=messages, temperature=temperature,
                max_tokens=max_tokens, response_format=response_format,
            )
            content = completion.choices[0].message.content
            return json.loads(content) if is_json else content.strip()
        except Exception as e:
            print(f"Error querying OpenAI: {e}")
            return None
    elif provider == 'gemini':
        try:
            # Apply rate limiting before API call
            _rate_limit_gemini()
            
            genai.configure(api_key=api_key)
            
            # Configure safety settings to reduce false positives for academic content
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
            ]
            
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            if is_json:
                generation_config["response_mime_type"] = "application/json"
            
            system_instruction = LLM_SYSTEM_PROMPT if LLM_SYSTEM_PROMPT else None
            model = genai.GenerativeModel(
                model_name, 
                system_instruction=system_instruction,
                safety_settings=safety_settings
            )
            
            response = model.generate_content(prompt, generation_config=generation_config)
            
            # Check if response was blocked
            if not response.candidates:
                print(f"Warning: Gemini returned no candidates. Response may have been blocked.")
                return None
            
            candidate = response.candidates[0]
            
            # Check finish_reason
            # 0: UNSPECIFIED, 1: STOP (normal), 2: MAX_TOKENS, 3: SAFETY, 4: RECITATION, 5: LANGUAGE, 6: OTHER...
            if candidate.finish_reason == 2:  # MAX_TOKENS
                print(f"Warning: Gemini hit max_tokens limit ({max_tokens}). Response truncated.")
                # For some tasks (like summaries), truncated response might still be usable
                # For JSON tasks, this will likely fail parsing
            elif candidate.finish_reason == 3:  # SAFETY
                print(f"Warning: Gemini blocked response due to safety filters.")
                if hasattr(candidate, 'safety_ratings'):
                    print(f"Safety ratings: {candidate.safety_ratings}")
                return None
            elif candidate.finish_reason == 4:  # RECITATION
                print(f"Warning: Gemini blocked response due to recitation concerns.")
                return None
            elif candidate.finish_reason not in [0, 1, 2]:  # Not normal or max_tokens
                print(f"Warning: Gemini stopped with reason code {candidate.finish_reason}")
            
            # Check if content exists
            if not candidate.content or not candidate.content.parts:
                print(f"Warning: Gemini response has no content parts.")
                return None
            
            content = response.text
            
            # Parse JSON with better error handling
            if is_json:
                try:
                    return json.loads(content)
                except json.JSONDecodeError as je:
                    print(f"Warning: Failed to parse Gemini JSON response: {je}")
                    print(f"Raw content (first 500 chars): {content[:500]}")
                    return None
            else:
                return content.strip()
                
        except Exception as e:
            print(f"Error querying Gemini: {e}")
            return None
    else:
        print(f"Unsupported LLM provider: {provider}")
        return None