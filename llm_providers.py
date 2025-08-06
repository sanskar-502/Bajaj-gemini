"""
LLM Provider Interface
Handles different LLM providers (OpenAI, Gemini) with a unified interface
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import google.generativeai as genai
from openai import OpenAI

from config import Config


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM"""
        pass
    
    @abstractmethod
    def generate_structured_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a structured response (JSON) from the LLM"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider implementation"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=api_key)
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response using OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate_structured_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a structured response using OpenAI"""
        try:
            # Add JSON formatting instruction to the prompt
            json_prompt = f"{prompt}\n\nReturn your response as valid JSON only."
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": json_prompt}],
                temperature=0.1,
                **kwargs
            )
            response_text = response.choices[0].message.content
            
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"error": "Could not parse JSON response", "raw_response": response_text}
        except Exception as e:
            return {"error": f"Error generating structured response: {str(e)}"}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI model information"""
        return {
            "provider": "openai",
            "model": self.model,
            "type": "chat",
            "supports_structured_output": True
        }


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider implementation"""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
        self.api_key = api_key
        self.model = model
        genai.configure(api_key=api_key)
        self.model_instance = genai.GenerativeModel(model)
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response using Gemini"""
        try:
            response = self.model_instance.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate_structured_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a structured response using Gemini"""
        try:
            # Add JSON formatting instruction to the prompt
            json_prompt = f"{prompt}\n\nReturn your response as valid JSON only."
            
            response = self.model_instance.generate_content(json_prompt)
            response_text = response.text
            
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"error": "Could not parse JSON response", "raw_response": response_text}
        except Exception as e:
            return {"error": f"Error generating structured response: {str(e)}"}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Gemini model information"""
        return {
            "provider": "gemini",
            "model": self.model,
            "type": "generative",
            "supports_structured_output": True
        }


class LLMFactory:
    """Factory class to create LLM providers"""
    
    @staticmethod
    def create_provider(provider_type: str = None) -> LLMProvider:
        """Create an LLM provider based on configuration"""
        if provider_type is None:
            provider_type = Config.LLM_PROVIDER
        
        if provider_type.lower() == "openai":
            if not Config.OPENAI_API_KEY:
                raise ValueError("OpenAI API key is required")
            return OpenAIProvider(Config.OPENAI_API_KEY, Config.OPENAI_MODEL)
        
        elif provider_type.lower() == "gemini":
            if not Config.GEMINI_API_KEY:
                raise ValueError("Gemini API key is required")
            return GeminiProvider(Config.GEMINI_API_KEY, Config.GEMINI_MODEL)
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_type}")
    
    @staticmethod
    def get_available_providers() -> Dict[str, bool]:
        """Get available LLM providers based on API keys"""
        return {
            "openai": bool(Config.OPENAI_API_KEY),
            "gemini": bool(Config.GEMINI_API_KEY)
        }


# Convenience functions for easy usage
def get_llm_provider(provider_type: str = None) -> LLMProvider:
    """Get an LLM provider instance"""
    return LLMFactory.create_provider(provider_type)


def generate_response(prompt: str, provider_type: str = None, **kwargs) -> str:
    """Generate a response using the specified provider"""
    provider = get_llm_provider(provider_type)
    return provider.generate_response(prompt, **kwargs)


def generate_structured_response(prompt: str, provider_type: str = None, **kwargs) -> Dict[str, Any]:
    """Generate a structured response using the specified provider"""
    provider = get_llm_provider(provider_type)
    return provider.generate_structured_response(prompt, **kwargs) 