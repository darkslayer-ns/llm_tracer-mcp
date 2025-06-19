"""
LLM Response models for structured responses.
"""

import json
import re
from typing import Any, Dict, Optional, Type, TypeVar, Generic
from pydantic import BaseModel, Field

from .base import BaseFrameworkModel

T = TypeVar('T', bound=BaseModel)


class LLMResp(BaseFrameworkModel, Generic[T]):
    """
    Structured LLM response containing raw, clean, and parsed components.
    """
    raw_response: str = Field(..., description="Complete raw response from LLM")
    clean_text: str = Field(..., description="Text-only response without JSON components")
    parsed_data: Optional[T] = Field(None, description="Parsed Pydantic model from JSON")
    json_extracted: Optional[Dict[str, Any]] = Field(None, description="Raw JSON data extracted from response")
    parsing_success: bool = Field(False, description="Whether JSON parsing was successful")
    parsing_error: Optional[str] = Field(None, description="Error message if parsing failed")


class JSONExtractor:
    """
    Utility class for extracting and parsing JSON from LLM responses.
    """
    
    @staticmethod
    def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from text using various patterns.
        
        Args:
            text: Text that may contain JSON
            
        Returns:
            Extracted JSON data or None if not found
        """
        # Pattern 1: JSON wrapped in code blocks
        json_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(json_block_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Pattern 2: JSON object at the start/end of text
        json_object_patterns = [
            r'^\s*(\{.*?\})\s*$',  # Entire text is JSON
            r'(\{.*?\})\s*$',      # JSON at the end
            r'^\s*(\{.*?\})',      # JSON at the start
        ]
        
        for pattern in json_object_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
        
        # Pattern 3: Find any JSON-like structure
        brace_count = 0
        start_idx = -1
        
        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    try:
                        json_str = text[start_idx:i+1]
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        continue
        
        return None
    
    @staticmethod
    def clean_text_from_json(text: str) -> str:
        """
        Remove JSON components from text to get clean text only.
        
        Args:
            text: Original text with potential JSON
            
        Returns:
            Clean text without JSON components
        """
        # Remove JSON code blocks
        text = re.sub(r'```(?:json)?\s*\{.*?\}\s*```', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove standalone JSON objects
        text = re.sub(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', '', text)
        
        # Clean up extra whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r'^\s+|\s+$', '', text)    # Leading/trailing whitespace
        
        return text
    
    @classmethod
    def parse_response(cls, text: str, model_class: Optional[Type[T]] = None) -> LLMResp[T]:
        """
        Parse LLM response into structured format.
        
        Args:
            text: Raw LLM response text
            model_class: Optional Pydantic model class to parse JSON into
            
        Returns:
            Structured LLMResp with parsed components
        """
        # Extract JSON
        json_data = cls.extract_json_from_text(text)
        
        # Clean text
        clean_text = cls.clean_text_from_json(text)
        
        # Parse into model if provided
        parsed_data = None
        parsing_success = False
        parsing_error = None
        
        if json_data and model_class:
            try:
                parsed_data = model_class(**json_data)
                parsing_success = True
            except Exception as e:
                parsing_error = str(e)
        elif json_data:
            parsing_success = True
        
        return LLMResp(
            raw_response=text,
            clean_text=clean_text,
            parsed_data=parsed_data,
            json_extracted=json_data,
            parsing_success=parsing_success,
            parsing_error=parsing_error
        )