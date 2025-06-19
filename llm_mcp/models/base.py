"""
Base Pydantic models for the generic LLM-MCP framework.
"""

from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class BaseFrameworkModel(BaseModel):
    """
    Base model for all framework models with common configuration.
    """
    class Config:
        # Allow extra fields for maximum flexibility
        extra = "allow"
        # Use enum values instead of names
        use_enum_values = True
        # Validate assignment
        validate_assignment = True
        # Allow arbitrary types for maximum flexibility
        arbitrary_types_allowed = True
        # Populate by name for API compatibility (Pydantic V2)
        populate_by_name = True


class TimestampedModel(BaseFrameworkModel):
    """
    Base model with automatic timestamp tracking.
    """
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    
    def update_timestamp(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.utcnow()


class IdentifiedModel(BaseFrameworkModel):
    """
    Base model with ID tracking for sessions, requests, etc.
    """
    id: str = Field(default_factory=lambda: str(__import__('uuid').uuid4()), description="Unique identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ConfigurableModel(BaseFrameworkModel):
    """
    Base model for configurable components.
    """
    name: str = Field(..., description="Component name")
    description: Optional[str] = Field(None, description="Component description")
    config: Dict[str, Any] = Field(default_factory=dict, description="Configuration parameters")
    enabled: bool = Field(True, description="Whether component is enabled")