"""
Base model for MongoDB collections.
This module provides a base class for all MongoDB collection models.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid


class BaseModel:
    """Base class for all MongoDB collection models."""
    
    collection_name = None
    
    def __init__(self, **kwargs):
        """Initialize a new model instance."""
        self._data = kwargs
    
    @property
    def data(self) -> Dict[str, Any]:
        """Get the model data as a dictionary."""
        return self._data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary for MongoDB storage."""
        return self._data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseModel':
        """Create a model instance from a dictionary."""
        return cls(**data)
    
    @staticmethod
    def generate_id(prefix: str = '') -> str:
        """Generate a unique ID with an optional prefix."""
        return f"{prefix}{uuid.uuid4().hex}"
    
    @staticmethod
    def current_timestamp() -> datetime:
        """Get the current timestamp."""
        return datetime.utcnow()