"""
Customer model for MongoDB collection.
This module provides a class for the customer collection model.
"""

from typing import Dict, Any, Optional
from datetime import datetime, date
import random

from .base_model import BaseModel
from config.test_config import STATUS_VALUES


class Customer(BaseModel):
    """Customer model for MongoDB collection."""
    
    collection_name = "customers"
    
    def __init__(
        self,
        customer_id: Optional[str] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        address: Optional[Dict[str, str]] = None,
        date_of_birth: Optional[date] = None,
        ssn: Optional[str] = None,
        credit_score: Optional[int] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        **kwargs
    ):
        """Initialize a new Customer instance."""
        super().__init__(
            customer_id=customer_id or self.generate_id("CUST_"),
            first_name=first_name,
            last_name=last_name,
            email=email,
            phone=phone,
            address=address or {},
            date_of_birth=date_of_birth,
            ssn=ssn,
            credit_score=credit_score or random.randint(300, 850),
            created_at=created_at or self.current_timestamp(),
            updated_at=updated_at or self.current_timestamp(),
            **kwargs
        )
    
    @property
    def customer_id(self) -> str:
        """Get the customer ID."""
        return self._data["customer_id"]
    
    @property
    def full_name(self) -> str:
        """Get the customer's full name."""
        return f"{self._data['first_name']} {self._data['last_name']}"
    
    @property
    def credit_score(self) -> int:
        """Get the customer's credit score."""
        return self._data["credit_score"]
    
    @credit_score.setter
    def credit_score(self, value: int) -> None:
        """Set the customer's credit score."""
        if not 300 <= value <= 850:
            raise ValueError("Credit score must be between 300 and 850")
        self._data["credit_score"] = value
        self._data["updated_at"] = self.current_timestamp()
    
    def update_address(self, address: Dict[str, str]) -> None:
        """Update the customer's address."""
        required_fields = ["street", "city", "state", "zip", "country"]
        for field in required_fields:
            if field not in address:
                raise ValueError(f"Address must include {field}")
        
        self._data["address"] = address
        self._data["updated_at"] = self.current_timestamp()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the customer to a dictionary for MongoDB storage."""
        # Convert date objects to datetime for MongoDB compatibility
        data = self._data.copy()
        if isinstance(data.get("date_of_birth"), date):
            data["date_of_birth"] = datetime.combine(
                data["date_of_birth"], datetime.min.time()
            )
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Customer':
        """Create a Customer instance from a dictionary."""
        # Convert datetime objects to date for date_of_birth
        if "date_of_birth" in data and isinstance(data["date_of_birth"], datetime):
            data["date_of_birth"] = data["date_of_birth"].date()
        return cls(**data)